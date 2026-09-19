# The backend vocabulary

Every name a shared body may use whose correct spelling differs by environment,
listed once. This file is the contract; `<backend>/backend.hpp` and
`<backend>/dispatch.hpp` are one implementation of it per backend, and the
build selects which one by include path.

## The rules this file exists to enforce

1. **A shared source file contains no environment conditional and no
   define-if-absent fallback.** It opens with one unconditional
   `#include <backend.hpp>` and spells every name below through the
   vocabulary.
2. **A wrapper defines every name here.** There is no fallback, so a name a
   wrapper omits is a compile error naming the missing symbol at its first use,
   which is the earliest point it can be caught. A default would instead hand
   the omitting backend whatever the default happened to be, which is a silent
   fallback in the one place this project says never to have one.
3. **Environment branching that a backend genuinely needs lives inside that
   backend's wrapper**, and nowhere else. CUDA's `__CUDA_ARCH__` split between
   the host and device passes of a `__host__ __device__` function is the known
   case; it belongs in `cuda/backend.hpp`.
4. **Adding a name here means adding it to every wrapper**, in the same change.
   CI checks the wrappers against this list rather than waiting for a build to
   discover a gap.

Angle brackets in the include are load-bearing: a quoted include resolves
relative to the including file first, so it would find a neighboring header
rather than the selected backend.

## Execution space and linkage

| name | meaning |
|---|---|
| `SM_INLINE` | inline function visible to device code |
| `SM_INLINE_DEVICE` | inline function reachable only from device code |
| `SM_DEVICE` | qualifier for a pointer or reference into the backend's global memory |
| `SM_THREAD` | qualifier for a pointer or reference into thread-private memory |
| `SM_THREADGROUP` | qualifier for a pointer or reference into threadgroup-shared memory |
| `SM_CONSTANT` | a compile-time constant that must remain a variable |

On CUDA and on the host the three address-space qualifiers expand to nothing.
They are not decoration: Metal rejects an unqualified pointer in a device-visible
signature, and a shared body that omits them stops compiling there. Keeping them
in the CUDA and CPU builds is what makes that a compile error on one backend
rather than a divergence between two.

## Math

Every entry is float-throughout. GPU compute in this project is single
precision only, and that rule reaches whatever a wrapper supplies here: `sinf`,
`cosf`, `expf` and `logf`
reduce or scale their argument in double, so a kernel that merely calls one
emits double-precision instructions with no `double` in the source. The CUDA
wrapper therefore routes those five through `float_math.hpp`, and the release
build's FP64 SASS guard fails on any that slips.

| name | meaning |
|---|---|
| `SM_ABS(x)` | absolute value |
| `SM_MIN(a, b)` / `SM_MAX(a, b)` | minimum / maximum |
| `SM_SQRT(x)` | square root |
| `SM_DIV(a, b)` | division |
| `SM_FMA(a, b, c)` | fused multiply-add, which must genuinely fuse |
| `SM_ACOS(x)` / `SM_ATAN2(y, x)` | inverse trigonometry, arguments bounded by construction |
| `SM_COS(x)` | cosine |
| `SM_SIN_BOUNDED(x)` / `SM_COS_BOUNDED(x)` | sine / cosine with `abs(x)` asserted inside the bounded range |
| `SM_SIN_PERIODIC(x)` | sine after argument reduction |
| `SM_EXP(x)` / `SM_LOG(x)` | exponential / logarithm |
| `SM_ISNAN(x)` / `SM_ISINF(x)` / `SM_ISFINITE(x)` | classification. Never spell the `x != x` idiom, which fast math deletes |
| `SM_NEXTAFTER(a, b)` | next representable value |
| `SM_INFINITY` | positive infinity |
| `SM_NUMERIC_MAX(T)` | largest finite value of `T` |

**The backends compute different values at the transcendental entries**, at
roughly 2^-21 relative on CUDA's special-function units and roughly 2^-24 in a
host libm. That is a characterized-tolerance class declared in advance rather
than a finding to adjudicate later; see plan Decision 2. Every other entry above
is bit-exact comparable when the operation order is fixed.

**The declared figure is the error AT the seam, and what a caller delivers can be
far larger.** This was measured rather than assumed, and it is the one thing the
class above does not tell a reader. `SM_COS_BOUNDED` is `__cosf` on CUDA and
`std::cos` in the host libm, about 3.05e-07 apart, comfortably inside the 2^-21
budget. `linalg::eig::symm3x3` calls it to build a symmetric eigendecomposition
in closed form, and the tet elastic path decomposes a table whose spectrum spans
five orders. The seam's 3.05e-07 arrives in the fused 12x12 Hessian as 3.02e-04:
a thousandfold amplification, entirely from the conditioning of the consumer.

Two consequences for anyone adding a seam or a body:

- **A body that reaches a transcendental seam cannot be gated bit-exact**, and
  its tolerance is a property of the body rather than of the seam. Measure it;
  do not derive it from the table above.
- **The gap is not closable at the wrapper.** Pointing the CUDA side at the host
  libm's cosine does close it, from 3.02e-04 to 3.27e-07 in that Hessian, and it
  injects FP64 into the device binary: measured on an sm_89 build of
  `test_tet_stage_parity.cu`, 0 FP64 instructions with `__cosf` against 4 with
  `std::cos`. The release build's FP64 SASS guard rejects that, so the accurate
  cosine is unavailable on the device by a rule that outranks parity.

## Bit manipulation

| name | meaning |
|---|---|
| `SM_CLZ(v)` | count leading zeros of a 32-bit unsigned, 32 for an input of 0 |
| `SM_POPCOUNT(v)` | population count of a 32-bit unsigned |
| `SM_MULHI(a, b)` | high 32 bits of a signed 32x32 product |

`SM_MULHI` returns the high 32 bits of a signed 32-by-32 product, which is the
part a 32-bit multiply discards. It is one of the names the Metal prologue
defines that no shared body names today, so the first body to use it is also
the first test of it on the other backends.

## Cross-lane and threadgroup, required only of a backend that HAS lanes

| name | meaning |
|---|---|
| `SM_SIMD_WIDTH` | lanes per SIMD group |
| `SM_SHUFFLE_DOWN(v, o)` / `SM_SHUFFLE_UP(v, o)` | lane shuffle |
| `SM_SIMD_BALLOT(pred)` | ballot mask, valid only in uniform control flow |
| `SM_THREADGROUP_BARRIER()` | threadgroup barrier |

**A backend with no lanes supplies none of these and says so**, with
`#define BACKEND_HAS_LANES 0`. The contract check then exempts this group for
that wrapper, and reaching one of these names from such a build is a compile
error naming the missing macro. That is the correct diagnosis rather than an
inconvenience: it says you have reached a cross-lane primitive from a build that
has no lanes.

**Defining them at width 1 instead is a silent wrong answer, and this is
measured rather than argued.** `warp_reduce` loops
`for (offset = SM_SIMD_WIDTH / 2; offset > 0; offset >>= 1)`, so at width 1 the
loop body never runs and the function returns its input UNREDUCED. It compiles,
it runs, and it yields a plausible number that is not a reduction. The CPU
wrapper shipped exactly that for one revision.

## Atomics, split by what the caller does with the result

An **accumulate** folds a contribution into a total and nothing reads the
previous value. A **claim** exists so the returned slot is unique. The
distinction is invisible on CUDA, where both are `atomicAdd`, and decisive on a
backend whose accumulate is a plain `+=`.

| name | kind | meaning |
|---|---|---|
| `SM_ATOMIC_UINT` / `SM_ATOMIC_FLOAT` | type | the atomic-qualified scalar types |
| `SM_ATOMIC_ADD_UINT(p, v)` / `SM_ATOMIC_ADD_FLOAT(p, v)` | accumulate | fold a contribution into a total; the result is not read |
| `SM_ATOMIC_LOAD_UINT(p)` / `SM_ATOMIC_STORE_UINT(p, v)` | access | atomic load / store |
| `SM_ATOMIC_FETCH_ADD_UINT(p, v)` | **claim** | reserve a slot and RETURN it; must be a real atomic on every backend |

The CPU backend's assembly is partitioned by destination row (plan Decision 4),
so no two threads fold into the same total and the accumulates are sound as
plain operations. **That partition cannot make a claim safe**, because the
contention is over the counter rather than over a row: two threads would receive
the same slot, one contribution would be lost, the count would under-report, and
an overflow latch keyed on it might never fire. The contact pair cache, the
intersection record ring and `Row::push`'s head bump are all claims.

**Still missing from this vocabulary**, and each sits textually inside a body
that is to be shared, so the split is not yet available for them:
`atomicCAS` (`contact.cu:49`, `lbvh.cu:189`, `:204`), `atomicMin`
(`contact.cu:2378`, `:2389`), `atomicMax` (`lbvh.cu:465`,
`translation_lock.hpp:1011`) and `atomicSub` (`schwarz.cu:150`). Each is
resolved by hoisting the latch to an out-parameter in the CUDA source, as
`dev-mac-metal-port` already did for `accd::OverlapInfo`, or by growing this
list. Do the first where the value is a diagnostic and the second where it is
part of the algorithm.

## Elementwise dispatch, in `<backend>/dispatch.hpp`

The second header the include path selects, and the only part of the contract
that is not a `SM_` name. It is separate from `backend.hpp` because it is
the one entry a wrapper answers with a control-flow skeleton rather than a
spelling, and because `backend.hpp` must stay includable from anywhere,
including from headers this seam depends on.

Shared source reaches it through one unconditional `#include "dispatcher.hpp"`,
which is `#include <dispatch.hpp>` and nothing else.

| name | meaning |
|---|---|
| `DISPATCH_START(n)` | open a dispatch over `[0, n)`; the call site follows it with the body as a lambda |
| `DISPATCH_END` | terminate the lambda, run the body over every index, close the scope |
| `DISPATCH_QUEUE_START(n, q)` | the same, on a caller-owned queue handle `q` |
| `DISPATCH_QUEUE_END` | terminate, submit, and return WITHOUT waiting: the caller owns ordering |

The four impose one shape on a call site, so the lambda between START and END is
a single body that every backend compiles. `START` leaves `auto kernel =`
dangling and `END` supplies the terminating `;`, which is why the two halves are
macros rather than a function taking a callable: the body has to be written
inline at the call site to keep the captures readable.

Two properties a wrapper owes:

- **`DISPATCH_END` must have completed the work when it returns**, so a caller
  may read the result on the host on the next line. CUDA gets this from a
  stream synchronize after an asynchronous launch; a serial loop has it for
  free.
- **`DISPATCH_QUEUE_END` may return with the work outstanding**, which is the
  entire reason the queue form exists (the PCG fast loop chains a whole CG
  iteration with no host round-trip). A backend that runs the body immediately
  satisfies this too, by delivering more than was asked for. What a backend may
  NOT do is drop the ordering: if it defers, the handle has to carry the
  dependency.

**The CPU dispatch runs the body serially, in ascending index order, and that is
a decision rather than a first cut.** It makes a CPU run reproducible bit for
bit where a CUDA run is not, which is what lets it serve as an oracle; and it is
sound under either atomic vocabulary above, including for the contact kernels
indexed by PAIR, which have no row-owner partition to make a plain `+=` safe.
Parallelizing it is therefore not a local change to that one macro.
