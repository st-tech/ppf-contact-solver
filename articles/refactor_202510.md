### 🔄 Refactor Summary (October 2025)

---

This note summarizes the refactor made in October 2025.

- **🔧 Reliability**: Custom CUDA kernels replace problematic Thrust dependency
- **🎯 Per-Object Parameters**: Parameters can now be assigned per object instead of globally
- **📐 Code Quality**: Better code organization and centralized parameter management

### 🎮 Custom CUDA Kernels

NVIDIA Thrust library reports runtime errors with driver version `560.94+`.
A set of custom CUDA kernels are implemented as workaround in `crates/ppf-cts-solver/src/cpp/kernels/`:

- `reduce.cu` / `reduce.hpp` - Reduction operations ⚡
- `exclusive_scan.cu` / `exclusive_scan.hpp` - Prefix sum operations 🔢
- `vec_ops.cu` / `vec_ops.hpp` - Vector operations 📊

Now the `use-thrust` option has been removed.
Our internal benchmarks show that the new kernels perform as efficiently as thrust for our specific use cases. ⚡

### 🧩 Parameter System Refactor

- Parameters can now be defined per object (e.g., Young's modulus, contact gaps) instead of globally 🎯
- ⚠️ This introduces breaking changes to our Python APIs

### 🐳 Docker Improvements

- Python packages now install locally to `~/.local/share/ppf-cts/venv` instead of globally 🐍

### 📊 Code Metrics


| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Rust source | `3,426` lines | `3,891` lines | `+13.6%` 📈 |
| Python frontend | `6,161` lines | `7,908` lines | `+28.4%` 🚀 |
| CUDA source | `~15KB` | `~39KB` | `+160%` 🎮 |


---

*Last Updated: October 3, 2025* 📅
