// File: reduce.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../common.hpp"
#include "../buffer/buffer.hpp"
#include "reduce.hpp"

namespace kernels {

__host__ __device__ inline unsigned umin(unsigned a, unsigned b) {
    return (a < b) ? a : b;
}

template <typename T, typename Op> __device__ T warp_reduce(T val, Op func) {
    for (unsigned offset = 16; offset > 0; offset /= 2) {
        val = func(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T, typename Op>
__device__ T block_reduce(T val, Op func, T init_val) {
    const unsigned WARPS_PER_BLOCK = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const unsigned MAX_WARPS = 8; // 256 threads max / 32 = 8 warps
    __shared__ T warp_results[MAX_WARPS];

    const unsigned lane_id = threadIdx.x % WARP_SIZE;
    const unsigned warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce(val, func);

    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    val =
        (threadIdx.x < WARPS_PER_BLOCK) ? warp_results[threadIdx.x] : init_val;
    if (warp_id == 0) {
        val = warp_reduce(val, func);
    }

    return val;
}

template <class Y, typename Op>
__global__ void reduce_op_kernel_1(const Y *input, Y *output, Op func,
                                   Y init_val, unsigned n) {
    Y thread_sum = init_val;

    unsigned tid = threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;
    for (unsigned i = global_idx; i < n; i += grid_size) {
        thread_sum = func(thread_sum, input[i]);
    }
    Y block_sum = block_reduce(thread_sum, func, init_val);
    if (tid == 0) {
        output[blockIdx.x] = block_sum;
    }
}

template <class Y, typename Op>
__global__ void reduce_op_kernel_2(const Y *input1, const Y *input2, Y *output,
                                   Op func, Y init_val, unsigned n) {
    Y thread_sum = init_val;

    unsigned tid = threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;
    for (unsigned i = global_idx; i < n; i += grid_size) {
        thread_sum = func(thread_sum, input1[i], input2[i]);
    }
    auto add_func = [=] __device__(Y a, Y b) { return a + b; };
    Y block_sum = block_reduce(thread_sum, add_func, init_val);
    if (tid == 0) {
        output[blockIdx.x] = block_sum;
    }
}

// Forward declaration of generic template
template <typename T>
__global__ void inner_product_kernel_optimized(const T *input1, const T *input2, T *output, unsigned n);

// Optimized inner product kernel for float using vectorized loads and FMA
template <>
__global__ void inner_product_kernel_optimized<float>(const float *input1, const float *input2, float *output, unsigned n) {
    float thread_sum = 0.0f;

    unsigned tid = threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;

    const float2 *vec_input1 = reinterpret_cast<const float2*>(input1);
    const float2 *vec_input2 = reinterpret_cast<const float2*>(input2);

    unsigned vec_n = n / 8;

    for (unsigned i = global_idx; i < vec_n; i += grid_size) {
        unsigned base = i * 4;  // Each thread processes 4 float2 values (8 floats)

        float2 a0 = vec_input1[base];
        float2 a1 = vec_input1[base + 1];
        float2 a2 = vec_input1[base + 2];
        float2 a3 = vec_input1[base + 3];

        float2 b0 = vec_input2[base];
        float2 b1 = vec_input2[base + 1];
        float2 b2 = vec_input2[base + 2];
        float2 b3 = vec_input2[base + 3];

        thread_sum = __fmaf_rn(a0.x, b0.x, thread_sum);
        thread_sum = __fmaf_rn(a0.y, b0.y, thread_sum);
        thread_sum = __fmaf_rn(a1.x, b1.x, thread_sum);
        thread_sum = __fmaf_rn(a1.y, b1.y, thread_sum);
        thread_sum = __fmaf_rn(a2.x, b2.x, thread_sum);
        thread_sum = __fmaf_rn(a2.y, b2.y, thread_sum);
        thread_sum = __fmaf_rn(a3.x, b3.x, thread_sum);
        thread_sum = __fmaf_rn(a3.y, b3.y, thread_sum);
    }

    unsigned processed = vec_n * 8;
    for (unsigned i = processed + global_idx; i < n; i += grid_size) {
        thread_sum = __fmaf_rn(input1[i], input2[i], thread_sum);
    }

    auto add_func = [] __device__(float a, float b) { return a + b; };
    float block_sum = block_reduce(thread_sum, add_func, 0.0f);

    if (tid == 0) {
        output[blockIdx.x] = block_sum;
    }
}

// Generic version for other types
template <typename T>
__global__ void inner_product_kernel_optimized(const T *input1, const T *input2, T *output, unsigned n) {
    T thread_sum = T(0);

    unsigned tid = threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;

    unsigned vec_n = n / 4;

    for (unsigned i = global_idx; i < vec_n; i += grid_size) {
        unsigned base = i * 4;
        T sum0 = input1[base] * input2[base];
        T sum1 = input1[base + 1] * input2[base + 1];
        T sum2 = input1[base + 2] * input2[base + 2];
        T sum3 = input1[base + 3] * input2[base + 3];

        thread_sum += sum0 + sum1 + sum2 + sum3;
    }

    unsigned processed = vec_n * 4;
    for (unsigned i = processed + global_idx; i < n; i += grid_size) {
        thread_sum += input1[i] * input2[i];
    }

    auto add_func = [] __device__(T a, T b) { return a + b; };
    T block_sum = block_reduce(thread_sum, add_func, T(0));

    if (tid == 0) {
        output[blockIdx.x] = block_sum;
    }
}

template <typename Y, typename Op>
__global__ void final_reduce_kernel(Y *data, Y *output, Op func, Y init_val, unsigned n) {
    Y thread_sum = init_val;
    unsigned tid = threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;
    for (unsigned i = global_idx; i < n; i += grid_size) {
        thread_sum = func(thread_sum, data[i]);
    }
    Y result = block_reduce(thread_sum, func, init_val);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = result;
    }
}

template <class Y, typename Op1, typename Op2>
Y reduce(const Y *d_input1, const Y *d_input2, Op1 func1, Op2 func2, Y init_val,
         unsigned n) {
    if (n == 0) {
        return init_val;
    }

    const unsigned max_blocks = 1024;

    // Allocate buffers from pool
    buffer::MemoryPool &pool = buffer::get();
    auto block_sums = pool.get<Y>(max_blocks);
    auto temp = pool.get<Y>(max_blocks);
    auto final_result = pool.get<Y>(1);

    Y result;

    unsigned block_size = choose_block_size(n);
    const unsigned blocks_needed = (n + block_size - 1) / block_size;
    unsigned grid_size = umin(blocks_needed, max_blocks);

    Y *d_output = block_sums.data;

    if (d_input2) {
        reduce_op_kernel_2<Y><<<grid_size, block_size>>>(
            d_input1, d_input2, d_output, func2, init_val, n);
    } else {
        reduce_op_kernel_1<Y>
            <<<grid_size, block_size>>>(d_input1, d_output, func1, init_val, n);
    }

    if (grid_size == 1) {
        cudaMemcpy(&result, d_output, sizeof(Y), cudaMemcpyDeviceToHost);
        return result;
    }

    if (grid_size <= block_size) {
        Y *d_final_result = final_result.data;
        unsigned final_block_size = choose_block_size(grid_size);
        final_reduce_kernel<Y><<<1, final_block_size>>>(
            d_output, d_final_result, func1,
            init_val, grid_size);

        cudaMemcpy(&result, d_final_result, sizeof(Y), cudaMemcpyDeviceToHost);
    } else {
        Y *d_temp = temp.data;
        cudaMemcpy(d_temp, d_output, grid_size * sizeof(Y),
                   cudaMemcpyDeviceToDevice);

        while (grid_size > 1) {
            unsigned loop_block_size = choose_block_size(grid_size);
            unsigned new_grid_size =
                umin((grid_size + loop_block_size - 1) / loop_block_size,
                     max_blocks);
            reduce_op_kernel_1<Y><<<new_grid_size, loop_block_size>>>(
                d_temp, d_output, func1,
                init_val, grid_size);

            if (new_grid_size == 1) {
                break;
            }

            cudaMemcpy(d_temp, d_output, new_grid_size * sizeof(Y),
                       cudaMemcpyDeviceToDevice);
            grid_size = new_grid_size;
        }

        cudaMemcpy(&result, d_output, sizeof(Y), cudaMemcpyDeviceToHost);
    }

    return result;
}

template <class Y, typename Op>
Y reduce_1(const Y *d_input, Op func, Y init_val, unsigned n) {
    auto dummy = [] __device__(Y a, Y b, Y c) { return a; };
    return reduce<Y>(d_input, nullptr, func, dummy, init_val, n);
}

template <class Y, typename Op>
Y reduce_2(const Y *d_input1, const Y *d_input2, Op func, Y init_val,
           unsigned n) {
    auto add_func = [] __device__(Y a, Y b) { return a + b; };
    return reduce<Y>(d_input1, d_input2, add_func, func, init_val, n);
}

template <class T>
T inner_product(const T *array1, const T *array2, unsigned size) {
    if (size == 0) return T();

    unsigned block_size = choose_block_size(size);
    unsigned max_blocks = 1024;
    unsigned grid_size = umin((size + block_size - 1) / block_size, max_blocks);

    // Allocate buffers from pool
    buffer::MemoryPool &pool = buffer::get();
    auto block_sums = pool.get<T>(max_blocks);
    auto temp = pool.get<T>(max_blocks);
    auto final_result = pool.get<T>(1);

    T *d_output = block_sums.data;

    inner_product_kernel_optimized<T><<<grid_size, block_size>>>(
        array1, array2, d_output, size);

    T result;
    if (grid_size == 1) {
        cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
        return result;
    }

    T *d_final_result = final_result.data;
    if (grid_size <= block_size) {
        unsigned final_block_size = choose_block_size(grid_size);
        auto add_func = [] __device__(T a, T b) { return a + b; };
        final_reduce_kernel<T><<<1, final_block_size>>>(
            d_output, d_final_result, add_func, T(), grid_size);
        cudaMemcpy(&result, d_final_result, sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        T *d_temp = temp.data;
        cudaMemcpy(d_temp, d_output, grid_size * sizeof(T), cudaMemcpyDeviceToDevice);

        while (grid_size > 1) {
            unsigned loop_block_size = choose_block_size(grid_size);
            unsigned new_grid_size = umin((grid_size + loop_block_size - 1) / loop_block_size, max_blocks);
            auto add_func = [] __device__(T a, T b) { return a + b; };
            reduce_op_kernel_1<T><<<new_grid_size, loop_block_size>>>(
                d_temp, d_output, add_func, T(), grid_size);

            if (new_grid_size == 1) break;

            cudaMemcpy(d_temp, d_output, new_grid_size * sizeof(T), cudaMemcpyDeviceToDevice);
            grid_size = new_grid_size;
        }
        cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
    }

    return result;
}

template <class T> T sum_array(const T *array, unsigned size) {
    return reduce_1<T>(
        array, [] __device__(T sum, T val) { return sum + val; }, T(), size);
}

template <class T> T min_array(const T *array, unsigned size, T init_val) {
    return reduce_1<T>(
        array,
        [] __device__(T min_val, T val) {
            return val < min_val ? val : min_val;
        },
        init_val, size);
}

template <class T> T max_array(const T *array, unsigned size, T init_val) {
    return reduce_1<T>(
        array,
        [] __device__(T max_val, T val) {
            return val > max_val ? val : max_val;
        },
        init_val, size);
}

// Reduce `count` partial sums held in `cur` down to a single value written to
// the caller's device pointer `out`, ping-ponging between the `cur` and `nxt`
// scratch buffers (each sized >= max_blocks). The host reduce path copies
// partials back with a device-to-device cudaMemcpy between passes, which
// synchronizes the host; swapping pointers instead keeps every pass on `queue`
// with no host round-trip. Kept at namespace (external-linkage) scope, like the
// other reductions here, because nvcc requires that for a function whose
// extended __device__ lambda instantiates a __global__ template.
void reduce_partials_into(float *cur, float *nxt, unsigned count, float *out,
                          cudaStream_t queue) {
    const unsigned max_blocks = 1024;
    auto add_func = [] __device__(float a, float b) { return a + b; };
    while (count > 1) {
        unsigned block_size = choose_block_size(count);
        unsigned new_grid =
            umin((count + block_size - 1) / block_size, max_blocks);
        float *dst = (new_grid == 1) ? out : nxt;
        reduce_op_kernel_1<float><<<new_grid, block_size, 0, queue>>>(
            cur, dst, add_func, 0.0f, count);
        CUDA_HANDLE_ERROR(cudaGetLastError());
        float *swap = cur;
        cur = nxt;
        nxt = swap;
        count = new_grid;
    }
}

void inner_product_into(const float *a, const float *b, float *out, unsigned n,
                        cudaStream_t queue) {
    if (n == 0) {
        CUDA_HANDLE_ERROR(cudaMemsetAsync(out, 0, sizeof(float), queue));
        return;
    }
    const unsigned max_blocks = 1024;
    unsigned block_size = choose_block_size(n);
    unsigned grid_size = umin((n + block_size - 1) / block_size, max_blocks);

    if (grid_size == 1) {
        // Single block: the optimized kernel writes its block sum straight to
        // the caller's device scalar.
        inner_product_kernel_optimized<float>
            <<<1, block_size, 0, queue>>>(a, b, out, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
        return;
    }

    buffer::MemoryPool &pool = buffer::get();
    auto scratch_a = pool.get<float>(max_blocks);
    auto scratch_b = pool.get<float>(max_blocks);
    inner_product_kernel_optimized<float>
        <<<grid_size, block_size, 0, queue>>>(a, b, scratch_a.data, n);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    reduce_partials_into(scratch_a.data, scratch_b.data, grid_size, out, queue);
}

void sum_into(const float *in, float *out, unsigned n, cudaStream_t queue) {
    if (n == 0) {
        CUDA_HANDLE_ERROR(cudaMemsetAsync(out, 0, sizeof(float), queue));
        return;
    }
    const unsigned max_blocks = 1024;
    unsigned block_size = choose_block_size(n);
    unsigned grid_size = umin((n + block_size - 1) / block_size, max_blocks);
    auto add_func = [] __device__(float a, float b) { return a + b; };

    if (grid_size == 1) {
        reduce_op_kernel_1<float>
            <<<1, block_size, 0, queue>>>(in, out, add_func, 0.0f, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
        return;
    }

    buffer::MemoryPool &pool = buffer::get();
    auto scratch_a = pool.get<float>(max_blocks);
    auto scratch_b = pool.get<float>(max_blocks);
    reduce_op_kernel_1<float>
        <<<grid_size, block_size, 0, queue>>>(in, scratch_a.data, add_func,
                                              0.0f, n);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    reduce_partials_into(scratch_a.data, scratch_b.data, grid_size, out, queue);
}

template float sum_array(const float *array, unsigned size);
template unsigned sum_array(const unsigned *array, unsigned size);
template float min_array(const float *array, unsigned size, float init_val);
template float max_array(const float *array, unsigned size, float init_val);
template char min_array(const char *array, unsigned size, char init_val);
template char max_array(const char *array, unsigned size, char init_val);
template unsigned min_array(const unsigned *array, unsigned size,
                            unsigned init_val);
template unsigned max_array(const unsigned *array, unsigned size,
                            unsigned init_val);
template float inner_product(const float *array1, const float *array2,
                             unsigned size);

} // namespace kernels
