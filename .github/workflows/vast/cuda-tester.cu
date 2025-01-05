#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 1024; // Vector size
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc A");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc B");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc C");

    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy B");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization");

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy result");

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Verification failed at index %d: %f != %f + %f\n", i,
                   h_C[i], h_A[i], h_B[i]);
            success = false;
            break;
        }
    }
    checkCudaError(cudaFree(d_A), "cudaFree A");
    checkCudaError(cudaFree(d_B), "cudaFree B");
    checkCudaError(cudaFree(d_C), "cudaFree C");
    free(h_A);
    free(h_B);
    free(h_C);

    if (success) {
        printf("CUDA test passed successfully!\n");
    }

    return 0;
}
