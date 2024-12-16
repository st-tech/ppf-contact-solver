// File: cuda_version.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include <cuda.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    printf("%d", runtime_version);
    return 0;
}
