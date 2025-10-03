// Comprehensive test for reduce.cu operations
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include "../reduce.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test configurations
struct TestConfig {
    std::vector<unsigned> sizes;
    bool verbose;

    TestConfig() : verbose(false) {
        // Comprehensive size list: small, boundaries, and large
        sizes = {
            1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
            511, 512, 513, 1023, 1024, 1025, 2048, 4096, 8192, 16384,
            32768, 65536, 131072, 262144, 524288, 1048576, 10000000
        };
    }
};

// Test sum operation
bool test_sum(const TestConfig& config) {
    std::cout << "\nTesting SUM operation:" << std::endl;
    bool all_passed = true;

    for (unsigned size : config.sizes) {
        std::vector<float> h_data(size);
        double expected = 0;

        // Fill with simple pattern
        for (unsigned i = 0; i < size; ++i) {
            h_data[i] = 1.0f + (i % 100) * 0.01f;
            expected += h_data[i];
        }

        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        float result = kernels::sum_array(d_data, size);

        double rel_error = (expected > 0) ? std::abs((result - expected) / expected) : 0;
        bool passed = rel_error < 1e-4;  // 0.01% tolerance

        if (!passed || config.verbose) {
            std::cout << "  Size " << std::setw(8) << size << ": "
                      << (passed ? "PASS" : "FAIL")
                      << " (expected=" << expected << ", got=" << result
                      << ", error=" << (rel_error * 100) << "%)" << std::endl;
        }

        all_passed &= passed;
        CUDA_CHECK(cudaFree(d_data));
    }

    if (!config.verbose && all_passed) {
        std::cout << "  All " << config.sizes.size() << " tests PASSED" << std::endl;
    }

    return all_passed;
}

// Test min/max operations
bool test_min_max(const TestConfig& config) {
    std::cout << "\nTesting MIN/MAX operations:" << std::endl;
    bool all_passed = true;

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);

    for (unsigned size : config.sizes) {
        std::vector<float> h_data(size);
        float expected_min = 1e9f;
        float expected_max = -1e9f;

        for (unsigned i = 0; i < size; ++i) {
            h_data[i] = dis(gen);
            expected_min = std::min(expected_min, h_data[i]);
            expected_max = std::max(expected_max, h_data[i]);
        }

        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        float min_result = kernels::min_array(d_data, size, 1e9f);
        float max_result = kernels::max_array(d_data, size, -1e9f);

        bool min_passed = std::abs(min_result - expected_min) < 1e-5;
        bool max_passed = std::abs(max_result - expected_max) < 1e-5;
        bool passed = min_passed && max_passed;

        if (!passed || config.verbose) {
            std::cout << "  Size " << std::setw(8) << size << ": "
                      << (passed ? "PASS" : "FAIL")
                      << " (min: " << min_result << "/" << expected_min
                      << ", max: " << max_result << "/" << expected_max << ")" << std::endl;
        }

        all_passed &= passed;
        CUDA_CHECK(cudaFree(d_data));
    }

    if (!config.verbose && all_passed) {
        std::cout << "  All " << config.sizes.size() << " tests PASSED" << std::endl;
    }

    return all_passed;
}

// Test inner product operation
bool test_inner_product(const TestConfig& config) {
    std::cout << "\nTesting INNER PRODUCT operation:" << std::endl;
    bool all_passed = true;

    for (unsigned size : config.sizes) {
        std::vector<float> h_data1(size);
        std::vector<float> h_data2(size);
        double expected = 0;

        // Fill with pattern
        for (unsigned i = 0; i < size; ++i) {
            h_data1[i] = 1.0f + (i % 50) * 0.02f;
            h_data2[i] = 0.5f + (i % 25) * 0.02f;
            expected += static_cast<double>(h_data1[i]) * static_cast<double>(h_data2[i]);
        }

        float *d_data1, *d_data2;
        CUDA_CHECK(cudaMalloc(&d_data1, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data2, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data1, h_data1.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data2, h_data2.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        float result = kernels::inner_product(d_data1, d_data2, size);

        double rel_error = (expected > 0) ? std::abs((result - expected) / expected) : 0;
        bool passed = rel_error < 1e-3;  // 0.1% tolerance

        if (!passed || config.verbose) {
            std::cout << "  Size " << std::setw(8) << size << ": "
                      << (passed ? "PASS" : "FAIL")
                      << " (expected=" << expected << ", got=" << result
                      << ", error=" << (rel_error * 100) << "%)" << std::endl;
        }

        all_passed &= passed;
        CUDA_CHECK(cudaFree(d_data1));
        CUDA_CHECK(cudaFree(d_data2));
    }

    if (!config.verbose && all_passed) {
        std::cout << "  All " << config.sizes.size() << " tests PASSED" << std::endl;
    }

    return all_passed;
}

// Test edge cases
bool test_edge_cases() {
    std::cout << "\nTesting EDGE CASES:" << std::endl;
    bool all_passed = true;

    // Test 1: Empty array
    {
        float *d_null = nullptr;
        float result = kernels::sum_array(d_null, 0);
        bool passed = (result == 0.0f);
        std::cout << "  Empty array: " << (passed ? "PASS" : "FAIL")
                  << " (got=" << result << ")" << std::endl;
        all_passed &= passed;
    }

    // Test 2: Single element
    {
        float h_val = 42.0f;
        float *d_val;
        CUDA_CHECK(cudaMalloc(&d_val, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_val, &h_val, sizeof(float), cudaMemcpyHostToDevice));

        float sum = kernels::sum_array(d_val, 1);
        float min = kernels::min_array(d_val, 1, 100.0f);
        float max = kernels::max_array(d_val, 1, -100.0f);

        bool passed = (sum == 42.0f && min == 42.0f && max == 42.0f);
        std::cout << "  Single element: " << (passed ? "PASS" : "FAIL")
                  << " (sum=" << sum << ", min=" << min << ", max=" << max << ")" << std::endl;
        all_passed &= passed;

        CUDA_CHECK(cudaFree(d_val));
    }

    // Test 3: All same values
    {
        unsigned size = 1000;
        std::vector<float> h_data(size, 7.5f);
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        float sum = kernels::sum_array(d_data, size);
        float min = kernels::min_array(d_data, size, 100.0f);
        float max = kernels::max_array(d_data, size, -100.0f);

        bool passed = (std::abs(sum - 7500.0f) < 1e-3) && (min == 7.5f) && (max == 7.5f);
        std::cout << "  All same (1000x7.5): " << (passed ? "PASS" : "FAIL")
                  << " (sum=" << sum << ", min=" << min << ", max=" << max << ")" << std::endl;
        all_passed &= passed;

        CUDA_CHECK(cudaFree(d_data));
    }

    return all_passed;
}

int main(int argc, char** argv) {
    // Check for verbose flag
    TestConfig config;
    config.verbose = (argc > 1 && std::string(argv[1]) == "-v");

    std::cout << "=== REDUCE.CU COMPREHENSIVE TEST ===" << std::endl;
    std::cout << "Testing with " << config.sizes.size() << " different sizes" << std::endl;
    std::cout << "Largest size: " << config.sizes.back() << " elements" << std::endl;

    if (config.verbose) {
        std::cout << "Verbose mode: showing all test results" << std::endl;
    }

    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;

    bool all_passed = true;

    // Run all tests
    all_passed &= test_sum(config);
    all_passed &= test_min_max(config);
    all_passed &= test_inner_product(config);
    all_passed &= test_edge_cases();

    // Final result
    std::cout << "\n=== RESULT: " << (all_passed ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗") << " ===" << std::endl;

    return all_passed ? 0 : 1;
}