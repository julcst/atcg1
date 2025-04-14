#include <cuda_runtime.h>
#include "opg/hostdevice/random.h"
#include "opg/glmwrapper.h"
#include "opg/hostdevice/misc.h"
#include "opg/exception.h"
#include <cstdint>

#include "kernels.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format = OBJ

__global__ void generateRandom(float* out, uint32_t n) {
    // Get the thread index
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return; // Check if the thread index is within bounds

    // Generate a seed and init RNG
    const auto tea = sampleTEA32(idx, 42);
    PCG32 rng{};
    rng.seed(tea, 0);

    // Generate a random float in the range [0, 1)
    out[idx] = rng.nextFloat();
}

__global__ void countThreshold(const float* in, uint32_t* count, float threshold, uint32_t n) {
    // Get the thread index
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return; // Check if the thread index is within bounds

    // Count the number of elements above the threshold
    if (in[idx] > threshold) atomicAdd(count, 1);
}

void sampleBernoulli(uint32_t* out, float threshold, uint32_t n) {
    uint32_t* aboveThresholdCount; // compute this value
    cudaMallocManaged(&aboveThresholdCount, sizeof(int));

    float* randoms;
    cudaMallocManaged(&randoms, n * sizeof(float));

    uint32_t blockSize = 256;
    uint32_t blockCount = (n + blockSize - 1) / blockSize;

    generateRandom<<<blockCount, blockSize>>>(randoms, n);
    CUDA_SYNC_CHECK();
    countThreshold<<<blockCount, blockSize>>>(randoms, aboveThresholdCount, threshold, n);
    CUDA_SYNC_CHECK();

    *out = *aboveThresholdCount; // Copy the result to the output variable

    cudaFree(randoms);
    cudaFree(aboveThresholdCount);
}

__global__ void matMult(const float* A, int A_rows, int A_cols,
                        const float* B, int B_rows, int B_cols,
                        float* C, int C_rows, int C_cols) {
    // Get the thread index
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C_rows && col < C_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * C_cols + col] = sum;
    }
}

void matrixMultiply(const float* A, int A_rows, int A_cols,
                    const float* B, int B_rows, int B_cols,
                    float* C, int C_rows, int C_cols) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_rows * A_cols * sizeof(float));
    cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, B_rows * B_cols * sizeof(float));
    cudaMemcpy(d_B, B, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_C, C_rows * C_cols * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((C_cols + blockSize.x - 1) / blockSize.x, (C_rows + blockSize.y - 1) / blockSize.y);

    matMult<<<gridSize, blockSize>>>(d_A, A_rows, A_cols, d_B, B_rows, B_cols, d_C, C_rows, C_cols);
    CUDA_SYNC_CHECK();
    
    cudaMemcpy(C, d_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}