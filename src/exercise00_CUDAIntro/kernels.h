#pragma once

// Declare your kernel functions here:

void sampleBernoulli(uint32_t* out, float threshold, uint32_t n);

void matrixMultiply(const float* A, int A_rows, int A_cols, const float* B, int B_rows, int B_cols, float* C, int C_rows, int C_cols);