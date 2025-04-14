#pragma once

// Declare your kernel functions here:
void launchVecMulConst(int* dataArray, int N, int constant);

__global__ void vecMulConst(int* dataArray, int N, int constant);

void launchConvolution2D(const unsigned char* image, const int* kernel, int* output, int image_width, int image_height, int kernel_size);

__global__ void convolution2D(const unsigned char* image, const int* kernel, int* output, int image_width, int image_height, int kernel_size);