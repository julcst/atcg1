#include <cuda_runtime.h>
#include "opg/hostdevice/random.h"
#include "opg/glmwrapper.h"
#include "opg/hostdevice/misc.h"
#include <cstdint>

#include "kernels.h"

// By default, .cu files are compiled into .ptx files in our framework, that are then loaded by OptiX and compiled
// into a ray-tracing pipeline. In this case, we want the kernels.cu to be compiled as a "normal" .obj file that is
// linked against the application such that we can simply call the functions defined in the kernels.cu file.
// The following custom pragma notifies our build system that this file should be compiled into a "normal" .obj file.
#pragma cuda_source_property_format = OBJ

__global__ void VecMulConst(int* dataArray, int N, int constant)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        dataArray[i] *= constant;
}

__global__ void convolution2D(const unsigned char* image, const int* kernel, int* output, int image_width, int image_height, int kernel_size)
{
    //pixel index of flattened image
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < image_width * image_height) {
        int kernel_radius = kernel_size / 2;
        int row = index / image_width;
        int col = index % image_width;
        int sum = 0;
        for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
            for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                //index of neighbor pixel
                int n_row = min(max(row + ky, 0), height - 1);
                int n_col = min(max(col + kx, 0), width - 1);
                int n_index = n_row * width + n_col;

                //perform convolution
                sum += input[n_index] * kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
            }
        }

        output[index] = sum;
    }
}