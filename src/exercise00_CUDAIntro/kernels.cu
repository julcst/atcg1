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