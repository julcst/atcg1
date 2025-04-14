#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>

#include "opg/opg.h"
#include "opg/exception.h"
#include "opg/imagedata.h"
#include "opg/memory/devicebuffer.h"
#include "opg/glmwrapper.h"

#include "kernels.h"

void taskA()
{
    /* Task:
     * Using the GLM math library:
     * - Consider the equation $<n, x> + d = 0$ in 3d space that holds for all points on a plane, where n is a unit-length vector and d is a scalar.
     *   Determine the parameters n, d that describes the plane spanned by the given plane points.
     * - Swap the y and z components of the plane normal.
     * - Find the intersection point between the *new* plane (modified n, old d) and the ray.
     *    - Note: the ray parameters are given as CUDA-builtin float3 vectors.
     *    - Convert the ray parameters to glm::vec3 using the cuda2glm() function.
     *    - Implement the ray-plane intersection.
     *    - Convert the intersection point back to a CUDA-builtin float3 vector using the glm2cuda() function.
     */

    std::cout << "taskA output:" << std::endl;

    // Plane equiation: dot(n, x) + d = 0
    glm::vec3 plane_point_1 = glm::vec3(3, 0, 2);
    glm::vec3 plane_point_2 = glm::vec3(4, 3, 2);
    glm::vec3 plane_point_3 = glm::vec3(1, 2, 4);

    // Determine the parameters n, d that describes the plane spanned by the given plane points
    glm::vec3 plane_normal;
    float plane_d;

    plane_normal = glm::normalize(glm::cross(plane_point_2-plane_point_1, plane_point_3-plane_point_1));
    plane_d = -glm::dot(plane_normal, plane_point_1);

    std::cout << "\tplane normal:             " << plane_normal << ", plane offset: " << plane_d << std::endl;

    //swap plane_normal y,z components
    std::swap(plane_normal.y, plane_normal.z);

    std::cout << "\tswizzled plane normal:    " << plane_normal << std::endl;

    // Ray equation: x = origin + t * dir
    float3 ray_origin_cudafloat3 = make_float3(1, 4, 2);
    float3 ray_dir_cudafloat3 = make_float3(2, 4, 3);

    float3 intersection_cudafloat3;

    // - convert ray parameters to GLM vectors using cuda2glm() function.
    // - implement ray-plane intersection.
    // - convert intersection point back to float3 type using glm2cuda() function.

    glm::vec3 ray_origin = cuda2glm(ray_origin_cudafloat3);
    glm::vec3 ray_dir = cuda2glm(ray_dir_cudafloat3);

    float t = (glm::dot(plane_normal, ray_origin) + plane_d) / glm::dot(plane_normal, ray_dir);
    glm::vec3 intersection = ray_origin + t * ray_dir;
    intersection_cudafloat3 = glm2cuda(intersection);

    std::cout << "\tray-plane intersection at (" << intersection_cudafloat3.x << " " << intersection_cudafloat3.y << " " << intersection_cudafloat3.z << ")" << std::endl;
}

void taskB()
{
    /* Task:
     * - Generate an array containing integer numbers 1..1e7 and copy it to the GPU.
     * - Use a CUDA kernel to multiply each number by a constant in parallel.
     * - Copy the result back to the CPU host memory.
     */

    // Use dataArray as your host memory
    std::vector<int> dataArray;
    // Use d_dataArray as a pointer to device memory
    int *d_dataArray;

    // constant to multiply each number by
    int constant = 2;

    // init vector
    const int N = 10000000;
    dataArray.reserve(N); 
    dataArray.resize(N);

    // fill the vector with the numbers 1 to N
    std::iota(dataArray.begin(), dataArray.end(), 1);

    //print input
    std::cout << "taskB input:" << std::endl;
    std::cout << "\tfirst 10 entries:";
    for (int i = 0; i < std::min<int>(dataArray.size(), 10); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
    std::cout << "\tlast 10 entries:";
    for (int i = std::max<int>(dataArray.size() - 10, 0); i < dataArray.size(); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
    //

    //copy vector to gpu
    cudaMalloc(&d_dataArray, N * sizeof(int));
    cudaMemcpy(d_dataArray, dataArray.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    //kernel call
    launchVecMulConst(d_dataArray, N, constant);

    //copy back from gpu to host
    cudaMemcpy(dataArray.data(), d_dataArray, N * sizeof(int), cudaMemcpyDeviceToHost);

    //print output
    std::cout << "taskB output:" << std::endl;
    std::cout << "\tfirst 10 entries:";
    for (int i = 0; i < std::min<int>(dataArray.size(), 10); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;
    std::cout << "\tlast 10 entries:";
    for (int i = std::max<int>(dataArray.size() - 10, 0); i < dataArray.size(); ++i)
    {
        std::cout << " " << dataArray[i];
    }
    std::cout << std::endl;

    //free gpu memory
    cudaFree(d_dataArray);
}

void taskC()
{
    /* Task:
     * - Apply the separable Sobel filter to an image.
     *   Use two kernel executions to compute a single 2D convolution.
     * - Output G = sqrt(G_x^2 + G_y^2)
     */

    opg::ImageData imageData;
    // opg::getRootPath returns the absolute path to the root directory
    // of this framework independent of the current working directory
    std::string filename = opg::getRootPath() + "/data/exercise00_CUDAIntro/Valve.png";
    opg::readImage(filename.c_str(), imageData);
    uint32_t channelSize = opg::getImageFormatChannelSize(imageData.format);
    uint32_t channelCount = opg::getImageFormatChannelCount(imageData.format);

    // Each color channel stores a single uint8_t (or unsigned char) value per pixel.
    OPG_CHECK(channelSize == 1);

    //convert to grayscale
    if(channelCount == 3){
        std::vector<unsigned char> data_grayscale(imageData.width * imageData.height, 0);
        for (int i = 0; i < imageData.width * imageData.height; i++) {
            float R = static_cast<float>(imageData.data[i * 3]);
            float G = static_cast<float>(imageData.data[i * 3 + 1]);
            float B = static_cast<float>(imageData.data[i * 3 + 2]);
            float grayValue = 0.2989f * R + 0.5870f * G + 0.1140f * B;
            data_grayscale[i] = static_cast<unsigned char>(grayValue);
        }
        imageData.data.assign(data_grayscale.begin(), data_grayscale.end());
        imageData.format = opg::ImageFormat::FORMAT_R_UINT8;
    }

    //copy to gpu
    unsigned char* d_imageData;
    cudaMalloc(&d_imageData, imageData.width * imageData.height * sizeof(unsigned char));
    cudaMemcpy(d_imageData, imageData.data.data(), imageData.width * imageData.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //apply convolution G_x
    std::vector<int> g_x(imageData.width * imageData.height, 0);
    int* d_g_x;
    cudaMalloc(&d_g_x, imageData.width * imageData.height * sizeof(int));
    cudaMemcpy(d_g_x, g_x.data(), imageData.width * imageData.height * sizeof(int), cudaMemcpyHostToDevice);

    int kernel_size = 3;
    std::vector<int> kernel_x {-1,0,1,-2,0,2,-1,0,1};
    int* d_kernel_x;
    cudaMalloc(&d_kernel_x, imageData.width * imageData.height * sizeof(int));
    cudaMemcpy(d_kernel_x, kernel_x.data(), imageData.width * imageData.height * sizeof(int), cudaMemcpyHostToDevice);

    launchConvolution2D(d_imageData, d_kernel_x, d_g_x, imageData.width, imageData.height, kernel_size);

    cudaMemcpy(g_x.data(), d_g_x, imageData.width * imageData.height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_g_x);
    cudaFree(d_kernel_x);

    //apply convolution G_y
    std::vector<int> g_y(imageData.width * imageData.height, 0);
    int* d_g_y;
    cudaMalloc(&d_g_y, imageData.width * imageData.height * sizeof(int));
    cudaMemcpy(d_g_y, g_y.data(), imageData.width * imageData.height * sizeof(int), cudaMemcpyHostToDevice);

    kernel_size = 3;
    std::vector<int> kernel_y {-1,-2,-1,0,0,0,1,2,1};
    int* d_kernel_y;
    cudaMalloc(&d_kernel_y, imageData.width * imageData.height * sizeof(int));
    cudaMemcpy(d_kernel_y, kernel_y.data(), imageData.width * imageData.height * sizeof(int), cudaMemcpyHostToDevice);

    launchConvolution2D(d_imageData, d_kernel_y, d_g_y, imageData.width, imageData.height, kernel_size);

    cudaMemcpy(g_y.data(), d_g_y, imageData.width * imageData.height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_g_y);
    cudaFree(d_kernel_y);
    cudaFree(d_imageData);

    //gradient magnitude
    std::vector<int> G(imageData.width * imageData.height, 0);
    for (int i = 0; i < imageData.width * imageData.height; i++) {
        G[i] = static_cast<int>(sqrt(g_x[i] * g_x[i] + g_y[i] * g_y[i]));
    }

    //convert to image data (normalize G, map to range 0-255)
    std::vector<unsigned char> G_norm(imageData.width * imageData.height, 0);
    auto max = std::max_element(G.begin(), G.end());
    for (int i = 0; i < imageData.width * imageData.height; i++) {
        G_norm[i] = static_cast<unsigned char>(std::clamp((G[i] / *max) * 255, 0, 255));
    }
    imageData.data.assign(G_norm.begin(), G_norm.end());

    // Write your solution back into the imageData.data array such that we can safe it as an image again
    opg::writeImagePNG("taskC_output.png", imageData);
}

void taskD()
{
    /* Task:
     * - Create an array of uniformly distributed (pseudo) random floats in [0, 1] on the GPU.
     *   You can use tea<4>(a, b) to initialize a random number generator, and rnd(seed) to generate a pseudo random number on the GPU. (See sutil/cuda/random.h)
     * - Count all entries that are greater than 0.5 in parallel.
     */

    float threshold = 0.5f;
    uint32_t totalCount = 10000000;
    uint32_t aboveThresholdCount = 0;

    sampleBernoulli(&aboveThresholdCount, threshold, totalCount);

    std::cout << "taskD output:" << std::endl;
    std::cout << "\t" << aboveThresholdCount << "/" << totalCount << " values are greater than " << threshold << std::endl;
}

void taskE()
{
    /* Task:
     * - Implement a matrix multiplication in a CUDA kernel.
     *   Store your matrices in row major order (https://en.wikipedia.org/wiki/Row-_and_column-major_order).
     */

    int lhsRows = 4;
    int lhsCols = 10;
    std::vector<float> lhs = {// 4x10
                              97, 95, 80, 31, 31, 72, 1, 2, 88, 93,
                              46, 58, 54, 94, 84, 59, 75, 4, 78, 62,
                              44, 92, 14, 83, 82, 47, 78, 88, 28, 12,
                              22, 96, 61, 93, 95, 77, 27, 35, 58, 53};
    int rhsRows = 10;
    int rhsCols = 3;
    std::vector<float> rhs = {// 10x3
                              79, 64, 67,
                              67, 14, 43,
                              68, 17, 43,
                              75, 4, 3,
                              94, 81, 83,
                              1, 56, 18,
                              52, 59, 0,
                              88, 89, 96,
                              93, 66, 76,
                              38, 33, 12};
    int outputRows = 4;
    int outputCols = 3;
    std::vector<float> output;
    output.resize(outputRows * outputCols);
    // Expected output:
    // 36725, 24679, 25982,
    // 40059, 27133, 23270,
    // 39432, 28626, 26127,
    // 40192, 26453, 26179

    matrixMultiply(lhs.data(), lhsRows, lhsCols, rhs.data(), rhsRows, rhsCols, output.data(), outputRows, outputCols);

    std::cout << "taskE output:" << std::endl;
    for (int r = 0; r < outputRows; ++r)
    {
        std::cout << "\t";
        for (int c = 0; c < outputCols; ++c)
        {
            std::cout << " " << output[r * outputCols + c];
        }
        std::cout << std::endl;
    }
}

int main()
{
    taskA();
    taskB();
    taskC();
    taskD();
    taskE();
}
