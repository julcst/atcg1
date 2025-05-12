#include "radiositykernels.h"

#include "opg/hostdevice/misc.h"
#include "opg/memory/devicebuffer.h"

#pragma cuda_source_property_format=OBJ

// Compute the radiosity iteratively using the Jacobi method:
// radiosity = radiosity + lambda * (emissions - K * radiosity)
// for K := identity - diag(albedo) * form_factor_matrix
__global__ void _applyRadiositySteps(
    const glm::vec3* emissions,
    const glm::vec3* albedos,
    const float* form_factor_matrix,
    const glm::vec3* old_radiosity,
    glm::vec3* new_radiosity,
    const size_t total_primitive_count,
    const float lambda
) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_primitive_count) return;

    glm::vec3 step = emissions[i];
    for (size_t j = 0; j < total_primitive_count; ++j) {
        auto K = -albedos[i] * form_factor_matrix[i * total_primitive_count + j];
        if (i == j) K += 1.0f; // Add the identity matrix
        step -= K * old_radiosity[j];
    }
    new_radiosity[i] = old_radiosity[i] + lambda * step;
}

void applyRadiositySteps(
    const glm::vec3* emissions,
    const glm::vec3* albedos,
    const float* form_factor_matrix,
    glm::vec3* radiosity,
    const size_t total_primitive_count,
    const float lambda,
    const uint32_t steps
) {
    const size_t block_size = 256;
    const auto grid_size = ceil_div(total_primitive_count, block_size);

    // Allocate temporary storage for the new radiosity values
    opg::DeviceBuffer<glm::vec3> temp;
    temp.alloc(total_primitive_count);

    for (uint32_t step = 0; step < steps; step++) {
        glm::vec3* old_radiosity = step % 2 == 0 ? radiosity : temp.data();
        glm::vec3* new_radiosity = step % 2 == 0 ? temp.data() : radiosity;
        _applyRadiositySteps<<<grid_size, block_size>>>(
            emissions, albedos, form_factor_matrix, old_radiosity, new_radiosity, total_primitive_count, lambda
        );
    }

    if (steps % 2 == 1) {
        // Copy the last result back to the original radiosity buffer
        cudaMemcpy(radiosity, temp.data(), total_primitive_count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }
}
