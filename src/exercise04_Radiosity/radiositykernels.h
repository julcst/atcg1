#pragma once

#include "opg/glmwrapper.h"

void applyRadiositySteps(
    const glm::vec3* emissions,
    const glm::vec3* albedos,
    const float* form_factor_matrix,
    glm::vec3* radiosity,
    const size_t total_primitive_count,
    const float lambda = 1.0f,
    const uint32_t steps = 1
);