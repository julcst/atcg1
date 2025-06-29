#include "phasefunctions.cuh"
#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/coordinates.h"

__forceinline__ __device__ glm::vec3 warp_square_to_sphere_uniform(const glm::vec2 uv)
{
    float z = uv.x * 2 - 1;
    float phi = uv.y * 2 * glm::pi<float>();

    float r = glm::sqrt(glm::max(0.0f, 1 - z * z));
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ float warp_square_to_sphere_uniform_pdf(const glm::vec3 &dir)
{
    return 1 / (4 * glm::pi<float>());
}

//

extern "C" __device__ PhaseFunctionEvalResult __direct_callable__henyeygreenstein_evalPhaseFunction(const MediumInteraction &interaction, const glm::vec3 &outgoing_ray_dir)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    /* Implement:
     * - Compute the phase-function value for `outgoing_ray_dir`.
     * - Compute the sampling probability of generating the `outgoing_ray_dir` using phase-function importance sampling.
     */

    PhaseFunctionEvalResult result;
    // Dummy implementation scatters uniformly into all directions.
    // result.sampling_pdf = 1/(4*glm::pi<float>());
    // result.phase_function_value = glm::vec3(1/(4*glm::pi<float>()));

    //
    float cos_theta = glm::dot(glm::normalize(interaction.incoming_ray_dir), glm::normalize(outgoing_ray_dir));
    float g = sbt_data->g;

    float p = ((1 - g * g) / 2) * glm::pow((1 + g * g - 2 * g * cos_theta), -1.5f);

    result.sampling_pdf = p;
    result.phase_function_value = glm::vec3(p);
    //

    return result;
}

extern "C" __device__ PhaseFunctionSamplingResult __direct_callable__henyeygreenstein_samplePhaseFunction(const MediumInteraction &interaction, PCG32 &rng)
{
    const HenyeyGreensteinPhaseFunctionData *sbt_data = *reinterpret_cast<const HenyeyGreensteinPhaseFunctionData **>(optixGetSbtDataPointer());

    PhaseFunctionSamplingResult result;
    // Dummy implementation samples the sphere uniformly.
    // result.outgoing_ray_dir = warp_square_to_sphere_uniform(rng.next2d());
    // result.sampling_pdf = warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir);
    // result.phase_function_weight = glm::vec3(warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir)) / warp_square_to_sphere_uniform_pdf(result.outgoing_ray_dir);
    // result.phase_function_weight = glm::vec3(1);

    /* Implement:
     * - Sample a direction from the henyey greenstein phase function using the g-parameter stored in the sbt_data.
     * - Compute the respective phase function value.
     * - Compute the respective sampling probability.
     */

    //
    float g = sbt_data->g;

    // outgoing ray dir
    float term_1 = (1 - g * g) / (1 - g + 2 * g * rng.next1d());
    float cos_theta = (g / 2) * (1 + g * g - term_1 * term_1);
    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));

    float phi = 2 * glm::pi<float>() * rng.next1d();

    glm::mat3 local_frame = opg::compute_local_frame(interaction.incoming_ray_dir);
    glm::vec3 outgoing_ray_dir = local_frame[0] * sin_theta * glm::cos(phi) + local_frame[1] * sin_theta * glm::sin(phi) + local_frame[2] * cos_theta;
    result.outgoing_ray_dir = glm::normalize(outgoing_ray_dir);

    // phase function value
    float p = ((1 - g * g) / 2) * glm::pow((1 + g * g - 2 * g * cos_theta), -1.5f);
    result.phase_function_weight = glm::vec3(p);

    // sampling probability
    result.sampling_pdf = p * (1 / (2 * glm::pi<float>()));
    //

    return result;
}
