#include "homogeneousmedium.cuh"

#include "opg/hostdevice/color.h"



// Here T is either `float` or `glm::vec3`
template <typename T>
__device__ T transmittance(float t, T sigma_t)
{
    return glm::exp(-sigma_t * t);
}


__device__ float warp_1d_sample_to_medium_event_distance(float u, float sigma_t)
{
    float t = -glm::log(u) / sigma_t;
    return t;
}

__device__ float warp_1d_sample_to_medium_event_distance_pdf(float t, float sigma_t)
{
    float pdf = sigma_t * glm::exp(-sigma_t * t);
    return pdf;
}


extern "C" __device__ glm::vec3 __direct_callable__homogeneousMedium_evalTransmittance(const opg::Ray &ray, float distance, PCG32 &unused_rng)
{
    const HomogeneousMediumData *sbt_data = *reinterpret_cast<const HomogeneousMediumData **>(optixGetSbtDataPointer());

    // Evaluate the probability of the light *not* interacting with the medium.
    glm::vec3 sigma_t = sbt_data->sigma_a + sbt_data->sigma_s;
    return transmittance(distance, sigma_t);
}

extern "C"  __device__ MediumSamplingResult __direct_callable__homogeneousMedium_sampleMediumEvent(const opg::Ray &ray, float max_distance, PCG32 &rng)
{
    const HomogeneousMediumData *sbt_data = *reinterpret_cast<const HomogeneousMediumData **>(optixGetSbtDataPointer());

    // Absorbtion, scattering and extinction coefficients...
    glm::vec3 sigma_a = sbt_data->sigma_a;
    glm::vec3 sigma_s = sbt_data->sigma_s;
    glm::vec3 sigma_t = sigma_a + sigma_s;

    // Scalar projection of scattering coefficient, used to sample the next medium scattering event.
    float sigma_s_scalar = rgb_to_scalar_weight(sbt_data->sigma_s);


    MediumSamplingResult result;
    // Dummy implementation:
    // Effectively no medium event.
    result.interaction = MediumInteraction::empty();
    result.interaction.incoming_ray_dir = ray.direction;
    result.transmittance_weight = glm::vec3(1);

    // If the scattering coefficient is 0, we don't need to sample a medium event where the phase function is then evaulated.
    // The case of absorbtion will be handled when evaluating the transmittance between two surface interactions later in the raygeneration shader.
    if (sigma_s_scalar == 0)
    {
        result.interaction.set_invalid();
        result.transmittance_weight = transmittance(max_distance, sigma_t);
        return result;
    }

    // Sample the free-flight distance proportional to sigma_s_scalar.
    float sampled_distance = warp_1d_sample_to_medium_event_distance(rng.next1d(), sigma_s_scalar);

    if (sampled_distance < max_distance)
    {
        // Medium event!
        // The sampling succeeded and a scattering event was found at the given distance
        result.interaction.incoming_distance = sampled_distance;
        // Compute the position of the medium interaction as well
        result.interaction.position = ray.at(result.interaction.incoming_distance);

        // Compute the transmittance including the scattering coeficient sigma_s, divided by sampling probability.
        // float sampling_pdf = warp_1d_sample_to_medium_event_distance_pdf(sampled_distance, sigma_s_scalar);
        // result.transmittance_weight = sbt_data->sigma_s * transmittance(sampled_distance, sigma_t) / sampling_pdf;
        result.transmittance_weight = sbt_data->sigma_s / sigma_s_scalar * transmittance(sampled_distance, sigma_t - sigma_s_scalar);
    }
    else
    {
        // No medium event...
        // The sampling did not succeed, and there is no scattering event *before* the max_distance.
        result.interaction.set_invalid();

        // All no medium events are *the same* event, so we need to compute the transmittance and sampling_pdf for
        // *any* such case, i.e. marginalize over all sampled distances >= max_distance.
        // float sampling_pdf = transmittance(max_distance, sigma_s_scalar);
        // result.transmittance_weight = transmittance(max_distance, sigma_t) / sampling_pdf;
        result.transmittance_weight = transmittance(max_distance, sigma_t - sigma_s_scalar);
    }

    return result;
}
