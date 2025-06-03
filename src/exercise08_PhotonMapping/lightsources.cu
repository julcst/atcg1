#include "lightsources.cuh"

#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/binarysearch.h"
#include "opg/hostdevice/coordinates.h"
#include "common.h"

#include <optix.h>


extern "C" __device__ EmitterSamplingResult __direct_callable__pointlight_sampleLight(const Interaction &si, PCG32 &unused_rng)
{
    const PointLightData *sbt_data = *reinterpret_cast<const PointLightData **>(optixGetSbtDataPointer());

    glm::vec3 dir_to_light = sbt_data->position - si.position;

    EmitterSamplingResult result;
    result.radiance_weight_at_receiver = sbt_data->intensity / glm::dot(dir_to_light, dir_to_light);
    result.direction_to_light = glm::normalize(dir_to_light);
    result.distance_to_light = glm::length(dir_to_light);
    result.sampling_pdf = 1;
    return result;
}

extern "C" __device__ EmitterPhotonSamplingResult __direct_callable__pointlight_samplePhoton(PCG32 &rng)
{
    const PointLightData *sbt_data = *reinterpret_cast<const PointLightData **>(optixGetSbtDataPointer());

    EmitterPhotonSamplingResult result;
    result.position = sbt_data->position;
    result.direction = warp_square_to_sphere_uniform(rng.next2d());
    result.sampling_pdf = warp_square_to_sphere_uniform_pdf(result.direction);
    result.radiance_weight = sbt_data->intensity / result.sampling_pdf;
    return result;
}


extern "C" __device__ EmitterSamplingResult __direct_callable__directionallight_sampleLight(const Interaction &si, PCG32 &unused_rng)
{
    const DirectionalLightData *sbt_data = *reinterpret_cast<const DirectionalLightData **>(optixGetSbtDataPointer());

    glm::vec3 dir_to_light = sbt_data->direction;

    EmitterSamplingResult result;
    result.radiance_weight_at_receiver = sbt_data->irradiance_at_receiver;
    result.direction_to_light = glm::normalize(dir_to_light);
    result.distance_to_light = std::numeric_limits<float>::infinity();
    result.sampling_pdf = 1;
    return result;
}

extern "C" __device__ EmitterPhotonSamplingResult __direct_callable__directionallight_samplePhoton(PCG32 &rng)
{
    // Photon sampling for directional light not implemented
    // We would need to know a bounding volume of the scene, and then sample a position uniformly distributed on the bounding volume projected orthogonal to the light direction...
    EmitterPhotonSamplingResult result;
    result.radiance_weight = glm::vec3(0);
    result.sampling_pdf = 0; // invalid sample
    return result;
}






extern "C" __device__ glm::vec3 __direct_callable__spherelight_evalLight(const SurfaceInteraction &si)
{
    const SphereLightData *sbt_data = *reinterpret_cast<const SphereLightData **>(optixGetSbtDataPointer());
    // We can assume that si is actually on the surface of the light source

    // The emitted radiance is constant across the light source
    return sbt_data->radiance;
}

extern "C" __device__ EmitterSamplingResult __direct_callable__spherelight_sampleLight(const Interaction &si, PCG32 &rng)
{
    const SphereLightData *sbt_data = *reinterpret_cast<const SphereLightData **>(optixGetSbtDataPointer());

    // Some useful quantities
    glm::vec3 light_center_dir = glm::normalize(sbt_data->position - si.position);
    float light_center_distance = glm::length(sbt_data->position - si.position);
    float cap_height = 1 - glm::sqrt(light_center_distance*light_center_distance - sbt_data->radius*sbt_data->radius) / light_center_distance;

    // A transformation matrix that rotates the z axis to the light_center_dir
    glm::mat3 local_frame = opg::compute_local_frame(light_center_dir);


    EmitterSamplingResult result;

    if (light_center_distance < sbt_data->radius)
    {
        result.radiance_weight_at_receiver = glm::vec3(0);
        result.sampling_pdf = 0;
        return result;
    }

    // Light source sampling
    glm::vec3 local_direction_to_light = warp_square_to_spherical_cap_uniform(rng.next2d(), cap_height);
    result.direction_to_light = local_frame * local_direction_to_light;

    float spherical_cap_area = 2*glm::pi<float>() * cap_height;

    // Radiance divided by sampling pdf
    result.radiance_weight_at_receiver = sbt_data->radiance * spherical_cap_area;

    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / spherical_cap_area;


    // Compute light_ray_length for occlusion query
    {
        glm::vec3 O = si.position - sbt_data->position;
        glm::vec3 D = result.direction_to_light;

        float p = glm::dot(D, O) / glm::dot(D, D); // p/2 actually
        float q = (glm::dot(O, O) - sbt_data->radius * sbt_data->radius) / glm::dot(D, D);

        float k = glm::max(p*p - q, 0.0f);
        // Usually we would have to check for k < 0, but by construction the ray intersects the light source.

        // Assuming the surface element is outside of the sphere, the first intersection is what we want!
        result.distance_to_light = -p - glm::sqrt(k);
    }

    return result;
}

extern "C" __device__ float __direct_callable__spherelight_evalLightSamplingPDF(const Interaction &si, const SurfaceInteraction &si_on_light)
{
    const SphereLightData *sbt_data = *reinterpret_cast<const SphereLightData **>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    float light_center_distance = glm::length(sbt_data->position - si.position);
    float cap_height = 1 - glm::sqrt(light_center_distance*light_center_distance - sbt_data->radius*sbt_data->radius) / light_center_distance;

    // Probability of sampling this direction via light source sampling
    return 1 / (2*M_PIf * cap_height);
}

extern "C" __device__ EmitterPhotonSamplingResult __direct_callable__spherelight_samplePhoton(PCG32 &rng)
{
    const SphereLightData *sbt_data = *reinterpret_cast<const SphereLightData **>(optixGetSbtDataPointer());

    EmitterPhotonSamplingResult result;
    result.sampling_pdf = 0; // invalid photon

    /*
     * Implement:
     * - Sample a photon from the surface of an emissive sphere.
     *     - Select a random surface normal of the sphere.
     *     - Compute the corresponding world-space position.
     *     - Sample a light direction at the selected surface element.
     *     - Compute the sampling pdf and radiance of this photon.
     * Hint: Since we assume diffuse emission of light, the emitted radiance is `glm::dot(surface_normal, light_dir) * sbt_data->radiance`.
     */

    //

    return result;
}



extern "C" __device__ glm::vec3 __direct_callable__meshlight_evalLight(const SurfaceInteraction &si)
{
    const MeshLightData *sbt_data = *reinterpret_cast<const MeshLightData **>(optixGetSbtDataPointer());
    // We can assume that si is actually on the surface of the light source

    // The emitted radiance is constant across the light source
    return sbt_data->radiance;
}

extern "C" __device__ EmitterSamplingResult __direct_callable__meshlight_sampleLight(const Interaction &si, PCG32 &rng)
{
    const MeshLightData *sbt_data = *reinterpret_cast<const MeshLightData **>(optixGetSbtDataPointer());

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    triangle_index = opg::binary_search(sbt_data->mesh_cdf, rng.next1d());

    triangle_barys = rng.next2d();
    // Mirror barys at diagonal line to cover a triangle instead of a square
    if (triangle_barys.x + triangle_barys.y > 1)
        triangle_barys = glm::vec2(1) - triangle_barys;


    // Compute the `light_position` using the triangle_index and the triangle_barys on the mesh:

    // Indices of triangle vertices in the mesh
    glm::uvec3 vertex_indices = glm::uvec3(0u);
    if (sbt_data->mesh_indices.elmt_byte_size == sizeof(glm::u32vec3))
    {
        // Indices stored as 32-bit unsigned integers
        const glm::u32vec3* indices = reinterpret_cast<glm::u32vec3*>(sbt_data->mesh_indices.data);
        vertex_indices = glm::uvec3(indices[triangle_index]);
    }
    else
    {
        // Indices stored as 16-bit unsigned integers
        const glm::u16vec3* indices = reinterpret_cast<glm::u16vec3*>(sbt_data->mesh_indices.data);
        vertex_indices = glm::uvec3(indices[triangle_index]);
    }

    // Vertex positions of selected triangle
    glm::vec3 P0 = sbt_data->mesh_positions[vertex_indices.x];
    glm::vec3 P1 = sbt_data->mesh_positions[vertex_indices.y];
    glm::vec3 P2 = sbt_data->mesh_positions[vertex_indices.z];

    // Compute local position
    glm::vec3 local_light_position = (1.0f-triangle_barys.x-triangle_barys.y)*P0 + triangle_barys.x*P1 + triangle_barys.y*P2;
    // Transform local position to world position
    glm::vec3 light_position = glm::vec3(sbt_data->local_to_world * glm::vec4(local_light_position, 1));

    // Compute local normal
    glm::vec3 local_light_normal = glm::cross(P1-P0, P2-P0);
    // Normals are transformed by (A^-1)^T instead of A
    glm::vec3 light_normal = glm::normalize(glm::transpose(glm::mat3(sbt_data->world_to_local)) * local_light_normal);


    // Assemble sampling result

    EmitterSamplingResult result;
    result.sampling_pdf = 0; // initialize with invalid sample

    // light source sampling
    result.direction_to_light = glm::normalize(light_position - si.position);
    result.distance_to_light = glm::length(light_position - si.position);

    float one_over_light_position_pdf = sbt_data->total_surface_area;
    float cos_theta_on_light = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / (result.distance_to_light * result.distance_to_light);

    result.radiance_weight_at_receiver = sbt_data->radiance * one_over_light_direction_pdf;

    // Probability of sampling this direction via light source sampling
    result.sampling_pdf = 1 / one_over_light_direction_pdf;

    return result;
}

extern "C" __device__ float __direct_callable__meshlight_evalLightSamplingPDF(const Interaction &si, const SurfaceInteraction &si_on_light)
{
    const MeshLightData *sbt_data = *reinterpret_cast<const MeshLightData **>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal = si_on_light.normal;
    glm::vec3 light_ray_dir = si_on_light.incoming_ray_dir; // glm::normalize(si_on_light.position - si.position);
    float light_ray_length = si_on_light.incoming_distance; // glm::length(si_on_light.position - si.position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = 1 / sbt_data->total_surface_area;

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length * light_ray_length / cos_theta_on_light;

    return light_direction_pdf;
}

extern "C" __device__ EmitterPhotonSamplingResult __direct_callable__meshlight_samplePhoton(PCG32 &rng)
{
    const MeshLightData *sbt_data = *reinterpret_cast<const MeshLightData **>(optixGetSbtDataPointer());

    EmitterPhotonSamplingResult result;
    result.sampling_pdf = 0; // invalid photon

    /*
     * Implement:
     * - Sample a photon from the surface of an emissive mesh.
     *     - Select a random position on the surface of the mesh (like in the ..._sampleLight() method)
     *     - Sample a light direction at the selected surface element
     *     - Compute the sampling pdf and radiance of this photon.
     * Hint: Since we assume diffuse emission of light, the emitted radiance is `glm::dot(surface_normal, light_dir) * sbt_data->radiance`.
     */

    //

    return result;
}
