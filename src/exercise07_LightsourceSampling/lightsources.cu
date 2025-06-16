#include "lightsources.cuh"

#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/interaction.cuh"
#include "opg/hostdevice/binarysearch.h"
#include "opg/hostdevice/coordinates.h"

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


//



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
    result.sampling_pdf = 0; // initialize with invalid sample

    /* Implement:
     * - Sample the direction from the given shading point (si) towards a random point on the sphere that is the light source.
     * Hint: From the point of view of the given shading point (si), the sphere light source simply looks like a uniform disk on the sky.
     * Consider the spherical cap [1] that corresponds to the projection of the light source onto the sphere of all directions at the given surface interaction.
     * [1] https://en.wikipedia.org/wiki/Spherical_cap
     */

    // TODO implement
    // Sample a random point on the spherical cap
    glm::vec2 rand = rng.next2d();
    float cosTheta = 1 - rand.x * cap_height; // in [1 - h, 1]
    float phi = 2.0f * glm::pi<float>() * rand.y; // in [0, 2pi]
    // Convert spherical coordinates to Cartesian coordinates in the local frame
    glm::vec3 local_light_dir;
    local_light_dir.x = glm::sqrt(1 - cosTheta * cosTheta) * glm::cos(phi);
    local_light_dir.y = glm::sqrt(1 - cosTheta * cosTheta) * glm::sin(phi);
    local_light_dir.z = cosTheta;
    glm::vec3 light_direction = normalize(local_frame * local_light_dir); // Transform to world coordinates
    result.direction_to_light = -light_direction;
    // Compute the position of the light source
    glm::vec3 light_position = sbt_data->position + light_center_distance * light_direction;
    // Compute the normal at the light source
    glm::vec3 light_normal = light_direction;
    result.normal_at_light = light_normal;
    // Compute the distance to the light source
    float light_distance = glm::length(light_position - si.position);
    result.distance_to_light = light_distance;
    // Compute the radiance weight at the receiver
    result.radiance_weight_at_receiver = sbt_data->radiance * glm::dot(light_normal, light_direction) / (light_distance * light_distance);
    // Compute the sampling PDF
    result.sampling_pdf = glm::pi<float>() * sbt_data->radius * sbt_data->radius / (light_distance * light_distance);
    result.radiance_weight_at_receiver = sbt_data->radiance;

    return result;
}

extern "C" __device__ float __direct_callable__spherelight_evalLightSamplingPDF(const Interaction &si, const SurfaceInteraction &si_on_light)
{
    const SphereLightData *sbt_data = *reinterpret_cast<const SphereLightData **>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    float light_center_distance = glm::length(sbt_data->position - si.position);
    float cap_height = 1 - glm::sqrt(light_center_distance*light_center_distance - sbt_data->radius*sbt_data->radius) / light_center_distance;

    /* Implement:
     * - Compute the probability of sampling the direction towards the surface interaction on the light source (si_on_light) from the surface interaction at a shading point (si) via light source sampling.
     */

    float sampling_pdf = 0;

    // TODO implement
    sampling_pdf = glm::pi<float>() * sbt_data->radius * sbt_data->radius / (light_center_distance * light_center_distance);
    //

    return sampling_pdf;
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

    /* Implement:
     * - Sample the direction from the given shading point (si) towards a random point on the mesh that is the light source.
     *     - Select a (random) triangle from the mesh with probability proportional to its surface area.
     *     - Compute the barycentric coordinates of a random point on the unit triangle.
     *     - Fill the `EmitterSamplingResult` structure at the end of this method.
     * Hint: Do a binary search in the cummulative probability distribution over the triangles (sbt_data->mesh_cdf).
     *       The selected triangle and barycentric coordinates on the unit triangle are subsequently used to construct the corresponding point on the triangle in world space.
     */

    // Select the triangle to sample a direction from uniformly at random, proportional to its surface area
    uint32_t triangle_index = 0;
    // Sample the barycentric coordinates on the triangle uniformly.
    glm::vec2 triangle_barys = glm::vec2(0, 0);

    // TODO implement
    triangle_index = binary_search(sbt_data->mesh_cdf, rng.next1d());
    const auto triangle_pdf = sbt_data->mesh_cdf[triangle_index] / sbt_data->total_surface_area; // The probability of sampling this triangle is proportional to its area
    // 


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

    // TODO implement
    result.distance_to_light = glm::length(light_position - si.position);
    result.direction_to_light = glm::normalize(light_position - si.position);
    const auto cosThetaL = abs(glm::dot(light_normal, result.direction_to_light));
    result.sampling_pdf = result.distance_to_light * result.distance_to_light * triangle_pdf / cosThetaL;
    result.normal_at_light = light_normal;
    result.radiance_weight_at_receiver = sbt_data->radiance;
    //

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

    /* Implement:
     * - Compute the probability of sampling the direction towards the surface interaction on the light source (si_on_light) from the surface interaction at a shading point (si) via light source sampling.
     */

    float light_direction_pdf = 0;

    // TODO implement
    const auto triangle_pdf = sbt_data->mesh_cdf[si_on_light.primitive_index] / sbt_data->total_surface_area; // The probability of sampling this triangle is proportional to its area
    const auto cosThetaL = abs(glm::dot(light_normal, light_ray_dir));
    light_direction_pdf =  light_ray_length * light_ray_length * triangle_pdf / (cosThetaL);
    //

    return light_direction_pdf;
}
