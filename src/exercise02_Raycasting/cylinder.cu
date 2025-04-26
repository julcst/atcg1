#include "cylinder.cuh"

#include "opg/scene/utility/interaction.cuh"
#include "opg/scene/utility/trace.cuh"

#include <optix.h>
#include "opg/raytracing/optixglm.h"


extern "C" __global__ void __intersection__cylinder()
{
    const glm::vec3 center = glm::vec3(0, 0, 0);
    const float     radius = 1;
    const float     half_height = 1.0f;
    const glm::vec3 axis   = glm::vec3(0, 0, 1);

    const glm::vec3 ray_orig = optixGetObjectRayOriginGLM();
    const glm::vec3 ray_dir  = optixGetObjectRayDirectionGLM();
    const float     ray_tmin = optixGetRayTmin();
    const float     ray_tmax = optixGetRayTmax();

    /* Implement:
     * - Ray-cylinder intersection.
     * Hint: Use the function `bool optixReportIntersection(float hitT, unsigned int hitKind)` to report an intersection.
     * The hitKind can be set to 0 since we do not need it.
     */

    glm::vec3 isec_pos;
    float z_min = (center - half_height * axis).z; // = -1
    float z_max = (center + half_height * axis).z; // = 1

    //intersection with infinitely long cylinder
    //coefficients of at^2+bt+c=0
    float a = ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y;
    float b = 2 * (ray_dir.x * ray_orig.x + ray_dir.y * ray_orig.y);
    float c = ray_orig.x * ray_orig.x + ray_orig.y * ray_orig.y - radius * radius;

    float d = b * b - 4.0f * a * c;
    if(d > 0.0f){
        float sqrt_d = sqrtf(d);

        //first possible intersection
        float t1 = (-b + sqrt_d) / 2.0f * a;
        if (t1 > ray_tmin && t1 < ray_tmax) {
            //check if intersection point inside z range defined by cylinder height
            isec_pos = ray_orig + t1 * ray_dir;
            if (isec_pos.z >= z_min && isec_pos.z <= z_max) {
                optixReportIntersection(t1, 0);
            }
        }

        //second possible intersection
        float t2 = (-b - sqrt_d) / 2.0f * a;
        if (t2 > ray_tmin && t2 < ray_tmax) {
            //check if intersection point inside z range defined by cylinder height
            isec_pos = ray_orig + t2 * ray_dir;
            if (isec_pos.z >= z_min && isec_pos.z <= z_max) {
                optixReportIntersection(t2, 0);
            }
        }
    }

    //intersection with bottom cap
    float t3 = (z_min - ray_orig.z) / ray_dir.z;
    if (t3 > ray_tmin && t3 < ray_tmax) {
        //check if intersection point is inside radius of cap
        isec_pos = ray_orig + t3 * ray_dir;
        if(isec_pos.x * isec_pos.x + isec_pos.y * isec_pos.y <= radius * radius){
            optixReportIntersection(t3, 0);
        }
    }

    //intersection with top cap
    float t4 = (z_max - ray_orig.z) / ray_dir.z;
    if (t4 > ray_tmin && t4 < ray_tmax) {
        //check if intersection point is inside radius of cap
        isec_pos = ray_orig + t4 * ray_dir;
        if(isec_pos.x * isec_pos.x + isec_pos.y * isec_pos.y <= radius * radius){
            optixReportIntersection(t4, 0);
        }
    }
}

extern "C" __global__ void __closesthit__cylinder()
{
    SurfaceInteraction *si = getPayloadDataPointer<SurfaceInteraction>();
    const ShapeInstanceHitGroupSBTData* sbt_data = reinterpret_cast<const ShapeInstanceHitGroupSBTData*>(optixGetSbtDataPointer());

    const glm::vec3 world_ray_origin = optixGetWorldRayOriginGLM();
    const glm::vec3 world_ray_dir    = optixGetWorldRayDirectionGLM();
    const float     tmax             = optixGetRayTmax();

    // NOTE: optixGetObjectRayOrigin() and optixGetObjectRayDirection() are not available in closest hit programs.
    // const glm::vec3 object_ray_origin = optixGetObjectRayOriginGLM();
    // const glm::vec3 object_ray_dir    = optixGetObjectRayDirectionGLM();

    const glm::vec3 local_axis = glm::vec3(0, 0, 1);
    const float half_height = 1.0f;
    const glm::vec3 center = glm::vec3(0, 0, 0);
    const float radius = 1;


    // Set incoming ray direction and distance
    si->incoming_ray_dir = world_ray_dir;
    si->incoming_distance = tmax;


    /* Implement:
     * - Compute the position surface normal and tangent vector of the ray-cylinder intersection.
     * - Store these values in the SurfaceInteraction si.
     */

    // World space postion
    si->position = world_ray_origin + tmax * world_ray_dir;

    // Transform position into local object space to compute normals and friends
    glm::vec3 local_position = optixTransformPointFromWorldToObjectSpace(si->position);

    float u = 0.0f;
    float v = 0.0f;
    float z_bottom = (center - half_height * local_axis).z;
    float z_top = (center + half_height * local_axis).z;
    float eps = 0.0001f;
    if(local_position.z < z_top + eps && local_position.z > z_top - eps && local_position.x * local_position.x + local_position.y * local_position.y <= radius * radius){
        //top cap
        si->normal = glm::normalize(local_axis);

        si->tangent = glm::vec3(1.0f, 0.0f, 0.0f);

        u = (glm::atan2(local_position.y, local_position.x) + glm::two_pi<float>()) / glm::two_pi<float>();
        v = sqrtf(local_position.x * local_position.x + local_position.y * local_position.y) / radius;
    }
    else if(local_position.z < z_bottom + eps && local_position.z > z_bottom - eps && local_position.x * local_position.x + local_position.y * local_position.y <= radius * radius){
        //bottom cap
        si->normal = glm::normalize(-local_axis);

        si->tangent = glm::vec3(1.0f, 0.0f, 0.0f);

        u = (glm::atan2(local_position.y, local_position.x) + glm::two_pi<float>()) / glm::two_pi<float>();
        v = sqrtf(local_position.x * local_position.x + local_position.y * local_position.y) / radius;
    }
    else{
        //cylinder side
        //intersection point projected on cylinder axis
        glm::vec3 p = center + glm::dot(local_axis, local_position) * local_axis;
        si->normal = glm::normalize(local_position - p);

        si->tangent = glm::cross(local_axis, si->normal);

        u = (glm::atan2(local_position.y, local_position.x) + glm::two_pi<float>()) / glm::two_pi<float>();
        v = (local_position.z + half_height) / 2*half_height;
    }

    // Transform local object space normal to world space normal
    si->normal = optixTransformNormalFromObjectToWorldSpace(si->normal);
    si->normal = glm::normalize(si->normal);

    si->tangent = optixTransformNormalFromObjectToWorldSpace(si->tangent);
    si->tangent = glm::normalize(si->tangent);

    si->uv = glm::vec2(u,v);
    //

    si->primitive_index = optixGetPrimitiveIndex();

    si->bsdf = sbt_data->bsdf;
    si->emitter = sbt_data->emitter;
}
