#pragma once

#include "opg/scene/interface/raygenerator.h"
#include "opg/scene/interface/emitter.h"

#include "pathtracingraygenerator.cuh"

class PathtracingRayGenerator : public opg::RayGenerator
{
public:
    PathtracingRayGenerator(PrivatePtr<opg::Scene> scene, const opg::Properties &props);
    ~PathtracingRayGenerator();

    virtual void launchFrame(CUstream stream, const opg::TensorView<glm::vec3, 2> &output_buffer) override;

    virtual void finalize() override;

protected:
    virtual void initializePipeline(opg::RayTracingPipeline *pipeline, opg::ShaderBindingTable *sbt) override;

private:
    // This buffer is filled with the launch parameters before each ray tracing launch.
    opg::DeviceBuffer<PathtracingLaunchParams>  m_launch_params_buffer;

    // List of all emitters that are used for direct illumination on the GPU.
    opg::DeviceBuffer<const EmitterVPtrTable*>  m_emitters_buffer;

    // Store the one-sample-per-pixel estimate for the current subframe.
    opg::DeviceBuffer<glm::vec4> m_sample_buffer;

    // In every frame one ray is randomly shot from the camera.
    // The result of the individual rays is averaged here over multiple frames.
    opg::DeviceBuffer<glm::vec4> m_accum_buffer;
    uint32_t m_accum_buffer_width;
    uint32_t m_accum_buffer_height;

    // Count how many frames we have rendered from the current camera view.
    // This is important to generate different rays in every ray tracing launch, and also for updating the accumulation buffer.
    uint32_t m_accum_sample_count;

    // The "revision" of the camera changes every time it is modified.
    // Checking if the revision changed is easier than to check for changes in individual parameters.
    uint32_t m_camera_revision;

    // Used to seed the random number generator differently in each subframe. The subframe index is never reset.
    uint32_t m_subframe_index       = 0;


    // Indices of shader programs in the shader binding table
    uint32_t m_raygen_index         = 0;
    uint32_t m_surface_miss_index   = 0;
    uint32_t m_occlusion_miss_index = 0;
};
