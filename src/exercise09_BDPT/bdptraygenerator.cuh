#pragma once

#include "opg/glmwrapper.h"
#include "opg/memory/bufferview.h"
#include "opg/memory/tensorview.h"

#include "opg/scene/components/camera.cuh"
#include "opg/scene/interface/emitter.cuh"
#include "opg/scene/utility/trace.cuh"

#include "opg/memory/vector.h"
#include "opg/scene/utility/interaction.cuh"

#include <optix.h>

struct PathVertex
{
    SurfaceInteraction si;
    // The througput between the vertex and the endpoint (camera/light). This includes division by all interpediate sampling pdfs!
    glm::vec3 throughput_weight;
    // Sampling PDF of this vertex, given the previous vertex.
    //float sampling_pdf_to_prev;
    //float sampling_pdf_from_prev;
    //float sampling_pdf_subpath;
};


#define MAX_SUB_PATH_VERTEX_COUNT 8

struct PerPixelData // GBuffer
{
    // This is stored where the GBuffer is stored, but the content is not that.
    // NOTE: The PathVertices are stored "inline" in this structure and don't reference additional memory.
    opg::Vector<PathVertex, MAX_SUB_PATH_VERTEX_COUNT> light_subpath;
    opg::Vector<PathVertex, MAX_SUB_PATH_VERTEX_COUNT> camera_subpath;
};


struct PathtracingLaunchParams
{
    // Small floating point number used to move rays away from surfaces when tracing to avoid self intersections.
    float scene_epsilon;

    // The index of the current frame is used to generate a different random seed for each frame
    uint32_t      subframe_index;

    // 2D output buffer storing the final linear color values for the current subframe only.
    // Each thread in the __raygen__main shader program computes one pixel value in this array.
    // Access via output_buffer[y][x].value() or output_buffer(glm::uvec2(x, y)).value()
    opg::TensorView<glm::vec3, 2> output_radiance;
    // Additional data that is stored per pixel, like the GBuffer in real-time renderers, or in our case the light and camera subpaths.
    opg::TensorView<PerPixelData, 2> per_pixel_data;

    // Size of the image we are generating
    uint32_t image_width;
    uint32_t image_height;

    // Camera parameters that are used to spawn rays through the pixels of the image
    CameraData camera;

    // Sum over the weight of all emitters in the scene (Used to compute the emitter selection probability)
    float                                     emitters_total_weight;
    // The cummulative distribution over all emitters in the scene (Used to select an emitter during next-event estimation)
    opg::BufferView<float>                    emitters_cdf;
    // List of all emitters in the scene
    opg::BufferView<const EmitterVPtrTable *> emitters;

    // Additional parameters used to initiate ray tracing operations
    TraceParameters surface_interaction_trace_params;
    TraceParameters occlusion_trace_params;
    OptixTraversableHandle traversable_handle; // top-level scene IAS
};

#define MAX_TRACE_OPS   (MAX_SUB_PATH_VERTEX_COUNT)
#define MAX_TRACE_DEPTH (MAX_SUB_PATH_VERTEX_COUNT)
