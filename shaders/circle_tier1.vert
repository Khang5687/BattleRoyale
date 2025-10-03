// Vertex shader for Tier 1 (SIMPLE_SHAPE) - Procedural circle rendering
// Renders small circles without texture (2-10px)
#version 450

layout(location = 0) in vec2 inPos;         // quad corners in [-1, 1]
layout(location = 1) in vec2 inCenter;      // pixel space center
layout(location = 2) in float inRadius;     // pixel radius
layout(location = 3) in vec4 inColor;       // RGBA
layout(location = 4) in uint inTextureIndex; // Unused for Tier 1

layout(push_constant) uniform Push {
    vec2 viewport; // effective viewport size
} pc;

layout(location = 0) out vec2 vPos;   // pass the local quad pos for SDF
layout(location = 1) out vec4 vColor; // pass color

const float CIRCLE_DEPTH_RADIUS_SCALE = 600.0;
const float CIRCLE_DEPTH_RANGE = 0.9;

void main() {
    vec2 world = inCenter + inPos * inRadius;
    vec2 ndc = (world / pc.viewport) * 2.0 - 1.0;
    float depthFactor = clamp(inRadius / CIRCLE_DEPTH_RADIUS_SCALE, 0.0, 1.0);
    float depth = 1.0 - depthFactor * CIRCLE_DEPTH_RANGE;
    gl_Position = vec4(ndc, depth, 1.0);
    vPos = inPos;
    vColor = inColor;
}
