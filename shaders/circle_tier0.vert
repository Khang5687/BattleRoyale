// Vertex shader for Tier 0 (PIXEL_DUST) - Point sprite rendering
// Renders sub-pixel circles as single-vertex point sprites (no texture)
#version 450

layout(location = 0) in vec2 inPos;         // Unused for point sprites
layout(location = 1) in vec2 inCenter;      // pixel space center
layout(location = 2) in float inRadius;     // pixel radius (typically <2px)
layout(location = 3) in vec4 inColor;       // RGBA team color
layout(location = 4) in uint inTextureIndex; // Unused for Tier 0

layout(push_constant) uniform Push {
    vec2 viewport; // effective viewport size
} pc;

layout(location = 0) out vec4 vColor; // pass color to fragment shader

void main() {
    // Transform world position to NDC
    vec2 ndc = (inCenter / pc.viewport) * 2.0 - 1.0;
    
    // Depth based on radius (same as regular circles for consistency)
    const float CIRCLE_DEPTH_RADIUS_SCALE = 600.0;
    const float CIRCLE_DEPTH_RANGE = 0.9;
    float depthFactor = clamp(inRadius / CIRCLE_DEPTH_RADIUS_SCALE, 0.0, 1.0);
    float depth = 1.0 - depthFactor * CIRCLE_DEPTH_RANGE;
    
    gl_Position = vec4(ndc, depth, 1.0);
    gl_PointSize = 1.0; // Single pixel
    vColor = inColor;
}
