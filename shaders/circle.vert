// Vertex shader for instanced SDF circles
#version 450

layout(location = 0) in vec2 inPos;       // quad corners in [-1, 1]
layout(location = 1) in vec2 inCenter;    // pixel space center
layout(location = 2) in float inRadius;   // pixel radius
layout(location = 3) in vec4 inColor;     // RGBA

layout(push_constant) uniform Push {
    vec2 viewport; // framebuffer size in pixels
} pc;

layout(location = 0) out vec2 vPos;   // pass the local quad pos for SDF
layout(location = 1) out vec4 vColor; // pass color

void main() {
    vec2 world = inCenter + inPos * inRadius;
    vec2 ndc = (world / pc.viewport) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vPos = inPos;
    vColor = inColor;
}








