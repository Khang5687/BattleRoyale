// Vertex shader for instanced health bars
#version 450

layout(location = 0) in vec2 inPos;       // Quad corners in [-1, 1]
layout(location = 1) in vec2 inCenter;    // World-space bar center
layout(location = 2) in vec2 inSize;      // Bar width/height in world units
layout(location = 3) in float inFill;     // 0..1 health ratio

layout(push_constant) uniform Push {
    vec2 cameraOffset; // World-space top-left of the camera
    vec2 viewport;     // Visible world size
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out float vFill;

void main() {
    vec2 halfSize = inSize * 0.5;
    vec2 world = inCenter + vec2(inPos.x * halfSize.x, inPos.y * halfSize.y);
    vec2 ndc = ((world - pc.cameraOffset) / pc.viewport) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = inPos * 0.5 + 0.5;
    vFill = clamp(inFill, 0.0, 1.0);
}
