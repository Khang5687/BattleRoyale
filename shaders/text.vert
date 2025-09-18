// HUD text vertex shader
#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform Push {
    vec2 viewport; // effective viewport size (framebuffer size / zoom factor)
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

void main() {
    vec2 normalized = vec2(inPosition.x / pc.viewport.x, inPosition.y / pc.viewport.y);
    vec2 ndc = vec2(normalized.x * 2.0 - 1.0, normalized.y * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = inUV;
    vColor = inColor;
}
