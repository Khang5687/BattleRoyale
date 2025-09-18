// HUD text fragment shader
#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uFontAtlas;

void main() {
    float alpha = texture(uFontAtlas, vUV).r;
    outColor = vec4(vColor.rgb, vColor.a * alpha);
}
