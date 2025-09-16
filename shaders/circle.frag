// Fragment shader rendering SDF circle with smooth edge
#version 450

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    float d = length(vPos);
    if (d > 1.0) {
        discard; // outside circle
    }
    // Smooth edge using derivative-based width
    float edge = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);
    outColor = vec4(vColor.rgb, vColor.a * alpha);
}



