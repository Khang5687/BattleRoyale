// Fragment shader rendering SDF circle with smooth edge and texture support
#version 450

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vImageLayer;
layout(location = 3) in vec2 vTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2DArray uTextureAtlas;

void main() {
    float d = length(vPos);
    if (d > 1.0) {
        discard; // outside circle
    }

    // Smooth edge using derivative-based width
    float edge = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);

    vec4 finalColor;
    if (vImageLayer >= 0.0) {
        // Sample from texture atlas
        vec3 texCoord = vec3(vTexCoord, vImageLayer);
        vec4 texColor = texture(uTextureAtlas, texCoord);
        // Blend with health color for visual feedback
        finalColor = mix(texColor, vColor, 0.3);
    } else {
        // Use flat color
        finalColor = vColor;
    }

    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}



