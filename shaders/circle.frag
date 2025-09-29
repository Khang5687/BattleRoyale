// Fragment shader rendering SDF circle with smooth edge and texture support
#version 450

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in flat uint vTextureIndex;
layout(location = 3) in vec2 vTexCoord;
layout(location = 0) out vec4 outColor;

// Texture atlas - using atlas approach for now, will upgrade to bindless later
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
    if (vTextureIndex != 0xFFFFFFFFu) { // Not INVALID_TEXTURE_INDEX
        // Use atlas layer (extracting from textureIndex)
        uint atlasLayer = vTextureIndex & 0x7FFFFFFFu; // Remove high bit if set
        vec3 texCoord = vec3(vTexCoord, float(atlasLayer));
        vec4 texColor = texture(uTextureAtlas, texCoord);
        // Blend with health color for visual feedback
        finalColor = mix(texColor, vColor, 0.3);
    } else {
        // Use flat color
        finalColor = vColor;
    }

    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}



