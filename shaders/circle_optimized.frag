// Optimized fragment shader with aggressive LOD and early-out
#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in flat uint vTextureIndex;
layout(location = 3) in vec2 vTexCoord;
layout(location = 4) in flat uint vLodLevel;

layout(location = 0) out vec4 outColor;

// Bindless texture array (simplified for compatibility)
layout(set = 0, binding = 0) uniform sampler2D uTextures[2048];

// LOD levels
const uint LOD_PIXEL_DUST = 0;
const uint LOD_SIMPLE_SHAPE = 1;
const uint LOD_BASIC_TEXTURE = 2;
const uint LOD_FULL_DETAIL = 3;

void main() {
    float d = length(vPos);
    if (d > 1.0) {
        discard; // Outside circle
    }
    
    // Ultra-fast path for tiny circles
    if (vLodLevel == LOD_PIXEL_DUST) {
        // Single pixel - no texture sampling, no edge smoothing
        outColor = vColor;
        return;
    }
    
    // Calculate edge smoothing only for larger circles
    float alpha = 1.0;
    if (vLodLevel >= LOD_SIMPLE_SHAPE) {
        float edge = fwidth(d);
        alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);
    }
    
    vec4 finalColor;
    
    // Skip texture sampling for simple shapes
    if (vLodLevel <= LOD_SIMPLE_SHAPE || vTextureIndex == 0xFFFFFFFF) {
        finalColor = vColor;
    } else {
        // Sample texture with LOD bias based on level
        float lodBias = (vLodLevel == LOD_BASIC_TEXTURE) ? 2.0 : 0.0;
        vec4 texColor = textureLod(uTextures[nonuniformEXT(vTextureIndex)], vTexCoord, lodBias);

        // Simplified health blend
        finalColor = mix(texColor, vColor, 0.3);
    }
    
    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}
