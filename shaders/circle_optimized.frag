// Optimized fragment shader with aggressive LOD, early-out, and virtual texture support
#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_descriptor_indexing : enable

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vImageLayer;
layout(location = 3) in vec2 vTexCoord;
layout(location = 4) in flat uint vTextureIndex;
layout(location = 5) in flat uint vLodLevel;

layout(location = 0) out vec4 outColor;

// Bindless texture array
layout(set = 0, binding = 0) uniform sampler2D uTextures[];
layout(set = 0, binding = 1) uniform sampler uTextureSampler;

// Virtual texture system
layout(set = 0, binding = 2) uniform sampler2D uIndirectionTexture;
layout(set = 0, binding = 3) uniform sampler2DArray uPhysicalTextureArray;
layout(set = 0, binding = 4) restrict buffer FeedbackBuffer {
    uint feedbackBuffer[];
};

// Virtual texture constants (must match C++ constants)
#define VIRTUAL_TEXTURE_SIZE 4096
#define INDIRECTION_SIZE 4096

// LOD levels
const uint LOD_PIXEL_DUST = 0;
const uint LOD_SIMPLE_SHAPE = 1;
const uint LOD_BASIC_TEXTURE = 2;
const uint LOD_FULL_DETAIL = 3;

// Virtual texture sampling function
vec4 sampleVirtualTexture(vec2 uv, uint textureId) {
    // Calculate mip level based on derivatives
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    float maxDerivative = max(length(dx), length(dy));
    float mipLevel = log2(maxDerivative * float(VIRTUAL_TEXTURE_SIZE));

    // Sample indirection texture
    vec4 indirection = texture(uIndirectionTexture, uv);

    // Decode physical page coordinates
    vec2 physicalUV = fract(uv * float(INDIRECTION_SIZE)) * indirection.zw + indirection.xy;

    // Write to feedback buffer for page requests
    if (indirection.a < 0.5) {
        // Page not loaded, request it
        atomicAdd(feedbackBuffer[textureId], 1);
    }

    // Sample from physical texture cache
    return texture(uPhysicalTextureArray, vec3(physicalUV, indirection.a));
}

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
        vec4 texColor;

        // Choose sampling method - virtual texture for high LOD, bindless for others
        if (vLodLevel >= LOD_FULL_DETAIL && vTextureIndex < 8192) {
            // Use virtual texture sampling for high detail
            texColor = sampleVirtualTexture(vTexCoord, vTextureIndex);
        } else {
            // Use bindless texture indexing for basic sampling
            float lodBias = (vLodLevel == LOD_BASIC_TEXTURE) ? 2.0 : 0.0;
            texColor = textureLod(sampler2D(uTextures[nonuniformEXT(vTextureIndex)], uTextureSampler),
                                  vTexCoord, lodBias);
        }

        // Simplified health blend
        finalColor = mix(texColor, vColor, 0.3);
    }
    
    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}
