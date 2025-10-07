// Fragment shader rendering SDF circle with screen-space aware texture sampling
#version 450

#ifdef GL_EXT_shader_demote_to_helper_invocation
#extension GL_EXT_shader_demote_to_helper_invocation : enable
#endif

layout(constant_id = 0) const float uLodRadiusScale = 1.0;
layout(constant_id = 1) const bool uUseHelperDemote = false;

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in flat uint vTextureIndex;
layout(location = 3) in vec2 vTexCoord;
layout(location = 4) in float vScreenRadius;
layout(location = 0) out vec4 outColor;

// Texture atlas - using atlas approach for now, will upgrade to bindless later
layout(set = 0, binding = 0) uniform sampler2DArray uTextureAtlas;

const float PIXEL_DUST_SCREEN_RADIUS = 0.75;
const float MIP_SAMPLE_THRESHOLD = 2.0;
const float MIN_SAMPLE_RADIUS = 1e-3;
const float MAX_ATLAS_LOD = 8.0;

bool hasAtlasTexture() {
    return (vTextureIndex != 0xFFFFFFFFu) && ((vTextureIndex & 0x80000000u) != 0u);
}

vec4 sampleAtlasColor(vec3 texCoord) {
    return texture(uTextureAtlas, texCoord);
}

vec4 sampleAtlasColorLod(vec3 texCoord, float lod) {
    return textureLod(uTextureAtlas, texCoord, lod);
}

void main() {
    float d = length(vPos);
    if (d > 1.0) {
#if defined(GL_EXT_shader_demote_to_helper_invocation)
        if (uUseHelperDemote) {
            demoteToHelperInvocationEXT();
            return;
        }
#endif
        discard; // outside circle
    }

    // Smooth edge using derivative-based width
    float edge = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);

    vec4 finalColor = vColor;
    float screenRadius = max(vScreenRadius, MIN_SAMPLE_RADIUS);
    float lodScale = max(uLodRadiusScale, MIN_SAMPLE_RADIUS);
    float dustCutoff = PIXEL_DUST_SCREEN_RADIUS * lodScale;
    float mipCutoff = MIP_SAMPLE_THRESHOLD * lodScale;
    bool canSampleAtlas = hasAtlasTexture();

    if (canSampleAtlas && screenRadius > dustCutoff) {
        uint atlasLayer = vTextureIndex & 0x7FFFFFFFu;
        vec3 texCoord = vec3(vTexCoord, float(atlasLayer));
        vec4 texColor;

        if (screenRadius < mipCutoff) {
            float lod = clamp(log2(mipCutoff / screenRadius), 0.0, MAX_ATLAS_LOD);
            texColor = sampleAtlasColorLod(texCoord, lod);
        } else {
            texColor = sampleAtlasColor(texCoord);
        }

        // Blend with health color for visual feedback
        finalColor = mix(texColor, vColor, 0.3);
    }

    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}
