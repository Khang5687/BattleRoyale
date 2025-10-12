// Fragment shader rendering SDF circle with screen-space aware texture sampling
#version 450

#ifdef GL_EXT_shader_demote_to_helper_invocation
#extension GL_EXT_shader_demote_to_helper_invocation : enable
#endif

layout(constant_id = 0) const bool uUseHelperDemote = false;

layout(push_constant) uniform Push {
    vec2 viewport;
    float lodPixelDust;
    float lodSimpleShape;
    float lodBasicTexture;
} pc;

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in flat uint vTextureIndex;
layout(location = 3) in vec2 vTexCoord;
layout(location = 4) in float vScreenRadius;
layout(location = 0) out vec4 outColor;

// Texture atlas - using atlas approach for now, will upgrade to bindless later
layout(set = 0, binding = 0) uniform sampler2DArray uTextureAtlas;
layout(set = 0, binding = 3) readonly buffer AtlasThumbnails {
    vec4 colors[];
} uAtlasThumbnails;

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
    float dustCutoff = max(pc.lodPixelDust, MIN_SAMPLE_RADIUS);
    float simpleShapeCutoff = max(pc.lodSimpleShape, dustCutoff + MIN_SAMPLE_RADIUS);
    float basicTextureCutoff = max(pc.lodBasicTexture, simpleShapeCutoff + MIN_SAMPLE_RADIUS);
    bool canSampleAtlas = hasAtlasTexture();

    if (canSampleAtlas) {
        uint atlasLayer = vTextureIndex & 0x7FFFFFFFu;
        vec4 thumbColor = uAtlasThumbnails.colors[atlasLayer];

        // PIXEL_DUST tier: Use thumbnail color only
        if (screenRadius <= dustCutoff) {
            vec4 blended = mix(thumbColor, vColor, 0.3);
            outColor = vec4(blended.rgb, blended.a * alpha);
            return;
        }

        // SIMPLE_SHAPE tier: Use thumbnail color only (no texture sampling)
        if (screenRadius <= simpleShapeCutoff) {
            vec4 blended = mix(thumbColor, vColor, 0.3);
            outColor = vec4(blended.rgb, blended.a * alpha);
            return;
        }

        vec3 texCoord = vec3(vTexCoord, float(atlasLayer));
        vec4 texColor;

        // BASIC_TEXTURE tier: Sample with LOD bias
        if (screenRadius < basicTextureCutoff) {
            float lod = clamp(log2(basicTextureCutoff / screenRadius), 0.0, MAX_ATLAS_LOD);
            texColor = sampleAtlasColorLod(texCoord, lod);
        } else {
            // FULL_DETAIL tier: Full resolution sampling
            texColor = sampleAtlasColor(texCoord);
        }

        finalColor = mix(texColor, vColor, 0.3);
    }

    outColor = vec4(finalColor.rgb, finalColor.a * alpha);
}
