// Fragment shader for Tier 0 (PIXEL_DUST) - Simple colored pixel
// No texture fetch, no SDF calculation, just flat color for maximum performance
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    // Simple flat color output - no texture, no SDF, no smoothing
    outColor = vColor;
}
