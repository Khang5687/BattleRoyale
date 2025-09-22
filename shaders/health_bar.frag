// Fragment shader for instanced health bars
#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in float vFill;

layout(location = 0) out vec4 outColor;

vec3 gradientColor(float fill) {
    vec3 low = vec3(0.85, 0.2, 0.2);    // Red
    vec3 mid = vec3(0.95, 0.85, 0.2);   // Yellow
    vec3 high = vec3(0.2, 0.85, 0.25);  // Green
    float toMid = clamp(fill * 2.0, 0.0, 1.0);
    float toHigh = clamp((fill - 0.5) * 2.0, 0.0, 1.0);
    vec3 color = mix(low, mid, toMid);
    color = mix(color, high, toHigh);
    return color;
}

void main() {
    float fill = clamp(vFill, 0.0, 1.0);
    vec3 background = vec3(0.06, 0.06, 0.07);
    vec3 fillColor = gradientColor(fill);

    // Smooth mask for fill edge to avoid harsh transition
    float fillMask = 1.0 - smoothstep(fill - 0.008, fill + 0.008, vUV.x);
    vec3 color = mix(background, fillColor, fillMask);

    // Soften edges slightly for a pill-shaped appearance
    float edgeDistance = min(min(vUV.x, 1.0 - vUV.x), min(vUV.y, 1.0 - vUV.y));
    float edgeFade = clamp(edgeDistance * 24.0, 0.0, 1.0);
    float alpha = mix(0.7, 1.0, edgeFade);

    outColor = vec4(color, alpha);
}
