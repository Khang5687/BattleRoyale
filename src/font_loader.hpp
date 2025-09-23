#pragma once

#include <filesystem>
#include <vector>
#include <cstdint>
#include <unordered_map>

struct stbtt_fontinfo;

namespace br5 {

// Returns the first existing font path among the default search locations.
std::filesystem::path findDefaultFontAsset();

// Loads the binary contents of a font file into the provided buffer.
bool loadFontFile(const std::filesystem::path& path, std::vector<unsigned char>& outData);

struct PackedGlyph {
    float xoff = 0.0f;
    float yoff = 0.0f;
    float xoff2 = 0.0f;
    float yoff2 = 0.0f;
    float xadvance = 0.0f;
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 0.0f;
    float v1 = 0.0f;
    uint32_t glyphIndex = 0;
};

struct FontAtlas {
    float basePixelHeight = 0.0f;
    float ascent = 0.0f;
    float descent = 0.0f;
    float lineGap = 0.0f;
    float lineAdvance = 0.0f;
    float scale = 1.0f;
    uint32_t atlasWidth = 0;
    uint32_t atlasHeight = 0;
    std::vector<unsigned char> pixels;
    std::unordered_map<uint32_t, PackedGlyph> glyphs;
};

// Builds a simple single-channel font atlas for ASCII glyphs at the requested pixel height.
bool buildFontAtlas(const std::vector<unsigned char>& fontData, float pixelHeight, FontAtlas& outAtlas, stbtt_fontinfo& outFontInfo);

} // namespace br5
