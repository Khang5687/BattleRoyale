#include "font_loader.hpp"

#include <array>
#include <fstream>
#include "../stb/stb_truetype.h"

namespace br5 {
namespace {
const std::array<std::filesystem::path, 4> kDefaultFontCandidates = {
    std::filesystem::path("assets/fonts/hud.ttf"),
    std::filesystem::path("assets/fonts/Roboto-Regular.ttf"),
    std::filesystem::path("assets/hud.ttf"),
    std::filesystem::path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
};
} // namespace

std::filesystem::path findDefaultFontAsset() {
    for (const auto& candidate : kDefaultFontCandidates) {
        if (!candidate.empty() && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

bool loadFontFile(const std::filesystem::path& path, std::vector<unsigned char>& outData) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        outData.clear();
        return false;
    }

    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::streamsize>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (size < 0) {
        outData.clear();
        return false;
    }

    outData.resize(static_cast<size_t>(size));
    if (size > 0) {
        file.read(reinterpret_cast<char*>(outData.data()), size);
        if (!file) {
            outData.clear();
            return false;
        }
    }

    return true;
}

bool buildFontAtlas(const std::vector<unsigned char>& fontData, float pixelHeight, FontAtlas& outAtlas, stbtt_fontinfo& outFontInfo) {
    if (fontData.empty() || pixelHeight <= 0.0f) {
        return false;
    }

    int fontOffset = stbtt_GetFontOffsetForIndex(fontData.data(), 0);
    if (fontOffset < 0) {
        return false;
    }

    if (!stbtt_InitFont(&outFontInfo, fontData.data(), fontOffset)) {
        return false;
    }

    outAtlas.basePixelHeight = pixelHeight;
    outAtlas.scale = stbtt_ScaleForPixelHeight(&outFontInfo, pixelHeight);

    int ascent = 0;
    int descent = 0;
    int lineGap = 0;
    stbtt_GetFontVMetrics(&outFontInfo, &ascent, &descent, &lineGap);
    outAtlas.ascent = static_cast<float>(ascent) * outAtlas.scale;
    outAtlas.descent = static_cast<float>(descent) * outAtlas.scale;
    outAtlas.lineGap = static_cast<float>(lineGap) * outAtlas.scale;
    outAtlas.lineAdvance = outAtlas.ascent - outAtlas.descent + outAtlas.lineGap;

    constexpr int kFirstChar = 32;
    constexpr int kCharCount = 95; // ASCII printable

    struct PackAttempt {
        int atlasSize;
        unsigned oversample;
    };

    const std::array<PackAttempt, 4> attempts = {
        PackAttempt{512, 2},
        PackAttempt{512, 1},
        PackAttempt{1024, 2},
        PackAttempt{1024, 1}
    };

    std::vector<stbtt_packedchar> packedChars(kCharCount);

    for (const auto& attempt : attempts) {
        outAtlas.atlasWidth = static_cast<uint32_t>(attempt.atlasSize);
        outAtlas.atlasHeight = static_cast<uint32_t>(attempt.atlasSize);
        outAtlas.pixels.assign(static_cast<size_t>(attempt.atlasSize) * attempt.atlasSize, 0u);

        stbtt_pack_context packContext{};
        if (!stbtt_PackBegin(&packContext, outAtlas.pixels.data(), attempt.atlasSize, attempt.atlasSize, 0, 1, nullptr)) {
            continue;
        }
        stbtt_PackSetOversampling(&packContext, attempt.oversample, attempt.oversample);

        stbtt_pack_range range{};
        range.font_size = pixelHeight;
        range.first_unicode_codepoint_in_range = kFirstChar;
        range.array_of_unicode_codepoints = nullptr;
        range.num_chars = kCharCount;
        range.chardata_for_range = packedChars.data();

        bool packed = stbtt_PackFontRanges(&packContext, fontData.data(), 0, &range, 1) != 0;
        stbtt_PackEnd(&packContext);

        if (!packed) {
            continue;
        }

        outAtlas.glyphs.clear();
        outAtlas.glyphs.reserve(static_cast<size_t>(kCharCount) + 1u);

        const float invAtlasWidth = 1.0f / static_cast<float>(outAtlas.atlasWidth);
        const float invAtlasHeight = 1.0f / static_cast<float>(outAtlas.atlasHeight);

        for (int i = 0; i < kCharCount; ++i) {
            const stbtt_packedchar& pc = packedChars[static_cast<size_t>(i)];
            PackedGlyph glyph{};
            glyph.xoff = pc.xoff;
            glyph.yoff = pc.yoff;
            glyph.xoff2 = pc.xoff2;
            glyph.yoff2 = pc.yoff2;
            glyph.xadvance = pc.xadvance;
            glyph.u0 = static_cast<float>(pc.x0) * invAtlasWidth;
            glyph.v0 = static_cast<float>(pc.y0) * invAtlasHeight;
            glyph.u1 = static_cast<float>(pc.x1) * invAtlasWidth;
            glyph.v1 = static_cast<float>(pc.y1) * invAtlasHeight;
            glyph.glyphIndex = stbtt_FindGlyphIndex(&outFontInfo, kFirstChar + i);
            outAtlas.glyphs[static_cast<uint32_t>(kFirstChar + i)] = glyph;
        }

        if (outAtlas.glyphs.find(static_cast<uint32_t>('?')) == outAtlas.glyphs.end()) {
            const uint32_t fallbackCode = static_cast<uint32_t>('?');
            auto fallbackIt = outAtlas.glyphs.find(static_cast<uint32_t>(' '));
            if (fallbackIt != outAtlas.glyphs.end()) {
                outAtlas.glyphs[fallbackCode] = fallbackIt->second;
            }
        }

        return true;
    }

    outAtlas.pixels.clear();
    outAtlas.glyphs.clear();
    return false;
}

} // namespace br5
