#include <GLFW/glfw3.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <fstream>
#include <ratio>

#define STB_TRUETYPE_IMPLEMENTATION
#include "../stb/stb_truetype.h"

#include "font_loader.hpp"

#include "damage_curve.hpp"

// Action history system for undo/redo
enum class ActionType {
    ADD_POINT,
    REMOVE_POINT,
    MOVE_POINT,
    LOAD_PRESET,
    LOAD_CONFIGURATION
};

struct Action {
    ActionType type;
    std::vector<CurvePoint> beforePoints;
    std::vector<CurvePoint> afterPoints;
    CurvePreset presetBefore = CurvePreset::CUSTOM;
    CurvePreset presetAfter = CurvePreset::CUSTOM;
    int affectedPointIndex = -1;
    CurvePoint oldPoint;
    CurvePoint newPoint;
};

class ActionHistory {
private:
    std::vector<Action> actions;
    size_t currentIndex = 0; // Points to next undo action
    static constexpr size_t MAX_HISTORY = 50;

public:
    void recordAction(Action action) {
        // Remove any actions after current index (when doing new action after undo)
        if (currentIndex < actions.size()) {
            actions.resize(currentIndex);
        }

        actions.push_back(action);
        currentIndex = actions.size();

        // Limit history size
        if (actions.size() > MAX_HISTORY) {
            actions.erase(actions.begin());
            currentIndex--;
        }
    }

    bool canUndo() const {
        return currentIndex > 0;
    }

    bool canRedo() const {
        return currentIndex < actions.size();
    }

    Action* getUndoAction() {
        if (!canUndo()) return nullptr;
        return &actions[currentIndex - 1];
    }

    Action* getRedoAction() {
        if (!canRedo()) return nullptr;
        return &actions[currentIndex];
    }

    void undo() {
        if (canUndo()) {
            currentIndex--;
        }
    }

    void redo() {
        if (canRedo()) {
            currentIndex++;
        }
    }

    void clear() {
        actions.clear();
        currentIndex = 0;
    }
};

// Configuration constants
const int INITIAL_WINDOW_WIDTH = 1200;
const int INITIAL_WINDOW_HEIGHT = 800;
const int GRID_SIZE = 20;
const float CURVE_MARGIN = 50.0f;
const float POINT_RADIUS = 6.0f;
const float POINT_HOVER_RADIUS = 8.0f;
const int PRESET_BUTTON_HEIGHT = 30;
const float UI_PANEL_WIDTH_RATIO = 0.25f; // 25% of window width

// Colors
const float GRID_COLOR[3] = {0.3f, 0.3f, 0.3f};
const float AXIS_COLOR[3] = {0.6f, 0.6f, 0.6f};
const float CURVE_COLOR[3] = {0.2f, 0.8f, 0.2f};
const float POINT_COLOR[3] = {0.8f, 0.2f, 0.2f};
const float POINT_HOVER_COLOR[3] = {1.0f, 0.4f, 0.4f};
const float BACKGROUND_COLOR[3] = {0.1f, 0.1f, 0.1f};
const float TEXT_COLOR[3] = {0.9f, 0.9f, 0.9f};

struct FontRenderer {
    GLuint textureId = 0;
    float pixelHeight = 32.0f;
    bool ready = false;
    std::vector<unsigned char> fontData;
    br5::FontAtlas atlas;
    stbtt_fontinfo fontInfo{};
};

static FontRenderer g_fontRenderer;

void destroyFontRenderer();

bool initializeFontRenderer(float pixelHeight) {
    if (g_fontRenderer.ready) {
        if (std::fabs(pixelHeight - g_fontRenderer.pixelHeight) < 0.5f) {
            return true;
        }
        destroyFontRenderer();
    }

    const std::filesystem::path fontPath = br5::findDefaultFontAsset();
    if (fontPath.empty()) {
        std::cerr << "Config font not found. Expected assets/fonts/hud.ttf or compatible fallback." << std::endl;
        return false;
    }

    g_fontRenderer.fontData.clear();
    if (!br5::loadFontFile(fontPath, g_fontRenderer.fontData)) {
        std::cerr << "Failed to read config font file: " << fontPath << std::endl;
        return false;
    }

    g_fontRenderer.pixelHeight = pixelHeight;

    if (!br5::buildFontAtlas(g_fontRenderer.fontData, pixelHeight, g_fontRenderer.atlas, g_fontRenderer.fontInfo)) {
        std::cerr << "Failed to build font atlas for config UI" << std::endl;
        g_fontRenderer.fontData.clear();
        return false;
    }

    std::vector<unsigned char> atlasRgba(g_fontRenderer.atlas.pixels.size() * 4, 255);
    for (size_t i = 0; i < g_fontRenderer.atlas.pixels.size(); ++i) {
        unsigned char alpha = g_fontRenderer.atlas.pixels[i];
        atlasRgba[i * 4 + 0] = 255;
        atlasRgba[i * 4 + 1] = 255;
        atlasRgba[i * 4 + 2] = 255;
        atlasRgba[i * 4 + 3] = alpha;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &g_fontRenderer.textureId);
    glBindTexture(GL_TEXTURE_2D, g_fontRenderer.textureId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 static_cast<GLsizei>(g_fontRenderer.atlas.atlasWidth),
                 static_cast<GLsizei>(g_fontRenderer.atlas.atlasHeight),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, atlasRgba.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    g_fontRenderer.ready = true;

    std::cout << "Config font ready from '" << fontPath.string() << "' ("
              << g_fontRenderer.atlas.atlasWidth << "x" << g_fontRenderer.atlas.atlasHeight
              << ", glyphs " << g_fontRenderer.atlas.glyphs.size() << ")" << std::endl;
    return true;
}

void destroyFontRenderer() {
    if (g_fontRenderer.textureId != 0) {
        glDeleteTextures(1, &g_fontRenderer.textureId);
        g_fontRenderer.textureId = 0;
    }

    g_fontRenderer.fontData.clear();
    g_fontRenderer.atlas.pixels.clear();
    g_fontRenderer.atlas.glyphs.clear();
    g_fontRenderer.ready = false;
}

// UI state
struct UIState {
    DamageCurve curve;
    ActionHistory actionHistory;
    std::vector<CurvePoint> displayPoints;
    int selectedPoint = -1;
    int hoveredPoint = -1;
    bool isDragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    // Drag state for undo/redo
    bool dragStarted = false;
    CurvePoint dragStartPoint;

    // Clipboard for copy/paste
    std::vector<CurvePoint> clipboardPoints;
    bool hasClipboardData = false;

    // Bezier curve editing
    int selectedControlPoint = -1; // -1 = none, 0 = point, 1 = control in, 2 = control out
    bool showControlHandles = true;

    // Curve validation
    bool showValidation = true;
    DamageCurve::CurveValidationResult lastValidation;

    // Status messages
    std::string statusMessage;
    float statusMessageTimer = 0.0f;
    static constexpr float STATUS_MESSAGE_DURATION = 3.0f; // seconds

    // Dynamic UI layout (updated on window resize)
    int currentWindowWidth = INITIAL_WINDOW_WIDTH;
    int currentWindowHeight = INITIAL_WINDOW_HEIGHT;
    int currentFramebufferWidth = INITIAL_WINDOW_WIDTH;
    int currentFramebufferHeight = INITIAL_WINDOW_HEIGHT;
    float dpiScaleX = 1.0f;
    float dpiScaleY = 1.0f;
    float uiPanelWidth = 0.0f;
    float curveAreaLeft = 0.0f;
    float curveAreaRight = 0.0f;
    float curveAreaTop = CURVE_MARGIN;
    float curveAreaBottom = 0.0f;
    float curveWidth = 0.0f;
    float curveHeight = 0.0f;

    bool showGrid = true;
    bool snapToGrid = false;
    float gridSnapSize = 0.05f; // 5% snap increments
    CurvePreset currentPreset = CurvePreset::BATTLE_ROYALE;

    // Undo/redo helper methods
    void recordAction(ActionType type, const std::vector<CurvePoint>& beforePoints,
                     const std::vector<CurvePoint>& afterPoints,
                     CurvePreset presetBefore = CurvePreset::CUSTOM,
                     CurvePreset presetAfter = CurvePreset::CUSTOM,
                     int affectedIndex = -1, const CurvePoint& oldPoint = {},
                     const CurvePoint& newPoint = {}) {
        Action action;
        action.type = type;
        action.beforePoints = beforePoints;
        action.afterPoints = afterPoints;
        action.presetBefore = presetBefore;
        action.presetAfter = presetAfter;
        action.affectedPointIndex = affectedIndex;
        action.oldPoint = oldPoint;
        action.newPoint = newPoint;

        actionHistory.recordAction(action);
        currentPreset = presetAfter;
    }

    void undo() {
        if (actionHistory.canUndo()) {
            Action* action = actionHistory.getUndoAction();
            if (action) {
                curve.setPoints(action->beforePoints);
                currentPreset = action->presetBefore;
                actionHistory.undo();
            }
        }
    }

    void redo() {
        if (actionHistory.canRedo()) {
            Action* action = actionHistory.getRedoAction();
            if (action) {
                curve.setPoints(action->afterPoints);
                currentPreset = action->presetAfter;
                actionHistory.redo();
            }
        }
    }

    // Copy/paste methods
    void copyCurve() {
        clipboardPoints = curve.getPoints();
        hasClipboardData = true;
        setStatusMessage("Curve copied to clipboard (" + std::to_string(clipboardPoints.size()) + " points)");
    }

    void pasteCurve() {
        if (!hasClipboardData) {
            setStatusMessage("No curve data in clipboard");
            return;
        }

        auto beforePoints = curve.getPoints();
        CurvePreset beforePreset = currentPreset;

        curve.setPoints(clipboardPoints);
        currentPreset = CurvePreset::CUSTOM;

        auto afterPoints = curve.getPoints();
        recordAction(ActionType::LOAD_CONFIGURATION, beforePoints, afterPoints,
                    beforePreset, currentPreset);

        setStatusMessage("Curve pasted from clipboard (" + std::to_string(clipboardPoints.size()) + " points)");
    }

    // Bezier curve editing methods
    void cycleInterpolationType(int pointIndex) {
        if (pointIndex < 0 || pointIndex >= static_cast<int>(curve.getPoints().size())) return;

        auto points = curve.getPoints();
        InterpolationType currentType = points[pointIndex].type;

        // Cycle through interpolation types
        switch (currentType) {
            case InterpolationType::LINEAR:
                points[pointIndex].type = InterpolationType::SPLINE;
                break;
            case InterpolationType::SPLINE:
                points[pointIndex].type = InterpolationType::BEZIER;
                // Initialize control points for Bezier
                if (pointIndex > 0 && pointIndex < static_cast<int>(points.size()) - 1) {
                    float prevX = points[pointIndex - 1].playerRatio;
                    float nextX = points[pointIndex + 1].playerRatio;
                    float currentX = points[pointIndex].playerRatio;
                    float dx = (nextX - prevX) * 0.25f; // Quarter distance to neighbors

                    points[pointIndex].controlInX = -dx * 0.5f;
                    points[pointIndex].controlOutX = dx * 0.5f;
                    points[pointIndex].controlInY = 0.0f;
                    points[pointIndex].controlOutY = 0.0f;
                }
                break;
            case InterpolationType::BEZIER:
                points[pointIndex].type = InterpolationType::LINEAR;
                break;
        }

        curve.setPoints(points);
        std::string typeName = points[pointIndex].type == InterpolationType::LINEAR ? "LINEAR" :
                              points[pointIndex].type == InterpolationType::SPLINE ? "SPLINE" : "BEZIER";
        setStatusMessage("Point " + std::to_string(pointIndex) + " interpolation: " + typeName);
    }

    void toggleControlHandles() {
        showControlHandles = !showControlHandles;
        setStatusMessage("Control handles " + std::string(showControlHandles ? "shown" : "hidden"));
    }

    void updateValidation() {
        lastValidation = curve.validateCurve();
    }

    void toggleValidation() {
        showValidation = !showValidation;
        std::cout << "Validation display " << (showValidation ? "enabled" : "disabled") << std::endl;
    }

    // Status message system
    void setStatusMessage(const std::string& message) {
        statusMessage = message;
        statusMessageTimer = STATUS_MESSAGE_DURATION;
        std::cout << message << std::endl; // Also log to console
    }

    void updateStatusMessage(float deltaTime) {
        if (statusMessageTimer > 0.0f) {
            statusMessageTimer -= deltaTime;
            if (statusMessageTimer <= 0.0f) {
                statusMessage.clear();
            }
        }
    }

    // Snap-to-grid helper functions
    float snapValueToGrid(float value, float snapSize) {
        if (!snapToGrid) return value;
        return std::round(value / snapSize) * snapSize;
    }

    void toggleSnapToGrid() {
        snapToGrid = !snapToGrid;
        setStatusMessage("Snap to grid " + std::string(snapToGrid ? "enabled" : "disabled"));
    }

    void deleteSelectedPoint() {
        if (selectedPoint >= 0 && curve.getPoints().size() > 2) {
            auto beforePoints = curve.getPoints();
            curve.removePoint(selectedPoint);
            auto afterPoints = curve.getPoints();

            recordAction(ActionType::REMOVE_POINT, beforePoints, afterPoints,
                        currentPreset, currentPreset, selectedPoint, beforePoints[selectedPoint]);
            selectedPoint = -1;
            selectedControlPoint = -1;
            setStatusMessage("Point deleted");
        }
    }

    void nudgeSelectedPoint(float deltaX, float deltaY) {
        if (selectedPoint >= 0 && selectedControlPoint == 0) {
            auto points = curve.getPoints();
            points[selectedPoint].playerRatio = std::clamp(points[selectedPoint].playerRatio + deltaX, 0.0f, 1.0f);
            points[selectedPoint].damageMultiplier = snapValueToGrid(
                std::clamp(points[selectedPoint].damageMultiplier + deltaY, 0.01f, 5.0f), gridSnapSize);
            curve.setPoints(points);
            setStatusMessage("Point nudged");
        }
    }
};

UIState g_uiState;

// Frame timing for status message updates
static auto g_lastFrameTime = std::chrono::high_resolution_clock::now();

// Coordinate conversion functions - REWRITTEN FOR CORRECTNESS
// Player ratio: 0.0 (left) to 1.0 (right)
// Damage multiplier: 0.01 (bottom) to 5.0 (top)

float screenToPlayerRatio(float screenX) {
    // Simple linear mapping: screen X to 0.0-1.0 range
    if (g_uiState.curveWidth <= 0) return 0.0f;
    return std::clamp((screenX - g_uiState.curveAreaLeft) / g_uiState.curveWidth, 0.0f, 1.0f);
}

float screenToDamageMultiplier(float screenY) {
    // Map screen Y to damage range 0.01-5.0
    if (g_uiState.curveHeight <= 0) return 1.0f;

    // Normalize screen Y to 0-1 range (0 = bottom, 1 = top)
    float normalizedY = (screenY - g_uiState.curveAreaTop) / g_uiState.curveHeight;
    normalizedY = std::clamp(normalizedY, 0.0f, 1.0f);

    // Map to damage multiplier range: 0.01 (bottom) to 5.0 (top)
    // Use logarithmic mapping for better control at low values
    return std::pow(10.0f, -2.0f + normalizedY * 2.7f); // 0.01 to ~5.0
}

float playerRatioToScreen(float playerRatio) {
    // Convert player ratio back to screen X coordinate
    return g_uiState.curveAreaLeft + std::clamp(playerRatio, 0.0f, 1.0f) * g_uiState.curveWidth;
}

float damageMultiplierToScreen(float damageMultiplier) {
    // Convert damage multiplier back to screen Y coordinate

    // Clamp to our supported range
    damageMultiplier = std::clamp(damageMultiplier, 0.01f, 5.0f);

    // Convert from logarithmic scale back to linear screen coordinate
    // damageMultiplier = 10^(-2.0 + normalizedY * 2.7)
    // So: log10(damageMultiplier) = -2.0 + normalizedY * 2.7
    // normalizedY = (log10(damageMultiplier) + 2.0) / 2.7
    float normalizedY = (std::log10(damageMultiplier) + 2.0f) / 2.7f;
    normalizedY = std::clamp(normalizedY, 0.0f, 1.0f);

    // Map to screen Y coordinate
    return g_uiState.curveAreaTop + normalizedY * g_uiState.curveHeight;
}

bool isPointNearMouse(float screenX, float screenY, double mouseX, double mouseY, float radius) {
    float dx = screenX - static_cast<float>(mouseX);
    float dy = screenY - static_cast<float>(mouseY);
    return dx * dx + dy * dy <= radius * radius;
}

void updateCurveArea() {
    // Calculate UI panel width based on current framebuffer size (for rendering)
    g_uiState.uiPanelWidth = g_uiState.currentFramebufferWidth * UI_PANEL_WIDTH_RATIO;

    // Update curve area bounds using framebuffer coordinates
    g_uiState.curveAreaLeft = g_uiState.uiPanelWidth + CURVE_MARGIN * g_uiState.dpiScaleX;
    g_uiState.curveAreaRight = g_uiState.currentFramebufferWidth - CURVE_MARGIN * g_uiState.dpiScaleX;
    g_uiState.curveAreaTop = CURVE_MARGIN * g_uiState.dpiScaleY;
    g_uiState.curveAreaBottom = g_uiState.currentFramebufferHeight - CURVE_MARGIN * g_uiState.dpiScaleY;

    // Calculate curve area dimensions
    g_uiState.curveWidth = g_uiState.curveAreaRight - g_uiState.curveAreaLeft;
    g_uiState.curveHeight = g_uiState.curveAreaBottom - g_uiState.curveAreaTop;
}

// OpenGL rendering functions
void drawLine(float x1, float y1, float x2, float y2, const float color[3], float width = 1.0f) {
    glColor3f(color[0], color[1], color[2]);
    glLineWidth(width);
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

void drawCircle(float centerX, float centerY, float radius, const float color[3], bool filled = true) {
    glColor3f(color[0], color[1], color[2]);

    if (filled) {
        glBegin(GL_TRIANGLE_FAN);
    } else {
        glBegin(GL_LINE_LOOP);
    }

    const int segments = 32;
    for (int i = 0; i < segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = centerX + radius * cos(angle);
        float y = centerY + radius * sin(angle);
        glVertex2f(x, y);
    }
    glEnd();
}

// Simple bitmap font rendering using OpenGL
void drawText(float x, float bottom, const std::string& text, const float color[3] = TEXT_COLOR) {
    if (!g_fontRenderer.ready || text.empty()) {
        return;
    }

    float cursorX = x;
    float baseline = bottom - g_fontRenderer.atlas.descent;
    const float lineHeight = g_fontRenderer.atlas.lineAdvance > 0.0f
        ? g_fontRenderer.atlas.lineAdvance
        : (g_fontRenderer.atlas.ascent - g_fontRenderer.atlas.descent);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_fontRenderer.textureId);
    glColor4f(color[0], color[1], color[2], 1.0f);

    glBegin(GL_QUADS);
    uint32_t prevGlyphIndex = 0;
    for (unsigned char ch : text) {
        if (ch == '\n') {
            cursorX = x;
            baseline -= lineHeight;
            prevGlyphIndex = 0;
            continue;
        }

        uint32_t codepoint = static_cast<uint32_t>(ch);
        auto glyphIt = g_fontRenderer.atlas.glyphs.find(codepoint);
        if (glyphIt == g_fontRenderer.atlas.glyphs.end()) {
            glyphIt = g_fontRenderer.atlas.glyphs.find(static_cast<uint32_t>('?'));
            if (glyphIt == g_fontRenderer.atlas.glyphs.end()) {
                continue;
            }
        }

        const auto& glyph = glyphIt->second;

        float x0 = cursorX + glyph.xoff;
        float x1 = cursorX + glyph.xoff2;
        float yTop = baseline - glyph.yoff;
        float yBottom = baseline - glyph.yoff2;

        glTexCoord2f(glyph.u0, glyph.v0); glVertex2f(x0, yTop);
        glTexCoord2f(glyph.u1, glyph.v0); glVertex2f(x1, yTop);
        glTexCoord2f(glyph.u1, glyph.v1); glVertex2f(x1, yBottom);
        glTexCoord2f(glyph.u0, glyph.v1); glVertex2f(x0, yBottom);

        float advance = glyph.xadvance;
        if (prevGlyphIndex != 0 && glyph.glyphIndex != 0) {
            advance += static_cast<float>(stbtt_GetGlyphKernAdvance(&g_fontRenderer.fontInfo, prevGlyphIndex, glyph.glyphIndex))
                       * g_fontRenderer.atlas.scale;
        }

        cursorX += advance;
        prevGlyphIndex = glyph.glyphIndex;
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void drawGrid() {
    if (!g_uiState.showGrid) return;

    // Vertical grid lines
    for (float x = g_uiState.curveAreaLeft; x <= g_uiState.curveAreaRight; x += GRID_SIZE) {
        drawLine(x, g_uiState.curveAreaTop, x, g_uiState.curveAreaBottom, GRID_COLOR);
    }

    // Horizontal grid lines
    for (float y = g_uiState.curveAreaTop; y <= g_uiState.curveAreaBottom; y += GRID_SIZE) {
        drawLine(g_uiState.curveAreaLeft, y, g_uiState.curveAreaRight, y, GRID_COLOR);
    }

    // Axes
    float centerX = g_uiState.curveAreaLeft + g_uiState.curveWidth * 0.5f;
    float centerY = g_uiState.curveAreaTop + g_uiState.curveHeight * 0.5f;

    // X-axis
    drawLine(g_uiState.curveAreaLeft, centerY, g_uiState.curveAreaRight, centerY, AXIS_COLOR, 2.0f);

    // Y-axis
    drawLine(centerX, g_uiState.curveAreaTop, centerX, g_uiState.curveAreaBottom, AXIS_COLOR, 2.0f);

    // Y-axis numerical labels (damage multipliers)
    const float labelColor[3] = {0.7f, 0.7f, 0.7f};
    float labelOffsetX = -35.0f * g_uiState.dpiScaleX;

    // Bottom label (0.01x)
    drawText(centerX + labelOffsetX, g_uiState.curveAreaBottom - 5.0f * g_uiState.dpiScaleY, "0.01x", labelColor);

    // Middle-bottom label (0.1x)
    float midBottomY = g_uiState.curveAreaTop + g_uiState.curveHeight * 0.75f;
    drawText(centerX + labelOffsetX, midBottomY, "0.1x", labelColor);

    // Middle label (1.0x)
    drawText(centerX + labelOffsetX, centerY, "1.0x", labelColor);

    // Middle-top label (2.0x)
    float midTopY = g_uiState.curveAreaTop + g_uiState.curveHeight * 0.25f;
    drawText(centerX + labelOffsetX, midTopY, "2.0x", labelColor);

    // Top label (5.0x)
    drawText(centerX + labelOffsetX, g_uiState.curveAreaTop + 5.0f * g_uiState.dpiScaleY, "5.0x", labelColor);
}

void drawCurve() {
    const auto& points = g_uiState.curve.getPoints();
    if (points.size() < 2) return;

    glColor3f(CURVE_COLOR[0], CURVE_COLOR[1], CURVE_COLOR[2]);
    glLineWidth(3.0f);
    glBegin(GL_LINE_STRIP);

    // Sample the curve at regular intervals
    const int samples = 200;
    for (int i = 0; i <= samples; ++i) {
        float playerRatio = static_cast<float>(i) / samples;
        float damageMultiplier = g_uiState.curve.evaluateCurve(playerRatio);

        float screenX = playerRatioToScreen(playerRatio);
        float screenY = damageMultiplierToScreen(damageMultiplier);

        glVertex2f(screenX, screenY);
    }

    glEnd();
}

void drawControlPoints() {
    const auto& points = g_uiState.curve.getPoints();

    for (size_t i = 0; i < points.size(); ++i) {
        float screenX = playerRatioToScreen(points[i].playerRatio);
        float screenY = damageMultiplierToScreen(points[i].damageMultiplier);

        const float* color = POINT_COLOR;
        float radius = POINT_RADIUS;

        if (static_cast<int>(i) == g_uiState.hoveredPoint) {
            color = POINT_HOVER_COLOR;
            radius = POINT_HOVER_RADIUS;
        }

        // Highlight selected point
        if (static_cast<int>(i) == g_uiState.selectedPoint) {
            const float selectedColor[3] = {1.0f, 0.8f, 0.2f};
            drawCircle(screenX, screenY, radius + 4.0f, selectedColor, false);
        }

        drawCircle(screenX, screenY, radius, color, true);
        drawCircle(screenX, screenY, radius + 2.0f, AXIS_COLOR, false);

        // Draw interpolation type indicator
        if (points[i].type != InterpolationType::LINEAR) {
            const float typeColor[3] = {
                points[i].type == InterpolationType::SPLINE ? 0.2f : 0.8f,
                points[i].type == InterpolationType::SPLINE ? 0.8f : 0.2f,
                0.2f
            };
            drawCircle(screenX + radius + 6.0f, screenY - radius - 6.0f, 3.0f, typeColor, true);
        }
    }

    // Draw Bezier control handles if enabled
    if (g_uiState.showControlHandles) {
        for (size_t i = 0; i < points.size(); ++i) {
            if (points[i].type == InterpolationType::BEZIER) {
                float pointX = playerRatioToScreen(points[i].playerRatio);
                float pointY = damageMultiplierToScreen(points[i].damageMultiplier);

                // Draw control lines and handles
                float cInX, cInY, cOutX, cOutY;
                points[i].getControlInPoint(cInX, cInY);
                points[i].getControlOutPoint(cOutX, cOutY);

                float screenCInX = playerRatioToScreen(cInX);
                float screenCInY = damageMultiplierToScreen(cInY);
                float screenCOutX = playerRatioToScreen(cOutX);
                float screenCOutY = damageMultiplierToScreen(cOutY);

                // Control lines
                const float lineColor[3] = {0.5f, 0.5f, 0.5f};
                drawLine(pointX, pointY, screenCInX, screenCInY, lineColor, 1.0f);
                drawLine(pointX, pointY, screenCOutX, screenCOutY, lineColor, 1.0f);

                // Control handles
                const float handleColor[3] = {0.7f, 0.7f, 0.2f};
                drawCircle(screenCInX, screenCInY, 4.0f, handleColor, true);
                drawCircle(screenCOutX, screenCOutY, 4.0f, handleColor, true);

                // Highlight selected control handle
                if (static_cast<int>(i) == g_uiState.selectedPoint && g_uiState.selectedControlPoint > 0) {
                    const float selectedHandleColor[3] = {1.0f, 1.0f, 0.2f};
                    if (g_uiState.selectedControlPoint == 1) {
                        drawCircle(screenCInX, screenCInY, 6.0f, selectedHandleColor, false);
                    } else if (g_uiState.selectedControlPoint == 2) {
                        drawCircle(screenCOutX, screenCOutY, 6.0f, selectedHandleColor, false);
                    }
                }
            }
        }
    }
}

void drawUI() {
    // Draw background panel using framebuffer coordinates
    glColor3f(0.05f, 0.05f, 0.05f);
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(g_uiState.uiPanelWidth, 0);
    glVertex2f(g_uiState.uiPanelWidth, g_uiState.currentFramebufferHeight);
    glVertex2f(0, g_uiState.currentFramebufferHeight);
    glEnd();

    // Draw panel border
    drawLine(g_uiState.uiPanelWidth, 0, g_uiState.uiPanelWidth, g_uiState.currentFramebufferHeight, AXIS_COLOR, 2.0f);

    // Draw preset buttons (simplified representation)
    const std::vector<std::string> presetNames = {
        "Linear", "Exponential", "Logarithmic", "S-Curve",
        "Battle Royale", "Speedrun", "Endurance"
    };

    // Calculate button positions from top of window (since Y=0 is now at bottom)
    float buttonMargin = 10.0f * g_uiState.dpiScaleX;
    float buttonSpacing = 10.0f * g_uiState.dpiScaleY;
    float startY = g_uiState.currentFramebufferHeight - 60.0f * g_uiState.dpiScaleY; // Start from top with margin

    for (size_t i = 0; i < presetNames.size(); ++i) {
        float scaledButtonHeight = PRESET_BUTTON_HEIGHT * g_uiState.dpiScaleY;
        float y = startY - i * (scaledButtonHeight + buttonSpacing);

        // Button background
        glColor3f(0.2f, 0.2f, 0.2f);
        if (static_cast<size_t>(g_uiState.currentPreset) == i) {
            glColor3f(0.3f, 0.5f, 0.3f); // Highlight selected preset
        }

        glBegin(GL_QUADS);
        glVertex2f(buttonMargin, y - scaledButtonHeight);
        glVertex2f(g_uiState.uiPanelWidth - buttonMargin, y - scaledButtonHeight);
        glVertex2f(g_uiState.uiPanelWidth - buttonMargin, y);
        glVertex2f(buttonMargin, y);
        glEnd();

        // Button border
        const float* borderColor = AXIS_COLOR;
        if (static_cast<size_t>(g_uiState.currentPreset) == i) {
            // Thicker border for selected preset
            const float selectedBorderColor[3] = {0.5f, 0.8f, 0.5f};
            borderColor = selectedBorderColor;
            drawLine(buttonMargin, y - scaledButtonHeight, g_uiState.uiPanelWidth - buttonMargin, y - scaledButtonHeight, borderColor, 2.0f);
            drawLine(g_uiState.uiPanelWidth - buttonMargin, y - scaledButtonHeight, g_uiState.uiPanelWidth - buttonMargin, y, borderColor, 2.0f);
            drawLine(g_uiState.uiPanelWidth - buttonMargin, y, buttonMargin, y, borderColor, 2.0f);
            drawLine(buttonMargin, y, buttonMargin, y - scaledButtonHeight, borderColor, 2.0f);
        } else {
            drawLine(buttonMargin, y - scaledButtonHeight, g_uiState.uiPanelWidth - buttonMargin, y - scaledButtonHeight, borderColor);
            drawLine(g_uiState.uiPanelWidth - buttonMargin, y - scaledButtonHeight, g_uiState.uiPanelWidth - buttonMargin, y, borderColor);
            drawLine(g_uiState.uiPanelWidth - buttonMargin, y, buttonMargin, y, borderColor);
            drawLine(buttonMargin, y, buttonMargin, y - scaledButtonHeight, borderColor);
        }

        // Button text
        float textX = buttonMargin + 5.0f * g_uiState.dpiScaleX;
        float textY = y - scaledButtonHeight/2 - 6.0f * g_uiState.dpiScaleY; // Center vertically
        const float* textColor = TEXT_COLOR;
        if (static_cast<size_t>(g_uiState.currentPreset) == i) {
            const float selectedTextColor[3] = {0.9f, 1.0f, 0.9f};
            textColor = selectedTextColor;
        }
        drawText(textX, textY, presetNames[i], textColor);
    }

    // Add title and instructions
    float baseSpacing = g_fontRenderer.ready ?
        (g_fontRenderer.atlas.lineAdvance > 0.0f ? g_fontRenderer.atlas.lineAdvance : (g_fontRenderer.atlas.ascent - g_fontRenderer.atlas.descent))
        : 18.0f * g_uiState.dpiScaleY;
    const float titleColor[3] = {1.0f, 1.0f, 1.0f};
    float titleBaseline = g_uiState.currentFramebufferHeight - 35.0f * g_uiState.dpiScaleY;
    drawText(15.0f * g_uiState.dpiScaleX, titleBaseline, "Damage Curve Presets:", titleColor);

    // Current preset indicator
    if (g_fontRenderer.ready && g_uiState.currentPreset != CurvePreset::CUSTOM) {
        float indicatorBaseline = titleBaseline - (baseSpacing + 8.0f * g_uiState.dpiScaleY);
        const float indicatorColor[3] = {0.5f, 1.0f, 0.5f};
        const std::string currentText = "Active: " + presetNames[static_cast<int>(g_uiState.currentPreset)];
        drawText(15.0f * g_uiState.dpiScaleX, indicatorBaseline, currentText, indicatorColor);
    }

    // Validation status
    if (g_uiState.showValidation && g_fontRenderer.ready) {
        float validationBaseline = titleBaseline - (2 * baseSpacing + 16.0f * g_uiState.dpiScaleY);
        const auto& validation = g_uiState.lastValidation;

        std::string statusText = "Curve Status: ";
        const float* statusColor;

        if (!validation.isValid) {
            statusText += "INVALID";
            statusColor = new float[3]{1.0f, 0.2f, 0.2f}; // Red
        } else if (validation.hasDiscontinuities) {
            statusText += "DISCONTINUOUS";
            statusColor = new float[3]{1.0f, 0.4f, 0.2f}; // Orange-red
        } else if (validation.hasSharpCorners) {
            statusText += "SHARP CORNERS";
            statusColor = new float[3]{1.0f, 0.8f, 0.2f}; // Orange
        } else {
            statusText += "SMOOTH";
            statusColor = new float[3]{0.2f, 1.0f, 0.2f}; // Green
        }

        drawText(15.0f * g_uiState.dpiScaleX, validationBaseline, statusText, statusColor);

        // Smoothness score
        float smoothness = g_uiState.curve.estimateSmoothness();
        char smoothnessBuf[32];
        std::snprintf(smoothnessBuf, sizeof(smoothnessBuf), "Smoothness: %.1f%%", smoothness * 100.0f);
        drawText(15.0f * g_uiState.dpiScaleX, validationBaseline - baseSpacing, smoothnessBuf, TEXT_COLOR);

        delete[] statusColor;
    }

    // Add instructions at bottom of panel
    float instructBaseline = 95.0f * g_uiState.dpiScaleY;
    const float instructColor[3] = {0.7f, 0.7f, 0.7f};
    float instructX = 15.0f * g_uiState.dpiScaleX;
    float instructSpacing = baseSpacing + 6.0f * g_uiState.dpiScaleY;
    drawText(instructX, instructBaseline, "Ctrl+S: Save", instructColor);
    drawText(instructX, instructBaseline - instructSpacing, "Ctrl+L: Load", instructColor);
    drawText(instructX, instructBaseline - 2 * instructSpacing, "Ctrl+Z/Y: Undo/Redo", instructColor);
    drawText(instructX, instructBaseline - 3 * instructSpacing, "Ctrl+C/V: Copy/Paste", instructColor);
    drawText(instructX, instructBaseline - 4 * instructSpacing, "J/K/B: Export JSON/CSV/B64", instructColor);
    drawText(instructX, instructBaseline - 5 * instructSpacing, "I: Cycle Interp Type", instructColor);
    drawText(instructX, instructBaseline - 6 * instructSpacing, "H: Toggle Handles", instructColor);
    drawText(instructX, instructBaseline - 7 * instructSpacing, "G: Toggle Grid", instructColor);
    drawText(instructX, instructBaseline - 8 * instructSpacing, "N: Toggle Snap-to-Grid", instructColor);
    drawText(instructX, instructBaseline - 9 * instructSpacing, "Arrows: Nudge Point", instructColor);
    drawText(instructX, instructBaseline - 10 * instructSpacing, "Delete: Remove Point", instructColor);
    drawText(instructX, instructBaseline - 11 * instructSpacing, "V: Toggle Validation", instructColor);
    drawText(instructX, instructBaseline - 12 * instructSpacing, "Right-click: Del Point", instructColor);

    // Add axis labels for the curve area
    const float labelColor[3] = {0.8f, 0.8f, 0.8f};

    // X-axis label (bottom)
    float xLabelX = g_uiState.curveAreaLeft + g_uiState.curveWidth/2 - 60.0f * g_uiState.dpiScaleX;
    float xLabelY = 20.0f * g_uiState.dpiScaleY;
    drawText(xLabelX, xLabelY, "Player Elimination Ratio", labelColor);

    // Y-axis label (left side, rotated effect by spacing letters vertically)
    float yLabelX = g_uiState.curveAreaLeft - 40.0f * g_uiState.dpiScaleX;
    float yLabelY = g_uiState.curveAreaTop + g_uiState.curveHeight/2 + 40.0f * g_uiState.dpiScaleY;
    float yLabelSpacing = 12.0f * g_uiState.dpiScaleY;
    drawText(yLabelX, yLabelY, "D", labelColor);
    drawText(yLabelX, yLabelY - yLabelSpacing, "a", labelColor);
    drawText(yLabelX, yLabelY - 2 * yLabelSpacing, "m", labelColor);
    drawText(yLabelX, yLabelY - 3 * yLabelSpacing, "a", labelColor);
    drawText(yLabelX, yLabelY - 4 * yLabelSpacing, "g", labelColor);
    drawText(yLabelX, yLabelY - 5 * yLabelSpacing, "e", labelColor);

}

void drawValidationIndicators() {
    if (!g_uiState.showValidation) return;

    const auto& validation = g_uiState.lastValidation;

    // Draw discontinuity indicators
    if (validation.hasDiscontinuities) {
        const float discontinuityColor[3] = {1.0f, 0.2f, 0.2f}; // Red for discontinuities

        for (size_t pointIndex : validation.discontinuityPoints) {
            if (pointIndex < g_uiState.curve.getPoints().size()) {
                const auto& point = g_uiState.curve.getPoints()[pointIndex];
                float screenX = playerRatioToScreen(point.playerRatio);
                float screenY = damageMultiplierToScreen(point.damageMultiplier);

                // Draw warning triangle above the point
                glColor3f(discontinuityColor[0], discontinuityColor[1], discontinuityColor[2]);
                glBegin(GL_TRIANGLES);
                glVertex2f(screenX, screenY + 15.0f);
                glVertex2f(screenX - 8.0f, screenY + 5.0f);
                glVertex2f(screenX + 8.0f, screenY + 5.0f);
                glEnd();

                // Draw exclamation mark
                drawText(screenX - 3.0f, screenY + 8.0f, "!", discontinuityColor);
            }
        }
    }

    // Draw sharp corner indicators
    if (validation.hasSharpCorners) {
        const float cornerColor[3] = {1.0f, 0.8f, 0.2f}; // Orange for sharp corners

        for (size_t pointIndex : validation.sharpCornerPoints) {
            if (pointIndex < g_uiState.curve.getPoints().size()) {
                const auto& point = g_uiState.curve.getPoints()[pointIndex];
                float screenX = playerRatioToScreen(point.playerRatio);
                float screenY = damageMultiplierToScreen(point.damageMultiplier);

                // Draw warning diamond to the right of the point
                glColor3f(cornerColor[0], cornerColor[1], cornerColor[2]);
                glBegin(GL_QUADS);
                glVertex2f(screenX + 12.0f, screenY);
                glVertex2f(screenX + 6.0f, screenY + 6.0f);
                glVertex2f(screenX + 12.0f, screenY + 12.0f);
                glVertex2f(screenX + 18.0f, screenY + 6.0f);
                glEnd();
            }
        }
    }
}

void drawStatusMessage() {
    if (g_uiState.statusMessage.empty()) return;

    // Calculate fade out effect
    float alpha = 1.0f;
    if (g_uiState.statusMessageTimer < 1.0f) {
        alpha = g_uiState.statusMessageTimer; // Fade out in last second
    }

    // Draw background box
    float boxWidth = 400.0f * g_uiState.dpiScaleX;
    float boxHeight = 40.0f * g_uiState.dpiScaleY;
    float boxX = (g_uiState.currentFramebufferWidth - boxWidth) / 2.0f;
    float boxY = g_uiState.currentFramebufferHeight - 100.0f * g_uiState.dpiScaleY;

    // Semi-transparent background
    glColor4f(0.1f, 0.1f, 0.1f, 0.8f * alpha);
    glBegin(GL_QUADS);
    glVertex2f(boxX, boxY);
    glVertex2f(boxX + boxWidth, boxY);
    glVertex2f(boxX + boxWidth, boxY + boxHeight);
    glVertex2f(boxX, boxY + boxHeight);
    glEnd();

    // Border
    glColor4f(0.5f, 0.5f, 0.5f, alpha);
    glBegin(GL_LINE_LOOP);
    glVertex2f(boxX, boxY);
    glVertex2f(boxX + boxWidth, boxY);
    glVertex2f(boxX + boxWidth, boxY + boxHeight);
    glVertex2f(boxX, boxY + boxHeight);
    glEnd();

    // Text
    float textX = boxX + 20.0f * g_uiState.dpiScaleX;
    float textY = boxY + 12.0f * g_uiState.dpiScaleY;
    const float textColor[3] = {0.9f, 0.9f, 0.9f};
    glColor4f(textColor[0], textColor[1], textColor[2], alpha);
    drawText(textX, textY, g_uiState.statusMessage);
}

void render() {
    // Calculate delta time for status message timer
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> deltaTime = currentTime - g_lastFrameTime;
    g_lastFrameTime = currentTime;

    // Update status message timer
    g_uiState.updateStatusMessage(deltaTime.count());

    // Update validation on each frame
    if (g_uiState.showValidation) {
        g_uiState.updateValidation();
    }

    glClear(GL_COLOR_BUFFER_BIT);

    drawGrid();
    drawCurve();
    if (g_uiState.showValidation) {
        drawValidationIndicators();
    }
    drawControlPoints();
    drawUI();
    drawStatusMessage(); // Draw status messages last (on top)
}

// Event handlers - COMPLETELY REWRITTEN FOR ROBUSTNESS
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    double rawMouseX, rawMouseY;
    glfwGetCursorPos(window, &rawMouseX, &rawMouseY);

    // Convert GLFW mouse coordinates (Y=0 at top) to OpenGL framebuffer coordinates (Y=0 at bottom)
    // Scale from window coordinates to framebuffer coordinates for high-DPI displays
    float mouseX = static_cast<float>(rawMouseX * g_uiState.dpiScaleX);
    float mouseY = static_cast<float>(g_uiState.currentFramebufferHeight - rawMouseY * g_uiState.dpiScaleY);

    // Debug output disabled in final version

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            // Clear previous state
            g_uiState.selectedPoint = -1;
            g_uiState.isDragging = false;

            // STEP 1: Check if clicking in UI panel (preset buttons)
            if (mouseX >= 0 && mouseX <= g_uiState.uiPanelWidth) {

                float buttonMargin = 10.0f * g_uiState.dpiScaleX;
                float buttonSpacing = 10.0f * g_uiState.dpiScaleY;
                float scaledButtonHeight = PRESET_BUTTON_HEIGHT * g_uiState.dpiScaleY;
                float startY = g_uiState.currentFramebufferHeight - 60.0f * g_uiState.dpiScaleY;

                if (mouseX >= buttonMargin && mouseX <= g_uiState.uiPanelWidth - buttonMargin) {
                    for (int i = 0; i < 7; ++i) {
                        float buttonTop = startY - i * (scaledButtonHeight + buttonSpacing);
                        float buttonBottom = buttonTop - scaledButtonHeight;

                        if (mouseY >= buttonBottom && mouseY <= buttonTop) {
                            // Record action before loading preset
                            auto beforePoints = g_uiState.curve.getPoints();
                            CurvePreset beforePreset = g_uiState.currentPreset;

                            g_uiState.currentPreset = static_cast<CurvePreset>(i);
                            g_uiState.curve.loadPreset(g_uiState.currentPreset);

                            // Record action after loading preset
                            auto afterPoints = g_uiState.curve.getPoints();
                            g_uiState.recordAction(ActionType::LOAD_PRESET, beforePoints, afterPoints,
                                                  beforePreset, g_uiState.currentPreset);

                            // Show status message
                            const std::string presetNames[] = {"Linear", "Exponential", "Logarithmic", "S-Curve", "Battle Royale", "Speedrun", "Endurance"};
                            g_uiState.setStatusMessage("Loaded preset: " + std::string(presetNames[i]));

                            return; // Exit early - button handled
                        }
                    }
                }
                return; // Click in UI panel but no button - ignore
            }

            // STEP 2: Check if clicking in curve area
            if (mouseX >= g_uiState.curveAreaLeft && mouseX <= g_uiState.curveAreaRight &&
                mouseY >= g_uiState.curveAreaTop && mouseY <= g_uiState.curveAreaBottom) {

                // STEP 2A: Check if clicking on Bezier control handles first (higher priority)
                const auto& points = g_uiState.curve.getPoints();
                if (g_uiState.showControlHandles) {
                    for (size_t i = 0; i < points.size(); ++i) {
                        if (points[i].type == InterpolationType::BEZIER) {
                            // Check control handles
                            float cInX, cInY, cOutX, cOutY;
                            points[i].getControlInPoint(cInX, cInY);
                            points[i].getControlOutPoint(cOutX, cOutY);

                            float screenCInX = playerRatioToScreen(cInX);
                            float screenCInY = damageMultiplierToScreen(cInY);
                            float screenCOutX = playerRatioToScreen(cOutX);
                            float screenCOutY = damageMultiplierToScreen(cOutY);

                            if (isPointNearMouse(screenCInX, screenCInY, mouseX, mouseY, 6.0f)) {
                                g_uiState.selectedPoint = static_cast<int>(i);
                                g_uiState.selectedControlPoint = 1; // Control in
                                g_uiState.isDragging = true;
                                return;
                            }
                            if (isPointNearMouse(screenCOutX, screenCOutY, mouseX, mouseY, 6.0f)) {
                                g_uiState.selectedPoint = static_cast<int>(i);
                                g_uiState.selectedControlPoint = 2; // Control out
                                g_uiState.isDragging = true;
                                return;
                            }
                        }
                    }
                }

                // STEP 2B: Check if clicking on existing control point
                for (size_t i = 0; i < points.size(); ++i) {
                    float pointScreenX = playerRatioToScreen(points[i].playerRatio);
                    float pointScreenY = damageMultiplierToScreen(points[i].damageMultiplier);

                    if (isPointNearMouse(pointScreenX, pointScreenY, mouseX, mouseY, POINT_HOVER_RADIUS)) {
                        g_uiState.selectedPoint = static_cast<int>(i);
                        g_uiState.selectedControlPoint = 0; // Point itself
                        g_uiState.isDragging = true;
                        g_uiState.dragStarted = true;
                        g_uiState.dragStartPoint = points[i]; // Store original position
                        return; // Exit early - point selected
                    }
                }

                // STEP 2B: No point clicked - add new point
                float playerRatio = g_uiState.snapValueToGrid(
                    screenToPlayerRatio(mouseX), g_uiState.gridSnapSize);
                float damageMultiplier = g_uiState.snapValueToGrid(
                    screenToDamageMultiplier(mouseY), g_uiState.gridSnapSize);

                // Record action before adding point
                auto beforePoints = g_uiState.curve.getPoints();

                // Add the point (validation happens in DamageCurve class)
                g_uiState.curve.addPoint(playerRatio, damageMultiplier);

                // Record action after adding point
                auto afterPoints = g_uiState.curve.getPoints();
                g_uiState.recordAction(ActionType::ADD_POINT, beforePoints, afterPoints,
                                      g_uiState.currentPreset, g_uiState.currentPreset,
                                      static_cast<int>(afterPoints.size() - 1));
                return;
            }

        } else if (action == GLFW_RELEASE) {
            // Record move action if we were dragging a point
            if (g_uiState.dragStarted && g_uiState.selectedPoint >= 0 && g_uiState.selectedControlPoint == 0) {
                auto afterPoints = g_uiState.curve.getPoints();
                auto beforePoints = afterPoints;
                beforePoints[g_uiState.selectedPoint] = g_uiState.dragStartPoint;

                g_uiState.recordAction(ActionType::MOVE_POINT, beforePoints, afterPoints,
                                      g_uiState.currentPreset, g_uiState.currentPreset,
                                      g_uiState.selectedPoint, g_uiState.dragStartPoint,
                                      afterPoints[g_uiState.selectedPoint]);
            }
            g_uiState.isDragging = false;
            g_uiState.selectedPoint = -1;
            g_uiState.selectedControlPoint = -1;
            g_uiState.dragStarted = false;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        // Right-click to remove point
        const auto& points = g_uiState.curve.getPoints();

        for (size_t i = 0; i < points.size(); ++i) {
            float screenX = playerRatioToScreen(points[i].playerRatio);
            float screenY = damageMultiplierToScreen(points[i].damageMultiplier);

            if (isPointNearMouse(screenX, screenY, mouseX, mouseY, POINT_HOVER_RADIUS)) {
                if (points.size() > 2) { // Keep at least 2 points
                    // Record action before removing point
                    auto beforePoints = g_uiState.curve.getPoints();

                    // Remove the point
                    g_uiState.curve.removePoint(i);

                    // Record action after removing point
                    auto afterPoints = g_uiState.curve.getPoints();
                    g_uiState.recordAction(ActionType::REMOVE_POINT, beforePoints, afterPoints,
                                          g_uiState.currentPreset, g_uiState.currentPreset,
                                          static_cast<int>(i), beforePoints[i]);
                }
                break;
            }
        }
    }
}

void cursorPosCallback(GLFWwindow* window, double rawX, double rawY) {
    // Convert GLFW mouse coordinates (Y=0 at top) to OpenGL framebuffer coordinates (Y=0 at bottom)
    // Scale from window coordinates to framebuffer coordinates for high-DPI displays
    float mouseX = static_cast<float>(rawX * g_uiState.dpiScaleX);
    float mouseY = static_cast<float>(g_uiState.currentFramebufferHeight - rawY * g_uiState.dpiScaleY);

    // Update hover state (only if not currently dragging)
    if (!g_uiState.isDragging) {
        const auto& points = g_uiState.curve.getPoints();
        g_uiState.hoveredPoint = -1;

        // Only check hover in curve area
        if (mouseX >= g_uiState.curveAreaLeft && mouseX <= g_uiState.curveAreaRight &&
            mouseY >= g_uiState.curveAreaTop && mouseY <= g_uiState.curveAreaBottom) {

            for (size_t i = 0; i < points.size(); ++i) {
                float screenX = playerRatioToScreen(points[i].playerRatio);
                float screenY = damageMultiplierToScreen(points[i].damageMultiplier);

                if (isPointNearMouse(screenX, screenY, mouseX, mouseY, POINT_HOVER_RADIUS)) {
                    g_uiState.hoveredPoint = static_cast<int>(i);
                    break;
                }
            }
        }
    }

    // Handle dragging
    if (g_uiState.isDragging && g_uiState.selectedPoint >= 0) {
        // Only allow dragging within curve area
        if (mouseX >= g_uiState.curveAreaLeft && mouseX <= g_uiState.curveAreaRight &&
            mouseY >= g_uiState.curveAreaTop && mouseY <= g_uiState.curveAreaBottom) {

            if (g_uiState.selectedControlPoint == 0) {
                // Dragging a point
                float playerRatio = g_uiState.snapValueToGrid(screenToPlayerRatio(mouseX), g_uiState.gridSnapSize);
                float damageMultiplier = g_uiState.snapValueToGrid(screenToDamageMultiplier(mouseY), g_uiState.gridSnapSize);
                g_uiState.curve.movePoint(g_uiState.selectedPoint, playerRatio, damageMultiplier);
            } else if (g_uiState.selectedControlPoint == 1 || g_uiState.selectedControlPoint == 2) {
                // Dragging a control handle
                float controlX = screenToPlayerRatio(mouseX);
                float controlY = screenToDamageMultiplier(mouseY);

                auto points = g_uiState.curve.getPoints();
                if (g_uiState.selectedControlPoint == 1) {
                    // Control in
                    points[g_uiState.selectedPoint].setControlInPoint(controlX, controlY);
                } else {
                    // Control out
                    points[g_uiState.selectedPoint].setControlOutPoint(controlX, controlY);
                }
                g_uiState.curve.setPoints(points);
            }
        }
    }

    g_uiState.lastMouseX = mouseX;
    g_uiState.lastMouseY = mouseY;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_S:
                if (mods & GLFW_MOD_CONTROL) {
                    try {
                        g_uiState.curve.saveConfiguration("simulation_config.txt");
                        g_uiState.setStatusMessage("Configuration saved to simulation_config.txt");

                        // Create backup
                        auto now = std::chrono::system_clock::now();
                        auto time = std::chrono::system_clock::to_time_t(now);
                        std::stringstream backupName;
                        backupName << "backup/simulation_config_backup_" << time << ".txt";
                        g_uiState.curve.saveConfiguration(backupName.str());
                        // Don't show backup message as primary status
                    } catch (const std::exception& e) {
                        g_uiState.setStatusMessage("Save failed: " + std::string(e.what()));
                    }
                }
                break;
            case GLFW_KEY_L:
                if (mods & GLFW_MOD_CONTROL) {
                    try {
                        auto beforePoints = g_uiState.curve.getPoints();
                        CurvePreset beforePreset = g_uiState.currentPreset;

                        g_uiState.curve.loadConfiguration("simulation_config.txt");
                        g_uiState.setStatusMessage("Configuration loaded from simulation_config.txt");

                        auto afterPoints = g_uiState.curve.getPoints();
                        g_uiState.recordAction(ActionType::LOAD_CONFIGURATION, beforePoints, afterPoints,
                                              beforePreset, CurvePreset::CUSTOM);
                    } catch (const std::exception& e) {
                        g_uiState.setStatusMessage("Load failed: " + std::string(e.what()));
                    }
                }
                break;
            case GLFW_KEY_Z:
                if (mods & GLFW_MOD_CONTROL) {
                    if (g_uiState.actionHistory.canUndo()) {
                        g_uiState.undo();
                        g_uiState.setStatusMessage("Undo performed");
                    } else {
                        g_uiState.setStatusMessage("Nothing to undo");
                    }
                }
                break;
            case GLFW_KEY_Y:
                if (mods & GLFW_MOD_CONTROL) {
                    if (g_uiState.actionHistory.canRedo()) {
                        g_uiState.redo();
                        g_uiState.setStatusMessage("Redo performed");
                    } else {
                        g_uiState.setStatusMessage("Nothing to redo");
                    }
                }
                break;
            case GLFW_KEY_C:
                if (mods & GLFW_MOD_CONTROL) {
                    g_uiState.copyCurve();
                    // copyCurve already sets status message
                }
                break;
            case GLFW_KEY_V:
                if (mods & GLFW_MOD_CONTROL) {
                    g_uiState.pasteCurve();
                    // pasteCurve already sets status message
                }
                break;
            case GLFW_KEY_I:
                // Cycle interpolation type for selected point
                if (g_uiState.selectedPoint >= 0) {
                    g_uiState.cycleInterpolationType(g_uiState.selectedPoint);
                    // cycleInterpolationType already sets status message
                } else {
                    g_uiState.setStatusMessage("No point selected");
                }
                break;
            case GLFW_KEY_H:
                // Toggle control handles
                g_uiState.toggleControlHandles();
                // toggleControlHandles already sets status message
                break;
            case GLFW_KEY_E:
                if (mods & GLFW_MOD_CONTROL) {
                    // Export menu - show quick export options
                    g_uiState.setStatusMessage("Quick exports: J=JSON, K=CSV, B=Base64");
                }
                break;
            case GLFW_KEY_J:
                // Quick export to JSON
                try {
                    g_uiState.curve.exportToJson("curve_config.json");
                    g_uiState.setStatusMessage("Exported to curve_config.json");
                } catch (const std::exception& e) {
                    g_uiState.setStatusMessage("Export failed: " + std::string(e.what()));
                }
                break;
            case GLFW_KEY_K:
                // Quick export to CSV
                try {
                    g_uiState.curve.exportToCsv("curve_config.csv");
                    g_uiState.setStatusMessage("Exported to curve_config.csv");
                } catch (const std::exception& e) {
                    g_uiState.setStatusMessage("Export failed: " + std::string(e.what()));
                }
                break;
            case GLFW_KEY_B:
                // Export to Base64 for sharing
                try {
                    std::string base64 = g_uiState.curve.exportToBase64();
                    std::ofstream file("curve_config.b64");
                    file << base64;
                    g_uiState.setStatusMessage("Exported to curve_config.b64 (shareable)");
                } catch (const std::exception& e) {
                    g_uiState.setStatusMessage("Export failed: " + std::string(e.what()));
                }
                break;
            case GLFW_KEY_DELETE:
                // Delete selected point
                g_uiState.deleteSelectedPoint();
                // deleteSelectedPoint already sets status message
                break;
            case GLFW_KEY_LEFT:
                // Nudge left
                g_uiState.nudgeSelectedPoint(-0.01f, 0.0f);
                // nudgeSelectedPoint already sets status message
                break;
            case GLFW_KEY_RIGHT:
                // Nudge right
                g_uiState.nudgeSelectedPoint(0.01f, 0.0f);
                // nudgeSelectedPoint already sets status message
                break;
            case GLFW_KEY_UP:
                // Nudge up
                g_uiState.nudgeSelectedPoint(0.0f, 0.1f);
                // nudgeSelectedPoint already sets status message
                break;
            case GLFW_KEY_DOWN:
                // Nudge down
                g_uiState.nudgeSelectedPoint(0.0f, -0.1f);
                // nudgeSelectedPoint already sets status message
                break;
            case GLFW_KEY_N:
                // Toggle snap to grid
                g_uiState.toggleSnapToGrid();
                // toggleSnapToGrid already sets status message
                break;
            case GLFW_KEY_G:
                g_uiState.showGrid = !g_uiState.showGrid;
                break;
        }
    }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);

    // Get both window size and framebuffer size for DPI scaling
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    // Update UI state with both window and framebuffer sizes
    g_uiState.currentWindowWidth = windowWidth;
    g_uiState.currentWindowHeight = windowHeight;
    g_uiState.currentFramebufferWidth = width;
    g_uiState.currentFramebufferHeight = height;

    // Calculate DPI scale factors
    g_uiState.dpiScaleX = (windowWidth > 0) ? static_cast<float>(width) / static_cast<float>(windowWidth) : 1.0f;
    g_uiState.dpiScaleY = (windowHeight > 0) ? static_cast<float>(height) / static_cast<float>(windowHeight) : 1.0f;

    // Update projection matrix using framebuffer size - keep Y axis going up (standard OpenGL)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);  // Use framebuffer size for projection
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Update curve area using framebuffer coordinates
    updateCurveArea();
    initializeFontRenderer(28.0f * g_uiState.dpiScaleY);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create window
    GLFWwindow* window = glfwCreateWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT,
                                          "BattleRoyale5 - Damage Curve Editor", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Set callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Initialize OpenGL
    glClearColor(BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2], 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Set up initial projection with actual window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    framebufferSizeCallback(window, width, height);

    if (!initializeFontRenderer(28.0f * g_uiState.dpiScaleY)) {
        std::cerr << "Config font unavailable; UI text disabled." << std::endl;
    }

    // Initialize UI state
    updateCurveArea();

    // Try to load previously saved configuration, fall back to preset if none exists
    try {
        g_uiState.curve.loadConfiguration("simulation_config.txt");
        g_uiState.currentPreset = CurvePreset::CUSTOM; // Mark as custom since it's a saved config
        std::cout << "Loaded previously saved configuration with "
                  << g_uiState.curve.getPointCount() << " control points" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "No saved configuration found, loading default BATTLE_ROYALE preset" << std::endl;
        g_uiState.curve.loadPreset(CurvePreset::BATTLE_ROYALE);
        g_uiState.currentPreset = CurvePreset::BATTLE_ROYALE;
    }

    std::cout << "BattleRoyale5 Damage Curve Editor" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Left click: Add point or select/drag existing point/handle" << std::endl;
    std::cout << "  Right click: Remove point" << std::endl;
    std::cout << "  Ctrl+S: Save configuration" << std::endl;
    std::cout << "  Ctrl+L: Load configuration" << std::endl;
    std::cout << "  Ctrl+Z/Y: Undo/Redo" << std::endl;
    std::cout << "  Ctrl+C/V: Copy/Paste curve" << std::endl;
    std::cout << "  I: Cycle interpolation type (Linear -> Spline -> Bezier)" << std::endl;
    std::cout << "  H: Toggle Bezier control handles" << std::endl;
    std::cout << "  G: Toggle grid" << std::endl;
    std::cout << "  V: Toggle validation display" << std::endl;
    std::cout << "  J: Export to JSON" << std::endl;
    std::cout << "  K: Export to CSV" << std::endl;
    std::cout << "  B: Export to Base64 (for sharing)" << std::endl;
    std::cout << "  N: Toggle snap-to-grid" << std::endl;
    std::cout << "  Arrow keys: Nudge selected point" << std::endl;
    std::cout << "  Delete: Remove selected point" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        render();

        glfwSwapBuffers(window);
    }

    // Cleanup
    destroyFontRenderer();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}