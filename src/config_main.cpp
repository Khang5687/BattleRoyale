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

#include "damage_curve.hpp"

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

// UI state
struct UIState {
    DamageCurve curve;
    std::vector<CurvePoint> displayPoints;
    int selectedPoint = -1;
    int hoveredPoint = -1;
    bool isDragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

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
    CurvePreset currentPreset = CurvePreset::BATTLE_ROYALE;
};

UIState g_uiState;

// Coordinate conversion functions - REWRITTEN FOR CORRECTNESS
// Player ratio: 0.0 (left) to 1.0 (right)
// Damage multiplier: 0.5 (bottom) to 4.5 (top)

float screenToPlayerRatio(float screenX) {
    // Simple linear mapping: screen X to 0.0-1.0 range
    if (g_uiState.curveWidth <= 0) return 0.0f;
    return std::clamp((screenX - g_uiState.curveAreaLeft) / g_uiState.curveWidth, 0.0f, 1.0f);
}

float screenToDamageMultiplier(float screenY) {
    // Map screen Y to damage range 0.5-4.5
    if (g_uiState.curveHeight <= 0) return 1.0f;

    // Normalize screen Y to 0-1 range (0 = bottom, 1 = top)
    float normalizedY = (screenY - g_uiState.curveAreaTop) / g_uiState.curveHeight;
    normalizedY = std::clamp(normalizedY, 0.0f, 1.0f);

    // Map to damage multiplier range: 0.5 (bottom) to 4.5 (top)
    return 0.5f + normalizedY * 4.0f;
}

float playerRatioToScreen(float playerRatio) {
    // Convert player ratio back to screen X coordinate
    return g_uiState.curveAreaLeft + std::clamp(playerRatio, 0.0f, 1.0f) * g_uiState.curveWidth;
}

float damageMultiplierToScreen(float damageMultiplier) {
    // Convert damage multiplier back to screen Y coordinate

    // Normalize damage multiplier to 0-1 range
    float normalizedDamage = (std::clamp(damageMultiplier, 0.5f, 4.5f) - 0.5f) / 4.0f;

    // Map to screen Y coordinate
    return g_uiState.curveAreaTop + normalizedDamage * g_uiState.curveHeight;
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
void drawText(float x, float y, const std::string& text, const float color[3] = TEXT_COLOR) {
    glColor3f(color[0], color[1], color[2]);

    // Simple character rendering - each character is 8x12 pixels
    const float charWidth = 8.0f;
    const float charHeight = 12.0f;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        float charX = x + i * charWidth;

        // Skip non-printable characters
        if (c < 32 || c > 126) continue;

        // Draw a simple rectangle representation for each character
        // This is a minimal implementation - could be enhanced with actual bitmap fonts
        glBegin(GL_LINE_LOOP);
        glVertex2f(charX, y);
        glVertex2f(charX + charWidth - 2, y);
        glVertex2f(charX + charWidth - 2, y + charHeight);
        glVertex2f(charX, y + charHeight);
        glEnd();

        // Draw some simple character patterns for common letters
        glBegin(GL_LINES);
        if (c >= 'A' && c <= 'Z') {
            // Draw a simple pattern for uppercase letters
            glVertex2f(charX + 2, y + 2);
            glVertex2f(charX + charWidth - 4, y + charHeight - 2);
        } else if (c >= 'a' && c <= 'z') {
            // Draw a simple pattern for lowercase letters
            glVertex2f(charX + 2, y + charHeight/2);
            glVertex2f(charX + charWidth - 4, y + charHeight/2);
        } else if (c >= '0' && c <= '9') {
            // Draw a simple pattern for numbers
            glVertex2f(charX + charWidth/2, y + 2);
            glVertex2f(charX + charWidth/2, y + charHeight - 2);
        }
        glEnd();
    }
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

        drawCircle(screenX, screenY, radius, color, true);
        drawCircle(screenX, screenY, radius + 2.0f, AXIS_COLOR, false);
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
    float titleY = g_uiState.currentFramebufferHeight - 20.0f * g_uiState.dpiScaleY;
    const float titleColor[3] = {1.0f, 1.0f, 1.0f};
    drawText(15.0f * g_uiState.dpiScaleX, titleY, "Damage Curve Presets:", titleColor);

    // Add instructions at bottom of panel
    float instructY = 80.0f * g_uiState.dpiScaleY;
    const float instructColor[3] = {0.7f, 0.7f, 0.7f};
    float instructX = 15.0f * g_uiState.dpiScaleX;
    float instructLineSpacing = 15.0f * g_uiState.dpiScaleY;
    drawText(instructX, instructY, "Ctrl+S: Save", instructColor);
    drawText(instructX, instructY - instructLineSpacing, "Ctrl+L: Load", instructColor);
    drawText(instructX, instructY - 2 * instructLineSpacing, "G: Toggle Grid", instructColor);
    drawText(instructX, instructY - 3 * instructLineSpacing, "Right-click: Del Point", instructColor);

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

    // Current preset indicator
    if (g_uiState.currentPreset != CurvePreset::CUSTOM) {
        float indicatorY = startY + 40.0f * g_uiState.dpiScaleY;
        const float indicatorColor[3] = {0.5f, 1.0f, 0.5f};
        const std::string currentText = "Active: " + presetNames[static_cast<int>(g_uiState.currentPreset)];
        drawText(15.0f * g_uiState.dpiScaleX, indicatorY, currentText, indicatorColor);
    }
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT);

    drawGrid();
    drawCurve();
    drawControlPoints();
    drawUI();
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
                            g_uiState.currentPreset = static_cast<CurvePreset>(i);
                            g_uiState.curve.loadPreset(g_uiState.currentPreset);
                            return; // Exit early - button handled
                        }
                    }
                }
                return; // Click in UI panel but no button - ignore
            }

            // STEP 2: Check if clicking in curve area
            if (mouseX >= g_uiState.curveAreaLeft && mouseX <= g_uiState.curveAreaRight &&
                mouseY >= g_uiState.curveAreaTop && mouseY <= g_uiState.curveAreaBottom) {

                // STEP 2A: Check if clicking on existing control point
                const auto& points = g_uiState.curve.getPoints();
                for (size_t i = 0; i < points.size(); ++i) {
                    float pointScreenX = playerRatioToScreen(points[i].playerRatio);
                    float pointScreenY = damageMultiplierToScreen(points[i].damageMultiplier);

                    if (isPointNearMouse(pointScreenX, pointScreenY, mouseX, mouseY, POINT_HOVER_RADIUS)) {
                        g_uiState.selectedPoint = static_cast<int>(i);
                        g_uiState.isDragging = true;
                        return; // Exit early - point selected
                    }
                }

                // STEP 2B: No point clicked - add new point
                float playerRatio = screenToPlayerRatio(mouseX);
                float damageMultiplier = screenToDamageMultiplier(mouseY);

                // Add the point (validation happens in DamageCurve class)
                g_uiState.curve.addPoint(playerRatio, damageMultiplier);
                return;
            }

        } else if (action == GLFW_RELEASE) {
            g_uiState.isDragging = false;
            g_uiState.selectedPoint = -1;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        // Right-click to remove point
        const auto& points = g_uiState.curve.getPoints();

        for (size_t i = 0; i < points.size(); ++i) {
            float screenX = playerRatioToScreen(points[i].playerRatio);
            float screenY = damageMultiplierToScreen(points[i].damageMultiplier);

            if (isPointNearMouse(screenX, screenY, mouseX, mouseY, POINT_HOVER_RADIUS)) {
                if (points.size() > 2) { // Keep at least 2 points
                    g_uiState.curve.removePoint(i);
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

            float playerRatio = screenToPlayerRatio(mouseX);
            float damageMultiplier = screenToDamageMultiplier(mouseY);

            // Move the point (validation happens in DamageCurve class)
            g_uiState.curve.movePoint(g_uiState.selectedPoint, playerRatio, damageMultiplier);
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
                    g_uiState.curve.saveConfiguration("simulation_config.txt");
                    std::cout << "Configuration saved to simulation_config.txt" << std::endl;
                }
                break;
            case GLFW_KEY_L:
                if (mods & GLFW_MOD_CONTROL) {
                    try {
                        g_uiState.curve.loadConfiguration("simulation_config.txt");
                        std::cout << "Configuration loaded from simulation_config.txt" << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "Failed to load configuration: " << e.what() << std::endl;
                    }
                }
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

    // Initialize UI state
    updateCurveArea();
    g_uiState.curve.loadPreset(CurvePreset::BATTLE_ROYALE);

    std::cout << "BattleRoyale5 Damage Curve Editor" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Left click: Add point or select/drag existing point" << std::endl;
    std::cout << "  Right click: Remove point" << std::endl;
    std::cout << "  Ctrl+S: Save configuration" << std::endl;
    std::cout << "  Ctrl+L: Load configuration" << std::endl;
    std::cout << "  G: Toggle grid" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        render();

        glfwSwapBuffers(window);
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}