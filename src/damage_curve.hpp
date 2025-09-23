#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// Interpolation methods for curve evaluation
enum class InterpolationType {
    LINEAR,    // Linear interpolation between points
    SPLINE     // Catmull-Rom spline interpolation
};

// Curve presets for common battle royale scenarios
enum class CurvePreset {
    LINEAR,           // Constant pacing: 1.0 damage throughout
    EXPONENTIAL,      // Slow start, explosive finale: 0.3 → 3.0
    LOGARITHMIC,      // Fast start, gradual finale: 2.0 → 0.5
    S_CURVE,          // Dramatic tension: slow→fast→slow→explosive
    BATTLE_ROYALE,    // Optimized for BR games: 0.5→0.8→1.5→2.5
    SPEEDRUN,         // Fast elimination: 1.5→2.0→3.0
    ENDURANCE,        // Long battles: 0.2→0.4→0.8→1.2
    CUSTOM            // User-defined curve
};

// Control point for damage curve
struct CurvePoint {
    float playerRatio;       // X-axis: 0.0 (all players) → 1.0 (1 player left)
    float damageMultiplier;  // Y-axis: damage scaling factor
    InterpolationType type;  // Interpolation method for this segment

    CurvePoint() : playerRatio(0.0f), damageMultiplier(1.0f), type(InterpolationType::LINEAR) {}
    CurvePoint(float ratio, float multiplier, InterpolationType interpType = InterpolationType::LINEAR)
        : playerRatio(ratio), damageMultiplier(multiplier), type(interpType) {}

    // Comparison operator for sorting
    bool operator<(const CurvePoint& other) const {
        return playerRatio < other.playerRatio;
    }
};

// Configuration structure for damage curve system
struct DamageCurveConfig {
    static constexpr float DEFAULT_MIN_DAMAGE_MULTIPLIER = 0.1f;
    static constexpr float DEFAULT_MAX_DAMAGE_MULTIPLIER = 5.0f;
    static constexpr size_t DEFAULT_CACHE_SIZE = 1000;
    static constexpr float DEFAULT_SMOOTHING_FACTOR = 0.1f;

    float minDamageMultiplier = DEFAULT_MIN_DAMAGE_MULTIPLIER;
    float maxDamageMultiplier = DEFAULT_MAX_DAMAGE_MULTIPLIER;
    size_t cacheSize = DEFAULT_CACHE_SIZE;
    float smoothingFactor = DEFAULT_SMOOTHING_FACTOR;
    bool enableCurveSmoothing = true;
    uint32_t curveUpdateFrequency = 60; // Hz
};

// Main damage curve class for evaluating damage scaling based on player elimination ratio
class DamageCurve {
public:
    DamageCurve();
    explicit DamageCurve(const DamageCurveConfig& config);

    // Core curve evaluation
    float evaluateCurve(float playerRatio) const;

    // Point manipulation
    void addPoint(float playerRatio, float damageMultiplier, InterpolationType type = InterpolationType::LINEAR);
    void removePoint(size_t index);
    void movePoint(size_t index, float playerRatio, float damageMultiplier);
    void clearPoints();

    // Preset management
    void loadPreset(CurvePreset preset);
    void loadConfiguration(const std::string& filename);
    void saveConfiguration(const std::string& filename) const;

    // Curve access and validation
    const std::vector<CurvePoint>& getPoints() const { return points_; }
    size_t getPointCount() const { return points_.size(); }
    bool isValid() const;
    void validateAndClamp();

    // Configuration
    void setConfig(const DamageCurveConfig& config) { config_ = config; }
    const DamageCurveConfig& getConfig() const { return config_; }

    // Performance optimization
    void rebuildCache();
    void clearCache();

private:
    std::vector<CurvePoint> points_;
    DamageCurveConfig config_;

    // Performance cache for common evaluations
    mutable std::vector<std::pair<float, float>> cache_;
    mutable bool cacheValid_ = false;

    // Helper methods
    float linearInterpolation(float t, float y0, float y1) const;
    float catmullRomSpline(float t, float y0, float y1, float y2, float y3) const;
    size_t findSegment(float playerRatio) const;
    void sortPoints();
    void ensureDefaultPoints();
    float clampDamageMultiplier(float value) const;
};

// Utility function for main simulation integration
float calculateDynamicDamage(const DamageCurve& curve, float baseDamage, uint32_t aliveCount, uint32_t totalPlayers);