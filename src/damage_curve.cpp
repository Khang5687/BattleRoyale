#include "damage_curve.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

DamageCurve::DamageCurve() : config_(DamageCurveConfig{}) {
    ensureDefaultPoints();
}

DamageCurve::DamageCurve(const DamageCurveConfig& config) : config_(config) {
    ensureDefaultPoints();
}

float DamageCurve::evaluateCurve(float playerRatio) const {
    // Clamp input to valid range
    playerRatio = std::clamp(playerRatio, 0.0f, 1.0f);

    // Handle edge cases
    if (points_.empty()) {
        return 1.0f; // Default multiplier if no points
    }
    if (points_.size() == 1) {
        return clampDamageMultiplier(points_[0].damageMultiplier);
    }

    // Find the segment containing this ratio
    size_t segmentIndex = findSegment(playerRatio);

    // Handle boundary cases
    if (segmentIndex == 0) {
        return clampDamageMultiplier(points_[0].damageMultiplier);
    }
    if (segmentIndex >= points_.size()) {
        return clampDamageMultiplier(points_.back().damageMultiplier);
    }

    // Get interpolation parameters
    const CurvePoint& p0 = points_[segmentIndex - 1];
    const CurvePoint& p1 = points_[segmentIndex];

    float t = (playerRatio - p0.playerRatio) / (p1.playerRatio - p0.playerRatio);
    t = std::clamp(t, 0.0f, 1.0f);

    float result;

    // Use interpolation method from the target point
    switch (p1.type) {
        case InterpolationType::LINEAR:
            result = linearInterpolation(t, p0.damageMultiplier, p1.damageMultiplier);
            break;

        case InterpolationType::SPLINE:
            // For spline, we need 4 points. Use linear fallback if not enough points.
            if (points_.size() >= 4 && segmentIndex >= 2 && segmentIndex < points_.size() - 1) {
                const CurvePoint& p_1 = points_[segmentIndex - 2];
                const CurvePoint& p2 = points_[segmentIndex + 1];
                result = catmullRomSpline(t, p_1.damageMultiplier, p0.damageMultiplier,
                                        p1.damageMultiplier, p2.damageMultiplier);
            } else {
                result = linearInterpolation(t, p0.damageMultiplier, p1.damageMultiplier);
            }
            break;

        default:
            result = linearInterpolation(t, p0.damageMultiplier, p1.damageMultiplier);
            break;
    }

    return clampDamageMultiplier(result);
}

void DamageCurve::addPoint(float playerRatio, float damageMultiplier, InterpolationType type) {
    points_.emplace_back(playerRatio, damageMultiplier, type);
    sortPoints();
    cacheValid_ = false;
}

void DamageCurve::removePoint(size_t index) {
    if (index < points_.size()) {
        points_.erase(points_.begin() + index);
        cacheValid_ = false;
    }
}

void DamageCurve::movePoint(size_t index, float playerRatio, float damageMultiplier) {
    if (index < points_.size()) {
        points_[index].playerRatio = playerRatio;
        points_[index].damageMultiplier = damageMultiplier;
        sortPoints();
        cacheValid_ = false;
    }
}

void DamageCurve::clearPoints() {
    points_.clear();
    cacheValid_ = false;
}

void DamageCurve::loadPreset(CurvePreset preset) {
    clearPoints();

    switch (preset) {
        case CurvePreset::LINEAR:
            addPoint(0.0f, 1.0f);
            addPoint(1.0f, 1.0f);
            break;

        case CurvePreset::EXPONENTIAL:
            addPoint(0.0f, 0.3f);
            addPoint(0.3f, 0.5f);
            addPoint(0.7f, 1.2f);
            addPoint(1.0f, 3.0f);
            break;

        case CurvePreset::LOGARITHMIC:
            addPoint(0.0f, 2.0f);
            addPoint(0.3f, 1.2f);
            addPoint(0.7f, 0.8f);
            addPoint(1.0f, 0.5f);
            break;

        case CurvePreset::S_CURVE:
            addPoint(0.0f, 0.4f, InterpolationType::SPLINE);
            addPoint(0.2f, 0.6f, InterpolationType::SPLINE);
            addPoint(0.5f, 1.0f, InterpolationType::SPLINE);
            addPoint(0.8f, 1.8f, InterpolationType::SPLINE);
            addPoint(1.0f, 2.5f, InterpolationType::SPLINE);
            break;

        case CurvePreset::BATTLE_ROYALE:
            addPoint(0.0f, 0.5f);
            addPoint(0.3f, 0.8f);
            addPoint(0.7f, 1.5f);
            addPoint(1.0f, 2.5f);
            break;

        case CurvePreset::SPEEDRUN:
            addPoint(0.0f, 1.5f);
            addPoint(0.5f, 2.0f);
            addPoint(1.0f, 3.0f);
            break;

        case CurvePreset::ENDURANCE:
            addPoint(0.0f, 0.2f);
            addPoint(0.4f, 0.4f);
            addPoint(0.8f, 0.8f);
            addPoint(1.0f, 1.2f);
            break;

        case CurvePreset::CUSTOM:
        default:
            ensureDefaultPoints();
            break;
    }

    validateAndClamp();
}

void DamageCurve::loadConfiguration(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open damage curve config file: " << filename << std::endl;
        loadPreset(CurvePreset::BATTLE_ROYALE); // Fallback to default
        return;
    }

    clearPoints();

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            std::getline(iss, value);

            if (key == "curve_points") {
                // Parse curve points: "0.0:0.3,0.2:0.5,0.5:0.8,0.8:1.2,1.0:2.0"
                std::istringstream pointStream(value);
                std::string point;
                while (std::getline(pointStream, point, ',')) {
                    size_t colonPos = point.find(':');
                    if (colonPos != std::string::npos) {
                        float ratio = std::stof(point.substr(0, colonPos));
                        float multiplier = std::stof(point.substr(colonPos + 1));
                        addPoint(ratio, multiplier);
                    }
                }
            } else if (key == "min_damage_multiplier") {
                config_.minDamageMultiplier = std::stof(value);
            } else if (key == "max_damage_multiplier") {
                config_.maxDamageMultiplier = std::stof(value);
            } else if (key == "enable_curve_smoothing") {
                config_.enableCurveSmoothing = (value == "true");
            } else if (key == "curve_update_frequency") {
                config_.curveUpdateFrequency = std::stoi(value);
            }
        }
    }

    if (points_.empty()) {
        ensureDefaultPoints();
    }

    validateAndClamp();
}

void DamageCurve::saveConfiguration(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write damage curve config file: " << filename << std::endl;
        return;
    }

    file << "# simulation_config.txt - Damage Scaling Configuration" << std::endl;
    file << "damage_curve_version=1.0" << std::endl;
    file << std::endl;

    file << "# Curve interpolation method: linear, spline" << std::endl;
    file << "curve_interpolation=spline" << std::endl;
    file << std::endl;

    file << "# Control points: playerRatio:damageMultiplier (sorted by playerRatio)" << std::endl;
    file << "curve_points=";
    for (size_t i = 0; i < points_.size(); ++i) {
        if (i > 0) file << ",";
        file << points_[i].playerRatio << ":" << points_[i].damageMultiplier;
    }
    file << std::endl;
    file << std::endl;

    file << "# Curve validation bounds" << std::endl;
    file << "min_damage_multiplier=" << config_.minDamageMultiplier << std::endl;
    file << "max_damage_multiplier=" << config_.maxDamageMultiplier << std::endl;
    file << std::endl;

    file << "# Additional simulation parameters" << std::endl;
    file << "enable_curve_smoothing=" << (config_.enableCurveSmoothing ? "true" : "false") << std::endl;
    file << "curve_update_frequency=" << config_.curveUpdateFrequency << std::endl;
}

bool DamageCurve::isValid() const {
    if (points_.empty()) {
        return false;
    }

    // Check if points are sorted
    for (size_t i = 1; i < points_.size(); ++i) {
        if (points_[i].playerRatio <= points_[i-1].playerRatio) {
            return false;
        }
    }

    // Check bounds
    for (const auto& point : points_) {
        if (point.playerRatio < 0.0f || point.playerRatio > 1.0f) {
            return false;
        }
        if (point.damageMultiplier < config_.minDamageMultiplier ||
            point.damageMultiplier > config_.maxDamageMultiplier) {
            return false;
        }
    }

    return true;
}

void DamageCurve::validateAndClamp() {
    if (points_.empty()) {
        ensureDefaultPoints();
        return;
    }

    // Clamp all values to valid ranges
    for (auto& point : points_) {
        point.playerRatio = std::clamp(point.playerRatio, 0.0f, 1.0f);
        point.damageMultiplier = clampDamageMultiplier(point.damageMultiplier);
    }

    sortPoints();
    cacheValid_ = false;
}

void DamageCurve::rebuildCache() {
    cache_.clear();
    cache_.reserve(config_.cacheSize);

    for (size_t i = 0; i < config_.cacheSize; ++i) {
        float ratio = static_cast<float>(i) / (config_.cacheSize - 1);
        float value = evaluateCurve(ratio);
        cache_.emplace_back(ratio, value);
    }

    cacheValid_ = true;
}

void DamageCurve::clearCache() {
    cache_.clear();
    cacheValid_ = false;
}

float DamageCurve::linearInterpolation(float t, float y0, float y1) const {
    return y0 + t * (y1 - y0);
}

float DamageCurve::catmullRomSpline(float t, float y0, float y1, float y2, float y3) const {
    float t2 = t * t;
    float t3 = t2 * t;

    return 0.5f * (
        2.0f * y1 +
        (-y0 + y2) * t +
        (2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3) * t2 +
        (-y0 + 3.0f * y1 - 3.0f * y2 + y3) * t3
    );
}

size_t DamageCurve::findSegment(float playerRatio) const {
    // Binary search for the segment
    auto it = std::upper_bound(points_.begin(), points_.end(), playerRatio,
        [](float ratio, const CurvePoint& point) {
            return ratio < point.playerRatio;
        });

    return std::distance(points_.begin(), it);
}

void DamageCurve::sortPoints() {
    std::sort(points_.begin(), points_.end());
}

void DamageCurve::ensureDefaultPoints() {
    if (points_.empty()) {
        // Default to BATTLE_ROYALE preset
        addPoint(0.0f, 0.5f);
        addPoint(0.3f, 0.8f);
        addPoint(0.7f, 1.5f);
        addPoint(1.0f, 2.5f);
    }
}

float DamageCurve::clampDamageMultiplier(float value) const {
    return std::clamp(value, config_.minDamageMultiplier, config_.maxDamageMultiplier);
}

// Utility function for main simulation integration
float calculateDynamicDamage(const DamageCurve& curve, float baseDamage, uint32_t aliveCount, uint32_t totalPlayers) {
    if (totalPlayers == 0) {
        return baseDamage; // Avoid division by zero
    }

    // Calculate player elimination ratio (0.0 = all alive, 1.0 = only winner left)
    float playerRatio = 1.0f - static_cast<float>(aliveCount) / static_cast<float>(totalPlayers);

    // Get curve multiplier and apply to base damage
    float curveMultiplier = curve.evaluateCurve(playerRatio);
    return baseDamage * curveMultiplier;
}