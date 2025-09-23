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

        case InterpolationType::BEZIER:
            // Cubic Bezier curve using control points
            result = cubicBezierInterpolation(t, p0, p1);
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

void DamageCurve::setPoints(const std::vector<CurvePoint>& newPoints) {
    points_ = newPoints;
    sortPoints();
    cacheValid_ = false;
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

float DamageCurve::cubicBezierInterpolation(float t, const CurvePoint& p0, const CurvePoint& p1) const {
    // Cubic Bezier curve: B(t) = (1-t)^3 * P0 + 3*(1-t)^2*t * P0_control_out + 3*(1-t)*t^2 * P1_control_in + t^3 * P1
    // But we need to interpolate in both X and Y dimensions since control points affect the curve shape

    // Get control points
    float p0x = p0.playerRatio;
    float p0y = p0.damageMultiplier;
    float p1x = p1.playerRatio;
    float p1y = p1.damageMultiplier;

    // Control points (absolute positions)
    float c0x, c0y, c1x, c1y;
    p0.getControlOutPoint(c0x, c0y);
    p1.getControlInPoint(c1x, c1y);

    // Clamp control points to reasonable bounds to prevent extreme curves
    c0x = std::clamp(c0x, p0x - 0.5f, p1x + 0.5f);
    c1x = std::clamp(c1x, p0x - 0.5f, p1x + 0.5f);
    c0y = std::clamp(c0y, 0.1f, 5.0f);
    c1y = std::clamp(c1y, 0.1f, 5.0f);

    // For damage multiplier interpolation, we use the Y values directly
    // since the X control points affect timing/curvature
    float t2 = t * t;
    float t3 = t2 * t;
    float oneMinusT = 1.0f - t;
    float oneMinusT2 = oneMinusT * oneMinusT;
    float oneMinusT3 = oneMinusT2 * oneMinusT;

    // Cubic Bezier formula for Y (damage multiplier)
    return oneMinusT3 * p0y + 3.0f * oneMinusT2 * t * c0y + 3.0f * oneMinusT * t2 * c1y + t3 * p1y;
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

DamageCurve::CurveValidationResult DamageCurve::validateCurve() const {
    CurveValidationResult result;

    if (points_.size() < 2) {
        result.isValid = false;
        return result;
    }

    // Check for basic validity
    for (const auto& point : points_) {
        if (point.playerRatio < 0.0f || point.playerRatio > 1.0f ||
            point.damageMultiplier < config_.minDamageMultiplier ||
            point.damageMultiplier > config_.maxDamageMultiplier) {
            result.isValid = false;
            return result;
        }
    }

    // Analyze curve smoothness
    const size_t sampleCount = 100;
    std::vector<float> samples(sampleCount);
    std::vector<float> derivatives(sampleCount - 1);

    // Sample the curve
    for (size_t i = 0; i < sampleCount; ++i) {
        float t = static_cast<float>(i) / (sampleCount - 1);
        samples[i] = evaluateCurve(t);
    }

    // Calculate first derivatives (slopes)
    for (size_t i = 0; i < sampleCount - 1; ++i) {
        float dt = 1.0f / (sampleCount - 1);
        derivatives[i] = (samples[i + 1] - samples[i]) / dt;
    }

    // Find discontinuities and sharp corners
    float discontinuityThreshold = 0.5f; // Large slope changes
    float sharpCornerThreshold = 0.3f;   // Moderate slope changes

    for (size_t i = 1; i < derivatives.size(); ++i) {
        float slopeChange = std::abs(derivatives[i] - derivatives[i - 1]);
        result.maxSlopeChange = std::max(result.maxSlopeChange, slopeChange);

        if (slopeChange > discontinuityThreshold) {
            result.hasDiscontinuities = true;
            // Map back to approximate point index
            float t = static_cast<float>(i) / (sampleCount - 1);
            size_t pointIndex = findSegment(t);
            if (std::find(result.discontinuityPoints.begin(), result.discontinuityPoints.end(), pointIndex) ==
                result.discontinuityPoints.end()) {
                result.discontinuityPoints.push_back(pointIndex);
            }
        } else if (slopeChange > sharpCornerThreshold) {
            result.hasSharpCorners = true;
            float t = static_cast<float>(i) / (sampleCount - 1);
            size_t pointIndex = findSegment(t);
            if (std::find(result.sharpCornerPoints.begin(), result.sharpCornerPoints.end(), pointIndex) ==
                result.sharpCornerPoints.end()) {
                result.sharpCornerPoints.push_back(pointIndex);
            }
        }
    }

    // Estimate curvature (second derivative approximation)
    std::vector<float> secondDerivatives(sampleCount - 2);
    for (size_t i = 0; i < secondDerivatives.size(); ++i) {
        float dt = 1.0f / (sampleCount - 1);
        secondDerivatives[i] = (derivatives[i + 1] - derivatives[i]) / dt;
        result.maxCurvature = std::max(result.maxCurvature, std::abs(secondDerivatives[i]));
    }

    return result;
}

float DamageCurve::estimateSmoothness(size_t sampleCount) const {
    if (points_.size() < 2) return 0.0f;

    // Sample the curve and analyze derivatives
    std::vector<float> samples(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i) {
        float t = static_cast<float>(i) / (sampleCount - 1);
        samples[i] = evaluateCurve(t);
    }

    // Calculate variance in first derivatives as smoothness measure
    // Lower variance = smoother curve
    float meanDerivative = 0.0f;
    std::vector<float> derivatives(sampleCount - 1);

    for (size_t i = 0; i < sampleCount - 1; ++i) {
        float dt = 1.0f / (sampleCount - 1);
        derivatives[i] = (samples[i + 1] - samples[i]) / dt;
        meanDerivative += derivatives[i];
    }
    meanDerivative /= derivatives.size();

    // Calculate variance
    float variance = 0.0f;
    for (float derivative : derivatives) {
        float diff = derivative - meanDerivative;
        variance += diff * diff;
    }
    variance /= derivatives.size();

    // Convert variance to smoothness score (0-1, higher is smoother)
    // Normalize by expected range of damage multipliers
    float expectedVariance = (config_.maxDamageMultiplier - config_.minDamageMultiplier) * 0.1f;
    return std::max(0.0f, 1.0f - variance / (expectedVariance * expectedVariance));
}

std::vector<float> DamageCurve::findDiscontinuities(float threshold) const {
    std::vector<float> discontinuities;

    if (points_.size() < 2) return discontinuities;

    // Check for interpolation type changes that might cause discontinuities
    for (size_t i = 1; i < points_.size(); ++i) {
        if (points_[i].type != points_[i - 1].type) {
            // Different interpolation types at segment boundary
            discontinuities.push_back(points_[i].playerRatio);
        }
    }

    // Sample curve and look for large derivative changes
    const size_t sampleCount = 200;
    std::vector<float> samples(sampleCount);

    for (size_t i = 0; i < sampleCount; ++i) {
        float t = static_cast<float>(i) / (sampleCount - 1);
        samples[i] = evaluateCurve(t);
    }

    for (size_t i = 2; i < sampleCount - 2; ++i) {
        // Check for large changes in local derivatives
        float dt = 1.0f / (sampleCount - 1);
        float deriv1 = (samples[i] - samples[i - 1]) / dt;
        float deriv2 = (samples[i + 1] - samples[i]) / dt;
        float derivChange = std::abs(deriv2 - deriv1);

        if (derivChange > threshold) {
            float t = static_cast<float>(i) / (sampleCount - 1);
            discontinuities.push_back(t);
        }
    }

    // Remove duplicates and sort
    std::sort(discontinuities.begin(), discontinuities.end());
    auto last = std::unique(discontinuities.begin(), discontinuities.end());
    discontinuities.erase(last, discontinuities.end());

    return discontinuities;
}

void DamageCurve::exportToJson(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for JSON export: " + filename);
    }

    file << "{\n";
    file << "  \"version\": \"1.0\",\n";
    file << "  \"description\": \"Battle Royale Damage Curve Configuration\",\n";
    file << "  \"points\": [\n";

    for (size_t i = 0; i < points_.size(); ++i) {
        const auto& point = points_[i];
        file << "    {\n";
        file << "      \"playerRatio\": " << point.playerRatio << ",\n";
        file << "      \"damageMultiplier\": " << point.damageMultiplier << ",\n";
        file << "      \"interpolationType\": \"";

        switch (point.type) {
            case InterpolationType::LINEAR: file << "linear"; break;
            case InterpolationType::SPLINE: file << "spline"; break;
            case InterpolationType::BEZIER: file << "bezier"; break;
        }

        file << "\",\n";
        file << "      \"controlPoints\": {\n";
        file << "        \"in\": {\"x\": " << (point.playerRatio + point.controlInX) << ", \"y\": " << (point.damageMultiplier + point.controlInY) << "},\n";
        file << "        \"out\": {\"x\": " << (point.playerRatio + point.controlOutX) << ", \"y\": " << (point.damageMultiplier + point.controlOutY) << "}\n";
        file << "      }\n";
        file << "    }";

        if (i < points_.size() - 1) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ],\n";
    file << "  \"config\": {\n";
    file << "    \"minDamageMultiplier\": " << config_.minDamageMultiplier << ",\n";
    file << "    \"maxDamageMultiplier\": " << config_.maxDamageMultiplier << ",\n";
    file << "    \"cacheSize\": " << config_.cacheSize << ",\n";
    file << "    \"smoothingFactor\": " << config_.smoothingFactor << ",\n";
    file << "    \"enableCurveSmoothing\": " << (config_.enableCurveSmoothing ? "true" : "false") << ",\n";
    file << "    \"curveUpdateFrequency\": " << config_.curveUpdateFrequency << "\n";
    file << "  }\n";
    file << "}\n";
}

void DamageCurve::importFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for JSON import: " + filename);
    }

    // Simple JSON parsing (in production, use a proper JSON library)
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    // For now, throw an error - proper JSON parsing would be complex
    throw std::runtime_error("JSON import not yet implemented. Use the existing text format.");
}

void DamageCurve::exportToCsv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for CSV export: " + filename);
    }

    // CSV header
    file << "playerRatio,damageMultiplier,interpolationType,controlInX,controlInY,controlOutX,controlOutY\n";

    for (const auto& point : points_) {
        file << point.playerRatio << ",";
        file << point.damageMultiplier << ",";

        switch (point.type) {
            case InterpolationType::LINEAR: file << "linear"; break;
            case InterpolationType::SPLINE: file << "spline"; break;
            case InterpolationType::BEZIER: file << "bezier"; break;
        }

        file << ",";
        file << point.controlInX << ",";
        file << point.controlInY << ",";
        file << point.controlOutX << ",";
        file << point.controlOutY << "\n";
    }
}

void DamageCurve::importFromCsv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for CSV import: " + filename);
    }

    points_.clear();
    std::string line;
    bool isHeader = true;

    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue; // Skip header
        }

        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        CurvePoint point;

        // Parse CSV fields
        std::getline(ss, token, ',');
        point.playerRatio = std::stof(token);

        std::getline(ss, token, ',');
        point.damageMultiplier = std::stof(token);

        std::getline(ss, token, ',');
        if (token == "linear") point.type = InterpolationType::LINEAR;
        else if (token == "spline") point.type = InterpolationType::SPLINE;
        else if (token == "bezier") point.type = InterpolationType::BEZIER;

        std::getline(ss, token, ',');
        point.controlInX = std::stof(token);

        std::getline(ss, token, ',');
        point.controlInY = std::stof(token);

        std::getline(ss, token, ',');
        point.controlOutX = std::stof(token);

        std::getline(ss, token, ',');
        point.controlOutY = std::stof(token);

        points_.push_back(point);
    }

    sortPoints();
    cacheValid_ = false;
}

std::string DamageCurve::exportToBase64() const {
    // Serialize to binary format first, then base64 encode
    // For simplicity, we'll use a text-based serialization
    std::stringstream ss;
    ss << "BRDCv1"; // Magic header
    ss << static_cast<uint32_t>(points_.size());

    for (const auto& point : points_) {
        ss.write(reinterpret_cast<const char*>(&point.playerRatio), sizeof(float));
        ss.write(reinterpret_cast<const char*>(&point.damageMultiplier), sizeof(float));
        ss.write(reinterpret_cast<const char*>(&point.type), sizeof(InterpolationType));
        ss.write(reinterpret_cast<const char*>(&point.controlInX), sizeof(float));
        ss.write(reinterpret_cast<const char*>(&point.controlInY), sizeof(float));
        ss.write(reinterpret_cast<const char*>(&point.controlOutX), sizeof(float));
        ss.write(reinterpret_cast<const char*>(&point.controlOutY), sizeof(float));
    }

    std::string binaryData = ss.str();

    // Simple base64 encoding (simplified implementation)
    static const char* base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    int val = 0;
    int valb = -6;

    for (unsigned char c : binaryData) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(base64Chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }

    if (valb > -6) {
        encoded.push_back(base64Chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }

    while (encoded.size() % 4) {
        encoded.push_back('=');
    }

    return encoded;
}

void DamageCurve::importFromBase64(const std::string& base64Data) {
    // Simple base64 decoding (simplified implementation)
    std::string binaryData;
    int val = 0;
    int valb = -8;

    for (char c : base64Data) {
        if (c == '=') break;

        static const std::string base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        auto pos = base64Chars.find(c);
        if (pos == std::string::npos) continue;

        val = (val << 6) + pos;
        valb += 6;
        if (valb >= 0) {
            binaryData.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }

    // Deserialize from binary
    std::stringstream ss(binaryData);
    char magic[7] = {0};
    ss.read(magic, 6);
    if (std::string(magic) != "BRDCv1") {
        throw std::runtime_error("Invalid base64 data format");
    }

    uint32_t pointCount;
    ss.read(reinterpret_cast<char*>(&pointCount), sizeof(uint32_t));

    points_.clear();
    for (uint32_t i = 0; i < pointCount; ++i) {
        CurvePoint point;
        ss.read(reinterpret_cast<char*>(&point.playerRatio), sizeof(float));
        ss.read(reinterpret_cast<char*>(&point.damageMultiplier), sizeof(float));
        ss.read(reinterpret_cast<char*>(&point.type), sizeof(InterpolationType));
        ss.read(reinterpret_cast<char*>(&point.controlInX), sizeof(float));
        ss.read(reinterpret_cast<char*>(&point.controlInY), sizeof(float));
        ss.read(reinterpret_cast<char*>(&point.controlOutX), sizeof(float));
        ss.read(reinterpret_cast<char*>(&point.controlOutY), sizeof(float));
        points_.push_back(point);
    }

    sortPoints();
    cacheValid_ = false;
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