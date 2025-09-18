#pragma once

#include "spatial_system.hpp"
#include "simd_verification.hpp"
#include <chrono>
#include <string>
#include <vector>

namespace spatial {

// Performance comparison between old and new spatial systems
class SpatialBenchmark {
public:
    struct BenchmarkResults {
        // Timing metrics
        std::chrono::nanoseconds oldSystemTime;
        std::chrono::nanoseconds newSystemTime;
        std::chrono::nanoseconds insertTime;
        std::chrono::nanoseconds queryTime;
        std::chrono::nanoseconds removeTime;

        // Accuracy metrics
        uint32_t oldSystemCollisions;
        uint32_t newSystemCollisions;
        uint32_t accuracyMatchCount;
        double accuracyPercentage;

        // Memory metrics
        size_t oldSystemMemoryUsage;
        size_t newSystemMemoryUsage;
        size_t memoryReduction;
        double memoryEfficiencyGain;

        // Scalability metrics
        double scalabilityFactor;
        uint32_t maxEntitiesOldSystem;
        uint32_t maxEntitiesNewSystem;

        // Cache performance
        SIMDVerification::MemoryLayoutMetrics simdMetrics;
        CacheProfiler::CacheMetrics cacheMetrics;
    };

    // Run comprehensive benchmark comparing old vs new spatial systems
    static BenchmarkResults runComprehensiveBenchmark(
        uint32_t entityCount = 10000,
        uint32_t queryCount = 1000,
        float worldWidth = 800.0f,
        float worldHeight = 600.0f
    );

    // Test specific scenarios
    static BenchmarkResults benchmarkInsertionPerformance(uint32_t entityCount);
    static BenchmarkResults benchmarkQueryPerformance(uint32_t entityCount, uint32_t queryCount);
    static BenchmarkResults benchmarkRemovalPerformance(uint32_t entityCount);

    // Stress testing for massive entity counts
    static BenchmarkResults stressTestMassiveScale(uint32_t maxEntities = 1000000);

    // Generate detailed performance report
    static std::string generatePerformanceReport(const BenchmarkResults& results);

private:
    // Helper functions for old system simulation
    static void simulateOldSpatialSystem(
        const std::vector<EntityData>& entities,
        float worldWidth, float worldHeight,
        std::chrono::nanoseconds& totalTime,
        std::vector<std::pair<uint32_t, uint32_t>>& collisions
    );

    // Test data generation
    static std::vector<EntityData> generateTestEntities(uint32_t count, float worldWidth, float worldHeight);
    static std::vector<vec2> generateQueryPoints(uint32_t count, float worldWidth, float worldHeight);

    // Memory usage calculation
    static size_t calculateMemoryUsage(const SpatialManager& manager);
    static size_t estimateOldSystemMemoryUsage(uint32_t entityCount);
};

// Integration test suite for spatial system validation
class SpatialIntegrationTests {
public:
    struct TestResults {
        bool allTestsPassed;
        uint32_t testsRun;
        uint32_t testsPassed;
        uint32_t testsFailed;
        std::vector<std::string> failureMessages;
    };

    // Run all integration tests
    static TestResults runAllTests();

    // Individual test categories
    static bool testBasicInsertion();
    static bool testBatchOperations();
    static bool testAccuracyAgainstReference();
    static bool testMemoryCoherence();
    static bool testCameraZoomIntegration();
    static bool testDynamicRebalancing();
    static bool testSIMDVectorization();

private:
    static void logTestResult(const std::string& testName, bool passed, TestResults& results);
};

} // namespace spatial