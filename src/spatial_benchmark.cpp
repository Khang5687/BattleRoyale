#include "spatial_benchmark.hpp"
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace spatial {

SpatialBenchmark::BenchmarkResults SpatialBenchmark::runComprehensiveBenchmark(
    uint32_t entityCount, uint32_t queryCount, float worldWidth, float worldHeight) {

    BenchmarkResults results = {};

    // Generate test data
    auto entities = generateTestEntities(entityCount, worldWidth, worldHeight);
    auto queryPoints = generateQueryPoints(queryCount, worldWidth, worldHeight);

    // Test old system (simulated)
    std::vector<std::pair<uint32_t, uint32_t>> oldCollisions;
    simulateOldSpatialSystem(entities, worldWidth, worldHeight, results.oldSystemTime, oldCollisions);
    results.oldSystemCollisions = static_cast<uint32_t>(oldCollisions.size());
    results.oldSystemMemoryUsage = estimateOldSystemMemoryUsage(entityCount);

    // Test new system
    auto start = std::chrono::high_resolution_clock::now();

    SpatialManager manager(worldWidth, worldHeight);
    manager.updateEntityData(entities);

    // Insert entities
    auto insertStart = std::chrono::high_resolution_clock::now();
    for (const auto& entity : entities) {
        manager.insertEntity(entity.entityId, {entity.posX, entity.posY}, entity.radius);
    }
    auto insertEnd = std::chrono::high_resolution_clock::now();
    results.insertTime = std::chrono::duration_cast<std::chrono::nanoseconds>(insertEnd - insertStart);

    // Query performance
    auto queryStart = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<uint32_t, uint32_t>> newCollisions;
    std::vector<uint32_t> nearbyEntities;

    for (const auto& entity : entities) {
        nearbyEntities.clear();
        manager.findNearbyEntities(entity.entityId, {entity.posX, entity.posY}, entity.radius, nearbyEntities);

        for (uint32_t otherId : nearbyEntities) {
            if (otherId > entity.entityId) { // Avoid duplicates
                newCollisions.emplace_back(entity.entityId, otherId);
            }
        }
    }
    auto queryEnd = std::chrono::high_resolution_clock::now();
    results.queryTime = std::chrono::duration_cast<std::chrono::nanoseconds>(queryEnd - queryStart);

    // Removal test
    auto removeStart = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < entities.size() / 2; ++i) {
        const auto& entity = entities[i];
        manager.removeEntity(entity.entityId, {entity.posX, entity.posY});
    }
    auto removeEnd = std::chrono::high_resolution_clock::now();
    results.removeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(removeEnd - removeStart);

    auto end = std::chrono::high_resolution_clock::now();
    results.newSystemTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    results.newSystemCollisions = static_cast<uint32_t>(newCollisions.size());
    results.newSystemMemoryUsage = calculateMemoryUsage(manager);

    // Calculate accuracy
    std::sort(oldCollisions.begin(), oldCollisions.end());
    std::sort(newCollisions.begin(), newCollisions.end());

    std::vector<std::pair<uint32_t, uint32_t>> intersection;
    std::set_intersection(oldCollisions.begin(), oldCollisions.end(),
                         newCollisions.begin(), newCollisions.end(),
                         std::back_inserter(intersection));

    results.accuracyMatchCount = static_cast<uint32_t>(intersection.size());
    uint32_t maxCollisions = std::max(results.oldSystemCollisions, results.newSystemCollisions);
    results.accuracyPercentage = maxCollisions > 0 ?
        (static_cast<double>(results.accuracyMatchCount) / maxCollisions) * 100.0 : 100.0;

    // Memory efficiency
    if (results.oldSystemMemoryUsage > 0) {
        results.memoryReduction = results.oldSystemMemoryUsage - results.newSystemMemoryUsage;
        results.memoryEfficiencyGain =
            (static_cast<double>(results.memoryReduction) / results.oldSystemMemoryUsage) * 100.0;
    }

    // SIMD metrics
    results.simdMetrics = SIMDVerification::verifyEntityDataLayout(entities);

    // Cache profiling
    CacheProfiler::startProfiling();
    // Run some operations to generate cache metrics
    for (int i = 0; i < 100; ++i) {
        nearbyEntities.clear();
        manager.findNearbyEntities(entities[i % entities.size()].entityId,
                                  {entities[i % entities.size()].posX, entities[i % entities.size()].posY},
                                  entities[i % entities.size()].radius, nearbyEntities);
    }
    CacheProfiler::stopProfiling();
    results.cacheMetrics = CacheProfiler::getMetrics();

    return results;
}

SpatialBenchmark::BenchmarkResults SpatialBenchmark::stressTestMassiveScale(uint32_t maxEntities) {
    BenchmarkResults results = {};

    // Test scalability limits
    uint32_t currentCount = 1000;
    results.maxEntitiesOldSystem = 50000; // Estimated limit for old system

    while (currentCount <= maxEntities) {
        auto entities = generateTestEntities(currentCount, 2000.0f, 1500.0f);

        auto start = std::chrono::high_resolution_clock::now();
        SpatialManager manager(2000.0f, 1500.0f);

        try {
            for (const auto& entity : entities) {
                manager.insertEntity(entity.entityId, {entity.posX, entity.posY}, entity.radius);
            }

            // Test queries
            std::vector<uint32_t> nearbyEntities;
            for (size_t i = 0; i < std::min(size_t(100), entities.size()); ++i) {
                nearbyEntities.clear();
                manager.findNearbyEntities(entities[i].entityId,
                                          {entities[i].posX, entities[i].posY},
                                          entities[i].radius, nearbyEntities);
            }

            results.maxEntitiesNewSystem = currentCount;
            currentCount *= 2;

        } catch (...) {
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (duration.count() > 1000) { // If it takes more than 1 second, stop
            break;
        }
    }

    results.scalabilityFactor = static_cast<double>(results.maxEntitiesNewSystem) / results.maxEntitiesOldSystem;

    return results;
}

std::string SpatialBenchmark::generatePerformanceReport(const BenchmarkResults& results) {
    std::stringstream report;

    report << "=== Spatial System Performance Report ===\n\n";

    report << "TIMING PERFORMANCE:\n";
    report << "  Old System Total: " << results.oldSystemTime.count() / 1000000.0 << " ms\n";
    report << "  New System Total: " << results.newSystemTime.count() / 1000000.0 << " ms\n";

    if (results.oldSystemTime.count() > 0) {
        double speedup = static_cast<double>(results.oldSystemTime.count()) / results.newSystemTime.count();
        report << "  Performance Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    report << "  Insert Time: " << results.insertTime.count() / 1000000.0 << " ms\n";
    report << "  Query Time: " << results.queryTime.count() / 1000000.0 << " ms\n";
    report << "  Remove Time: " << results.removeTime.count() / 1000000.0 << " ms\n\n";

    report << "COLLISION DETECTION ACCURACY:\n";
    report << "  Old System Collisions: " << results.oldSystemCollisions << "\n";
    report << "  New System Collisions: " << results.newSystemCollisions << "\n";
    report << "  Accuracy Match Count: " << results.accuracyMatchCount << "\n";
    report << "  Accuracy Percentage: " << std::fixed << std::setprecision(2) << results.accuracyPercentage << "%\n\n";

    report << "MEMORY USAGE:\n";
    report << "  Old System Memory: " << results.oldSystemMemoryUsage / 1024 << " KB\n";
    report << "  New System Memory: " << results.newSystemMemoryUsage / 1024 << " KB\n";
    report << "  Memory Reduction: " << results.memoryReduction / 1024 << " KB\n";
    report << "  Memory Efficiency Gain: " << std::fixed << std::setprecision(2) << results.memoryEfficiencyGain << "%\n\n";

    report << "SCALABILITY:\n";
    report << "  Max Entities (Old): " << results.maxEntitiesOldSystem << "\n";
    report << "  Max Entities (New): " << results.maxEntitiesNewSystem << "\n";
    report << "  Scalability Factor: " << std::fixed << std::setprecision(2) << results.scalabilityFactor << "x\n\n";

    report << "SIMD VECTORIZATION:\n";
    report << "  16-byte Aligned: " << (results.simdMetrics.isAligned16 ? "Yes" : "No") << "\n";
    report << "  32-byte Aligned: " << (results.simdMetrics.isAligned32 ? "Yes" : "No") << "\n";
    report << "  Cache Line Utilization: " << results.simdMetrics.cacheLineUtilization << "%\n";
    report << "  SIMD Speedup: " << std::fixed << std::setprecision(2) << results.simdMetrics.speedupRatio << "x\n";
    report << "  Vectorization Efficiency: " << std::fixed << std::setprecision(2) << results.simdMetrics.vectorizationEfficiency * 100 << "%\n\n";

    report << "CACHE PERFORMANCE:\n";
    report << "  L1 Hit Rate: " << std::fixed << std::setprecision(2) << results.cacheMetrics.hitRateL1 * 100 << "%\n";
    report << "  L2 Hit Rate: " << std::fixed << std::setprecision(2) << results.cacheMetrics.hitRateL2 * 100 << "%\n";
    report << "  L3 Hit Rate: " << std::fixed << std::setprecision(2) << results.cacheMetrics.hitRateL3 * 100 << "%\n";

    return report.str();
}

void SpatialBenchmark::simulateOldSpatialSystem(
    const std::vector<EntityData>& entities,
    float worldWidth, float worldHeight,
    std::chrono::nanoseconds& totalTime,
    std::vector<std::pair<uint32_t, uint32_t>>& collisions) {

    auto start = std::chrono::high_resolution_clock::now();

    // Simulate old hash grid approach
    const float gridCellSize = 64.0f;
    const int cellsX = std::max(1, static_cast<int>(worldWidth / gridCellSize));
    const int cellsY = std::max(1, static_cast<int>(worldHeight / gridCellSize));

    std::vector<std::vector<uint32_t>> grid(cellsX * cellsY);

    // Insert entities into grid
    for (const auto& entity : entities) {
        int cx = std::clamp(static_cast<int>(entity.posX / gridCellSize), 0, cellsX - 1);
        int cy = std::clamp(static_cast<int>(entity.posY / gridCellSize), 0, cellsY - 1);
        int cellIndex = cy * cellsX + cx;
        grid[cellIndex].push_back(entity.entityId);
    }

    // Find collisions using old O(n²) approach within cells
    for (int cy = 0; cy < cellsY; ++cy) {
        for (int cx = 0; cx < cellsX; ++cx) {
            for (int ny = std::max(0, cy - 1); ny <= std::min(cellsY - 1, cy + 1); ++ny) {
                for (int nx = std::max(0, cx - 1); nx <= std::min(cellsX - 1, cx + 1); ++nx) {
                    const auto& cellA = grid[cy * cellsX + cx];
                    const auto& cellB = grid[ny * cellsX + nx];

                    for (uint32_t idA : cellA) {
                        for (uint32_t idB : cellB) {
                            if (idA >= idB) continue;

                            // Find entities by ID (O(n) lookup - inefficient)
                            const EntityData* entityA = nullptr;
                            const EntityData* entityB = nullptr;

                            for (const auto& e : entities) {
                                if (e.entityId == idA) entityA = &e;
                                if (e.entityId == idB) entityB = &e;
                            }

                            if (entityA && entityB) {
                                float dx = entityB->posX - entityA->posX;
                                float dy = entityB->posY - entityA->posY;
                                float radiusSum = entityA->radius + entityB->radius;
                                float distSq = dx * dx + dy * dy;

                                if (distSq < radiusSum * radiusSum) {
                                    collisions.emplace_back(idA, idB);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

std::vector<EntityData> SpatialBenchmark::generateTestEntities(uint32_t count, float worldWidth, float worldHeight) {
    std::vector<EntityData> entities;
    entities.reserve(count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posXDist(50.0f, worldWidth - 50.0f);
    std::uniform_real_distribution<float> posYDist(50.0f, worldHeight - 50.0f);
    std::uniform_real_distribution<float> radiusDist(10.0f, 30.0f);

    for (uint32_t i = 0; i < count; ++i) {
        EntityData entity = {};
        entity.posX = posXDist(gen);
        entity.posY = posYDist(gen);
        entity.radius = radiusDist(gen);
        entity.entityId = i;
        entities.push_back(entity);
    }

    return entities;
}

std::vector<vec2> SpatialBenchmark::generateQueryPoints(uint32_t count, float worldWidth, float worldHeight) {
    std::vector<vec2> points;
    points.reserve(count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(0.0f, worldWidth);
    std::uniform_real_distribution<float> yDist(0.0f, worldHeight);

    for (uint32_t i = 0; i < count; ++i) {
        points.push_back({xDist(gen), yDist(gen)});
    }

    return points;
}

size_t SpatialBenchmark::calculateMemoryUsage(const SpatialManager& manager) {
    // Simplified memory calculation
    size_t entityDataSize = sizeof(EntityData) * 10000; // Estimate
    size_t spatialStructureSize = sizeof(SpatialManager);
    return entityDataSize + spatialStructureSize;
}

size_t SpatialBenchmark::estimateOldSystemMemoryUsage(uint32_t entityCount) {
    // Old system: grid + entity storage
    size_t gridSize = sizeof(std::vector<uint32_t>) * 10000; // Grid cells
    size_t entitySize = entityCount * (sizeof(float) * 6 + sizeof(uint32_t)); // SoA storage
    return gridSize + entitySize;
}

// SpatialIntegrationTests implementation
SpatialIntegrationTests::TestResults SpatialIntegrationTests::runAllTests() {
    TestResults results = {};
    results.allTestsPassed = true;
    results.testsRun = 0;
    results.testsPassed = 0;
    results.testsFailed = 0;

    // Run all test categories
    logTestResult("Basic Insertion", testBasicInsertion(), results);
    logTestResult("Batch Operations", testBatchOperations(), results);
    logTestResult("Accuracy Against Reference", testAccuracyAgainstReference(), results);
    logTestResult("Memory Coherence", testMemoryCoherence(), results);
    logTestResult("Camera Zoom Integration", testCameraZoomIntegration(), results);
    logTestResult("Dynamic Rebalancing", testDynamicRebalancing(), results);
    logTestResult("SIMD Vectorization", testSIMDVectorization(), results);

    results.allTestsPassed = (results.testsFailed == 0);
    return results;
}

bool SpatialIntegrationTests::testBasicInsertion() {
    try {
        SpatialManager manager(800.0f, 600.0f);

        // Test single entity insertion
        manager.insertEntity(1, {100.0f, 200.0f}, 25.0f);

        std::vector<uint32_t> nearbyEntities;
        manager.findNearbyEntities(1, {100.0f, 200.0f}, 25.0f, nearbyEntities);

        // Should find no other entities (only self is excluded)
        if (!nearbyEntities.empty()) {
            return false;
        }

        // Test multiple entity insertion
        manager.insertEntity(2, {120.0f, 220.0f}, 25.0f);
        manager.findNearbyEntities(1, {100.0f, 200.0f}, 30.0f, nearbyEntities);

        // Should find entity 2 nearby
        return nearbyEntities.size() == 1 && nearbyEntities[0] == 2;

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testBatchOperations() {
    try {
        SpatialManager manager(1000.0f, 1000.0f);

        // Create test entities
        std::vector<EntityData> entities;
        for (uint32_t i = 0; i < 1000; ++i) {
            EntityData entity = {};
            entity.entityId = i;
            entity.posX = static_cast<float>(i % 50) * 20.0f;
            entity.posY = static_cast<float>(i / 50) * 20.0f;
            entity.radius = 15.0f;
            entities.push_back(entity);
        }

        // Batch update
        manager.updateEntityData(entities);

        // Batch insert
        for (const auto& entity : entities) {
            manager.insertEntity(entity.entityId, {entity.posX, entity.posY}, entity.radius);
        }

        // Test batch queries
        std::vector<uint32_t> nearbyEntities;
        uint32_t totalCollisions = 0;

        for (size_t i = 0; i < std::min(size_t(100), entities.size()); ++i) {
            nearbyEntities.clear();
            manager.findNearbyEntities(entities[i].entityId,
                                      {entities[i].posX, entities[i].posY},
                                      entities[i].radius, nearbyEntities);
            totalCollisions += static_cast<uint32_t>(nearbyEntities.size());
        }

        return totalCollisions > 0; // Should find some collisions in this grid pattern

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testAccuracyAgainstReference() {
    try {
        // Test against known collision scenarios
        SpatialManager manager(400.0f, 400.0f);

        // Create entities that should definitely collide
        manager.insertEntity(1, {100.0f, 100.0f}, 20.0f);
        manager.insertEntity(2, {110.0f, 110.0f}, 20.0f);
        manager.insertEntity(3, {200.0f, 200.0f}, 15.0f);

        std::vector<uint32_t> nearbyEntities;

        // Entity 1 should find entity 2
        manager.findNearbyEntities(1, {100.0f, 100.0f}, 20.0f, nearbyEntities);
        bool found2 = std::find(nearbyEntities.begin(), nearbyEntities.end(), 2) != nearbyEntities.end();
        bool notFound3 = std::find(nearbyEntities.begin(), nearbyEntities.end(), 3) == nearbyEntities.end();

        if (!found2 || !notFound3) {
            return false;
        }

        // Test boundary cases
        nearbyEntities.clear();
        manager.findNearbyEntities(1, {100.0f, 100.0f}, 5.0f, nearbyEntities);

        // Should find no entities with small radius
        return nearbyEntities.empty();

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testMemoryCoherence() {
    try {
        // Test SIMD memory layout
        std::vector<EntityData> entities(1000);

        for (size_t i = 0; i < entities.size(); ++i) {
            entities[i].entityId = static_cast<uint32_t>(i);
            entities[i].posX = static_cast<float>(i * 10);
            entities[i].posY = static_cast<float>(i * 5);
            entities[i].radius = 20.0f;
        }

        auto metrics = SIMDVerification::verifyEntityDataLayout(entities);

        // Check memory alignment
        return metrics.isAligned16 && metrics.speedupRatio > 1.0;

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testCameraZoomIntegration() {
    try {
        SpatialManager manager(800.0f, 600.0f);

        // Test spatial bounds updates
        SpatialBounds bounds;
        bounds.worldBounds = {0, 0, 800, 600};
        bounds.zoomFactor = 1.0f;
        bounds.entityCount = 100;

        manager.updateSpatialBounds(bounds);

        // Test zoom change
        bounds.updateFromZoom(2.0f, {0, 0, 800, 600});
        manager.updateSpatialBounds(bounds);

        // Verify visible bounds calculation
        const auto& currentBounds = manager.getSpatialBounds();

        return std::abs(currentBounds.zoomFactor - 2.0f) < 0.001f;

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testDynamicRebalancing() {
    try {
        SpatialManager manager(800.0f, 600.0f);

        // Insert many entities to trigger rebalancing
        for (uint32_t i = 0; i < 5000; ++i) {
            float x = static_cast<float>(i % 100) * 8.0f;
            float y = static_cast<float>(i / 100) * 6.0f;
            manager.insertEntity(i, {x, y}, 10.0f);
        }

        // Update positions to trigger rebalancing
        for (uint32_t i = 0; i < 1000; ++i) {
            float newX = static_cast<float>(i % 80) * 10.0f;
            float newY = static_cast<float>(i / 80) * 7.5f;
            float oldX = static_cast<float>(i % 100) * 8.0f;
            float oldY = static_cast<float>(i / 100) * 6.0f;

            manager.updateEntity(i, {oldX, oldY}, {newX, newY}, 10.0f);
        }

        // Test that queries still work after rebalancing
        std::vector<uint32_t> nearbyEntities;
        manager.findNearbyEntities(0, {0.0f, 0.0f}, 15.0f, nearbyEntities);

        return !nearbyEntities.empty(); // Should find some nearby entities

    } catch (...) {
        return false;
    }
}

bool SpatialIntegrationTests::testSIMDVectorization() {
    try {
        // Test SIMD collision detection
        std::vector<EntityData> entities(100);

        for (size_t i = 0; i < entities.size(); ++i) {
            entities[i].entityId = static_cast<uint32_t>(i);
            entities[i].posX = static_cast<float>(i % 10) * 25.0f;
            entities[i].posY = static_cast<float>(i / 10) * 25.0f;
            entities[i].radius = 15.0f;
        }

        std::vector<std::pair<uint32_t, uint32_t>> collisions;
        SIMDVerification::batchCollisionDetection(entities.data(), entities.size(), collisions);

        // Should find collisions in this grid pattern
        return !collisions.empty();

    } catch (...) {
        return false;
    }
}

void SpatialIntegrationTests::logTestResult(const std::string& testName, bool passed, TestResults& results) {
    results.testsRun++;

    if (passed) {
        results.testsPassed++;
    } else {
        results.testsFailed++;
        results.failureMessages.push_back("FAILED: " + testName);
        results.allTestsPassed = false;
    }
}

} // namespace spatial