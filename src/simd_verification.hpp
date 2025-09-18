#pragma once

#include <vector>
#include <chrono>
#include <immintrin.h>
#include "spatial_system.hpp"

namespace spatial {

// SIMD memory layout verification utilities
class SIMDVerification {
public:
    struct MemoryLayoutMetrics {
        bool isAligned16;
        bool isAligned32;
        size_t cacheLineUtilization;
        double vectorizationEfficiency;
        std::chrono::nanoseconds scalarTime;
        std::chrono::nanoseconds simdTime;
        double speedupRatio;
    };

    // Verify EntityData alignment for SIMD operations
    static MemoryLayoutMetrics verifyEntityDataLayout(const std::vector<EntityData>& entities);

    // Test vectorized distance calculations vs scalar
    static void benchmarkDistanceCalculations(const std::vector<EntityData>& entities,
                                             const vec2& queryPoint, float radius,
                                             MemoryLayoutMetrics& metrics);

    // Verify cache-friendly data access patterns
    static void verifyCacheEfficiency(const std::vector<EntityData>& entities,
                                     MemoryLayoutMetrics& metrics);

    // SIMD-optimized batch collision detection
    static void batchCollisionDetection(const EntityData* entities, size_t count,
                                       std::vector<std::pair<uint32_t, uint32_t>>& collisions);

    // Memory prefetching utilities for large datasets
    static void prefetchEntityData(const EntityData* entities, size_t startIndex, size_t count);

private:
    // SIMD helper functions
    static __m128 loadEntityPositions(const EntityData& entity);
    static bool isMemoryAligned(const void* ptr, size_t alignment);
    static size_t calculateCacheLineUtilization(const std::vector<EntityData>& entities);
};

// Cache-miss profiling implementation
class CacheProfiler {
public:
    struct CacheMetrics {
        uint64_t l1Hits;
        uint64_t l1Misses;
        uint64_t l2Hits;
        uint64_t l2Misses;
        uint64_t l3Hits;
        uint64_t l3Misses;
        double hitRateL1;
        double hitRateL2;
        double hitRateL3;
    };

    static void startProfiling();
    static void stopProfiling();
    static CacheMetrics getMetrics();

private:
    static bool profilingActive;
    static CacheMetrics currentMetrics;
};

} // namespace spatial