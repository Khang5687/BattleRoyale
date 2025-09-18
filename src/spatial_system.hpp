#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <array>
#include <immintrin.h>

// Forward declarations
struct vec2 { float x, y; };
struct AABB { float minX, minY, maxX, maxY; };

namespace spatial {

// Cache-friendly entity data for SIMD operations
struct alignas(16) EntityData {
    float posX, posY, velX, velY;
    float radius;
    uint32_t entityId;
    uint16_t quadrantFlags; // For fast spatial queries
    uint16_t padding;
};

// Profiling hooks for cache-miss analysis
struct CacheProfilingHooks {
    static inline uint64_t totalQueries = 0;
    static inline uint64_t cacheHits = 0;
    static inline uint64_t cacheMisses = 0;

    static void recordQuery() { totalQueries++; }
    static void recordHit() { cacheHits++; }
    static void recordMiss() { cacheMisses++; }
    static double hitRate() { return totalQueries > 0 ? double(cacheHits) / totalQueries : 0.0; }
};

// QuadTree node for hierarchical spatial partitioning
class QuadTreeNode {
public:
    static constexpr uint32_t MAX_ENTITIES_PER_NODE = 16;
    static constexpr uint32_t MAX_DEPTH = 8;

    AABB bounds;
    std::vector<uint32_t> entityIds;
    std::array<std::unique_ptr<QuadTreeNode>, 4> children;
    uint32_t depth;
    bool isLeaf;

    QuadTreeNode(const AABB& bounds, uint32_t depth = 0);

    void insert(uint32_t entityId, const vec2& pos, float radius);
    void remove(uint32_t entityId, const vec2& pos);
    void clear();

    // Query entities in range with SIMD-optimized distance checks
    void queryRange(const AABB& range, std::vector<uint32_t>& results,
                   const EntityData* entityData) const;

    // Dynamic rebalancing when entities move
    void rebalance(const EntityData* entityData);

private:
    void subdivide();
    bool shouldSubdivide() const;
    uint32_t getChildIndex(const vec2& pos) const;
};

// Optimized hash grid for dense collision areas
class HashGrid {
public:
    static constexpr float DEFAULT_CELL_SIZE = 64.0f;

    HashGrid(float worldWidth, float worldHeight, float cellSize = DEFAULT_CELL_SIZE);

    void insert(uint32_t entityId, const vec2& pos, float radius);
    void remove(uint32_t entityId, const vec2& oldPos);
    void update(uint32_t entityId, const vec2& oldPos, const vec2& newPos, float radius);
    void clear();

    // SIMD-optimized collision detection within cells
    void findNearbyEntities(const vec2& pos, float radius,
                           std::vector<uint32_t>& results,
                           const EntityData* entityData) const;

    // Memory layout optimization for cache efficiency
    void compactCells();

    // Adaptive cell size based on entity density
    void updateCellSize(float averageEntityRadius, uint32_t entityCount);

private:
    float cellSize;
    int32_t cellsX, cellsY;
    float invCellSize;
    std::vector<std::vector<uint32_t>> cells;

    int32_t getCellIndex(const vec2& pos) const;
    void getCellRange(const vec2& pos, float radius, int32_t& minX, int32_t& minY,
                     int32_t& maxX, int32_t& maxY) const;
};

// Spatial bounds API for camera zoom integration
struct SpatialBounds {
    AABB worldBounds;
    AABB visibleBounds;
    float zoomFactor;
    uint32_t entityCount;

    void updateFromZoom(float newZoomFactor, const AABB& newWorldBounds);
    bool needsRepartitioning() const;
};

// Hybrid spatial manager coordinating QuadTree + HashGrid
class SpatialManager {
public:
    enum class Strategy {
        QUADTREE_ONLY,    // Low-density areas
        HASHGRID_ONLY,    // High-density areas
        HYBRID_AUTO       // Automatic selection based on density
    };

    SpatialManager(float worldWidth, float worldHeight);

    // Entity management
    void insertEntity(uint32_t entityId, const vec2& pos, float radius);
    void removeEntity(uint32_t entityId, const vec2& pos);
    void updateEntity(uint32_t entityId, const vec2& oldPos, const vec2& newPos, float radius);
    void clearAll();

    // Collision queries
    void findNearbyEntities(uint32_t entityId, const vec2& pos, float radius,
                           std::vector<uint32_t>& results) const;

    // Performance monitoring
    void collectPerformanceMetrics(struct SpatialPerformanceMetrics& metrics) const;

    // Camera integration
    void updateSpatialBounds(const SpatialBounds& bounds);
    const SpatialBounds& getSpatialBounds() const { return spatialBounds; }

    // SIMD-friendly data access
    void updateEntityData(const std::vector<EntityData>& entities);
    const EntityData* getEntityData() const { return entityData.data(); }

private:
    QuadTreeNode quadTree;
    HashGrid hashGrid;
    SpatialBounds spatialBounds;
    Strategy currentStrategy;

    // Cache-friendly entity storage
    std::vector<EntityData> entityData;
    std::unordered_map<uint32_t, size_t> entityToDataIndex;

    // Performance optimization
    mutable uint64_t queryCount = 0;
    mutable uint64_t quadTreeQueries = 0;
    mutable uint64_t hashGridQueries = 0;

    Strategy selectOptimalStrategy(const vec2& pos, float radius) const;
    void rebalanceIfNeeded();
};

// Performance metrics structure
struct SpatialPerformanceMetrics {
    uint64_t totalQueries;
    uint64_t quadTreeQueries;
    uint64_t hashGridQueries;
    double quadTreeHitRate;
    double hashGridHitRate;
    double averageEntitiesPerQuery;
    float rebalanceFrequency;
};

} // namespace spatial