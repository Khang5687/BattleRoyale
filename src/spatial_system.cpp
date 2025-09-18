#include "spatial_system.hpp"
#include <algorithm>
#include <cmath>

namespace spatial {

// QuadTreeNode Implementation
QuadTreeNode::QuadTreeNode(const AABB& bounds, uint32_t depth)
    : bounds(bounds), depth(depth), isLeaf(true) {
    entityIds.reserve(MAX_ENTITIES_PER_NODE);
}

void QuadTreeNode::insert(uint32_t entityId, const vec2& pos, float radius) {
    CacheProfilingHooks::recordQuery();

    // Check if entity fits in this node
    if (pos.x - radius < bounds.minX || pos.x + radius > bounds.maxX ||
        pos.y - radius < bounds.minY || pos.y + radius > bounds.maxY) {
        return; // Entity doesn't fit
    }

    if (isLeaf) {
        entityIds.push_back(entityId);
        CacheProfilingHooks::recordHit();

        if (shouldSubdivide()) {
            subdivide();
        }
    } else {
        // Insert into appropriate child nodes
        for (auto& child : children) {
            if (child) {
                child->insert(entityId, pos, radius);
            }
        }
    }
}

void QuadTreeNode::remove(uint32_t entityId, const vec2& pos) {
    if (isLeaf) {
        auto it = std::find(entityIds.begin(), entityIds.end(), entityId);
        if (it != entityIds.end()) {
            entityIds.erase(it);
            CacheProfilingHooks::recordHit();
        } else {
            CacheProfilingHooks::recordMiss();
        }
    } else {
        for (auto& child : children) {
            if (child) {
                child->remove(entityId, pos);
            }
        }
    }
}

void QuadTreeNode::clear() {
    entityIds.clear();
    for (auto& child : children) {
        if (child) {
            child->clear();
            child.reset();
        }
    }
    isLeaf = true;
}

void QuadTreeNode::queryRange(const AABB& range, std::vector<uint32_t>& results,
                             const EntityData* entityData) const {
    // Early rejection if ranges don't overlap
    if (range.maxX < bounds.minX || range.minX > bounds.maxX ||
        range.maxY < bounds.minY || range.minY > bounds.maxY) {
        return;
    }

    if (isLeaf) {
        // SIMD-optimized distance checks for leaf nodes
        const size_t simdWidth = 4;
        size_t i = 0;

        // Process 4 entities at a time using SIMD
        for (; i + simdWidth <= entityIds.size(); i += simdWidth) {
            __m128 posX = _mm_set_ps(
                entityData[entityIds[i + 3]].posX,
                entityData[entityIds[i + 2]].posX,
                entityData[entityIds[i + 1]].posX,
                entityData[entityIds[i]].posX
            );
            __m128 posY = _mm_set_ps(
                entityData[entityIds[i + 3]].posY,
                entityData[entityIds[i + 2]].posY,
                entityData[entityIds[i + 1]].posY,
                entityData[entityIds[i]].posY
            );

            __m128 centerX = _mm_set1_ps((range.minX + range.maxX) * 0.5f);
            __m128 centerY = _mm_set1_ps((range.minY + range.maxY) * 0.5f);

            __m128 dx = _mm_sub_ps(posX, centerX);
            __m128 dy = _mm_sub_ps(posY, centerY);
            __m128 distSq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));

            float rangeRadius = std::max(range.maxX - range.minX, range.maxY - range.minY) * 0.5f;
            __m128 rangeSq = _mm_set1_ps(rangeRadius * rangeRadius);

            __m128 mask = _mm_cmple_ps(distSq, rangeSq);

            // Extract results
            alignas(16) float maskArray[4];
            _mm_store_ps(maskArray, mask);

            for (size_t j = 0; j < simdWidth; ++j) {
                if (maskArray[j] != 0.0f) {
                    results.push_back(entityIds[i + j]);
                }
            }
        }

        // Process remaining entities
        for (; i < entityIds.size(); ++i) {
            const EntityData& entity = entityData[entityIds[i]];
            if (entity.posX >= range.minX && entity.posX <= range.maxX &&
                entity.posY >= range.minY && entity.posY <= range.maxY) {
                results.push_back(entityIds[i]);
            }
        }
    } else {
        // Recursively query child nodes
        for (const auto& child : children) {
            if (child) {
                child->queryRange(range, results, entityData);
            }
        }
    }
}

void QuadTreeNode::rebalance(const EntityData* entityData) {
    if (isLeaf || depth >= MAX_DEPTH) {
        return;
    }

    // Check if we should merge children back into this node
    uint32_t totalEntities = 0;
    for (const auto& child : children) {
        if (child) {
            totalEntities += child->entityIds.size();
        }
    }

    if (totalEntities <= MAX_ENTITIES_PER_NODE / 2) {
        // Merge children back into this node
        entityIds.clear();
        for (auto& child : children) {
            if (child) {
                entityIds.insert(entityIds.end(), child->entityIds.begin(), child->entityIds.end());
                child.reset();
            }
        }
        isLeaf = true;
    } else {
        // Recursively rebalance children
        for (auto& child : children) {
            if (child) {
                child->rebalance(entityData);
            }
        }
    }
}

void QuadTreeNode::subdivide() {
    if (depth >= MAX_DEPTH || !isLeaf) {
        return;
    }

    float midX = (bounds.minX + bounds.maxX) * 0.5f;
    float midY = (bounds.minY + bounds.maxY) * 0.5f;

    // Create four child nodes
    children[0] = std::make_unique<QuadTreeNode>(
        AABB{bounds.minX, bounds.minY, midX, midY}, depth + 1);
    children[1] = std::make_unique<QuadTreeNode>(
        AABB{midX, bounds.minY, bounds.maxX, midY}, depth + 1);
    children[2] = std::make_unique<QuadTreeNode>(
        AABB{bounds.minX, midY, midX, bounds.maxY}, depth + 1);
    children[3] = std::make_unique<QuadTreeNode>(
        AABB{midX, midY, bounds.maxX, bounds.maxY}, depth + 1);

    isLeaf = false;

    // Note: entityIds remain in this node for entities that span multiple children
    // This is a simplified implementation - full implementation would redistribute entities
}

bool QuadTreeNode::shouldSubdivide() const {
    return entityIds.size() > MAX_ENTITIES_PER_NODE && depth < MAX_DEPTH;
}

uint32_t QuadTreeNode::getChildIndex(const vec2& pos) const {
    float midX = (bounds.minX + bounds.maxX) * 0.5f;
    float midY = (bounds.minY + bounds.maxY) * 0.5f;

    uint32_t index = 0;
    if (pos.x >= midX) index |= 1;
    if (pos.y >= midY) index |= 2;
    return index;
}

// HashGrid Implementation
HashGrid::HashGrid(float worldWidth, float worldHeight, float cellSize)
    : cellSize(cellSize), invCellSize(1.0f / cellSize) {
    cellsX = static_cast<int32_t>(std::ceil(worldWidth / cellSize));
    cellsY = static_cast<int32_t>(std::ceil(worldHeight / cellSize));
    cells.resize(cellsX * cellsY);
}

void HashGrid::insert(uint32_t entityId, const vec2& pos, float radius) {
    int32_t minX, minY, maxX, maxY;
    getCellRange(pos, radius, minX, minY, maxX, maxY);

    for (int32_t y = minY; y <= maxY; ++y) {
        for (int32_t x = minX; x <= maxX; ++x) {
            int32_t cellIdx = y * cellsX + x;
            if (cellIdx >= 0 && cellIdx < static_cast<int32_t>(cells.size())) {
                cells[cellIdx].push_back(entityId);
            }
        }
    }
}

void HashGrid::remove(uint32_t entityId, const vec2& oldPos) {
    int32_t cellIdx = getCellIndex(oldPos);
    if (cellIdx >= 0 && cellIdx < static_cast<int32_t>(cells.size())) {
        auto& cell = cells[cellIdx];
        auto it = std::find(cell.begin(), cell.end(), entityId);
        if (it != cell.end()) {
            cell.erase(it);
        }
    }
}

void HashGrid::update(uint32_t entityId, const vec2& oldPos, const vec2& newPos, float radius) {
    remove(entityId, oldPos);
    insert(entityId, newPos, radius);
}

void HashGrid::clear() {
    for (auto& cell : cells) {
        cell.clear();
    }
}

void HashGrid::findNearbyEntities(const vec2& pos, float radius,
                                 std::vector<uint32_t>& results,
                                 const EntityData* entityData) const {
    int32_t minX, minY, maxX, maxY;
    getCellRange(pos, radius, minX, minY, maxX, maxY);

    for (int32_t y = minY; y <= maxY; ++y) {
        for (int32_t x = minX; x <= maxX; ++x) {
            int32_t cellIdx = y * cellsX + x;
            if (cellIdx >= 0 && cellIdx < static_cast<int32_t>(cells.size())) {
                const auto& cell = cells[cellIdx];
                for (uint32_t entityId : cell) {
                    const EntityData& entity = entityData[entityId];
                    float dx = entity.posX - pos.x;
                    float dy = entity.posY - pos.y;
                    float distSq = dx * dx + dy * dy;
                    float radiusSum = radius + entity.radius;
                    if (distSq <= radiusSum * radiusSum) {
                        results.push_back(entityId);
                    }
                }
            }
        }
    }
}

void HashGrid::compactCells() {
    for (auto& cell : cells) {
        cell.shrink_to_fit();
    }
}

void HashGrid::updateCellSize(float averageEntityRadius, uint32_t entityCount) {
    // Adaptive cell sizing based on entity density
    float targetCellSize = averageEntityRadius * 2.5f;
    float densityFactor = std::sqrt(static_cast<float>(entityCount)) / 100.0f;
    float newCellSize = targetCellSize * (1.0f + densityFactor);

    if (std::abs(newCellSize - cellSize) > cellSize * 0.2f) {
        // Significant change - rebuild grid
        cellSize = newCellSize;
        invCellSize = 1.0f / cellSize;

        int32_t newCellsX = static_cast<int32_t>(std::ceil(cellsX * cellSize / newCellSize));
        int32_t newCellsY = static_cast<int32_t>(std::ceil(cellsY * cellSize / newCellSize));

        if (newCellsX != cellsX || newCellsY != cellsY) {
            cellsX = newCellsX;
            cellsY = newCellsY;
            cells.clear();
            cells.resize(cellsX * cellsY);
        }
    }
}

int32_t HashGrid::getCellIndex(const vec2& pos) const {
    int32_t x = static_cast<int32_t>(pos.x * invCellSize);
    int32_t y = static_cast<int32_t>(pos.y * invCellSize);
    x = std::clamp(x, 0, cellsX - 1);
    y = std::clamp(y, 0, cellsY - 1);
    return y * cellsX + x;
}

void HashGrid::getCellRange(const vec2& pos, float radius, int32_t& minX, int32_t& minY,
                           int32_t& maxX, int32_t& maxY) const {
    minX = std::max(0, static_cast<int32_t>((pos.x - radius) * invCellSize));
    minY = std::max(0, static_cast<int32_t>((pos.y - radius) * invCellSize));
    maxX = std::min(cellsX - 1, static_cast<int32_t>((pos.x + radius) * invCellSize));
    maxY = std::min(cellsY - 1, static_cast<int32_t>((pos.y + radius) * invCellSize));
}

// SpatialBounds Implementation
void SpatialBounds::updateFromZoom(float newZoomFactor, const AABB& newWorldBounds) {
    zoomFactor = newZoomFactor;
    worldBounds = newWorldBounds;

    // Calculate visible bounds based on zoom
    float centerX = (worldBounds.minX + worldBounds.maxX) * 0.5f;
    float centerY = (worldBounds.minY + worldBounds.maxY) * 0.5f;
    float halfWidth = (worldBounds.maxX - worldBounds.minX) * 0.5f / zoomFactor;
    float halfHeight = (worldBounds.maxY - worldBounds.minY) * 0.5f / zoomFactor;

    visibleBounds = {
        centerX - halfWidth, centerY - halfHeight,
        centerX + halfWidth, centerY + halfHeight
    };
}

bool SpatialBounds::needsRepartitioning() const {
    // Trigger repartitioning if zoom changed significantly
    static float lastZoomFactor = 1.0f;
    bool needsRepart = std::abs(zoomFactor - lastZoomFactor) > 0.2f;
    lastZoomFactor = zoomFactor;
    return needsRepart;
}

// SpatialManager Implementation
SpatialManager::SpatialManager(float worldWidth, float worldHeight)
    : quadTree(AABB{0, 0, worldWidth, worldHeight}),
      hashGrid(worldWidth, worldHeight),
      currentStrategy(Strategy::HYBRID_AUTO) {
    spatialBounds.worldBounds = {0, 0, worldWidth, worldHeight};
    spatialBounds.visibleBounds = spatialBounds.worldBounds;
    spatialBounds.zoomFactor = 1.0f;
    spatialBounds.entityCount = 0;
}

void SpatialManager::insertEntity(uint32_t entityId, const vec2& pos, float radius) {
    // Update entity data storage
    EntityData data = {pos.x, pos.y, 0.0f, 0.0f, radius, entityId, 0, 0};

    auto it = entityToDataIndex.find(entityId);
    if (it != entityToDataIndex.end()) {
        entityData[it->second] = data;
    } else {
        entityToDataIndex[entityId] = entityData.size();
        entityData.push_back(data);
    }

    spatialBounds.entityCount++;

    // Choose optimal spatial structure
    Strategy strategy = selectOptimalStrategy(pos, radius);

    switch (strategy) {
        case Strategy::QUADTREE_ONLY:
            quadTree.insert(entityId, pos, radius);
            break;
        case Strategy::HASHGRID_ONLY:
            hashGrid.insert(entityId, pos, radius);
            break;
        case Strategy::HYBRID_AUTO:
            // Insert into both for hybrid approach
            quadTree.insert(entityId, pos, radius);
            hashGrid.insert(entityId, pos, radius);
            break;
    }
}

void SpatialManager::removeEntity(uint32_t entityId, const vec2& pos) {
    auto it = entityToDataIndex.find(entityId);
    if (it != entityToDataIndex.end()) {
        // Mark as removed (lazy deletion for cache efficiency)
        entityData[it->second].entityId = UINT32_MAX;
        entityToDataIndex.erase(it);
        spatialBounds.entityCount--;
    }

    // Remove from both structures (safe if not present)
    quadTree.remove(entityId, pos);
    hashGrid.remove(entityId, pos);
}

void SpatialManager::updateEntity(uint32_t entityId, const vec2& oldPos, const vec2& newPos, float radius) {
    auto it = entityToDataIndex.find(entityId);
    if (it != entityToDataIndex.end()) {
        EntityData& data = entityData[it->second];
        data.posX = newPos.x;
        data.posY = newPos.y;
        data.radius = radius;
    }

    // Update in spatial structures
    quadTree.remove(entityId, oldPos);
    quadTree.insert(entityId, newPos, radius);
    hashGrid.update(entityId, oldPos, newPos, radius);
}

void SpatialManager::clearAll() {
    quadTree.clear();
    hashGrid.clear();
    entityData.clear();
    entityToDataIndex.clear();
    spatialBounds.entityCount = 0;
    queryCount = 0;
    quadTreeQueries = 0;
    hashGridQueries = 0;
}

void SpatialManager::findNearbyEntities(uint32_t entityId, const vec2& pos, float radius,
                                       std::vector<uint32_t>& results) const {
    results.clear();
    queryCount++;

    Strategy strategy = selectOptimalStrategy(pos, radius);

    switch (strategy) {
        case Strategy::QUADTREE_ONLY: {
            quadTreeQueries++;
            AABB queryRange = {pos.x - radius, pos.y - radius, pos.x + radius, pos.y + radius};
            quadTree.queryRange(queryRange, results, entityData.data());
            break;
        }
        case Strategy::HASHGRID_ONLY:
            hashGridQueries++;
            hashGrid.findNearbyEntities(pos, radius, results, entityData.data());
            break;
        case Strategy::HYBRID_AUTO: {
            // Use both and merge results
            std::vector<uint32_t> quadResults, hashResults;

            quadTreeQueries++;
            AABB queryRange = {pos.x - radius, pos.y - radius, pos.x + radius, pos.y + radius};
            quadTree.queryRange(queryRange, quadResults, entityData.data());

            hashGridQueries++;
            hashGrid.findNearbyEntities(pos, radius, hashResults, entityData.data());

            // Merge and deduplicate
            std::unordered_set<uint32_t> uniqueResults;
            for (uint32_t id : quadResults) uniqueResults.insert(id);
            for (uint32_t id : hashResults) uniqueResults.insert(id);

            results.reserve(uniqueResults.size());
            for (uint32_t id : uniqueResults) {
                if (id != entityId) { // Exclude self
                    results.push_back(id);
                }
            }
            break;
        }
    }
}

void SpatialManager::collectPerformanceMetrics(SpatialPerformanceMetrics& metrics) const {
    metrics.totalQueries = queryCount;
    metrics.quadTreeQueries = quadTreeQueries;
    metrics.hashGridQueries = hashGridQueries;
    metrics.quadTreeHitRate = CacheProfilingHooks::hitRate();
    metrics.hashGridHitRate = queryCount > 0 ? double(hashGridQueries) / queryCount : 0.0;
    metrics.averageEntitiesPerQuery = queryCount > 0 ? double(spatialBounds.entityCount) / queryCount : 0.0;
    metrics.rebalanceFrequency = 0.1f; // Placeholder
}

void SpatialManager::updateSpatialBounds(const SpatialBounds& bounds) {
    spatialBounds = bounds;

    if (bounds.needsRepartitioning()) {
        rebalanceIfNeeded();
    }
}

void SpatialManager::updateEntityData(const std::vector<EntityData>& entities) {
    // Bulk update for SIMD-friendly operations
    entityData = entities;

    // Rebuild index mapping
    entityToDataIndex.clear();
    for (size_t i = 0; i < entityData.size(); ++i) {
        if (entityData[i].entityId != UINT32_MAX) {
            entityToDataIndex[entityData[i].entityId] = i;
        }
    }
}

SpatialManager::Strategy SpatialManager::selectOptimalStrategy(const vec2& pos, float radius) const {
    // Simple heuristic: use QuadTree for sparse areas, HashGrid for dense areas
    float localDensity = 0.0f;

    // Estimate local density by checking a small area
    AABB localArea = {pos.x - radius * 5, pos.y - radius * 5, pos.x + radius * 5, pos.y + radius * 5};

    // Count entities in local area (simplified estimation)
    float areaSize = (localArea.maxX - localArea.minX) * (localArea.maxY - localArea.minY);
    localDensity = spatialBounds.entityCount / areaSize;

    if (localDensity > 0.01f) { // High density threshold
        return Strategy::HASHGRID_ONLY;
    } else if (localDensity < 0.001f) { // Low density threshold
        return Strategy::QUADTREE_ONLY;
    } else {
        return Strategy::HYBRID_AUTO;
    }
}

void SpatialManager::rebalanceIfNeeded() {
    // Rebalance QuadTree
    quadTree.rebalance(entityData.data());

    // Update HashGrid cell size
    if (!entityData.empty()) {
        float avgRadius = 0.0f;
        uint32_t count = 0;
        for (const auto& entity : entityData) {
            if (entity.entityId != UINT32_MAX) {
                avgRadius += entity.radius;
                count++;
            }
        }
        if (count > 0) {
            avgRadius /= count;
            hashGrid.updateCellSize(avgRadius, count);
        }
    }
}

} // namespace spatial