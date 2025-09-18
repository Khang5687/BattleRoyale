# BattleRoyale Circles - Implementation Plan (Vulkan C++ on macOS / MoltenVK)

## Goals
- Vulkan + GLFW app on macOS (MoltenVK) rendering a simulation of image-backed circles (“players”).
- Bouncing physics with circle-circle and circle-wall collisions; health bars; eliminations.
- Winner sequence: collapse to single winner avatar centered at fixed size with "username wins" text in custom font.
- Massive player counts (up to MAX_PLAYER, with fake circles) and lazy image loading using a visibility threshold.
- Display HUD text: "Players left: X" top-left.
- Robustness: avoid segfaults, races; handle edge cases and large N efficiently.

## macOS Vulkan specifics
- Use loader + MoltenVK. Enable `VK_KHR_portability_enumeration` on instance and set `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`.
- Enable device extension `VK_KHR_portability_subset` and `VK_KHR_swapchain`.
- Validation layer `VK_LAYER_KHRONOS_validation` in debug builds.

## High-level Architecture
- App: window, Vulkan context (instance, device, queues, swapchain, render pass, frame sync, descriptor pools).
- Renderer: draw pass for circles, health bars, and text (HUD). Prefer one pipeline for sprites/billboards, instanced.
- Simulation: ECS-lite structs with SoA for hot loops (positions, velocities, radii, health, alive flags, imageId, bias).
- Asset system: image manager with lazy load tiers (fake -> placeholder -> real). O(1) lookup by player index; avoid per-frame traversal costs.
- Collision system: uniform spatial hash grid (or multi-level grids for varied radii). Broad-phase via grid buckets; narrow-phase exact tests. Deterministic resolution order to avoid races.
- Scheduler: single-threaded simulation step; async I/O for image decode/loading. Thread-safe queues with double buffers to apply asset state updates.

## Massive Scale Strategy (1M+ Circles) + Circle-Specific Optimizations

### 🎯 **Circle-Optimized Scaling Strategy for 1M+ Entities**

**Core Insight**: When circles become "specks of dust" on screen, exploit visual irrelevance for computational savings through adaptive simulation tiers.

### **Adaptive Simulation Architecture**
```cpp
class AdaptiveCircleSimulation {
    std::vector<Circle> individualCircles;        // <10k: Full simulation
    std::vector<DensityCluster> mediumClusters;   // 10k-100k: Grouped simulation
    std::vector<StatisticalCluster> dustClusters; // 100k-1M: Statistical simulation
};
```

### **Tier-Based Entity Management System**
```cpp
enum CircleSimulationTier {
    INDIVIDUAL,    // >10 pixels: Full circle rendering + physics
    CLUSTERED,     // 2-10 pixels: Group nearby circles into density blobs
    STATISTICAL,   // 0.5-2 pixels: Statistical cluster simulation
    INVISIBLE      // <0.5 pixels: Position tracking only
};
```

### **Performance Scaling Matrix**
| Entity Count | Rendering Method | Physics Method | Expected Performance |
|-------------|------------------|----------------|-------------------|
| 1-1k | Individual circles + SDF | Full collision detection | 120+ FPS |
| 1k-10k | Individual + simple LOD | Spatial grid collision | 60+ FPS |
| 10k-100k | Density clusters | Cluster-to-cluster physics | 30+ FPS |
| 100k-1M+ | Statistical clusters | Statistical approximation | 30+ FPS |

### **Density-Based Rendering Optimization**
```cpp
// Calculate apparent screen size for dynamic detail selection
float apparentRadius = physicalRadius * zoomFactor / distanceFromCamera;

if (apparentRadius < 0.5f) {
    renderAsPixel(averageColor);        // SUB-PIXEL: Single colored pixel
} else if (apparentRadius < 2.0f) {
    renderAsSquare(color);              // TINY: Simple colored square
} else if (apparentRadius < 10.0f) {
    renderAsSimpleCircle(color);        // SMALL: Simplified SDF
} else {
    renderAsDetailedCircle(texture);    // LARGE: Full detail + texture
}
```

### **Statistical Clustering for Dust-Level Circles**
```cpp
struct StatisticalCluster {
    vec2 centerOfMass;          // Cluster position
    float totalMass;            // Combined physics mass
    vec2 averageVelocity;       // Average movement direction
    uint32_t aliveCount;        // Circles in cluster
    vec3 dominantColor;         // Most common circle color

    // Simplified physics: treat cluster as single large circle
    void updatePhysics(float dt);

    // Statistical elimination within cluster
    void processEliminations(float damage);
};
```

### **Dynamic Tier Promotion/Demotion System**
```cpp
void updateSimulationTiers(float currentZoom) {
    // Promote clusters to individuals when zooming in
    for (auto& cluster : dustClusters) {
        if (cluster.getApparentSize(currentZoom) > INDIVIDUAL_PROMOTION_THRESHOLD) {
            promoteClusterToIndividuals(cluster);
        }
    }

    // Demote individuals to clusters when zooming out
    for (auto& circle : individualCircles) {
        if (circle.getApparentSize(currentZoom) < CLUSTER_DEMOTION_THRESHOLD) {
            demoteToCluster(circle);
        }
    }
}
```

### **Battle Royale Elimination Cascading**
```cpp
// Natural entity count reduction over time
if (cluster.aliveCount < CLUSTER_BREAKUP_THRESHOLD) {
    // Small clusters break apart into individuals for dramatic finale
    convertClusterToIndividuals(cluster);
}
```

### **Adaptive Performance Thresholds**
```cpp
// Adjust clustering aggressiveness based on performance
if (frameTime > TARGET_FRAME_TIME) {
    DUST_THRESHOLD *= 1.1f;     // More aggressive clustering
} else {
    DUST_THRESHOLD *= 0.99f;    // More detailed rendering
}
```

### GPU-Driven Rendering Architecture (2024 State-of-Art)
- **Compute Shader Culling**: Offload frustum, distance, and occlusion culling to GPU compute shaders
- **Indirect Drawing**: Use `vkCmdDrawIndexedIndirect` with GPU-generated command buffers to eliminate CPU draw call overhead
- **Performance Target**: Real-world 125k objects at 290 FPS rendering 40M+ triangles (RTX 2080 benchmark)

### Multi-Level Optimization Strategy
1. **GPU Culling Pipeline**: Compute shaders perform frustum culling, reducing CPU load exponentially
2. **Hierarchical LOD**: GPU-based distance culling with multiple detail levels based on effective viewport size
3. **Occlusion Culling**: Hi-Z buffer technique using previous frame depth for visibility testing
4. **Mesh Shaders**: Modern Vulkan mesh shaders for 10x geometry pipeline performance improvement

### Spatial Partitioning for Collision Performance
- **Hybrid Spatial Structure**: Quadtree for 2D spatial queries + optimized hash grid for collision detection
- **Performance Scaling**: Reduces collision checks from O(n²) to O(n) for clustered objects
- **Dynamic Rebalancing**: Efficient add/remove operations as entities move and get eliminated

### Data-Oriented Memory Layout (ECS-Inspired)
- **Structure of Arrays (SoA)**: Separate arrays for position, velocity, health, alive flags for cache efficiency
- **SIMD-Friendly Processing**: Contiguous memory layout enables vectorized operations on multiple entities
- **Memory Coherence**: Related data stored together to minimize cache misses during hot loops

### Advanced Asset Management
- Do not load all images or create 500k Vulkan image views. Maintain metadata list of file paths and a small LRU cache of decoded + GPU-resident textures (e.g., few thousands max).
- Use a texture atlas array (array of 2D layers) of fixed layer count (e.g., 2048) for visible, above-threshold players. Evict via LRU when layers are scarce. Index via `imageId -> layerIndex | placeholder`.
- If descriptor indexing (bindless) is absent or limited on MoltenVK, prefer a small set of per-frame descriptor sets for the atlas array; use an indirection buffer mapping instance to atlas layer.
- **GPU-Controlled Loading**: Compute shaders determine which textures to load based on visibility thresholds
- Lazy tiers:
  - Tier 0 (fake): no file I/O; rendered as flat color in compute/fragment; guaranteed to die before threshold.
  - Tier 1 (placeholder): tiny 1x1 or 4x4 texture (low memory) for small-but-visible sprites.
  - Tier 2 (real): high-quality mipmapped texture uploaded on demand when radius > IMAGE_LOAD_THRESHOLD.
- O(1) algorithmic constraint for image access: each entity holds stable index to metadata; atlas layer lookup is O(1) via a dense vector; LRU updates are O(1) amortized with linked-list + hashmap.

### Performance Monitoring & Adaptive Quality
- **Real-time Profiling**: Monitor frame time, draw calls, and memory usage
- **Adaptive LOD**: Dynamically adjust IMAGE_LOAD_THRESHOLD based on performance headroom
- **Quality Scaling**: Reduce texture resolution, disable effects when frame rate drops below target

## Rendering Approach
- Instanced rendering of quads (two triangles) for circles; circle appearance via fragment shader signed distance function to avoid per-vertex circles. Image sampled using per-instance atlas layer; health bar drawn as a second instanced pass or in same shader via a screen-space overlay.
- One command buffer per frame, one draw for circles (instanced), one draw for health bars, one draw for text.
- Push constants or storage buffer for per-frame params (time, viewport, scaling factor, thresholds, counts).

## Scaling Visual Size Without Resizing Massive Counts
- **DEPRECATED APPROACH**: Global scale factor applied to radius causes physics-rendering mismatch and collision issues.
- **RECOMMENDED**: Camera/viewport scaling approach - adjust effective viewport dimensions rather than object sizes.
- Only when an instance crosses IMAGE_LOAD_THRESHOLD do we request real texture; below threshold we render flat color/placeholder.

## Physics and Collisions
- Time step: fixed dt (e.g., 1/120s) using accumulator; decouple from render rate.
- Spatial hash grid with cell size ~= average diameter of common radius tier. For high variance, consider multi-grid (e.g., powers of two radii bins).
- Insert alive circles into grid per frame (SoA helps). Broad-phase: only test against items in same and neighboring cells (9 cells).
- Elastic collision response with damping; apply health damage based on impulse magnitude. Clamp minimum damage to ensure progress.
- Wall collisions via AABB checks; invert velocity component and apply small damping.
- Deterministic iteration order (by index) and atomic-free single-threaded physics step to avoid races. Optionally parallelize by tiling grid regions if needed later with careful conflict management.

## Elimination and Winner Flow
- When health <= 0: mark dead, remove from grid, free/evict texture layer (decrement refcount), decrement alive count.
- Winner: when alive count == 1, transition to "victory" state: interpolate position to center, scale up to WINNER_SCALE, display name (filename stem) in center. Freeze others.

## File/IO and Bias
- Enumerate image filenames from `assets/` once; store stems as player names; maintain mapping for bias damage reduction values (config file `bias.txt` or hardcoded map, modifiable at runtime).
- Bias system: players with bias values get damage reduction (not health/size changes). Applied as: `finalDamage = baseDamage * (1.0f - biasReduction)`.
- Bias only active when player count >= BIAS_ACTIVE_THRESHOLD (default 50) to prevent obvious advantages in small battles.
- MAX_PLAYER: if actual image files < MAX_PLAYER, spawn fake circles to reach MAX_PLAYER; fake circles have Tier 0 textures/colors.

## Constants (tunable)
- MAX_PLAYER
- IMAGE_LOAD_THRESHOLD_RADIUS
- MAX_CIRCLE_SIZE
- WINNER_SCALE
- DAMAGE_MULTIPLIER, WALL_DAMPING, COLLISION_DAMPING
- GRID_CELL_SIZE
- BIAS_ACTIVE_THRESHOLD (default: 50 - minimum player count for bias to be active)

### 1M+ Entity Scaling Constants
- INDIVIDUAL_PROMOTION_THRESHOLD (default: 10.0f - minimum apparent radius for individual simulation)
- CLUSTER_DEMOTION_THRESHOLD (default: 2.0f - maximum apparent radius for cluster demotion)
- DUST_THRESHOLD (default: 0.5f - minimum apparent radius for any rendering)
- CLUSTER_BREAKUP_THRESHOLD (default: 10 - minimum cluster size before breakup)
- MIN_CLUSTER_SIZE (default: 50 - minimum circles needed to form cluster)
- TARGET_FRAME_TIME (default: 16.67ms - 60 FPS target for adaptive thresholds)

## Robustness and Edge Cases
- Validation layers in debug; check all Vk results; RAII wrappers to prevent leaks.
- Guard zero devices/queues/swapchain formats; handle window resize.
- Clamp velocities/radii; avoid NaNs; ensure no division by zero in collision math.
- Thread safety: image loader thread communicates via lock-free MPMC queue or mutex-protected queue; main thread consumes at frame boundary.
- Prevent thundering herd on texture loads: coalesce requests; deduplicate per imageId; cancel when entity dies.

## Text Rendering (Players left)
- Simple path: bake minimal bitmap font (e.g., stb_easy_font) to a dynamic vertex buffer, or integrate Dear ImGui for HUD (no heavy styling). Keep draw order last.

## Milestones
1) ✅ Minimal Vulkan window (this repo). Verify presentation.
2) ✅ Create swapchain, render pass, frame sync, render clear color.
3) ✅ Instanced quad pipeline, draw N flat-color circles with SDF in shader.
4) ✅ Spatial grid + collisions + health; eliminations.
5) ✅ Texture atlas/array, placeholder + real mipmapped textures, lazy loading.
6) ✅ HUD text, players left counter, winner flow.
7) ✅ Bias configuration and tuning.
8) ✅ Performance instrumentation baseline with real-time diagnostics.
9) ❌ Stress test & profiling; refine grid and resource limits.

## Current Status (✅ = COMPLETED, 🔧 = NEEDS WORK)
✅ **Core Vulkan Setup**: MoltenVK initialization, swapchain, render pass, command buffers
✅ **Circle Rendering**: Instanced SDF-based circle rendering with smooth anti-aliased edges
✅ **Physics System**: Spatial hash grid collision detection, elastic collisions with damping
✅ **Health System**: Health-based eliminations with color-coded health visualization (green→red)
✅ **Winner Detection**: Victory state with winner centering, scaling, and name display
✅ **HUD Output**: Console-based "Players left: X" counter with winner announcement
✅ **Bias Configuration**: Damage reduction-based bias system with small-game protection
✅ **Speed Control**: Configurable speed multiplier constant for faster/slower gameplay
✅ **Asset Integration**: File enumeration from assets/ directory with bias application
✅ **Real Battle Simulation**: Battle royale with eliminations progressing to single winner
✅ **Image Avatar Loading System**: Complete texture atlas array with LRU cache and lazy loading tiers
🔧 **Dynamic Circle Scaling**: Global scale factor approach deprecated due to physics issues - needs camera/viewport scaling redesign

## ✅ **URGENT FIXES COMPLETED**
✅ **Circle Size Issues**: All circles now have uniform fixed radius (20.0px) - random size variation removed
✅ **Bias System Redesign**: Implemented damage reduction approach instead of health multipliers - bias is less visually obvious
✅ **Small Game Bias Protection**: Bias now only activates when player count ≥ 50 to prevent obvious advantages in small battles

## 🎉 Image Avatar Loading System - IMPLEMENTATION COMPLETE

The battle royale simulation now successfully runs with the new image avatar loading system! All major components have been implemented and are operational:

### ✅ Core Infrastructure
- **Texture Atlas Array**: 2048-layer 256x256 texture atlas for efficient GPU storage
- **LRU Cache System**: Efficient texture management with automatic eviction when atlas fills
- **Lazy Loading Tiers**:
  - Tier 0: Flat color (fake circles with small radius)
  - Tier 1: Placeholder textures
  - Tier 2: Real images (loaded when radius ≥ 20px threshold)

### ✅ Image Pipeline
- **STB Image Integration**: Background loading of JPEG/PNG files from assets/ directory
- **Multi-threaded Loading**: Background thread for image decoding without blocking simulation
- **Vulkan Upload System**: Proper memory management and image state transitions
- **Dynamic Scaling**: Images automatically resized to 256x256 atlas slots as needed

### ✅ Rendering Integration
- **Updated Shaders**: Support for texture sampling with health color blending
- **Descriptor Sets**: Proper Vulkan binding of texture atlas to shaders
- **Instance Data**: Extended vertex attributes for image layer indexing
- **Visual Feedback**: Health-based color tinting over loaded textures

### ✅ Performance Features
- **O(1) Atlas Lookup**: Hash-based image ID to atlas layer mapping
- **Efficient Eviction**: LRU-based texture replacement when atlas capacity reached
- **Threshold-based Loading**: Only load high-quality textures for large enough circles
- **Memory Management**: Proper staging buffers and resource cleanup

The system supports rendering image avatars for battle royale circles while maintaining high performance through lazy loading and efficient GPU memory usage. Circles start as flat colors and upgrade to real image textures when they become large enough to warrant the loading cost.

## 📊 Performance Instrumentation System - IMPLEMENTATION COMPLETE

### ✅ Core Performance Monitoring Infrastructure
- **Frame Timing System**: High-resolution CPU frame timing with 120-sample rolling average
- **Real-time FPS Display**: Calculated from rolling average frame times (1000ms / avgFrameTime)
- **Performance Metrics Structure**: Comprehensive timing data collection with `std::chrono`
- **Rolling Window Analysis**: 120-frame window providing 1-second performance averages

### ✅ On-Screen Diagnostics Overlay
- **Toggle Control**: Press F3 to show/hide performance diagnostics during runtime
- **Multi-metric Display**: FPS, frame time (ms), image loading stats, atlas usage, total frames
- **Visual Formatting**: Light green text with black drop shadows for readability
- **Non-intrusive Design**: Overlay positioned in top-left corner below "Players left" counter

### ✅ Performance Data Collection
- **Frame Time Capture**: Automatic timing at start of each render loop iteration
- **Image Loading Metrics**: Success/failure counters via atomic operations for thread safety
- **Atlas Utilization Tracking**: Real-time monitoring of texture atlas layer usage
- **Console Integration**: Periodic status updates every 120 frames (1-second intervals)

### ✅ Baseline Documentation
- **Performance Baseline Report**: Complete documentation in `PERFORMANCE_BASELINE.md`
- **Target Metrics Defined**: 60 FPS minimum, 125k+ entities stretch goal
- **Optimization Roadmap**: Clear path forward for GPU-driven rendering improvements
- **Monitoring Guidelines**: Usage instructions for real-time diagnostics

### 🎯 Performance Instrumentation Features
```cpp
struct PerformanceMetrics {
    static constexpr size_t SAMPLE_COUNT = 120; // 1 second at 120 FPS
    std::array<float, SAMPLE_COUNT> frameTimes{};
    size_t sampleIndex = 0;
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    float rollingAverage = 0.0f;
    bool showDiagnostics = false;
    uint64_t totalFrames = 0;
};
```

### 🔧 Real-time Diagnostics Display
- **FPS**: Current frames per second with frame time in milliseconds
- **Images**: Loaded texture count (success/failed ratio)
- **Atlas**: Texture atlas layer utilization (used/total capacity)
- **Frames**: Total frames rendered since application start
- **Help**: Key binding reference (F3 toggle)

The performance instrumentation provides comprehensive monitoring capabilities essential for measuring optimization effectiveness. All metrics are captured automatically with minimal performance overhead, enabling data-driven optimization decisions.

## 🔄 Dynamic Camera Scaling System - PLAYER COUNT-BASED ZOOM

**DESIGN GOAL**: Seamless camera scaling that makes circles appear larger as player count decreases, with physics remaining unchanged.

### 🎯 Core Concept: Player Count-Driven Camera Zoom

**Visual Effect**:
- **50,000 players**: Camera zoomed out → circles appear small but all visible
- **Decreasing count**: Camera gradually zooms in → circles appear progressively larger
- **Final players**: Camera zoomed in → circles appear large and dramatic
- **Seamless transition**: Viewers perceive circles as "growing" naturally

### 🎮 Implementation Approach: Viewport Scaling with Constraints

**Camera Zoom Factor Calculation**:
```cpp
// Zoom boundaries
static constexpr float MIN_ZOOM_FACTOR = 0.5f;  // Max zoom out (50k players)
static constexpr float MAX_ZOOM_FACTOR = 3.0f;  // Max zoom in (final players)
static constexpr uint32_t MAX_PLAYERS_FOR_MIN_ZOOM = 50000;
static constexpr uint32_t MIN_PLAYERS_FOR_MAX_ZOOM = 1;

float zoomFactor = calculateZoomFromPlayerCount(aliveCount);
vec2 effectiveViewport = realViewport / zoomFactor;
```

**Zoom Calculation Logic**:
- **High player count** (≥50k): Minimum zoom factor (0.5x) → maximum world area visible
- **Low player count** (≤1): Maximum zoom factor (3.0x) → minimum world area visible
- **Between counts**: Smooth interpolation for seamless scaling experience
- **Bounded range**: Min/max constraints prevent extreme zoom levels

### 🏗️ Technical Implementation

**Vertex Shader Modification**:
- Replace `pc.viewport` with `pc.effectiveViewport` in world-to-NDC conversion
- Physics world space remains unchanged at fixed 800x600 dimensions
- Collision walls stay at original `worldWidth/worldHeight` boundaries

**Push Constants Update**:
```cpp
struct PushConstants {
    vec2 effectiveViewport;  // Scaled viewport for zoom effect
    // Physics uses original worldWidth/worldHeight unchanged
};
```

**Collision System**:
- **Wall boundaries must scale** - collision walls need to match visible viewport area
- **Effective world boundaries**: `effectiveWorldWidth/Height = worldWidth/Height / zoomFactor`
- Circle physics continues at 40px radius with existing collision detection
- Spatial grid dimensions scale with effective world size for consistency

### ✨ Key Advantages

**Visual Benefits**:
- Circles appear to "grow" as battle progresses toward finale
- Smooth, cinematic zoom effect enhances drama
- No jarring scale jumps - continuous interpolation

**Technical Benefits**:
- **Consistent collision boundaries** - walls match visible screen edges
- **O(1) performance** - single calculation per frame for zoom factor
- **Proper viewport-collision alignment** - no invisible wall bouncing
- **Industry standard approach** - camera scaling with matched collision boundaries

**Constraint Benefits**:
- **Minimum zoom** prevents over-zooming with massive player counts
- **Maximum zoom** prevents excessive close-up with few players
- **Bounded experience** ensures consistent visual quality range

### 🎯 Implementation Priority

**IMPLEMENTATION STEPS**:
1. Add zoom factor calculation based on `sim.aliveCount()`
2. Calculate effective world boundaries: `effectiveWorld = worldSize / zoomFactor`
3. Update wall collision detection to use effective boundaries instead of fixed world size
4. Update spatial grid dimensions to match effective world size
5. Modify push constants to send `effectiveViewport` instead of `viewport`
6. Update vertex shader to use effective viewport for world-to-NDC conversion
7. Add min/max zoom factor constants for bounded scaling
8. Test smooth scaling behavior with proper wall collision alignment

**Expected Outcome**: Seamless camera zoom where circles appear to grow as the battle progresses, with collision walls properly aligned to visible screen edges, maintaining consistent 40px circle physics.

  ---

## DETAILED TODO LIST

### HIGH PRIORITY - Core Features Missing
- [x] **Image Avatar Loading System** (Milestone 5) ✅ **COMPLETED**
  - [x] Implement texture atlas array for GPU image storage (2048-layer 256x256 texture atlas)
  - [x] Create LRU cache system for texture management (efficient O(1) lookup and eviction)
  - [x] Add lazy loading tiers: Tier 0 (flat color/fake circles) → Tier 1 (placeholder) → Tier 2 (real images)
  - [x] Implement IMAGE_LOAD_THRESHOLD_RADIUS system (radius ≥ 20px threshold for real image loading)
  - [x] Add image decoding pipeline (STB Image integration with multi-threaded background loading)
  - [x] Create Vulkan texture upload system with proper memory management and image transitions
  - [x] Updated shaders with texture sampling and health color blending
  - [x] Extended instance data with image layer indexing
  - [x] Hash-based image ID to atlas layer mapping for O(1) access
  - [x] Dynamic scaling to 256x256 atlas slots with staging buffers

- [x] **✅ Circle Size & Bias System Fixes** **COMPLETED**
  - [x] **Uniform Circle Sizes**: Removed random radius variation - all circles now have identical fixed radius
    - [x] Replaced `std::uniform_real_distribution<float> distR(minRadius, maxRadius)` with `fixedRadius = 20.0f`
    - [x] All players now use consistent fixed radius for uniform visual appearance
    - [x] Removed bias-based radius scaling that made advantages too obvious
  - [x] **Redesigned Bias System**: Changed from health multipliers to damage reduction system
    - [x] Replaced `biasMultipliers` with `biasReductions` map for damage reduction approach
    - [x] Implemented damage reduction: `finalDamage = baseDamage * (1.0f - biasReduction)`
    - [x] All players start with uniform health (1.0f), bias only affects damage taken
    - [x] Bias is now less visually obvious while still providing competitive advantage
  - [x] **Bias Player Count Threshold**: Added configurable threshold for small games
    - [x] Added constant: `static constexpr uint32_t BIAS_ACTIVE_THRESHOLD = 50;`
    - [x] Bias only applies when `aliveCount() >= BIAS_ACTIVE_THRESHOLD`
    - [x] Below threshold, all players take equal damage (no bias applied)
    - [x] Prevents bias from being obvious in small battles

- [x] **Visual HUD Text Rendering**
  - [x] Replace console output with on-screen text overlay
  - [x] Implement simple bitmap font rendering or Dear ImGui integration
  - [x] Display "Players left: X" in top-left corner
  - [x] Add winner name display in center screen during victory

- [x] **Performance Instrumentation Baseline** ✅ **COMPLETED**
  - [x] Integrate frame-time capture (CPU & GPU) with rolling average output
  - [x] Add toggleable on-screen diagnostics overlay (FPS, alive count, image stats)
  - [x] Document baseline metrics pre-optimization (target player counts)

- [ ] **🔄 Dynamic Camera Scaling** (Player Count-Based Zoom) **DEFERRED – WAITING ON NEW SPATIAL PIPELINE**
  - [x] ~~Global scale factor approach~~ **DEPRECATED** - causes physics-rendering mismatch
  - [ ] **Prep work only**: define `CameraState` (min/max zoom, smoothing, winner focus)
  - [ ] Blocked: implement zoom logic after advanced spatial partitioning lands
  - [ ] Coordinate final implementation with **Enhanced Winner Sequence** tasks for shared camera polish
  - [ ] Post-blocker checklist (unlocked once new spatial system ships)
    - [ ] Add zoom factor calculation: `zoomFactor = calculateZoomFromPlayerCount(aliveCount)`
    - [ ] Add min/max zoom constants (MIN_ZOOM_FACTOR = 0.5f, MAX_ZOOM_FACTOR = 3.0f)
    - [ ] Calculate effective world boundaries: `effectiveWorld = worldSize / zoomFactor`
    - [ ] Update wall collision detection to use effective boundaries (not fixed world size)
    - [ ] Update spatial grid dimensions to match effective world size
    - [ ] Calculate effective viewport: `effectiveViewport = realViewport / zoomFactor`
    - [ ] Update push constants to send effectiveViewport instead of viewport
    - [ ] Update vertex shader to use pc.effectiveViewport for world-to-NDC conversion
    - [ ] Keep circle physics unchanged at fixed 40px radius
    - [ ] Test smooth scaling with proper wall collision alignment from 50k players to winner
  - [ ] **Constraint validation (post-blocker)**
    - [ ] Verify minimum zoom (0.5x) handles massive player counts without over-zooming
    - [ ] Verify maximum zoom (3.0x) provides dramatic final battle view
    - [ ] Test smooth interpolation between zoom levels

### MEDIUM PRIORITY - Polish & Performance
- [ ] **Health Bar Rendering**
  - [ ] Add visual health bars above/below circles
  - [ ] Implement second instanced rendering pass for health indicators
  - [ ] Color-code health bars (green → yellow → red)

- [ ] **Enhanced Winner Sequence**
  - [x] Simplify celebration to a single centered winner avatar with other circles removed
  - [x] Show custom-font "username wins" banner while the regular HUD stays hidden
  - [ ] Layer in optional celebratory VFX (confetti, particles, post effects)
  - [ ] Revisit camera polish or transitions once future zoom work is defined

### HIGH PRIORITY - 1M+ Entity Scaling Implementation
- [ ] **Adaptive Simulation Architecture Foundation**
  - [ ] Implement `AdaptiveCircleSimulation` class with tier-based entity management
  - [ ] Define `CircleSimulationTier` enum (INDIVIDUAL, CLUSTERED, STATISTICAL, INVISIBLE)
  - [ ] Create data structures for individual circles, density clusters, and statistical clusters
  - [ ] Add apparent screen size calculation: `apparentRadius = physicalRadius * zoomFactor / distanceFromCamera`

- [ ] **Density-Based Rendering System**
  - [ ] Implement dynamic detail selection based on apparent radius thresholds
  - [ ] Add pixel-level rendering for sub-pixel circles (aparentRadius < 0.5f)
  - [ ] Add square rendering for tiny circles (0.5f-2.0f pixels)
  - [ ] Add simplified SDF for small circles (2.0f-10.0f pixels)
  - [ ] Keep full detail rendering for large circles (>10.0f pixels)

- [ ] **Statistical Clustering Implementation**
  - [ ] Create `StatisticalCluster` data structure with center of mass, total mass, average velocity
  - [ ] Implement cluster formation logic for nearby dust-level circles
  - [ ] Add cluster-to-cluster physics simulation (simplified)
  - [ ] Implement statistical elimination within clusters
  - [ ] Add cluster breakup logic when count falls below threshold

- [ ] **Dynamic Tier Promotion/Demotion System**
  - [ ] Implement `updateSimulationTiers()` function with zoom-based promotion/demotion
  - [ ] Add promotion thresholds: clusters → individuals when zooming in
  - [ ] Add demotion thresholds: individuals → clusters when zooming out
  - [ ] Implement smooth transitions between simulation tiers

- [ ] **Adaptive Performance System**
  - [ ] Add real-time performance monitoring for tier threshold adjustment
  - [ ] Implement dynamic `DUST_THRESHOLD` adjustment based on frame time
  - [ ] Add performance headroom calculation for aggressive/conservative clustering
  - [ ] Create performance scaling matrix validation (1k→10k→100k→1M entities)

- [ ] **Circle-Specific Optimizations**
  - [ ] Implement battle royale elimination cascading (cluster breakup near finale)
  - [ ] Add pixel-cluster aggregation for overlapping circles
  - [ ] Optimize collision detection for clustered vs individual entities
  - [ ] Validate viewport-physics harmony across different simulation tiers

### MEDIUM PRIORITY - Advanced Performance
- [ ] **Performance Optimization** (Milestone 8) **RESEARCH-BACKED STRATEGIES**
  - [ ] **GPU-Driven Rendering – Stage 1 (Compute Culling Prototype)**
    - [ ] Stand up compute pass that frustum-culls instance data into a GPU-visible list
    - [ ] Define shared visibility buffer layout (include hook for future `CameraState` zoom factor)
    - [ ] Validate correctness against CPU path using instrumentation metrics
  - [ ] **GPU-Driven Rendering – Stage 2 (Indirect Draw Integration)**
    - [ ] Replace direct draws with `vkCmdDrawIndexedIndirect`
    - [ ] Add GPU-side instance count readback guards or fallback path
    - [ ] Benchmark draw-call reduction using new baseline instrumentation
  - [ ] **GPU-Driven Rendering – Stage 3 (Hi-Z Occlusion & Refinement)**
    - [ ] Build Hi-Z buffer from prior frame depth
    - [ ] Integrate occlusion test into compute culling
    - [ ] Target: 125k+ entities at 60+ FPS (industry benchmark achieved)
  - [ ] **Advanced Spatial Partitioning**
    - [ ] Replace simple hash grid with hybrid quadtree + hash grid system
    - [ ] Implement dynamic rebalancing for moving entities
    - [ ] Optimize collision detection from O(n²) to O(n) for clustered objects
    - [ ] Expose spatial bounds API consumed by deferred camera zoom feature
  - [ ] **Data-Oriented Memory Optimization**
    - [ ] Verify current SoA layout is SIMD-friendly for vectorized operations
    - [ ] Add memory coherence profiling for cache miss optimization
    - [ ] Implement batch processing for position/velocity updates
  - [ ] **Modern Vulkan Features** (Advanced)
    - [ ] Evaluate mesh shader implementation for 10x geometry performance
    - [ ] Consider GPU-controlled texture loading via compute shaders

### LOW PRIORITY - Advanced Features
- [ ] **Ultra-Massive Scale Support** (500k+ files) **BEYOND CURRENT SCOPE**
  - [ ] Implement ECS-style entity management for million+ entities (Unity DOTS-inspired)
  - [ ] Add background texture streaming with predictive loading
  - [ ] Create hierarchical culling system with mesh shaders
  - [ ] Research: Burst compiler equivalent for C++ SIMD optimization

- [ ] **Enhanced Collision System**
  - [ ] Add collision sound effects
  - [ ] Implement particle effects for impacts
  - [ ] Add circle trail/motion blur effects

- [ ] **Configuration & Tuning**
  - [ ] Create JSON-based configuration file for all constants
  - [ ] Add runtime parameter adjustment (speed, damage, etc.)
  - [ ] Implement save/load settings system

- [ ] **Debug & Development Tools**
  - [ ] Add debug visualization for spatial grid
  - [ ] Implement performance profiling overlay
  - [ ] Add collision count statistics
  - [ ] Create replay system for battles

---

## Notes on MoltenVK capabilities
- Descriptor indexing/bindless is limited; plan on atlas array + indirection buffer.
- Avoid geometry shaders; use instancing + SDF.
- Prefer fewer pipelines; Metal backend likes stable pipelines and small descriptor sets.
