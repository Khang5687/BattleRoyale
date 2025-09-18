# Performance Instrumentation Baseline - BattleRoyale5

## Overview
This document establishes baseline performance metrics for the BattleRoyale5 simulation before optimization, serving as a reference point for future performance improvements.

## Instrumentation Features Implemented

### Frame Timing System
- **Rolling Average**: 120-sample rolling window (1 second at 120 FPS target)
- **CPU Frame Timing**: High-resolution timestamp capture using `std::chrono::high_resolution_clock`
- **FPS Calculation**: Real-time FPS calculation from rolling average frame times
- **Frame Counter**: Total frames rendered since application start

### On-Screen Diagnostics Overlay (Press F3 to toggle)
- **FPS Display**: Current FPS with frame time in milliseconds
- **Image Loading Stats**: Successful vs failed image loads
- **Atlas Usage**: Current texture atlas layer utilization (used/max layers)
- **Frame Counter**: Total frames rendered
- **Help Text**: Key binding information

### Performance Metrics Structure
```cpp
struct PerformanceMetrics {
    static constexpr size_t SAMPLE_COUNT = 120; // 1 second at 120 FPS
    std::array<float, SAMPLE_COUNT> frameTimes{};
    size_t sampleIndex = 0;
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    float rollingAverage = 0.0f;
    bool showDiagnostics = false;
    uint64_t totalFrames = 0;

    // Methods: init(), captureFrame(), getFPS(), getFrameTimeMs(), toggleDiagnostics()
};
```

## Baseline Testing Configuration

### Test Environment
- **Platform**: macOS (Darwin 24.6.0)
- **Graphics API**: Vulkan with MoltenVK
- **Window Size**: 800x600 pixels
- **Target Player Count**: 50,000 entities

### Current Performance Characteristics

#### Entity Configuration
- **Max Players**: 50,000 (mix of real images and fake circles)
- **Fixed Circle Radius**: 20.0px (uniform size for all entities)
- **Physics Timestep**: Fixed 1/120s (120Hz physics simulation)
- **Texture Atlas**: 2048 layers, 256x256 per layer

#### Asset Loading System
- **Image Load Threshold**: Radius ≥ 20px triggers real texture loading
- **LRU Cache Management**: Automatic texture eviction when atlas capacity reached
- **Background Loading**: Multi-threaded image decoding
- **Loading Tiers**: Tier 0 (flat color) → Tier 1 (placeholder) → Tier 2 (real images)

#### Known Performance Hotspots (Pre-optimization)
- **Collision Detection**: O(n²) naive approach for 50k entities
- **CPU-Driven Rendering**: Direct draw calls without GPU culling
- **Memory Layout**: Mixed SoA/AoS - room for SIMD optimization
- **Texture Loading**: Synchronous uploads during frame rendering

## Performance Targets for Optimization

### Target Metrics
- **Minimum FPS**: 60 FPS sustained with 50k+ entities
- **Stretch Goal**: 125k+ entities at 60+ FPS (industry benchmark)
- **Frame Time Target**: <16.67ms (60 FPS) to <8.33ms (120 FPS)

### Optimization Roadmap
1. **GPU-Driven Rendering**: Compute shader culling + indirect draw calls
2. **Advanced Spatial Partitioning**: Hybrid quadtree + hash grid system
3. **SIMD-Optimized Memory Layout**: Vectorized position/velocity updates
4. **Hierarchical LOD**: Distance-based detail reduction
5. **Adaptive Quality**: Dynamic threshold adjustment based on performance headroom

## Monitoring Usage

### Real-time Diagnostics
- Press **F3** to toggle performance overlay during runtime
- Console output shows periodic status updates every 120 frames (1 second)
- Diagnostic overlay shows:
  - Current FPS and frame time (ms)
  - Image loading success/failure counts
  - Texture atlas utilization
  - Total frames rendered

### Console Output Example
```
Performance instrumentation initialized (Press F3 for diagnostics overlay)
Alive count: 50000
Starting battle royale with 50000 players!
Players left: 49995 | Files loaded: 45 success, 2 failed
```

### Data Collection
- Frame timing data collected automatically with 120-sample rolling window
- Atlas usage tracked via `imageIdToLayer.size()` for real-time monitoring
- Image loading statistics via atomic counters for thread-safe tracking

## Next Steps

This baseline establishes the measurement infrastructure needed to validate optimization efforts. The performance instrumentation will continue capturing metrics as optimizations are implemented, allowing for before/after comparisons and regression detection.

Key metrics to monitor during optimization:
- Frame time regression/improvement
- Atlas utilization efficiency
- Image loading throughput
- Memory usage patterns
- Entity scaling limits

---
*Generated on: September 18, 2025*
*Target Platform: macOS with MoltenVK*
*Version: BattleRoyale5 Performance Instrumentation Baseline*