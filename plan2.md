# Optimization Plan: Rendering 1M+ Textured Circles on MoltenVK/Mac

## Executive Summary

**Current State:**
- 5800 textured circles (256x256) cause significant lag
- Target: Smooth 10K circles immediately, scale to 1M circles
- Platform: MoltenVK on Apple Silicon (M1/M2/M3)

**Core Problem:**
At 1M circles, individual objects become sub-pixel (<1px) making full texture rendering wasteful. System needs hierarchical LOD with GPU-driven culling and aggressive texture streaming.

**Performance Target:**
- 10K circles @ 60 FPS (immediate goal)
- 1M circles @ 60 FPS (scale target with LOD/clustering)

---

## Phase 1: GPU-Driven Rendering Pipeline (HIGH PRIORITY)

### 1.1 Activate Compute Culling (Week 1)
**Status:** Infrastructure exists in `gpu_driven_rendering.hpp` but not deployed

**Implementation:**
1. **Compile compute shader** (`getGPUCullingComputeShader()`)
   - Add compute shader compilation to CMakeLists.txt
   - Generate SPIR-V from inline GLSL source
   - Location: `gpu_driven_rendering.hpp:61-179`

2. **Initialize GPU buffers**
   - Instance data buffer (SSBOs for all circle data)
   - Draw indirect buffer (VkDrawIndexedIndirectCommand array)
   - Visibility buffer (per-instance visibility flags)
   - Draw count buffer (atomic counter for visible objects)

3. **Integrate culling dispatch**
   - Before draw: Dispatch compute shader (64 threads/workgroup)
   - Frustum culling + pixel-size culling in compute
   - Output: Compacted draw indirect commands

4. **Replace CPU culling**
   - Remove per-frame CPU visibility checks
   - Use `vkCmdDrawIndexedIndirectCount()` for GPU-driven drawing

**Expected Performance:**
- Cull 1M objects in <0.5ms (GPU compute)
- Reduce CPU overhead by 80-90%

**MoltenVK Consideration:**
- **Issue:** Metal lacks native `drawCount` parameter in indirect draw ([MoltenVK #168](https://github.com/KhronosGroup/MoltenVK/issues/168))
- **Workaround:** Use `vkCmdDrawIndexedIndirectCount()` which MoltenVK implements via argument buffers
- **Alternative:** Fixed-size indirect buffer with instanceCount=0 for culled objects

---

## Phase 2: Hierarchical LOD System (Week 1-2)

### 2.1 GPU-Based LOD Classification
**Current:** `CircleRenderTier` enum exists but LOD selection is CPU-side

**Implement 4-Tier LOD in Compute Shader:**

```glsl
// Screen-space radius calculation
float screenRadius = worldRadius * zoomFactor / clipDepth;
float pixelRadius = screenRadius * screenWidth * 0.5;

// Tier classification
uint lodTier;
if (pixelRadius < 2.0) {
    lodTier = 0; // PIXEL_DUST - single colored pixel, no texture
} else if (pixelRadius < 10.0) {
    lodTier = 1; // SIMPLE_SHAPE - solid circle, no texture
} else if (pixelRadius < 50.0) {
    lodTier = 2; // BASIC_TEXTURE - 64x64 mipmap
} else {
    lodTier = 3; // FULL_DETAIL - 256x256 full texture
}
```

**Benefits:**
- Tier 0/1: 95% of objects at high zoom-out = massive texture bandwidth savings
- Tier 2: Low-res texture reduces memory reads by 16x
- Tier 3: Only for large, visible circles

### 2.2 Dynamic Instancing by LOD Tier

**Approach:** Sort/batch instances by LOD tier for efficient rendering

```cpp
// Compute shader writes to tier-specific instance buffers
struct TieredDrawCommands {
    VkDrawIndirectCommand tier0; // Pixel dust (point sprites)
    VkDrawIndirectCommand tier1; // Simple circles (no texture)
    VkDrawIndirectCommand tier2; // Basic texture (64x64)
    VkDrawIndirectCommand tier3; // Full detail (256x256)
};
```

**Multi-Draw Strategy:**
1. Single compute pass outputs 4 draw commands
2. Bind appropriate pipeline per tier
3. Tier 0: Point sprite rendering (1 vertex per circle)
4. Tier 1-3: Instanced quad rendering with texture

---

## Phase 3: Texture Streaming & Mipmap Generation (Week 2-3)

### 3.1 Generate Mipmaps for Atlas

**Critical Issue:** Current atlas has NO mipmaps
- Location: `TextureAtlas::ATLAS_SIZE = 256` (single mip level)
- Impact: 256x256 textures sampled even for 2px circles = 16384x wasted bandwidth

**Implementation:**
1. **Modify atlas creation** (src/main.cpp:5726+)
   ```cpp
   VkImageCreateInfo atlasInfo{};
   atlasInfo.mipLevels = 9; // log2(256) + 1 = 9 levels
   // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
   ```

2. **Generate mipmaps on upload**
   - After uploading layer, blit-generate mipchain
   - Use `vkCmdBlitImage()` for GPU-side downsampling
   - Transition layouts per-mip-level

3. **Update sampler** (src/main.cpp:5422+)
   ```cpp
   samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; // ✓ Already set
   samplerInfo.minLod = 0.0f;
   samplerInfo.maxLod = 9.0f; // Enable all mip levels
   ```

4. **Shader LOD selection**
   ```glsl
   // Fragment shader automatic mipmap selection via derivatives
   vec4 texColor = texture(uTextureAtlas, vec3(vTexCoord, atlasLayer));
   // Or explicit LOD:
   vec4 texColor = textureLod(uTextureAtlas, vec3(vTexCoord, atlasLayer), lodLevel);
   ```

**Memory Savings:**
- With mipmaps: Additional 33% memory (256+128+64+...+1 = 341 vs 256)
- Bandwidth savings: Up to 16x for small circles (64x64 vs 256x256)

### 3.2 Priority-Based Streaming

**Current System:** `ImageManager` already has priority system
- Enhance with LOD-aware priority:
  ```cpp
  float priority = baseScore;
  if (lodTier >= 3) priority *= 10.0f;  // High-detail visible circles
  if (lodTier == 2) priority *= 3.0f;   // Medium detail
  if (lodTier <= 1) priority *= 0.1f;   // Low priority, may not need texture
  ```

**Streaming Policy:**
- Preload Tier 3 textures first (large visible circles)
- Lazy-load Tier 2 (medium circles)
- Skip loading Tier 0/1 (sub-pixel, use solid colors)

---

## Phase 4: MoltenVK-Specific Optimizations (Week 3-4)

### 4.1 Leverage Apple Silicon Unified Memory

**Architecture Advantage:**
- CPU and GPU share physical memory (no PCIe transfer)
- `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` coexist

**Optimization:**
1. **Persistent mapped buffers**
   - Keep instance buffers persistently mapped
   - Write directly to GPU memory from CPU
   - No staging buffer needed

2. **Reduce synchronization**
   - Use Metal's `MTLStorageModeShared` (via MoltenVK)
   - Automatic coherency for shared allocations

**Code Changes:**
```cpp
// Allocate with shared memory properties
VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                             | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
// On Apple Silicon, this is zero-copy shared memory
```

### 4.2 Argument Buffers for Bindless Textures

**Current:** `BindlessTextureSystem` uses descriptor indexing
- MoltenVK translates to Metal argument buffers
- **Issue:** Descriptor indexing has overhead on Metal

**Optimization:**
1. **Use Metal 4 MTL4ArgumentTable** ([MoltenVK #2560](https://github.com/KhronosGroup/MoltenVK/issues/2560))
   - Avoids rebinding descriptors every frame
   - Persistent argument buffer binding

2. **Batch descriptor updates**
   - Update multiple texture descriptors at once
   - Use `vkUpdateDescriptorSets()` with large write arrays

3. **Descriptor set strategy**
   - Set 0: Global (never changes) - textures, samplers
   - Set 1: Per-frame (once/frame) - uniforms, culling data
   - Set 2: Per-draw (if needed) - push constants only

### 4.3 Metal Indirect Command Buffers (Future)

**Advanced Optimization:** Metal 3+ supports `MTLIndirectCommandBuffer`
- Encode entire draw sequence on GPU
- Reduces CPU overhead to near-zero

**Consideration:** Requires native Metal path, not available through MoltenVK Vulkan API

---

## Phase 5: Extreme Scale - Clustering & Impostors (Week 4-5)

### 5.1 Sub-Pixel Circle Clustering

**Problem:** At 1M circles with high zoom-out, most circles are <1px
- Rendering 1M quads wastes GPU cycles
- Most fragments get depth-tested out

**Solution: Density Field Clustering**

**Implementation:**
1. **Spatial binning** (compute shader)
   ```glsl
   // Bin circles into screen-space tiles (e.g., 32x32 px tiles)
   ivec2 tile = ivec2(screenPos.xy / 32.0);
   atomicAdd(densityGrid[tile.y * gridWidth + tile.x], 1);
   ```

2. **Render density blobs**
   - For tiles with >100 sub-pixel circles: Render single impostor sprite
   - Color = average team color, alpha = density
   - Eliminates 1M draw calls -> ~1K blob draws

3. **Hybrid rendering**
   - LOD Tier 0: Feed into clustering system
   - LOD Tier 1-3: Normal instanced rendering

**Expected Performance:**
- 1M circles at far zoom: ~1000 blob draws + ~10K normal draws = 60 FPS
- Near zoom: Normal LOD system (only ~50K visible circles)

### 5.2 Impostor Sprites for Medium Detail

**For Tier 2 (10-50px circles):**
- Pre-render common texture variants to atlas
- Use impostor quads with baked lighting/color
- Reduces fragment shader complexity

---

## Phase 6: Memory Optimization & Compression (Week 5-6)

### 6.1 Texture Compression

**Current:** R8G8B8A8_UNORM uncompressed (4 bytes/pixel)
- 256x256 texture = 256KB
- 5808 textures = 1.48 GB VRAM

**Implement ASTC Compression:**
- **ASTC 8x8** (Metal/iOS/macOS native):
  - Compression: 4bpp (8x reduction)
  - Quality: Excellent for photos/avatars
  - 256x256 texture: 32KB (vs 256KB)
  - Total: 185 MB (vs 1.48 GB) = **87% memory savings**

**Implementation:**
1. **Offline compression**
   - Use `astcenc` tool to compress source images
   - Store `.astc` files in assets/

2. **Runtime upload**
   ```cpp
   VkImageCreateInfo imgInfo{};
   imgInfo.format = VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
   // MoltenVK automatically uses Metal's ASTC decoder
   ```

3. **Fallback path**
   - Check `VK_FORMAT_ASTC_8x8_SRGB_BLOCK` support
   - Fallback to BC7 (desktop) or keep uncompressed

**Benefits:**
- 8x less memory bandwidth (critical bottleneck)
- 8x more textures fit in VRAM
- Near-lossless quality for photos

### 6.2 Virtual Texturing (Optional/Advanced)

**Current Infrastructure:** `VirtualTextureSystem` in `virtual_texturing.hpp`

**Activation Plan:**
1. Implement page-based texture cache (128x128 pages)
2. GPU feedback buffer for page requests
3. LRU eviction for cache management

**Benefits:**
- Theoretical support for unlimited textures
- Only load visible texture regions
- ~512 pages * 128x128 = 8 MB physical cache for entire dataset

**Complexity:** High - defer to Phase 7 if needed

---

## Phase 7: Advanced Techniques (Week 6+)

### 7.1 Hi-Z Occlusion Culling Integration

**Current:** `HiZPass` infrastructure exists (src/main.cpp:3034+)
- Hierarchical depth pyramid for occlusion queries

**Enhance:**
1. **Two-pass culling**
   - Pass 1: Frustum + size culling (compute)
   - Pass 2: Hi-Z occlusion test against depth pyramid (compute)

2. **Per-mip-level testing**
   - Test large objects against low-res mips (fast, conservative)
   - Test small objects against high-res mips (accurate)

**Performance:**
- Additional 30-40% cull rate in dense scenarios
- Prevent overdraw of occluded circles

### 7.2 Parallel Prefix Sum for Stream Compaction

**Current:** Culling outputs sparse indirect commands
- Wasted GPU threads on culled objects

**Optimization:**
1. **Implement GPU scan** (parallel prefix sum)
   - Compact visible instances into dense buffer
   - Reference: [NVIDIA GPU Gems 3, Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

2. **Stream compaction**
   ```glsl
   // Scan visibility flags to get write indices
   uint writeIndex = prefixSum[instanceID];
   if (visible) {
       compactedInstances[writeIndex] = instances[instanceID];
   }
   ```

**Benefits:**
- Perfect thread utilization (no wasted compute)
- Smaller indirect draw buffers
- Marginal but measurable perf gain (5-10%)

### 7.3 Async Compute for Culling

**Metal/MoltenVK:** Supports async compute queues
- Overlap culling compute with previous frame's rendering

**Implementation:**
1. Use separate compute queue for culling
2. Insert semaphore between culling and drawing
3. Pipeline: [Frame N render] || [Frame N+1 cull]

**Benefits:**
- Hide culling latency behind rendering
- Better GPU utilization

---

## Implementation Roadmap

### Week 1: Foundation
- [x] Research complete
- [ ] Compile GPU culling compute shader
- [ ] Initialize GPU-driven rendering buffers
- [ ] Integrate compute dispatch before draws
- [ ] Test with 10K circles

**Success Criteria:** 10K circles @ 60 FPS stable

### Week 2: LOD & Mipmaps
- [ ] Implement 4-tier LOD in compute shader
- [ ] Generate mipmap chains for texture atlas
- [ ] Update fragment shader for mip sampling
- [ ] Test LOD transitions (verify no popping)

**Success Criteria:** 50K circles @ 60 FPS, texture memory < 500 MB

### Week 3: MoltenVK Optimizations
- [ ] Enable unified memory optimizations
- [ ] Optimize descriptor set layout for Metal
- [ ] Profile with Metal frame capture
- [ ] Benchmark vs baseline

**Success Criteria:** 30% perf improvement on Apple Silicon

### Week 4: Clustering
- [ ] Implement density grid clustering
- [ ] Render impostor blobs for sub-pixel circles
- [ ] Hybrid rendering path
- [ ] Test with 1M circles

**Success Criteria:** 1M circles @ 60 FPS at far zoom

### Week 5: Compression
- [ ] Offline compress textures to ASTC 8x8
- [ ] Update loader for ASTC format
- [ ] Measure memory usage reduction
- [ ] Verify visual quality

**Success Criteria:** <200 MB texture memory, no visible quality loss

### Week 6: Polish & Optimization
- [ ] Hi-Z occlusion culling integration
- [ ] Async compute pipeline
- [ ] Performance profiling & tuning
- [ ] Stress testing at scale

**Success Criteria:** Stable 60 FPS from 10K to 1M circles

---

## Performance Projections

### Baseline (Current)
- **5.8K circles:** Laggy, ~20-30 FPS
- **Memory:** 1.5 GB textures
- **Bottleneck:** CPU culling, texture bandwidth

### After Phase 1-2 (GPU Culling + LOD)
- **10K circles:** 60 FPS stable ✓
- **100K circles:** 45 FPS (LOD helping)
- **Bottleneck:** Texture bandwidth for large circles

### After Phase 3 (Mipmaps + Streaming)
- **100K circles:** 60 FPS stable ✓
- **500K circles:** 50 FPS (mostly sub-pixel)
- **Memory:** 2 GB (with mipmaps, but better bandwidth)

### After Phase 4 (MoltenVK Optimizations)
- **100K circles:** 60 FPS + 30% GPU headroom
- **500K circles:** 60 FPS stable ✓

### After Phase 5 (Clustering)
- **1M circles:** 60 FPS @ far zoom ✓
- **1M circles:** 60 FPS @ medium zoom (most culled) ✓
- **Bottleneck:** Near zoom with 200K+ visible circles

### After Phase 6 (Compression)
- **1M circles:** 60 FPS all zoom levels ✓
- **Memory:** <300 MB total
- **Bandwidth:** 8x reduction = 480% perf headroom

---

## Technical Risks & Mitigations

### Risk 1: MoltenVK Indirect Draw Limitations
**Issue:** Metal lacks `drawCount` parameter ([Issue #168](https://github.com/KhronosGroup/MoltenVK/issues/168))

**Mitigation:**
- Use `vkCmdDrawIndexedIndirectCount()` (MoltenVK supports via workaround)
- Fallback: Fixed buffer with `instanceCount=0` for culled objects
- Alternative: Multi-draw with fixed batch sizes

### Risk 2: Descriptor Indexing Performance on Metal
**Issue:** Bindless textures may have overhead in MoltenVK translation

**Mitigation:**
- Profile descriptor binding cost with Metal frame capture
- Use argument buffers efficiently (batch updates)
- Consider hybrid: Atlas for common textures, bindless for rare

### Risk 3: Mipmap Generation Cost
**Issue:** Generating 9 mip levels on texture upload adds latency

**Mitigation:**
- Generate mipmaps asynchronously in transfer queue
- Precompute mipmaps offline and load compressed
- Show placeholder until mipmaps ready

### Risk 4: Clustering Visual Quality
**Issue:** Density blobs may look artificial at medium zoom

**Mitigation:**
- Use smooth LOD transitions (alpha blend between tiers)
- Tune clustering thresholds based on user testing
- Option to disable clustering for artistic control

---

## Monitoring & Metrics

### Key Performance Indicators (KPIs)
1. **Frame Time:** <16.67ms (60 FPS target)
2. **Culling Time:** <0.5ms for 1M circles
3. **Draw Call Count:** <5K per frame
4. **Texture Memory:** <300 MB with compression
5. **Bandwidth Usage:** <5 GB/s (measured via Metal profiler)

### Profiling Tools
- **Metal System Trace:** Frame timing, GPU utilization
- **Metal Frame Capture:** Shader performance, bandwidth analysis
- **RenderDoc:** Vulkan layer inspection (if needed)
- **Xcode Instruments:** CPU profiling

### Debug Visualizations
1. LOD tier heatmap (color code by tier)
2. Culling statistics overlay (visible/culled ratio)
3. Memory usage graph (real-time VRAM)
4. Draw call batching viewer (# draws per frame)

---

## Conclusion

This plan provides a systematic approach to scale from 5.8K laggy circles to 1M smooth circles on MoltenVK/Mac. The core strategy is:

1. **Move work to GPU** (culling, LOD selection)
2. **Aggressive LOD** (4 tiers based on pixel size)
3. **Smart texture streaming** (mipmaps, priority-based)
4. **MoltenVK-aware optimizations** (unified memory, argument buffers)
5. **Clustering for extreme scale** (density fields for sub-pixel objects)
6. **Compression** (ASTC 8x8 for 87% memory savings)

**Estimated Timeline:** 6 weeks for full implementation
**Risk Level:** Medium (MoltenVK quirks, but mitigations exist)
**Performance Target:** 1M circles @ 60 FPS ✓

The existing infrastructure (`gpu_driven_rendering.hpp`, `bindless_textures.hpp`, `virtual_texturing.hpp`) provides a strong foundation - main work is activation and integration.
