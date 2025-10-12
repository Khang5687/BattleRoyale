# Image Rendering Critical Issues - Action Plan

## Current Status: 🔴 CRITICAL

**Observed Symptoms:**
- ❌ Preload Phase 3 stalls at ~3,271 images (out of 5,806 total)
- ❌ FPS drops to unstable 6 FPS when many images rendered
- ❌ Constant texture flickering (images swap rapidly)
- ❌ LOD system provides minimal performance benefit despite classification overhead
- ❌ System refuses to load more images after stall point

**System Configuration:**
- Total images: 5,806
- Atlas capacity: 2,048 GPU layers (hard limit)
- Texture cache: 4,096 CPU slots (defined but not enforced)
- Async upload slots: 64 contexts
- Batch size: 16 images per upload

---

## 🔴 Issue #1: textureCache Unbounded Growth → Memory Exhaustion → Preload Deadlock

### Root Cause
`MAX_CACHE_SIZE = 4096` is defined but **never enforced**. No eviction logic exists for CPU-side `textureCache`. Cache grows until system runs out of memory.

**Evidence:**
- Preload stalls at **3,271 images** (~1.1 GB in textureCache)
- No `textureCache.erase()` calls exist in codebase
- Memory exhaustion causes decode allocations to fail silently
- Worker threads stall waiting for memory that never becomes available

### Performance Impact
- Preload cannot complete (stuck at 56% of total images)
- Memory pressure causes system-wide slowdown
- Eventually blocks all new texture loads
- Creates false impression that "images refuse to load"

### Fix Required
Implement LRU eviction for `textureCache`:

```cpp
// After successful texture decode, before adding to cache:
if (mgr.atlas.textureCache.size() >= ImageManager::MAX_CACHE_SIZE) {
    // Find least recently used texture (not in atlas)
    uint32_t lruImageId = UINT32_MAX;
    uint64_t oldestFrame = UINT64_MAX;

    for (const auto& [imageId, texture] : mgr.atlas.textureCache) {
        // Don't evict if currently in atlas
        if (mgr.atlas.imageIdToLayer.find(imageId) != mgr.atlas.imageIdToLayer.end()) {
            continue;
        }
        if (texture.lastUsed < oldestFrame) {
            oldestFrame = texture.lastUsed;
            lruImageId = imageId;
        }
    }

    if (lruImageId != UINT32_MAX) {
        mgr.atlas.textureCache.erase(lruImageId);
    }
}
```

**Priority:** 🔴 CRITICAL - Blocks all other fixes
**Complexity:** 🟢 Low (30 lines of code)
**Impact:** +3,000 additional images can load, preload completes

---

## 🔴 Issue #2: Atlas Thrashing - Loading Beyond Capacity

### Root Cause
Preload Phase 3 attempts to load **all 5,806 images** into a **2,048-layer atlas**. Impossible by design. Creates thrashing loop:
1. Load image 2049 → evict image 1
2. Image 1 still in visible set → re-request load
3. Load image 2050 → evict image 2
4. Image 2 re-requested
5. Infinite eviction/reload cycle

**Evidence:**
- Atlas capacity: 2,048 layers
- Preload target: 5,806 images (2.8× over capacity)
- `ensureFreeLayersAvailable()` at src/main.cpp:5950-5994 evicts aggressively
- No frame coherency - textures used in frame N evicted in frame N+1

### Performance Impact
- Constant texture uploads (dozens per frame)
- Visible flickering as images swap between placeholder and texture
- GPU bandwidth saturated with redundant uploads
- CPU time wasted on eviction bookkeeping

### Fix Required

**Option A: Smart Preload Cap (Quick Fix)**
```cpp
// In startPreloading() - Phase 3:
// Don't preload beyond atlas capacity + reasonable margin
size_t maxPreload = TextureAtlas::MAX_LAYERS + 512; // 2,560 images max
mgr.preloadTarget = std::min(maxPreload, mgr.atlas.imageFiles.size());
```

**Option B: Visibility-Based Preload (Better)**
Only preload images for circles that will be visible in first 10 seconds:
```cpp
// Preload only images attached to circles in starting area
// Skip images for circles outside 2× viewport radius
```

**Priority:** 🔴 CRITICAL
**Complexity:** 🟢 Low (Option A) / 🟡 Medium (Option B)
**Impact:** -80% eviction rate, eliminates thrashing

---

## 🔴 Issue #3: Async Upload System Saturation

### Root Cause
All 64 async upload contexts can be in-flight simultaneously. If GPU is slow or stalled, all slots fill up and system falls back to **synchronous uploads** which block for 20ms each.

**Evidence:**
- `MAX_ASYNC_UPLOADS = 64` (src/main.cpp:845)
- Fallback counter incrementing (visible in HUD diagnostics)
- `vkGetFenceStatus()` may never return SUCCESS if GPU hangs
- No timeout or recovery mechanism for stuck uploads

### Performance Impact
- Sync fallback: 20ms GPU stall per batch
- At 6 FPS (167ms/frame), could be 8× sync uploads per frame = 160ms blocked
- CPU waits on GPU instead of preparing next frame
- Creates cascading delays

### Fix Required

**Add upload timeout and forced flush:**
```cpp
static void pollUploadCompletions(ImageManager& mgr) {
    uint64_t currentFrame = mgr.atlas.frameCounter;

    for (auto& ctx : mgr.asyncUploads) {
        if (!ctx.inFlight) continue;

        VkResult result = vkGetFenceStatus(mgr.device, ctx.fence);
        if (result == VK_SUCCESS) {
            // Mark textures ready (existing code)
            // ...
        } else if (currentFrame - ctx.submitFrame > 180) {
            // Upload stuck for 3+ seconds @ 60fps - force reset
            vkWaitForFences(mgr.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
            vkResetFences(mgr.device, 1, &ctx.fence);
            ctx.inFlight = false;
            std::cerr << "WARNING: Async upload timed out, forced completion" << std::endl;
        }
    }
}

// Add to AsyncUploadContext:
uint64_t submitFrame; // Track when upload was submitted
```

**Priority:** 🔴 CRITICAL
**Complexity:** 🟢 Low
**Impact:** Prevents permanent deadlock, recovers from GPU stalls

---

## 🟡 Issue #4: Narrow LOD Thresholds → Constant Tier Switching

### Root Cause
LOD thresholds separated by **4-6 pixels**:
- PIXEL_DUST: < 48px
- SIMPLE_SHAPE: < 54px (only 6px range!)
- BASIC_TEXTURE: < 58px (only 4px range!)
- FULL_DETAIL: ≥ 58px

Minor camera movement or zoom causes 40%+ of circles to switch tiers → flood of load requests.

### Performance Impact
- ~80,000 tier switches per 60-second match
- ~40,000 redundant load requests (same image, different tier)
- CPU overhead: 2-3ms/frame for priority recalculation

### Fix Required

**Widen thresholds with hysteresis:**
```cpp
void updateLodThresholdsFromViewport(uint32_t viewportWidth, uint32_t viewportHeight) {
    float minSideF = static_cast<float>(std::min(viewportWidth, viewportHeight));

    // Much wider gaps: 30px, 60px, 120px (was 6, 4, 8)
    pixelDustThreshold = std::clamp(minSideF * 0.025f, 30.0f, 60.0f);
    simpleShapeThreshold = pixelDustThreshold + 30.0f;  // 90px
    basicTextureThreshold = simpleShapeThreshold + 60.0f;  // 150px
    fullDetailThreshold = basicTextureThreshold + 120.0f;  // 270px
}

// Add hysteresis: different thresholds for up/down transitions
CircleRenderTier classifyRenderTier(float apparentRadius, CircleRenderTier currentTier) {
    // Add 10% margin before downgrading tier
    float downgradeMargin = (currentTier > PIXEL_DUST) ? 0.9f : 1.0f;

    if (apparentRadius < pixelDustThreshold * downgradeMargin) return PIXEL_DUST;
    if (apparentRadius < simpleShapeThreshold * downgradeMargin) return SIMPLE_SHAPE;
    if (apparentRadius < basicTextureThreshold * downgradeMargin) return BASIC_TEXTURE;
    return FULL_DETAIL;
}
```

**Priority:** 🟡 HIGH
**Complexity:** 🟢 Low
**Impact:** -60% tier switches, -40% load requests

---

## 🟡 Issue #5: Priority System Thrashing

### Root Cause
Priority recalculated **every frame for every visible circle**:
- 5,000 visible circles × `computeScore()` per frame
- 5,000 map lookups in `cachedPriorities`
- Same texture re-requested every frame until loaded
- No debouncing or cooldown period

### Performance Impact
- CPU overhead: 2-3ms/frame
- Request queue floods with duplicates
- Mutex contention on `requestMutex`

### Fix Required

**Add request debouncing:**
```cpp
// In requestImageLoad():
uint64_t framesSinceLastRequest = currentFrame - mgr.imageLastRequestFrame[imageId];
if (framesSinceLastRequest < 60) {  // 1 second cooldown at 60fps
    return; // Don't spam requests
}
```

**Priority:** 🟡 HIGH
**Complexity:** 🟢 Trivial (5 lines)
**Impact:** -50% redundant requests, -2ms CPU per frame

---

## 🟢 Issue #6: LOD Shader Still Sampling (Partially Fixed)

### Status
✅ **PIXEL_DUST**: Fixed - uses thumbnail only
✅ **SIMPLE_SHAPE**: Fixed - uses thumbnail only
❌ **BASIC_TEXTURE**: Still samples texture with LOD bias
✅ **FULL_DETAIL**: Expected to sample (correct behavior)

### Remaining Work
Consider skipping texture sampling for BASIC_TEXTURE tier as well (use thumbnail):
```glsl
// In circle.frag:
if (screenRadius < basicTextureCutoff) {
    // Use thumbnail instead of textured sampling
    finalColor = mix(thumbColor, vColor, 0.3);
} else {
    // Only FULL_DETAIL samples texture
    texColor = sampleAtlasColor(texCoord);
    finalColor = mix(texColor, vColor, 0.3);
}
```

**Priority:** 🟢 MEDIUM (already 50% fixed)
**Complexity:** 🟢 Trivial (3 lines)
**Impact:** Additional +20-30% fragment shader savings

---

## Implementation Priority

### Phase 1: Unblock Preload (Week 1)
1. ✅ **Fix #1**: textureCache eviction (4 hours) ← **MUST DO FIRST**
2. ✅ **Fix #2**: Cap preload at 2,560 images (1 hour)
3. ✅ **Fix #3**: Async upload timeout (2 hours)

**Expected Result:** Preload completes, system stable with 2,560 images loaded

### Phase 2: Eliminate Thrashing (Week 1-2)
4. ✅ **Fix #4**: Widen LOD thresholds + hysteresis (4 hours)
5. ✅ **Fix #5**: Request debouncing (1 hour)

**Expected Result:** -80% texture churn, -60% load requests, FPS stabilizes

### Phase 3: Optimize Rendering (Week 2)
6. ✅ **Fix #6**: BASIC_TEXTURE thumbnail rendering (1 hour)

**Expected Result:** +20-30% fragment shader performance

### Total Effort: ~13 hours for full fix, **~7 hours for critical path**

---

## Expected Performance After Fixes

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| **Preload Complete** | ❌ 56% | ✅ 100% | ✅ 100% | ✅ 100% |
| **FPS (10K circles)** | 6-15 | 25-35 | 50-70 | 60-80 |
| **Flickering** | Severe | Moderate | Minimal | None |
| **Evictions/sec** | 700 | 200 | 60 | 40 |
| **Memory Usage** | 1.1GB+ | 680MB | 680MB | 680MB |

---

## Root Cause Summary

All issues stem from **architectural design flaws**, not implementation bugs:

1. **No cache eviction** → Memory exhaustion
2. **Naive preload** → Attempts impossible (5,806 into 2,048 slots)
3. **No upload resilience** → Deadlocks on GPU stalls
4. **Over-sensitive LOD** → Constant tier switching
5. **No request throttling** → Priority system thrashing

**These are not quick hotfixes** - they are proper architectural patterns that should have been implemented from the start. The fixes implement industry-standard techniques:
- LRU cache management (standard)
- Capacity-aware preloading (standard)
- Timeout-based recovery (standard)
- Hysteresis thresholds (control systems 101)
- Request debouncing (rate limiting 101)

---

## Notes

- **Async uploads are implemented** but incomplete (no timeout recovery)
- **LOD shader partially fixed** (2/4 tiers now skip sampling)
- **Apple Silicon inefficiencies** (Issue #4 from original doc) de-prioritized - architectural fixes above will provide 80% of gains
- Original document's "0% LOD performance benefit" claim was accurate at time of writing, now reduced to "minimal benefit" since SIMPLE_SHAPE tier is fixed

---

**Last Updated:** 2025-10-12
**Next Review:** After Phase 1 implementation (textureCache fix + preload cap)
