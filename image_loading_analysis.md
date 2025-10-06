# Image Loading & Rendering Pipeline

## 1. System Overview
- `Simulation::initializeFromAssets` enumerates on-disk avatars and seeds the `ImageManager` with filenames so both the loader threads and render loop operate on a shared ID space.
- `ImageManager` owns the texture atlas (`256×256×2048` array with 9 mip levels), a bindless descriptor scaffold, optional GPU texture streaming, and CPU staging resources.
- A lock-protected priority queue (`requestQueue`) drives a pool of stb-based decoder threads. Decoded RGBA8 data is resized, mip-mapped, and staged for GPU upload.
- The render loop calls `updateImageManager`/`finalizeImageManagerUpdate` each frame to flush CPU work, transition atlas layers, and launch the optional compute shader streamer before drawing with `shaders/circle.frag`.

## 2. Asset Enumeration & Simulation Coupling
Real image files are discovered when the simulation initializes; every player index keeps a stable `imageId`, and the `ImageManager` caches the corresponding filename list and lookup buffers.

```cpp
// src/main.cpp:907-923
if (imageManager) {
    imageManager->atlas.imageFiles = files;
    ensureAtlasLookupBuffer(*imageManager, imageManager->atlas.imageFiles.size());
    prepareImageManagerForImageCount(*imageManager, imageManager->atlas.imageFiles.size());
    requestPriorityRefresh(*imageManager);
}
```

`Simulation::updateImageTiers()` marks which circles are allowed to sample textures; tiers feed into the loader’s priority system and determine whether the renderer falls back to flat colors.

## 3. ImageManager Construction & Shared Resources
`initImageManager` wires every subsystem that image loading relies on: the atlas image, lookup buffers, bindless descriptor set, optional GPU streaming, and CPU staging memory.

```cpp
// src/main.cpp:5856-5908
createImage(physicalDevice, device,
    TextureAtlas::ATLAS_SIZE, TextureAtlas::ATLAS_SIZE, TextureAtlas::MAX_LAYERS,
    VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mgr.atlas.atlasArray, 9);

mgr.atlas.atlasArray.view = createImageView2DArray(device, mgr.atlas.atlasArray.image,
    VK_FORMAT_R8G8B8A8_UNORM, TextureAtlas::MAX_LAYERS, 9);
mgr.atlas.sampler = createTextureSampler(device);

if (!initBindlessTextures(mgr.bindless, physicalDevice, device, descriptorPool)) {
    std::cout << "[BINDLESS] Bindless textures not supported, falling back to texture atlas" << std::endl;
}
```

The descriptor layout created for the circle pipeline exposes:
- binding 0 – combined sampler for the atlas (active in fragment shader),
- binding 1 – storage buffer `imageLayerLookupBuffer` for ID → layer indirection,
- binding 2 – partially-bound sampled image array intended for future bindless support (`createCirclePipeline`, `src/main.cpp:3479-3506`).

If GPU streaming is enabled, `initImageManager` builds persistent SSBOs for requests/pixels, maps them, and creates the compute pipeline that feeds directly into the atlas (`src/main.cpp:5930-6078`).

CPU uploads use a 256 MB persistently-mapped staging buffer (`mappedStagingMemory`) plus a `BatchedUpload` list sized for 128 images per flush.

## 4. Load Prioritisation & Request Queue
Visibility-driven metrics decide which images should be decoded next. `LoadPriority::computeScore` blends distance, entity radius, recency, and starvation prevention.

```cpp
// src/main.cpp:488-521
float LoadPriority::computeScore() const {
    const float PROXIMITY_CULLING_RADIUS = 2000.0f;
    if (distanceToPlayer > PROXIMITY_CULLING_RADIUS) return 0.0f;
    const float MIN_VISIBLE_RADIUS = 1.5f;
    if (circleRadius < MIN_VISIBLE_RADIUS) return 0.0f;
    float distanceScore = 2000.0f / (1.0f + distanceToPlayer * distanceToPlayer * 0.01f);
    float radiusScore = std::min(circleRadius * 15.0f, 300.0f);
    float recencyScore = lastAccessFrame > 0 ? 800.0f : 0.0f;
    float agingScore = (currentFrame > lastRequestFrame)
        ? std::min(static_cast<float>(currentFrame - lastRequestFrame) * 8.0f, 400.0f)
        : 0.0f;
    float visibilityBonus = (distanceToPlayer < 50.0f && circleRadius > 5.0f) ? 1000.0f : 0.0f;
    return distanceScore + radiusScore + recencyScore + agingScore + visibilityBonus;
}
```

`updateLoaderPriorityCache` ( `src/main.cpp:2524-2568` ) refreshes cached scores for all live entities every frame, blends in adaptive LOD tiers, and tells the request queue to rebuild when priorities shift. The request API enforces single in-flight work per image:

```cpp
// src/main.cpp:6201-6231
std::lock_guard<std::mutex> lock(mgr.requestMutex);
if (mgr.atlas.textureCache.find(imageId) != mgr.atlas.textureCache.end()) return;
if (mgr.decodeInFlight.find(imageId) != mgr.decodeInFlight.end()) {
    mgr.imageLastRequestFrame[imageId] = priority.currentFrame;
    return;
}
uint64_t seq = mgr.requestSequence.fetch_add(1) + 1;
PendingInfo info{};
info.metrics = priority;
info.metrics.lastRequestFrame = priority.currentFrame;
info.score = priority.computeScore();
info.sequence = seq;
mgr.pendingInfos[imageId] = info;
mgr.requestQueue.push(PendingQueueEntry{imageId, info.score, info.metrics, info.sequence});
```

Decoder threads block on `requestCv` and pop the highest-score `PendingQueueEntry` via `dequeueNextImageRequest` (`src/main.cpp:5751-5781`).

## 5. CPU Decode & Mipmap Generation
Each decoder thread loops until `stopLoading` is set, using stb to load, resize, and mipmap a single image before handing it to the upload stage.

```cpp
// src/main.cpp:6139-6189
unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);
if (data) {
    LoadedTexture tex;
    const uint8_t* resizedData = nullptr;
    if (width != TextureAtlas::ATLAS_SIZE || height != TextureAtlas::ATLAS_SIZE) {
        unsigned char* result = stbir_resize_uint8_linear(
            data, width, height, 0,
            buffer.data.data(), TextureAtlas::ATLAS_SIZE, TextureAtlas::ATLAS_SIZE, 0,
            STBIR_RGBA);
        if (result) {
            resizedData = buffer.data.data();
        }
    } else {
        resizedData = data;
    }

    if (resizedData) {
        TextureWithMipmaps mipmapped = generateMipmaps(resizedData,
            TextureAtlas::ATLAS_SIZE, TextureAtlas::ATLAS_SIZE, 4);
        tex.width = mipmapped.width;
        tex.height = mipmapped.height;
        tex.mipLevels = mipmapped.mipLevels;
        tex.data = std::move(mipmapped.data);
        tex.mipOffsets = std::move(mipmapped.mipOffsets);
        tex.mipDimensions = std::move(mipmapped.mipDimensions);
        tex.refCount = 1;
        tex.lastUsed = mgr.atlas.frameCounter;
        {
            std::lock_guard<std::mutex> lock(mgr.uploadMutex);
            mgr.pendingUploads.emplace(imageId, std::move(tex));
        }
    }
    stbi_image_free(data);
}
```

`generateMipmaps` ( `src/main.cpp:2885-2951` ) computes all nine mip levels on the CPU using `stbir_resize_uint8_*`, storing byte offsets for each level so GPU uploads can address them contiguously.

## 6. GPU Upload Paths
When the render loop calls `updateImageManager`, CPU-decoded textures transfer from `pendingUploads` into `atlas.textureCache`. Actual GPU uploads happen on demand inside `getAtlasLayerForImage`.

### 6.1 Layer Assignment & LRU
Layers are assigned lazily, evicting via LRU when the atlas is full while honouring GPU-streaming in-flight layers.

```cpp
// src/main.cpp:7059-7134
auto it = mgr.atlas.imageIdToLayer.find(imageId);
if (it != mgr.atlas.imageIdToLayer.end()) { /* refresh LRU & return */ }

if (mgr.atlas.textureCache.find(imageId) == mgr.atlas.textureCache.end()) {
    float score = priority.computeScore();
    if (score > 0.0f && mgr.frameLoadCount < ImageManager::MAX_LOADS_PER_FRAME) {
        requestImageLoad(mgr, imageId, priority);
        mgr.frameLoadCount++;
    }
    return -1;
}

uint32_t layer = !mgr.atlas.freeLayers.empty()
    ? mgr.atlas.freeLayers.front()
    : /* pop LRU via ensureAtlasLayerCapacity */;
mgr.atlas.imageIdToLayer[imageId] = layer;
mgr.atlas.layerToImageId[layer] = imageId;
mgr.atlas.layerLayouts[layer] = VK_IMAGE_LAYOUT_UNDEFINED;
if (imageId < mgr.atlas.imageLayerLookupCPU.size()) {
    mgr.atlas.imageLayerLookupCPU[imageId] = static_cast<int32_t>(layer);
    mgr.atlas.lookupDirty = true;
}
uploadTextureToAtlasLayer(mgr, imageId, layer);
```

`ensureAtlasLayerCapacity` (`src/main.cpp:5637-5668`) maintains the free-list while skipping layers that are currently being written by the GPU streamer.

### 6.2 Upload Execution
`uploadTextureToAtlasLayer` chooses between three strategies: GPU compute streamer, batched buffer copy, or a one-off staging buffer fallback. All paths transition every mip level.

```cpp
// src/main.cpp:6830-6886
if (queueGpuStreamUpload(mgr, imageId, layer, texture)) {
    return; // compute shader handles it
}
if (addTextureToBatch(mgr, imageId, layer, texture)) {
    if (mgr.currentBatch.size() >= ImageManager::BATCH_SIZE) {
        flushBatchedUploads(mgr);
    }
    return;
}
if (!mgr.currentBatch.empty()) {
    flushBatchedUploads(mgr);
    if (addTextureToBatch(mgr, imageId, layer, texture)) return;
}
// Fallback: dedicated staging buffer + vkCmdCopyBufferToImage for all mip levels
```

`flushBatchedUploads` ( `src/main.cpp:6688-6830` ) builds vectors of `VkImageMemoryBarrier` objects and `VkBufferImageCopy` regions (one per mip level) before issuing a single `vkCmdCopyBufferToImage`. Layout state per layer is tracked in `atlas.layerLayouts` to avoid redundant transitions.

### 6.3 GPU Texture Streaming
If `gpuStream.enabled`, textures can bypass CPU staging by copying pixels directly into a mapped SSBO and executing `recordGpuStreamUploads` each frame.

```cpp
// shaders/texture_stream.comp
layout(set = 0, binding = 0, std430) buffer RequestBuffer { StreamRequest requests[]; };
layout(set = 0, binding = 1, std430) readonly buffer PixelBuffer { uint pixelData[]; };
layout(set = 0, binding = 2, rgba8) uniform writeonly image2DArray atlas;
...
if (req.state != REQUEST_STATE_READY) return;
for (uint y = gl_LocalInvocationID.y; y < height; y += gl_WorkGroupSize.y) {
    for (uint x = gl_LocalInvocationID.x; x < width; x += gl_WorkGroupSize.x) {
        uint index = req.pixelOffset + y * width + x;
        imageStore(atlas, ivec3(int(x), int(y), int(req.layer)), unpackColor(pixelData[index]));
    }
}
if (gl_LocalInvocationIndex == 0u) {
    memoryBarrierBuffer();
    requests[slot].state = REQUEST_STATE_COMPLETE;
}
```

`recordGpuStreamUploads` ( `src/main.cpp:6897-7040` ) inserts the necessary SSBO and image barriers, dispatches the compute shader (`GpuStreamContext::SLOT_COUNT` groups), and restores atlas layers to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.

## 7. Frame Update Hooks & Descriptor Maintenance
Every frame the renderer:
1. Calls `updateImageManager` to poll GPU-stream completions, move textures from `pendingUploads` into the RAM cache, and refresh loader metrics.
2. Invokes `finalizeImageManagerUpdate` to flush any partial CPU batches and rewrite the image ID → layer lookup buffer if `lookupDirty` was set.
3. Runs `recordGpuStreamUploads(cmd, imageManager)` immediately before GPU culling or draw call setup (`src/main.cpp:8248`).

Atlas descriptors are created alongside the circle pipeline and updated once (`src/main.cpp:7894-7908`). If the lookup buffer is resized, `updateAtlasLookupDescriptor` rebinds it (`src/main.cpp:5803-5817`).

## 8. Rendering Consumption
Instance data encodes whether a circle should sample the atlas or stay flat-coloured. The renderer sets the high bit of `textureIndex` when an atlas layer is available.

```cpp
// src/main.cpp:1615-1649
uint32_t bindlessIndex = getBindlessIndex(imageManager->bindless, imageId[i]);
if (bindlessIndex != BindlessTextureSystem::INVALID_TEXTURE_INDEX) {
    inst.textureIndex = bindlessIndex;
    inst.color[...] = 1.0f - h * 0.5f;
} else {
    int32_t layer = getAtlasLayerForImage(*imageManager, imageId[i]);
    if (layer >= 0) {
        inst.textureIndex = static_cast<uint32_t>(layer) | 0x80000000;
        inst.color[...] = 1.0f - h * 0.5f;
    } else {
        inst.textureIndex = BindlessTextureSystem::INVALID_TEXTURE_INDEX;
        inst.color[...] = placeholderTint;
    }
}
```

At draw time `shaders/circle.frag` uses the atlas (binding 0). If `textureIndex` is not `0xFFFFFFFF`, it strips the high bit to recover the array layer and mixes the sampled color with the health tint.

```glsl
// shaders/circle.frag:18-33
if (vTextureIndex != 0xFFFFFFFFu) {
    uint atlasLayer = vTextureIndex & 0x7FFFFFFFu;
    vec3 texCoord = vec3(vTexCoord, float(atlasLayer));
    vec4 texColor = texture(uTextureAtlas, texCoord);
    finalColor = mix(texColor, vColor, 0.3);
} else {
    finalColor = vColor;
}
```

Pipeline creation currently binds the atlas sampler plus lookup buffer; the bindless and virtual texture bindings are enabled at the descriptor level but unused by `circle.frag` (future work tracked by `circle_optimized.frag`).

## 9. Metrics, Budgeting & Diagnostics
- VRAM budgeting converts reported device-local memory into a per-layer limit (`VRAMBudget::calculateMaxLayers`, `src/main.cpp:476-506`).
- Loader throughput is tracked via atomics (`metricsImagesPerSecond`, `metricsAverageScore`, etc.) updated once per second in `updateLoaderMetricsWindow`.
- `ImageManager::MAX_LOADS_PER_FRAME` throttles request issuance to avoid per-frame hitches.
- Logs indicate when GPU streaming is enabled, when batches run slowly, and when VRAM queries fall back to defaults.

## 10. Interplay Summary & Current Gaps
- Simulation determines which `imageId` each agent uses and signals when high-detail textures are required; the loader honours those priorities and exposes ready data through `textureCache`.
- CPU decoding and GPU uploads are decoupled via `pendingUploads`, enabling background work without stalling the main thread.
- Atlas layer state (`layerLayouts`) and lookup buffers ensure the renderer never samples from an invalid layout.
- GPU streaming and CPU batching complement each other: streaming handles steady-state updates without CPU copies, while batching amortises large uploads during preloading or on systems without descriptor indexing support.
- Bindless and virtual-texturing infrastructure exists (descriptor slots, `BindlessTextureSystem`, `VirtualTextureSystem`) but shader consumption is still atlas-based; any optimisation plan must account for the gap between scaffolding and active usage.

This documentation reflects the exact behaviour in the current tree; any optimisation or redesign should maintain the contracts outlined above unless the corresponding shader, simulation, and upload code are updated in tandem.
