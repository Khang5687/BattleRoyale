# Rendering Optimization Plan (Mac M1 Pro)

## Objectives
- Prioritize smooth real-time rendering for 1K+ textured circles on Apple M1 Pro (UMA + tile-based GPU).
- Defer or degrade texture sampling when screen-space coverage is sub-pixel while preserving gameplay readability.
- Reduce bandwidth, state churn, and latency in the Vulkan/MoltenVK pipeline by aligning with Apple TBDR guidelines.

- Phase 0 summaries show every circle stays in BASIC/FULL detail with atlas sampling; no pixel clusters appear and apparent radii >10px almost immediately.
- `BR5_PHASE0_TELEMETRY` logging confirms the cheap tiers never trigger—Phase 1 must enforce screen-space LOD in the shader.

## Phase 0 – Profiling & Ground Truth
- [ ] Capture Metal System Trace and GPU counter snapshots (via Xcode + MoltenVK) while spawning 1K–5K circles to isolate CPU versus GPU stalls.
- [ ] Record per-frame CPU/GPU budgets (target ≤16 ms each @60 fps) and memory traffic to quantify texture sampling cost (`image_loading_analysis.md`).
- [x] Instrument existing LOD tier logic to log screen-space radius, atlas layer hit rate, and shader path usage for validation before changes (see Phase0 telemetry output in `phase0.log`).

## Phase 1 – Screen-Space LOD & Lazy Rendering
- [x] Extend `InstanceLayoutCPU` with `float screenPixelRadius` (computed from world radius * viewport scale) and store in staging uploads; telemetry now tracks dust/mip/full counts.
- [x] Update vertex shader to pass screen radius so fragment code can infer pixel coverage without extra uniforms.
- [x] Author a new fragment path that enforces tiered behaviour:
  - Tier 0 (`screenPixelRadius < 0.75`): skip texture fetch, output cached flat color.
  - Tier 1 (`0.75 ≤ radius < 2.0`): sample `textureLod` with explicit `lod = max(log2(radius / 0.75), minMip)` to pull from higher mip levels only.
  - Tier 2 (`radius ≥ 2.0`): existing trilinear sampling with health tint.
- [ ] Validate derivatives by enabling `VK_EXT_shader_demote_to_helper_invocation` fallback to avoid dFdx divergence on clipped quads.

## Phase 2 – Texture Data Simplification
- [ ] Precompute and cache “thumbnail” colors (e.g., 4×4 RGBA and average color) when generating mip chains; store alongside atlas metadata to feed Tier 0 shading.
- [ ] Add CPU-side hot path to reuse cached thumbnails for circles flagged as pixel dust, bypassing atlas lookups entirely.
- [ ] Investigate MoltenVK support for `VK_EXT_fragment_density_map`/`VK_KHR_fragment_shading_rate`; if present, prototype reducing shading rate for sub-4px circles.

## Phase 3 – GPU Pipeline Alignment with Apple TBDR
- [ ] Batch circle draws by LOD tier to minimize divergent branches and allow smaller descriptor updates (ref: tile-based best practices [[Apple TBDR](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering)]).
- [ ] Explore replacing `circle.frag` with the existing bindless-ready `circle_optimized.frag` and trim to required features; evaluate tile shader opportunities to keep accumulators in local memory.
- [ ] Audit render pass transitions to ensure atlas writes remain within a single render pass where possible (leveraging fast tile memory on M1).

## Phase 4 – Memory & Streaming Refinements
- [ ] Confirm persistent staging buffer resides in shared memory (`HOST_VISIBLE | HOST_COHERENT` already UMA-optimal) and eliminate redundant CPU copies during atlas updates.
- [ ] Gate background decode threads based on active Tier 2 circles to avoid uploading textures that never leave thumbnail state.
- [ ] Tune batch size and transfer queue usage to avoid monopolizing the unified memory bus during active rendering.

## Phase 5 – Verification & Iteration
- [ ] Re-run Metal traces after each phase; compare frame-time histograms and texture bandwidth against Phase 0 baseline.
- [ ] Build automated stress scene (5K circles, varying radii) to regression-test LOD thresholds and shader branching cost.
- [ ] Document findings and feed back into `image_loading_analysis.md` for future optimization rounds.
