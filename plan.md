# BattleRoyale Circles - Implementation Plan (Vulkan C++ on macOS / MoltenVK)

## Goals
- Vulkan + GLFW app on macOS (MoltenVK) rendering a simulation of image-backed circles (“players”).
- Bouncing physics with circle-circle and circle-wall collisions; health bars; eliminations.
- Winner sequence: last circle centers, scales up, shows filename as name.
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

## Massive Scale Strategy (>500k files)
- Do not load all images or create 500k Vulkan image views. Maintain metadata list of file paths and a small LRU cache of decoded + GPU-resident textures (e.g., few thousands max).
- Use a texture atlas array (array of 2D layers) of fixed layer count (e.g., 2048) for visible, above-threshold players. Evict via LRU when layers are scarce. Index via `imageId -> layerIndex | placeholder`.
- If descriptor indexing (bindless) is absent or limited on MoltenVK, prefer a small set of per-frame descriptor sets for the atlas array; use an indirection buffer mapping instance to atlas layer.
- Lazy tiers:
  - Tier 0 (fake): no file I/O; rendered as flat color in compute/fragment; guaranteed to die before threshold.
  - Tier 1 (placeholder): tiny 1x1 or 4x4 texture (low memory) for small-but-visible sprites.
  - Tier 2 (real): high-quality mipmapped texture uploaded on demand when radius > IMAGE_LOAD_THRESHOLD.
- O(1) algorithmic constraint for image access: each entity holds stable index to metadata; atlas layer lookup is O(1) via a dense vector; LRU updates are O(1) amortized with linked-list + hashmap.

## Rendering Approach
- Instanced rendering of quads (two triangles) for circles; circle appearance via fragment shader signed distance function to avoid per-vertex circles. Image sampled using per-instance atlas layer; health bar drawn as a second instanced pass or in same shader via a screen-space overlay.
- One command buffer per frame, one draw for circles (instanced), one draw for health bars, one draw for text.
- Push constants or storage buffer for per-frame params (time, viewport, scaling factor, thresholds, counts).

## Scaling Visual Size Without Resizing Massive Counts
- Global scale factor S derived from activeAliveCount; applied in vertex shader to instance radius. This avoids touching per-entity data for millions of circles. Clamp by MAX_CIRCLE_SIZE.
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
- Enumerate image filenames from `assets/` once; store stems as player names; maintain mapping for bias multipliers (config file `bias.json` or hardcoded map, modifiable at runtime).
- MAX_PLAYER: if actual image files < MAX_PLAYER, spawn fake circles to reach MAX_PLAYER; fake circles have Tier 0 textures/colors.

## Constants (tunable)
- MAX_PLAYER
- IMAGE_LOAD_THRESHOLD_RADIUS
- MAX_CIRCLE_SIZE
- WINNER_SCALE
- DAMAGE_MULTIPLIER, WALL_DAMPING, COLLISION_DAMPING
- GRID_CELL_SIZE

## Robustness and Edge Cases
- Validation layers in debug; check all Vk results; RAII wrappers to prevent leaks.
- Guard zero devices/queues/swapchain formats; handle window resize.
- Clamp velocities/radii; avoid NaNs; ensure no division by zero in collision math.
- Thread safety: image loader thread communicates via lock-free MPMC queue or mutex-protected queue; main thread consumes at frame boundary.
- Prevent thundering herd on texture loads: coalesce requests; deduplicate per imageId; cancel when entity dies.

## Text Rendering (Players left)
- Simple path: bake minimal bitmap font (e.g., stb_easy_font) to a dynamic vertex buffer, or integrate Dear ImGui for HUD (no heavy styling). Keep draw order last.

## Milestones
1) Minimal Vulkan window (this repo). Verify presentation.
2) Create swapchain, render pass, frame sync, render clear color.
3) Instanced quad pipeline, draw N flat-color circles with SDF in shader.
4) Spatial grid + collisions + health; eliminations.
5) Texture atlas/array, placeholder + real mipmapped textures, lazy loading.
6) HUD text, players left counter, winner flow.
7) Bias configuration and tuning.
8) Stress test & profiling; refine grid and resource limits.

## Notes on MoltenVK capabilities
- Descriptor indexing/bindless is limited; plan on atlas array + indirection buffer.
- Avoid geometry shaders; use instancing + SDF.
- Prefer fewer pipelines; Metal backend likes stable pipelines and small descriptor sets.

