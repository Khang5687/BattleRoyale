# Battle Royale Circle Simulation - Implementation Plan

## Overview
A Vulkan-based C++ application simulating a battle royale game where circular players bounce around an arena, collide with each other and walls, take damage, and ultimately declare a winner. The system must handle potentially 500,000+ players with optimized performance.

## Core Requirements
- **Players**: Circles with images from `/assets/` folder
- **Physics**: Bouncing off walls and other circles
- **Health System**: Health bars, damage on collision
- **Dynamic Scaling**: Circles grow as fewer players remain
- **Winner Declaration**: Last standing player scales up and shows name
- **Performance**: Handle 500k+ circles with lazy loading and fake circles
- **Bias System**: Player-specific win multipliers
- **UI**: Player count display

## Technical Architecture

### 1. Build System & Dependencies
```cmake
# CMakeLists.txt requirements:
- Vulkan SDK
- GLFW (window management)
- GLM (math library)
- STB Image (texture loading)
- Freetype (text rendering for player names/count)
- Threading support (std::thread or TBB)
```

### 2. Vulkan Pipeline Setup
- **Swapchain**: Double/triple buffering for smooth rendering
- **Render Pass**: Single pass with color and depth attachments
- **Graphics Pipeline**:
  - Vertex shader: Circle positions, sizes, health bars
  - Fragment shader: Texture sampling, health bar rendering
  - Instanced rendering for efficient circle drawing
- **Descriptor Sets**: Texture arrays, uniform buffers for transforms

### 3. Data Structures

#### Player/Circle Structure
```cpp
struct Player {
    std::string name;           // From filename
    float bias_multiplier;      // Win probability modifier (1.0-2.0)
    glm::vec2 position;         // Current position
    glm::vec2 velocity;         // Movement vector
    float radius;               // Current size
    float health;               // 0.0-100.0
    bool is_real;               // Real player vs fake circle
    int texture_index;          // Index in texture array
    bool alive;                 // Still in game
};
```

#### Constants Configuration
```cpp
const int MAX_CIRCLES = 500000;
const int FAKE_CIRCLES = 400000;        // Fake circles for initial mass
const float VISIBILITY_THRESHOLD = 2.0f; // Minimum pixels for visibility
const float MAX_CIRCLE_SIZE = 50.0f;    // Maximum radius when few players
const float BASE_HEALTH = 100.0f;
const float COLLISION_DAMAGE = 10.0f;
const float WALL_BOUNCE_DAMPING = 0.95f;
const float ARENA_SIZE = 1000.0f;
```

### 4. Physics Simulation

#### Collision Detection Strategy
**Challenge**: O(n²) collision detection for 500k circles is impossible
**Solution**: Spatial partitioning with grid-based approach

```cpp
class SpatialGrid {
    std::vector<std::vector<std::vector<Player*>>> grid;
    float cell_size;
    int grid_width, grid_height;

    // Methods:
    void insert(Player* player);
    void remove(Player* player);
    std::vector<Player*> get_nearby(Player* player);
    void update_all_positions();
};
```

**Threading**: Divide grid into chunks processed by worker threads
- Use thread pool (8-16 threads based on CPU cores)
- Each thread handles collision detection for its grid chunk
- Atomic operations for position updates to prevent race conditions

#### Physics Integration
```cpp
class PhysicsEngine {
    SpatialGrid spatial_grid;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::atomic<bool> simulation_running;

    void update_physics(float delta_time);
    void resolve_collisions(std::vector<Player*>& chunk);
    void handle_wall_collisions(Player* player);
};
```

### 5. Rendering System

#### Instanced Rendering
- Single vertex buffer for circle geometry (low-poly circle mesh)
- Instance buffer with per-circle data (position, size, texture index, health)
- Health bars rendered as separate instances or overlaid in fragment shader

#### Texture Management
```cpp
class TextureManager {
    std::vector<VkImage> textures;
    std::unordered_map<std::string, int> texture_indices;
    VkDescriptorSet texture_descriptor_set;

    // Lazy loading system
    void load_texture_async(const std::string& filename);
    void unload_distant_textures(); // Based on camera/player positions
};
```

**Memory Optimization**:
- Texture atlas for frequently used images
- Mipmapping for distance-based quality
- Compressed texture formats (BC/ASTC)

#### Dynamic Scaling Algorithm
```cpp
float calculate_circle_size(int alive_players) {
    float scale_factor = std::max(0.1f,
        std::min(1.0f, (float)MAX_CIRCLES / alive_players));
    return BASE_CIRCLE_SIZE * scale_factor * MAX_CIRCLE_SIZE;
}
```

### 6. Performance Optimizations

#### Fake Circles System
- Initial population: `real_players + FAKE_CIRCLES`
- Fake circles have no textures, simple physics
- All fake circles die before reaching `VISIBILITY_THRESHOLD`
- Prevents loading 500k textures initially

#### Lazy Loading Implementation
```cpp
class LazyLoader {
    std::queue<std::string> load_queue;
    std::vector<std::thread> loader_threads;
    std::mutex queue_mutex;

    void load_worker();
    bool should_load_texture(Player* player); // Based on size/screen position
};
```

#### Memory Management
- Object pooling for Player objects
- Texture streaming based on player proximity to camera
- Circular buffer for position history (for interpolation)

### 7. Threading and Synchronization

#### Race Condition Prevention
- **Collision Detection**: Use atomic operations for health updates
- **Position Updates**: Double-buffering - read from old buffer, write to new
- **Texture Loading**: Mutex-protected texture array
- **Player Removal**: Mark as dead, clean up in main thread

```cpp
std::atomic<float> player_health;
std::mutex texture_load_mutex;

// Thread-safe health reduction
void take_damage(Player* player, float damage) {
    float current = player->health.load(std::memory_order_relaxed);
    while (!player->health.compare_exchange_weak(current, current - damage));
}
```

### 8. Game State Management

#### State Machine
```cpp
enum class GameState {
    LOADING,
    SIMULATING,
    WINNER_DECLARATION,
    FINISHED
};

class GameManager {
    GameState current_state;
    std::vector<Player> players;
    Player* winner;

    void update_loading();
    void update_simulation();
    void update_winner_sequence();
};
```

#### Winner Declaration Sequence
1. Detect last alive player
2. Smooth interpolation to center of screen
3. Scale up over 2-3 seconds
4. Display player name with animation
5. Hold for 5 seconds then fade out

### 9. Input and Camera System

#### Camera Controls
- Mouse drag to pan
- Mouse wheel to zoom
- Follow winner during declaration
- Auto-zoom to fit all visible players

#### UI Elements
- Player count: Top-left text overlay
- Winner name: Centered, large text
- Health bars: Above each circle (scaled with circle size)

### 10. Error Handling and Edge Cases

#### Segfault Prevention
- Bounds checking on all array accesses
- Null pointer checks before dereference
- Vulkan validation layers in debug mode
- Exception handling for file I/O

#### Edge Cases
- **Zero Players**: Graceful shutdown
- **Single Player**: Immediate winner declaration
- **All Players Die Simultaneously**: Random winner selection
- **Memory Exhaustion**: Fallback to lower quality textures
- **Thread Crashes**: Thread monitoring and restart

### 11. Development Phases

#### Phase 1: Core Infrastructure
1. Vulkan window setup
2. Basic circle rendering (no textures)
3. Simple physics (wall bouncing only)

#### Phase 2: Collision System
1. Grid-based collision detection
2. Multi-threaded physics
3. Health system implementation

#### Phase 3: Visual Features
1. Texture loading system
2. Health bar rendering
3. UI text overlays

#### Phase 4: Performance Optimization
1. Instanced rendering
2. Lazy loading implementation
3. Fake circles system

#### Phase 5: Polish
1. Winner declaration animation
2. Dynamic scaling
3. Bias system integration

### 12. Testing Strategy

#### Unit Tests
- Physics calculations
- Collision detection accuracy
- Texture loading performance

#### Performance Benchmarks
- Frame time with varying player counts
- Memory usage scaling
- Thread utilization

#### Stress Testing
- 500k fake circles simulation
- Texture loading under memory pressure
- Long-running stability tests

### 13. Potential Challenges & Solutions

#### Challenge: Vulkan Instancing Limits
**Solution**: Multiple draw calls or indirect drawing

#### Challenge: Texture Memory
**Solution**: Streaming system with LRU cache eviction

#### Challenge: CPU Bottleneck in Physics
**Solution**: GPU compute shaders for physics calculations

#### Challenge: Floating Point Precision
**Solution**: Use double precision for positions, single for rendering

### 14. File Structure
```
src/
├── main.cpp
├── vulkan/
│   ├── VulkanContext.h/cpp
│   ├── Renderer.h/cpp
│   └── ShaderManager.h/cpp
├── game/
│   ├── Player.h/cpp
│   ├── PhysicsEngine.h/cpp
│   ├── GameManager.h/cpp
│   └── SpatialGrid.h/cpp
├── rendering/
│   ├── TextureManager.h/cpp
│   ├── UIManager.h/cpp
│   └── Camera.h/cpp
└── utils/
    ├── ThreadPool.h/cpp
    ├── Timer.h/cpp
    └── MathUtils.h/cpp
```

This plan provides a solid foundation for implementing the battle royale simulation with performance optimizations for handling massive player counts while maintaining visual fidelity and smooth gameplay.
