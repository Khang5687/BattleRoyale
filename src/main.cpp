#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdexcept>
#include <optional>
#include <iostream>
#include <limits>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <random>
#include <map>
#include <sstream>
#include <unordered_map>
#include <list>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <array>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb/stb_image_resize2.h"
#include "../stb/stb_easy_font.h"

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif
#ifndef VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
#define VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME "VK_KHR_get_physical_device_properties2"
#endif

static void glfwErrorCallback(int code, const char* desc) {
	std::fprintf(stderr, "GLFW error %d: %s\n", code, desc);
}

struct SwapchainObjects {
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	VkFormat colorFormat = VK_FORMAT_UNDEFINED;
	VkExtent2D extent{};
	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;
	std::vector<VkFramebuffer> framebuffers;
};

struct BufferWithMemory {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct PipelineObjects {
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
};

struct GeometryBuffers {
	BufferWithMemory quadVertexBuffer;
	BufferWithMemory instanceBuffer;
	uint32_t instanceCount = 0;
};

struct HudGeometry {
	BufferWithMemory vertexBuffer;
	VkDeviceSize bufferSize = 0;
};

static constexpr size_t HUD_INITIAL_VERTEX_CAPACITY = 8192;

struct InstanceLayoutCPU {
	float center[2];
	float radius;
	float pad;
	float color[4];
	float imageLayer; // Atlas layer index, -1 for flat color
	float pad2[3];    // Alignment padding
};

struct TextVertex {
	float position[2];
	float color[4];
};

// Image avatar loading system
struct ImageWithMemory {
	VkImage image = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t mipLevels = 1;
};

struct LoadedTexture {
	uint32_t width = 0;
	uint32_t height = 0;
	std::vector<uint8_t> data;
	uint32_t refCount = 0;
	int64_t lastUsed = 0; // For LRU
};

struct TextureAtlas {
	static constexpr uint32_t MAX_LAYERS = 2048;
	static constexpr uint32_t ATLAS_SIZE = 256; // Each layer is 256x256

	ImageWithMemory atlasArray;
	VkSampler sampler = VK_NULL_HANDLE;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

	// LRU management
	std::unordered_map<uint32_t, uint32_t> imageIdToLayer; // imageId -> layer index
	std::vector<uint32_t> layerToImageId; // layer index -> imageId (UINT32_MAX = free)
	std::list<uint32_t> lruOrder; // Most recently used first
	std::unordered_map<uint32_t, std::list<uint32_t>::iterator> layerToLruIter;
	std::queue<uint32_t> freeLayers;

	// Texture cache
	std::unordered_map<uint32_t, LoadedTexture> textureCache;
	std::vector<std::filesystem::path> imageFiles;

	uint32_t nextFreeLayer = 0;
	int64_t frameCounter = 0;
};

struct ImageManager {
	static constexpr float IMAGE_LOAD_THRESHOLD_RADIUS = 20.0f;
	static constexpr uint32_t MAX_CACHE_SIZE = 4096;

	TextureAtlas atlas;

	// Background loading thread
	std::thread loaderThread;
	std::atomic<bool> stopLoading{false};
	std::mutex requestMutex;
	std::queue<uint32_t> loadRequests;
	std::mutex uploadMutex;
	std::queue<std::pair<uint32_t, LoadedTexture>> pendingUploads;

	// Placeholder texture (small 4x4)
	ImageWithMemory placeholderTexture;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkCommandPool commandPool = VK_NULL_HANDLE;

	// Debug counters for file loading
	std::atomic<uint32_t> loadSuccessCount{0};
	std::atomic<uint32_t> loadFailCount{0};
	VkQueue graphicsQueue = VK_NULL_HANDLE;
};

// Forward declarations
static int32_t getAtlasLayerForImage(ImageManager& mgr, uint32_t imageId);
static void uploadTextureToAtlasLayer(ImageManager& mgr, uint32_t imageId, uint32_t layer);

struct Simulation {
	// Constants
	uint32_t maxPlayers = 50000;
	float minRadius = 12.0f;
	float maxRadius = 28.0f;
	float fixedRadius = 40.0f; // All circles have uniform size
	float wallDamping = 0.85f;
	float collisionDamping = 0.98f;
	float damageMultiplier = 0.2f;
	float minDamage = 0.05f;
	float gridCellSize = 64.0f;
	float speedMultiplier = 2.0f; // Speed multiplier for circle movement
	float constantSpeed = 140.0f * speedMultiplier; // Fixed speed magnitude for all circles
	static constexpr uint32_t BIAS_ACTIVE_THRESHOLD = 50; // Minimum player count for bias to be active

	// Bias system - now uses damage reduction instead of health multipliers
	std::map<std::string, float> biasReductions; // 0.0 = no reduction, 0.5 = 50% damage reduction

	// Helper function to normalize velocity to constant speed
	void normalizeVelocity(size_t i) {
		if (!alive[i]) return;
		float vx = velX[i];
		float vy = velY[i];
		float currentSpeed = std::sqrt(vx * vx + vy * vy);
		if (currentSpeed > 0.0f) {
			velX[i] = (vx / currentSpeed) * constantSpeed;
			velY[i] = (vy / currentSpeed) * constantSpeed;
		}
	}

	// World
	float worldWidth = 800.0f;
	float worldHeight = 600.0f;

	// State arrays
	std::vector<float> posX, posY;
	std::vector<float> velX, velY;
	std::vector<float> radius;
	std::vector<float> health; // 0..1
	std::vector<uint8_t> alive; // 0/1
	std::vector<uint32_t> imageId; // Index into image files array
	std::vector<uint8_t> imageTier; // 0=fake/flat color, 1=placeholder, 2=real image

	// Names for winner display
	std::vector<std::string> names;

	// Image manager
	ImageManager* imageManager = nullptr;

	// Temp
	std::mt19937 rng{std::random_device{}()};

	// Winner/victory state
	bool inVictory = false;
	int winnerIndex = -1;
	float victoryTime = 0.0f;

	void loadBiasConfig(const std::string& biasFile) {
		biasReductions.clear();
		std::ifstream file(biasFile);
		if (!file.is_open()) {
			std::cout << "No bias config found, using defaults\n";
			return;
		}
		std::string line;
		while (std::getline(file, line)) {
			size_t pos = line.find(':');
			if (pos != std::string::npos) {
				std::string name = line.substr(0, pos);
				float reduction = std::stof(line.substr(pos + 1));
				// Clamp reduction to [0.0, 0.9] to prevent invincibility
				reduction = std::clamp(reduction, 0.0f, 0.9f);
				biasReductions[name] = reduction;
			}
		}
		std::cout << "Loaded " << biasReductions.size() << " bias entries\n";
	}

	void initializeFromAssets(const std::string& assetsDir, uint32_t targetCount) {
		std::vector<std::filesystem::path> files;
		for (auto const& e : std::filesystem::directory_iterator(assetsDir)) {
			if (!e.is_regular_file()) continue;
			auto p = e.path();
			auto ext = p.extension().string();
			for (auto& c : ext) c = static_cast<char>(std::tolower(c));
			if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(p);
		}

		// Populate image manager with file list
		if (imageManager) {
			imageManager->atlas.imageFiles = files;
		}
		uint32_t count = std::min<uint32_t>(targetCount, static_cast<uint32_t>(files.size()));
		if (count == 0) count = targetCount; // fallback to fakes

		posX.resize(targetCount);
		posY.resize(targetCount);
		velX.resize(targetCount);
		velY.resize(targetCount);
		radius.resize(targetCount);
		health.resize(targetCount);
		alive.resize(targetCount);
		imageId.resize(targetCount);
		imageTier.resize(targetCount);
		names.resize(targetCount);

		std::uniform_real_distribution<float> distX(40.0f, worldWidth - 40.0f);
		std::uniform_real_distribution<float> distY(40.0f, worldHeight - 40.0f);
		std::uniform_real_distribution<float> distAngle(0.0f, 2.0f * 3.14159f); // Random angle for direction

		for (uint32_t i = 0; i < targetCount; ++i) {
			posX[i] = distX(rng);
			posY[i] = distY(rng);
			// Set velocity with constant speed and random direction
			float angle = distAngle(rng);
			velX[i] = std::cos(angle) * constantSpeed;
			velY[i] = std::sin(angle) * constantSpeed;
			radius[i] = fixedRadius; // All circles have uniform size
			health[i] = 1.0f; // All players start with same health
			alive[i] = 1;
			if (i < files.size()) {
				imageId[i] = i;
				imageTier[i] = 0; // Start with flat color, upgrade based on radius
				names[i] = files[i].stem().string();
				// Note: Bias is now applied during damage calculation, not initialization
			} else {
				imageId[i] = UINT32_MAX; // fake
				imageTier[i] = 0; // Fake always flat color
				names[i] = std::string("Fake_") + std::to_string(i);
			}
		}
	}

	uint32_t aliveCount() const {
		uint32_t c = 0; for (auto a : alive) if (a) ++c; return c;
	}

	void step(float dt) {
		// Integrate and wall collisions
		for (size_t i = 0; i < posX.size(); ++i) {
			if (!alive[i]) continue;
			posX[i] += velX[i] * dt;
			posY[i] += velY[i] * dt;
			float r = radius[i];
			bool wallHit = false;
			if (posX[i] - r < 0.0f) { posX[i] = r; velX[i] = -velX[i]; wallHit = true; }
			if (posX[i] + r > worldWidth) { posX[i] = worldWidth - r; velX[i] = -velX[i]; wallHit = true; }
			if (posY[i] - r < 0.0f) { posY[i] = r; velY[i] = -velY[i]; wallHit = true; }
			if (posY[i] + r > worldHeight) { posY[i] = worldHeight - r; velY[i] = -velY[i]; wallHit = true; }
			// Normalize velocity to maintain constant speed after wall collisions
			if (wallHit) {
				normalizeVelocity(i);
			}
		}

		// Spatial grid dimensions
		const int cellsX = std::max(1, static_cast<int>(worldWidth / gridCellSize));
		const int cellsY = std::max(1, static_cast<int>(worldHeight / gridCellSize));
		std::vector<std::vector<int>> grid(static_cast<size_t>(cellsX * cellsY));
		auto cellIndex = [&](float x, float y) {
			int cx = std::clamp(static_cast<int>(x / gridCellSize), 0, cellsX - 1);
			int cy = std::clamp(static_cast<int>(y / gridCellSize), 0, cellsY - 1);
			return cy * cellsX + cx;
		};
		for (int i = 0; i < static_cast<int>(posX.size()); ++i) {
			if (!alive[i]) continue;
			grid[cellIndex(posX[i], posY[i])].push_back(i);
		}

		// Collisions
		for (int cy = 0; cy < cellsY; ++cy) {
			for (int cx = 0; cx < cellsX; ++cx) {
				for (int ny = std::max(0, cy - 1); ny <= std::min(cellsY - 1, cy + 1); ++ny) {
					for (int nx = std::max(0, cx - 1); nx <= std::min(cellsX - 1, cx + 1); ++nx) {
						const auto& a = grid[cy * cellsX + cx];
						const auto& b = grid[ny * cellsX + nx];
						for (int i : a) {
							if (!alive[i]) continue;
							for (int j : b) {
								if (j <= i || !alive[j]) continue;
								float dx = posX[j] - posX[i];
								float dy = posY[j] - posY[i];
								float rr = radius[i] + radius[j];
								float dist2 = dx*dx + dy*dy;
								if (dist2 < rr*rr) {
									float dist = std::sqrt(std::max(dist2, 1e-6f));
									float nx2 = dx / dist;
									float ny2 = dy / dist;
									float penetration = rr - dist;
									// Separate
									posX[i] -= nx2 * (penetration * 0.5f);
									posY[i] -= ny2 * (penetration * 0.5f);
									posX[j] += nx2 * (penetration * 0.5f);
									posY[j] += ny2 * (penetration * 0.5f);
									// Velocities along normal
									float vi = velX[i]*nx2 + velY[i]*ny2;
									float vj = velX[j]*nx2 + velY[j]*ny2;
									float vi2 = vj;
									float vj2 = vi;
									float dvI = vi2 - vi;
									float dvJ = vj2 - vj;
									velX[i] += nx2 * dvI; velY[i] += ny2 * dvI;
									velX[j] += nx2 * dvJ; velY[j] += ny2 * dvJ;
									// Normalize velocities to maintain constant speed instead of damping
									normalizeVelocity(i);
									normalizeVelocity(j);
									// Damage based on impulse magnitude
									float impulse = std::abs(dvI) + std::abs(dvJ);
									float baseDamage = std::max(minDamage, impulse * damageMultiplier * 0.001f);
									
									// Apply bias damage reduction if bias is active (enough players)
									float finalDamageI = baseDamage;
									float finalDamageJ = baseDamage;
									
									if (aliveCount() >= BIAS_ACTIVE_THRESHOLD) {
										// Apply bias reduction for player i
										auto itI = biasReductions.find(names[i]);
										if (itI != biasReductions.end()) {
											finalDamageI = baseDamage * (1.0f - itI->second);
										}
										
										// Apply bias reduction for player j
										auto itJ = biasReductions.find(names[j]);
										if (itJ != biasReductions.end()) {
											finalDamageJ = baseDamage * (1.0f - itJ->second);
										}
									}
									
									health[i] -= finalDamageI; 
									health[j] -= finalDamageJ;
									if (health[i] <= 0.0f) { alive[i] = 0; }
									if (health[j] <= 0.0f) { alive[j] = 0; }
								}
							}
						}
					}
				}
			}
		}

		// Winner check
		if (!inVictory) {
			int last = -1; int cnt = 0; for (int i = 0; i < static_cast<int>(alive.size()); ++i) { if (alive[i]) { last = i; ++cnt; } }
			if (cnt <= 1 && last >= 0) {
				inVictory = true;
				winnerIndex = last;
				victoryTime = 0.0f;
			}
		} else {
			victoryTime += dt;
			// Freeze others
			for (int i = 0; i < static_cast<int>(alive.size()); ++i) {
				if (i == winnerIndex) continue;
				velX[i] = 0.0f; velY[i] = 0.0f;
			}
			// Move winner to center and scale up
			float targetX = worldWidth * 0.5f;
			float targetY = worldHeight * 0.5f;
			float t = std::min(1.0f, victoryTime * 0.5f);
			posX[winnerIndex] = posX[winnerIndex] + (targetX - posX[winnerIndex]) * t;
			posY[winnerIndex] = posY[winnerIndex] + (targetY - posY[winnerIndex]) * t;
			float baseR = std::max(30.0f, radius[winnerIndex]);
			radius[winnerIndex] = baseR * (1.0f + t * 2.5f);
		}
	}

	void updateImageTiers() {
		if (!imageManager) return;

		for (size_t i = 0; i < posX.size(); ++i) {
			if (!alive[i] || imageId[i] == UINT32_MAX) continue;

			float currentRadius = radius[i];

			// Upgrade tier based on radius threshold
			if (currentRadius >= ImageManager::IMAGE_LOAD_THRESHOLD_RADIUS) {
				if (imageTier[i] < 2) {
					imageTier[i] = 2; // Request real image
					// Queue for loading if not already loaded
					// This will be handled by the image manager
				}
			} else if (currentRadius >= 8.0f && imageTier[i] < 1) {
				imageTier[i] = 1; // Use placeholder
			}
		}
	}

	void writeInstances(std::vector<InstanceLayoutCPU>& out) const {
		out.clear();
		out.reserve(posX.size());
		for (size_t i = 0; i < posX.size(); ++i) {
			if (!alive[i]) continue;
			InstanceLayoutCPU inst{};
			inst.center[0] = posX[i]; inst.center[1] = posY[i];
			inst.radius = radius[i];
			float h = std::clamp(health[i], 0.0f, 1.0f);

			// Set image layer based on tier
			if (imageTier[i] == 0 || imageId[i] == UINT32_MAX) {
				// Flat color
				inst.imageLayer = -1.0f;
				// Green to red gradient by health
				inst.color[0] = 1.0f - h;
				inst.color[1] = h;
				inst.color[2] = 0.2f;
				inst.color[3] = 1.0f;
			} else if (imageTier[i] >= 2 && imageManager) {
				// Try to get texture layer from image manager
				int32_t layer = getAtlasLayerForImage(*imageManager, imageId[i]);
				if (layer >= 0) {
					inst.imageLayer = static_cast<float>(layer);
					// Health tint for texture
					inst.color[0] = 1.0f - h * 0.5f;
					inst.color[1] = 1.0f - h * 0.5f;
					inst.color[2] = 1.0f - h * 0.5f;
					inst.color[3] = 1.0f;
				} else {
					// Fallback to flat color while loading
					inst.imageLayer = -1.0f;
					inst.color[0] = 1.0f - h;
					inst.color[1] = h;
					inst.color[2] = 0.2f;
					inst.color[3] = 1.0f;
				}
			} else {
				// Placeholder or other tiers
				inst.imageLayer = -1.0f;
				inst.color[0] = 1.0f - h;
				inst.color[1] = h;
				inst.color[2] = 0.2f;
				inst.color[3] = 1.0f;
			}

			out.push_back(inst);
		}
	}
};

static VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
	for (const auto& f : formats) {
		if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_SRGB)
			&& f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return f;
		}
	}
	return formats[0];
}

static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
	for (const auto& m : modes) {
		if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
	}
	return VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
}

static VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) {
	if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return caps.currentExtent;
	}
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	VkExtent2D actual{};
	actual.width = static_cast<uint32_t>(width);
	actual.height = static_cast<uint32_t>(height);
	if (actual.width < caps.minImageExtent.width) actual.width = caps.minImageExtent.width;
	if (actual.height < caps.minImageExtent.height) actual.height = caps.minImageExtent.height;
	if (caps.maxImageExtent.width > 0 && actual.width > caps.maxImageExtent.width) actual.width = caps.maxImageExtent.width;
	if (caps.maxImageExtent.height > 0 && actual.height > caps.maxImageExtent.height) actual.height = caps.maxImageExtent.height;
	return actual;
}

static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format) {
	VkImageViewCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	ci.image = image;
	ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
	ci.format = format;
	ci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
	ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	ci.subresourceRange.baseMipLevel = 0;
	ci.subresourceRange.levelCount = 1;
	ci.subresourceRange.baseArrayLayer = 0;
	ci.subresourceRange.layerCount = 1;
	VkImageView view = VK_NULL_HANDLE;
	VkResult res = vkCreateImageView(device, &ci, nullptr, &view);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image view");
	}
	return view;
}

static VkRenderPass createRenderPass(VkDevice device, VkFormat colorFormat) {
	VkAttachmentDescription color{};
	color.format = colorFormat;
	color.samples = VK_SAMPLE_COUNT_1_BIT;
	color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorRef{};
	colorRef.attachment = 0;
	colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorRef;

	VkSubpassDependency dep{};
	dep.srcSubpass = VK_SUBPASS_EXTERNAL;
	dep.dstSubpass = 0;
	dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dep.srcAccessMask = 0;
	dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo rpci{};
	rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	rpci.attachmentCount = 1;
	rpci.pAttachments = &color;
	rpci.subpassCount = 1;
	rpci.pSubpasses = &subpass;
	rpci.dependencyCount = 1;
	rpci.pDependencies = &dep;

	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (vkCreateRenderPass(device, &rpci, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create render pass");
	}
	return renderPass;
}

static std::vector<char> readBinaryFile(const std::string& path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file) throw std::runtime_error("Failed to open file: " + path);
	const size_t size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(size);
	file.seekg(0);
	file.read(buffer.data(), size);
	return buffer;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
	VkShaderModuleCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = code.size();
	ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module");
	}
	return module;
}

static uint32_t findMemoryType(VkPhysicalDevice physical, uint32_t typeBits, VkMemoryPropertyFlags props) {
	VkPhysicalDeviceMemoryProperties memProps{};
	vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}
	throw std::runtime_error("Suitable memory type not found");
}

static void createBuffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, BufferWithMemory& out) {
	VkBufferCreateInfo bci{};
	bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.size = size;
	bci.usage = usage;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateBuffer(device, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer");
	}
	VkMemoryRequirements req{};
	vkGetBufferMemoryRequirements(device, out.buffer, &req);
	VkMemoryAllocateInfo mai{};
	mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = findMemoryType(physical, req.memoryTypeBits, props);
	if (vkAllocateMemory(device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate buffer memory");
	}
	vkBindBufferMemory(device, out.buffer, out.memory, 0);
}

static void* mapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size) {
	void* data = nullptr;
	vkMapMemory(device, memory, 0, size, 0, &data);
	return data;
}

static void unmapMemory(VkDevice device, VkDeviceMemory memory) {
	vkUnmapMemory(device, memory);
}

static PipelineObjects createCirclePipeline(VkDevice device, VkFormat colorFormat, VkRenderPass renderPass) {
	std::string baseDir = std::string(BR5_SHADER_DIR);
	std::string vertPath = baseDir + "/circle.vert.spv";
	std::string fragPath = baseDir + "/circle.frag.spv";
	auto vertCode = readBinaryFile(vertPath);
	auto fragCode = readBinaryFile(fragPath);
	VkShaderModule vert = createShaderModule(device, vertCode);
	VkShaderModule frag = createShaderModule(device, fragCode);

	VkPipelineShaderStageCreateInfo vs{}; vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; vs.stage = VK_SHADER_STAGE_VERTEX_BIT; vs.module = vert; vs.pName = "main";
	VkPipelineShaderStageCreateInfo fs{}; fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT; fs.module = frag; fs.pName = "main";
	VkPipelineShaderStageCreateInfo stages[] = { vs, fs };

	// Vertex bindings: 0 = quad vertices (vec2), 1 = instance data (center vec2, radius float, color vec4, imageLayer float)
	VkVertexInputBindingDescription bindings[2]{};
	bindings[0].binding = 0; bindings[0].stride = sizeof(float) * 2; bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	bindings[1].binding = 1; bindings[1].stride = sizeof(InstanceLayoutCPU); bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

	VkVertexInputAttributeDescription attrs[5]{};
	attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32_SFLOAT; attrs[0].offset = 0; // inPos
	attrs[1].location = 1; attrs[1].binding = 1; attrs[1].format = VK_FORMAT_R32G32_SFLOAT; attrs[1].offset = offsetof(InstanceLayoutCPU, center); // inCenter
	attrs[2].location = 2; attrs[2].binding = 1; attrs[2].format = VK_FORMAT_R32_SFLOAT; attrs[2].offset = offsetof(InstanceLayoutCPU, radius); // inRadius
	attrs[3].location = 3; attrs[3].binding = 1; attrs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrs[3].offset = offsetof(InstanceLayoutCPU, color); // inColor
	attrs[4].location = 4; attrs[4].binding = 1; attrs[4].format = VK_FORMAT_R32_SFLOAT; attrs[4].offset = offsetof(InstanceLayoutCPU, imageLayer); // inImageLayer

	VkPipelineVertexInputStateCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = bindings;
	vi.vertexAttributeDescriptionCount = 5; vi.pVertexAttributeDescriptions = attrs;

	VkPipelineInputAssemblyStateCreateInfo ia{}; ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkViewport vp{}; vp.x = 0.0f; vp.y = 0.0f; vp.width = 1.0f; vp.height = 1.0f; vp.minDepth = 0.0f; vp.maxDepth = 1.0f; // will be set by dynamic state or via scissor/viewport fixed? We'll set dynamic.
	VkRect2D sc{}; sc.offset = {0,0}; sc.extent = {1,1};

	VkPipelineViewportStateCreateInfo vps{}; vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;

	VkPipelineRasterizationStateCreateInfo rs{}; rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo ms{}; ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT; cba.blendEnable = VK_TRUE; cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; cba.colorBlendOp = VK_BLEND_OP_ADD; cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; cba.alphaBlendOp = VK_BLEND_OP_ADD;
	VkPipelineColorBlendStateCreateInfo cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; cb.attachmentCount = 1; cb.pAttachments = &cba;

	VkDynamicState dynamics[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dyn{}; dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynamics;

	// Push constants: vec2 viewport in vertex stage
	VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; pcr.offset = 0; pcr.size = sizeof(float) * 2;

	// Create descriptor set layout for texture atlas
	VkDescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 0;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.pImmutableSamplers = nullptr;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &samplerLayoutBinding;

	PipelineObjects po{};
	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &po.descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor set layout");
	}

	VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1; plci.pSetLayouts = &po.descriptorSetLayout;
	plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
	if (vkCreatePipelineLayout(device, &plci, nullptr, &po.layout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
		throw std::runtime_error("Failed to create pipeline layout");
	}

	VkPipelineRenderingCreateInfoKHR rendering{}; // not used with render pass pipeline

	VkGraphicsPipelineCreateInfo gpci{}; gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; gpci.stageCount = 2; gpci.pStages = stages; gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia; gpci.pViewportState = &vps; gpci.pRasterizationState = &rs; gpci.pMultisampleState = &ms; gpci.pDepthStencilState = nullptr; gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyn; gpci.layout = po.layout; gpci.renderPass = renderPass; gpci.subpass = 0;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &po.pipeline) != VK_SUCCESS) {
		vkDestroyPipelineLayout(device, po.layout, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
		throw std::runtime_error("Failed to create graphics pipeline");
	}

	vkDestroyShaderModule(device, frag, nullptr);
	vkDestroyShaderModule(device, vert, nullptr);
	return po;
}

static PipelineObjects createTextPipeline(VkDevice device, VkFormat colorFormat, VkRenderPass renderPass) {
	std::string baseDir = std::string(BR5_SHADER_DIR);
	std::string vertPath = baseDir + "/text.vert.spv";
	std::string fragPath = baseDir + "/text.frag.spv";
	auto vertCode = readBinaryFile(vertPath);
	auto fragCode = readBinaryFile(fragPath);
	VkShaderModule vert = createShaderModule(device, vertCode);
	VkShaderModule frag = createShaderModule(device, fragCode);

	VkPipelineShaderStageCreateInfo vs{}; vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; vs.stage = VK_SHADER_STAGE_VERTEX_BIT; vs.module = vert; vs.pName = "main";
	VkPipelineShaderStageCreateInfo fs{}; fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT; fs.module = frag; fs.pName = "main";
	VkPipelineShaderStageCreateInfo stages[] = { vs, fs };

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = sizeof(TextVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attrs[2]{};
	attrs[0].location = 0;
	attrs[0].binding = 0;
	attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
	attrs[0].offset = offsetof(TextVertex, position);
	attrs[1].location = 1;
	attrs[1].binding = 0;
	attrs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attrs[1].offset = offsetof(TextVertex, color);

	VkPipelineVertexInputStateCreateInfo vi{};
	vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vi.vertexBindingDescriptionCount = 1;
	vi.pVertexBindingDescriptions = &binding;
	vi.vertexAttributeDescriptionCount = 2;
	vi.pVertexAttributeDescriptions = attrs;

	VkPipelineInputAssemblyStateCreateInfo ia{};
	ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	ia.primitiveRestartEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo vps{};
	vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	vps.viewportCount = 1;
	vps.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rs{};
	rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rs.depthClampEnable = VK_FALSE;
	rs.rasterizerDiscardEnable = VK_FALSE;
	rs.polygonMode = VK_POLYGON_MODE_FILL;
	rs.cullMode = VK_CULL_MODE_NONE;
	rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rs.depthBiasEnable = VK_FALSE;
	rs.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo ms{};
	ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState cbAttach{};
	cbAttach.blendEnable = VK_TRUE;
	cbAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	cbAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	cbAttach.colorBlendOp = VK_BLEND_OP_ADD;
	cbAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	cbAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	cbAttach.alphaBlendOp = VK_BLEND_OP_ADD;
	cbAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo cb{};
	cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	cb.logicOpEnable = VK_FALSE;
	cb.attachmentCount = 1;
	cb.pAttachments = &cbAttach;

	VkDynamicState dynamics[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dyn{};
	dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dyn.dynamicStateCount = 2;
	dyn.pDynamicStates = dynamics;

	VkPushConstantRange pcr{};
	pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	pcr.offset = 0;
	pcr.size = sizeof(float) * 2;

	VkPipelineLayoutCreateInfo plci{};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 0;
	plci.pSetLayouts = nullptr;
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges = &pcr;

	PipelineObjects po{};
	if (vkCreatePipelineLayout(device, &plci, nullptr, &po.layout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
		throw std::runtime_error("Failed to create text pipeline layout");
	}

	VkGraphicsPipelineCreateInfo gpci{};
	gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	gpci.stageCount = 2;
	gpci.pStages = stages;
	gpci.pVertexInputState = &vi;
	gpci.pInputAssemblyState = &ia;
	gpci.pViewportState = &vps;
	gpci.pRasterizationState = &rs;
	gpci.pMultisampleState = &ms;
	gpci.pDepthStencilState = nullptr;
	gpci.pColorBlendState = &cb;
	gpci.pDynamicState = &dyn;
	gpci.layout = po.layout;
	gpci.renderPass = renderPass;
	gpci.subpass = 0;

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &po.pipeline) != VK_SUCCESS) {
		vkDestroyPipelineLayout(device, po.layout, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, vert, nullptr);
		throw std::runtime_error("Failed to create text pipeline");
	}

	vkDestroyShaderModule(device, frag, nullptr);
	vkDestroyShaderModule(device, vert, nullptr);
	return po;
}

static int hudMeasureWidth(const std::string& text) {
	if (text.empty()) return 0;
	return stb_easy_font_width(const_cast<char*>(text.c_str()));
}

static int hudMeasureHeight(const std::string& text) {
	if (text.empty()) return 0;
	return stb_easy_font_height(const_cast<char*>(text.c_str()));
}

static void appendHudText(std::vector<TextVertex>& out, float x, float y, const std::string& text, float scale, const std::array<float, 4>& color) {
	if (text.empty()) return;

	static std::vector<char> scratch;
	const size_t minBytes = std::max<size_t>(text.size() * 128, 4096);
	if (scratch.size() < minBytes) {
		scratch.resize(minBytes);
	}

	int quads = stb_easy_font_print(0.0f, 0.0f, const_cast<char*>(text.c_str()), nullptr, scratch.data(), static_cast<int>(scratch.size()));
	const size_t quadStride = 4 * 16;
	for (int qi = 0; qi < quads; ++qi) {
		const char* quadPtr = scratch.data() + qi * quadStride;
		float px[4];
		float py[4];
		for (int v = 0; v < 4; ++v) {
			const float* pos = reinterpret_cast<const float*>(quadPtr + v * 16);
			px[v] = x + pos[0] * scale;
			py[v] = y + pos[1] * scale;
		}

		TextVertex v0{};
		v0.position[0] = px[0];
		v0.position[1] = py[0];
		std::copy(color.begin(), color.end(), v0.color);
		TextVertex v1 = v0; v1.position[0] = px[1]; v1.position[1] = py[1];
		TextVertex v2 = v0; v2.position[0] = px[2]; v2.position[1] = py[2];
		TextVertex v3 = v0; v3.position[0] = px[3]; v3.position[1] = py[3];

		out.push_back(v0);
		out.push_back(v1);
		out.push_back(v2);
		out.push_back(v0);
		out.push_back(v2);
		out.push_back(v3);
	}
}

static std::vector<const char*> getRequiredInstanceExtensions() {
	uint32_t count = 0;
	const char** glfwExt = glfwGetRequiredInstanceExtensions(&count);
	std::vector<const char*> exts(glfwExt, glfwExt + count);
	exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	// Some MoltenVK/Loader combos require this when using portability
	exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	return exts;
}

static bool checkValidationLayerSupport(const std::vector<const char*>& layers) {
	uint32_t layerCount = 0;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	std::vector<VkLayerProperties> available(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, available.data());
	for (auto l : layers) {
		bool found = false;
		for (auto const& p : available) {
			if (std::string(p.layerName) == l) { found = true; break; }
		}
		if (!found) return false;
	}
	return true;
}

// ImageManager implementation
static void createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, uint32_t arrayLayers, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, ImageWithMemory& image) {
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = arrayLayers;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	if (vkCreateImage(device, &imageInfo, nullptr, &image.image) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, image.image, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(device, &allocInfo, nullptr, &image.memory) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate image memory");
	}

	vkBindImageMemory(device, image.image, image.memory, 0);
	image.width = width;
	image.height = height;
}

static VkImageView createImageView2DArray(VkDevice device, VkImage image, VkFormat format, uint32_t layerCount) {
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = layerCount;

	VkImageView imageView;
	if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create texture image view");
	}

	return imageView;
}

static VkSampler createTextureSampler(VkDevice device) {
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 0.0f;

	VkSampler sampler;
	if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create texture sampler");
	}

	return sampler;
}

static void initImageManager(ImageManager& mgr, VkPhysicalDevice physicalDevice, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue) {
	mgr.physicalDevice = physicalDevice;
	mgr.device = device;
	mgr.commandPool = commandPool;
	mgr.graphicsQueue = graphicsQueue;

	// Create texture atlas array
	createImage(physicalDevice, device,
		TextureAtlas::ATLAS_SIZE, TextureAtlas::ATLAS_SIZE, TextureAtlas::MAX_LAYERS,
		VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mgr.atlas.atlasArray);

	mgr.atlas.atlasArray.view = createImageView2DArray(device, mgr.atlas.atlasArray.image, VK_FORMAT_R8G8B8A8_UNORM, TextureAtlas::MAX_LAYERS);
	mgr.atlas.sampler = createTextureSampler(device);

	// Initialize LRU structures
	mgr.atlas.layerToImageId.resize(TextureAtlas::MAX_LAYERS, UINT32_MAX);
	for (uint32_t i = 0; i < TextureAtlas::MAX_LAYERS; ++i) {
		mgr.atlas.freeLayers.push(i);
	}

	// Create placeholder texture (4x4 magenta)
	createImage(physicalDevice, device, 4, 4, 1, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mgr.placeholderTexture);

	mgr.placeholderTexture.view = createImageView(device, mgr.placeholderTexture.image, VK_FORMAT_R8G8B8A8_UNORM);

	// Create background loading thread
	mgr.loaderThread = std::thread([&mgr]() {
		while (!mgr.stopLoading) {
			uint32_t imageId = UINT32_MAX;

			// Check for load requests
			{
				std::lock_guard<std::mutex> lock(mgr.requestMutex);
				if (!mgr.loadRequests.empty()) {
					imageId = mgr.loadRequests.front();
					mgr.loadRequests.pop();
				}
			}

			if (imageId == UINT32_MAX) {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				continue;
			}

			// Load image using stb_image
			if (imageId < mgr.atlas.imageFiles.size()) {
				std::string path = mgr.atlas.imageFiles[imageId].string();
				int width, height, channels;
				unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4); // Force RGBA

				if (data) {
					LoadedTexture tex;
					
					// Resize image to atlas size if needed
					if (width != TextureAtlas::ATLAS_SIZE || height != TextureAtlas::ATLAS_SIZE) {
						// Allocate buffer for resized image
						tex.data.resize(TextureAtlas::ATLAS_SIZE * TextureAtlas::ATLAS_SIZE * 4);
						
						// Resize using stb_image_resize2
						unsigned char* result = stbir_resize_uint8_linear(
							data, width, height, 0,
							tex.data.data(), TextureAtlas::ATLAS_SIZE, TextureAtlas::ATLAS_SIZE, 0,
							STBIR_RGBA
						);
						
						if (result == nullptr) {
							// Resize failed, skip this image
							stbi_image_free(data);
							mgr.loadFailCount++;
							continue;
						}
						
						tex.width = TextureAtlas::ATLAS_SIZE;
						tex.height = TextureAtlas::ATLAS_SIZE;
					} else {
						// Image is already the right size
						tex.width = static_cast<uint32_t>(width);
						tex.height = static_cast<uint32_t>(height);
						tex.data.resize(width * height * 4);
						std::memcpy(tex.data.data(), data, tex.data.size());
					}
					
					tex.refCount = 1;
					tex.lastUsed = mgr.atlas.frameCounter;

					stbi_image_free(data);

					// Queue for GPU upload
					{
						std::lock_guard<std::mutex> lock(mgr.uploadMutex);
						mgr.pendingUploads.emplace(imageId, std::move(tex));
					}
					mgr.loadSuccessCount++;
				} else {
					// Loading failed
					mgr.loadFailCount++;
					std::cout << "ERROR: Failed to load image ID " << imageId << std::endl;
				}
			}
		}
	});
}

static void requestImageLoad(ImageManager& mgr, uint32_t imageId) {
	std::lock_guard<std::mutex> lock(mgr.requestMutex);
	mgr.loadRequests.push(imageId);
}

static VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);
	return commandBuffer;
}

static void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkCommandBuffer commandBuffer) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerIndex) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = layerIndex;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else {
		throw std::invalid_argument("Unsupported layout transition");
	}

	vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	endSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
}

static void uploadTextureToAtlasLayer(ImageManager& mgr, uint32_t imageId, uint32_t layer) {
	auto texIt = mgr.atlas.textureCache.find(imageId);
	if (texIt == mgr.atlas.textureCache.end()) {
		return; // Texture not loaded
	}

	const LoadedTexture& texture = texIt->second;

	// Create staging buffer
	BufferWithMemory stagingBuffer;
	VkDeviceSize imageSize = texture.data.size();
	createBuffer(mgr.physicalDevice, mgr.device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);

	void* data = mapMemory(mgr.device, stagingBuffer.memory, imageSize);
	memcpy(data, texture.data.data(), static_cast<size_t>(imageSize));
	unmapMemory(mgr.device, stagingBuffer.memory);

	// Transition image layout for transfer
	transitionImageLayout(mgr.device, mgr.commandPool, mgr.graphicsQueue, mgr.atlas.atlasArray.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layer);

	// Copy buffer to image
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(mgr.device, mgr.commandPool);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = layer;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = {0, 0, 0};

	// Images are now pre-resized to atlas size
	region.imageExtent = {texture.width, texture.height, 1};

	vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.buffer, mgr.atlas.atlasArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(mgr.device, mgr.commandPool, mgr.graphicsQueue, commandBuffer);

	// Transition image layout for shader reading
	transitionImageLayout(mgr.device, mgr.commandPool, mgr.graphicsQueue, mgr.atlas.atlasArray.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, layer);

	// Cleanup staging buffer
	vkDestroyBuffer(mgr.device, stagingBuffer.buffer, nullptr);
	vkFreeMemory(mgr.device, stagingBuffer.memory, nullptr);
}

static int32_t getAtlasLayerForImage(ImageManager& mgr, uint32_t imageId) {
	mgr.atlas.frameCounter++;

	// Check if already in atlas
	auto it = mgr.atlas.imageIdToLayer.find(imageId);
	if (it != mgr.atlas.imageIdToLayer.end()) {
		uint32_t layer = it->second;
		// Update LRU
		if (mgr.atlas.layerToLruIter.find(layer) != mgr.atlas.layerToLruIter.end()) {
			mgr.atlas.lruOrder.erase(mgr.atlas.layerToLruIter[layer]);
		}
		mgr.atlas.lruOrder.push_front(layer);
		mgr.atlas.layerToLruIter[layer] = mgr.atlas.lruOrder.begin();
		return static_cast<int32_t>(layer);
	}

	// Check if texture is loaded but not in atlas
	auto texIt = mgr.atlas.textureCache.find(imageId);
	if (texIt == mgr.atlas.textureCache.end()) {
		// Request loading
		requestImageLoad(mgr, imageId);
		return -1; // Not available yet
	}

	// Find free layer or evict LRU
	uint32_t layer;
	if (!mgr.atlas.freeLayers.empty()) {
		layer = mgr.atlas.freeLayers.front();
		mgr.atlas.freeLayers.pop();
	} else if (!mgr.atlas.lruOrder.empty()) {
		// Evict LRU
		layer = mgr.atlas.lruOrder.back();
		uint32_t evictedImageId = mgr.atlas.layerToImageId[layer];
		mgr.atlas.imageIdToLayer.erase(evictedImageId);
		mgr.atlas.lruOrder.pop_back();
		mgr.atlas.layerToLruIter.erase(layer);
	} else {
		return -1; // No space
	}

	// Assign layer
	mgr.atlas.imageIdToLayer[imageId] = layer;
	mgr.atlas.layerToImageId[layer] = imageId;
	mgr.atlas.lruOrder.push_front(layer);
	mgr.atlas.layerToLruIter[layer] = mgr.atlas.lruOrder.begin();

	// Upload texture data to atlas layer
	uploadTextureToAtlasLayer(mgr, imageId, layer);
	return static_cast<int32_t>(layer);
}

static void updateImageManager(ImageManager& mgr) {
	// Process pending uploads
	std::lock_guard<std::mutex> lock(mgr.uploadMutex);
	while (!mgr.pendingUploads.empty()) {
		auto& [imageId, texture] = mgr.pendingUploads.front();
		mgr.atlas.textureCache[imageId] = std::move(texture);
		mgr.pendingUploads.pop();
	}
}

static void destroyImageManager(ImageManager& mgr) {
	if (mgr.device == VK_NULL_HANDLE) return;

	mgr.stopLoading = true;
	if (mgr.loaderThread.joinable()) {
		mgr.loaderThread.join();
	}

	if (mgr.atlas.sampler != VK_NULL_HANDLE) {
		vkDestroySampler(mgr.device, mgr.atlas.sampler, nullptr);
	}
	if (mgr.atlas.atlasArray.view != VK_NULL_HANDLE) {
		vkDestroyImageView(mgr.device, mgr.atlas.atlasArray.view, nullptr);
	}
	if (mgr.atlas.atlasArray.image != VK_NULL_HANDLE) {
		vkDestroyImage(mgr.device, mgr.atlas.atlasArray.image, nullptr);
	}
	if (mgr.atlas.atlasArray.memory != VK_NULL_HANDLE) {
		vkFreeMemory(mgr.device, mgr.atlas.atlasArray.memory, nullptr);
	}

	if (mgr.placeholderTexture.view != VK_NULL_HANDLE) {
		vkDestroyImageView(mgr.device, mgr.placeholderTexture.view, nullptr);
	}
	if (mgr.placeholderTexture.image != VK_NULL_HANDLE) {
		vkDestroyImage(mgr.device, mgr.placeholderTexture.image, nullptr);
	}
	if (mgr.placeholderTexture.memory != VK_NULL_HANDLE) {
		vkFreeMemory(mgr.device, mgr.placeholderTexture.memory, nullptr);
	}
}

int main() {
	glfwSetErrorCallback(glfwErrorCallback);
	if (!glfwInit()) {
		std::fprintf(stderr, "Failed to init GLFW\n");
		return 1;
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(800, 600, "BattleRoyale5 Vulkan", nullptr, nullptr);
	if (!window) {
		std::fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return 1;
	}

	std::vector<const char*> layers;
#ifndef NDEBUG
	layers.push_back("VK_LAYER_KHRONOS_validation");
	if (!checkValidationLayerSupport(layers)) layers.clear();
#endif

	auto extensions = getRequiredInstanceExtensions();
	VkApplicationInfo app{};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "BattleRoyale5";
	app.applicationVersion = VK_MAKE_VERSION(0,1,0);
	app.pEngineName = "None";
	app.engineVersion = VK_MAKE_VERSION(0,1,0);
	app.apiVersion = VK_API_VERSION_1_0; // MoltenVK safe default

	VkInstanceCreateInfo ici{};
	ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	ici.pApplicationInfo = &app;
	ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	ici.ppEnabledExtensionNames = extensions.data();
	ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
	ici.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

	VkInstance instance = VK_NULL_HANDLE;
	VkResult res = vkCreateInstance(&ici, nullptr, &instance);
	if (res != VK_SUCCESS) {
		std::fprintf(stderr, "vkCreateInstance failed: %d\n", res);
		glfwDestroyWindow(window);
		glfwTerminate();
		return 2;
	}

	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		std::fprintf(stderr, "Failed to create window surface\n");
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
		return 3;
	}

	uint32_t physicalCount = 0;
	vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
	if (physicalCount == 0) {
		std::fprintf(stderr, "No Vulkan physical devices found\n");
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
		return 4;
	}
	std::vector<VkPhysicalDevice> physicals(physicalCount);
	vkEnumeratePhysicalDevices(instance, &physicalCount, physicals.data());

	// Choose the first device that supports presentation
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	uint32_t graphicsQueueFamily = UINT32_MAX;
	uint32_t presentQueueFamily = UINT32_MAX;
	for (auto dev : physicals) {
		uint32_t qfCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
		std::vector<VkQueueFamilyProperties> qprops(qfCount);
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qprops.data());
		for (uint32_t i = 0; i < qfCount; ++i) {
			VkBool32 presentSupport = VK_FALSE;
			glfwGetPhysicalDevicePresentationSupport(instance, dev, i);
			vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
			bool graphics = (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
			if (graphics && presentSupport) {
				physical = dev;
				graphicsQueueFamily = i;
				presentQueueFamily = i;
				break;
			}
		}
		if (physical != VK_NULL_HANDLE) break;
	}
	if (physical == VK_NULL_HANDLE) {
		std::fprintf(stderr, "No suitable queue family found\n");
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
		return 5;
	}

	float priority = 1.0f;
	std::vector<VkDeviceQueueCreateInfo> qcis;
	{
		VkDeviceQueueCreateInfo qci{};
		qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci.queueFamilyIndex = graphicsQueueFamily;
		qci.queueCount = 1;
		qci.pQueuePriorities = &priority;
		qcis.push_back(qci);
	}

	std::vector<const char*> deviceExts = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
	};

	VkPhysicalDeviceFeatures features{};
	VkDeviceCreateInfo dci{};
	dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
	dci.pQueueCreateInfos = qcis.data();
	dci.pEnabledFeatures = &features;
	dci.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
	dci.ppEnabledExtensionNames = deviceExts.data();

	VkDevice device = VK_NULL_HANDLE;
	res = vkCreateDevice(physical, &dci, nullptr, &device);
	if (res != VK_SUCCESS) {
		std::fprintf(stderr, "vkCreateDevice failed: %d\n", res);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
		return 6;
	}

	VkQueue graphicsQueue = VK_NULL_HANDLE;
	vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
	VkQueue presentQueue = VK_NULL_HANDLE;
	vkGetDeviceQueue(device, presentQueueFamily, 0, &presentQueue);

	// Swapchain creation
	auto createSwapchainObjects = [&](SwapchainObjects& out, VkSwapchainKHR oldSwapchain) {
		VkSurfaceCapabilitiesKHR caps{};
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, nullptr);
		std::vector<VkSurfaceFormatKHR> formats(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, formats.data());
		VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);

		uint32_t presentModeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &presentModeCount, nullptr);
		std::vector<VkPresentModeKHR> presentModes(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &presentModeCount, presentModes.data());
		VkPresentModeKHR presentMode = choosePresentMode(presentModes);

		VkExtent2D extent = chooseExtent(caps, window);

		uint32_t minImageCount = caps.minImageCount + 1;
		if (caps.maxImageCount > 0 && minImageCount > caps.maxImageCount) {
			minImageCount = caps.maxImageCount;
		}

		VkSwapchainCreateInfoKHR sci{};
		sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		sci.surface = surface;
		sci.minImageCount = minImageCount;
		sci.imageFormat = surfaceFormat.format;
		sci.imageColorSpace = surfaceFormat.colorSpace;
		sci.imageExtent = extent;
		sci.imageArrayLayers = 1;
		sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		sci.preTransform = caps.currentTransform;
		sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		sci.presentMode = presentMode;
		sci.clipped = VK_TRUE;
		sci.oldSwapchain = oldSwapchain;

		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		VkResult scr = vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain);
		if (scr != VK_SUCCESS) {
			throw std::runtime_error("Failed to create swapchain");
		}

		uint32_t imageCount = 0;
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
		std::vector<VkImage> images(imageCount);
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());

		std::vector<VkImageView> views;
		views.reserve(images.size());
		for (auto img : images) {
			views.push_back(createImageView(device, img, surfaceFormat.format));
		}

		out.swapchain = swapchain;
		out.colorFormat = surfaceFormat.format;
		out.extent = extent;
		out.images = std::move(images);
		out.imageViews = std::move(views);
	};

	auto destroySwapchainObjects = [&](SwapchainObjects& sc) {
		for (auto fb : sc.framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
		sc.framebuffers.clear();
		for (auto v : sc.imageViews) vkDestroyImageView(device, v, nullptr);
		sc.imageViews.clear();
		if (sc.swapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device, sc.swapchain, nullptr); sc.swapchain = VK_NULL_HANDLE; }
	};

	SwapchainObjects sc{};
	createSwapchainObjects(sc, VK_NULL_HANDLE);

	VkRenderPass renderPass = createRenderPass(device, sc.colorFormat);
	PipelineObjects circlePipeline = createCirclePipeline(device, sc.colorFormat, renderPass);
	PipelineObjects textPipeline = createTextPipeline(device, sc.colorFormat, renderPass);

	// Create geometry buffers (quad vertices) and instance buffer
	GeometryBuffers geom{};
	{
		const float quadVertices[] = {
			-1.0f, -1.0f,
			 1.0f, -1.0f,
			 1.0f,  1.0f,
			-1.0f, -1.0f,
			 1.0f,  1.0f,
			-1.0f,  1.0f,
		};
		VkDeviceSize vbSize = sizeof(quadVertices);
		createBuffer(physical, device, vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, geom.quadVertexBuffer);
		std::memcpy(mapMemory(device, geom.quadVertexBuffer.memory, vbSize), quadVertices, vbSize);
		unmapMemory(device, geom.quadVertexBuffer.memory);

		// Instance buffer initial capacity; will update every frame
		VkDeviceSize ibSize = sizeof(InstanceLayoutCPU) * 2048;
		createBuffer(physical, device, ibSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, geom.instanceBuffer);
	}

	HudGeometry hud{};
	{
		VkDeviceSize vbSize = sizeof(TextVertex) * HUD_INITIAL_VERTEX_CAPACITY;
		createBuffer(physical, device, vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, hud.vertexBuffer);
		hud.bufferSize = vbSize;
	}

	// Framebuffers
	sc.framebuffers.resize(sc.imageViews.size());
	for (size_t i = 0; i < sc.imageViews.size(); ++i) {
		VkImageView attachments[] = { sc.imageViews[i] };
		VkFramebufferCreateInfo fci{};
		fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fci.renderPass = renderPass;
		fci.attachmentCount = 1;
		fci.pAttachments = attachments;
		fci.width = sc.extent.width;
		fci.height = sc.extent.height;
		fci.layers = 1;
		if (vkCreateFramebuffer(device, &fci, nullptr, &sc.framebuffers[i]) != VK_SUCCESS) {
			std::fprintf(stderr, "Failed to create framebuffer\n");
			return 7;
		}
	}

	// Command pool and buffers
	VkCommandPool commandPool = VK_NULL_HANDLE;
	{
		VkCommandPoolCreateInfo cpci{};
		cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cpci.queueFamilyIndex = graphicsQueueFamily;
		cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(device, &cpci, nullptr, &commandPool) != VK_SUCCESS) {
			std::fprintf(stderr, "Failed to create command pool\n");
			return 8;
		}
	}

	std::vector<VkCommandBuffer> commandBuffers(sc.framebuffers.size());
	{
		VkCommandBufferAllocateInfo cbai{};
		cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbai.commandPool = commandPool;
		cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
		if (vkAllocateCommandBuffers(device, &cbai, commandBuffers.data()) != VK_SUCCESS) {
			std::fprintf(stderr, "Failed to allocate command buffers\n");
			return 9;
		}
	}

	// Sync primitives per frame (use same count as swapchain images)
	std::vector<VkSemaphore> imageAvailableSemaphores(commandBuffers.size());
	std::vector<VkSemaphore> renderFinishedSemaphores(commandBuffers.size());
	std::vector<VkFence> inFlightFences(commandBuffers.size());
	{
		VkSemaphoreCreateInfo sci{}; sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		for (size_t i = 0; i < commandBuffers.size(); ++i) {
			vkCreateSemaphore(device, &sci, nullptr, &imageAvailableSemaphores[i]);
			vkCreateSemaphore(device, &sci, nullptr, &renderFinishedSemaphores[i]);
			vkCreateFence(device, &fci, nullptr, &inFlightFences[i]);
		}
	}

	// Create descriptor pool for texture atlases
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;

	VkDescriptorPool descriptorPool;
	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		std::fprintf(stderr, "Failed to create descriptor pool\n");
		return 10;
	}

	std::cout << "Vulkan initialized and swapchain created (MoltenVK)" << std::endl;
	std::cout.flush();

	// Initialize image manager
	std::cout << "Initializing image manager..." << std::endl; std::cout.flush();
	ImageManager imageManager{};
	initImageManager(imageManager, physical, device, commandPool, graphicsQueue);

	// Create descriptor set for texture atlas
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &circlePipeline.descriptorSetLayout;

	if (vkAllocateDescriptorSets(device, &allocInfo, &imageManager.atlas.descriptorSet) != VK_SUCCESS) {
		std::fprintf(stderr, "Failed to allocate descriptor sets\n");
		return 11;
	}

	// Update descriptor set
	VkDescriptorImageInfo imageInfo{};
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	imageInfo.imageView = imageManager.atlas.atlasArray.view;
	imageInfo.sampler = imageManager.atlas.sampler;

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = imageManager.atlas.descriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pImageInfo = &imageInfo;

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

	// Initialize simulation
	std::cout << "Creating simulation..." << std::endl; std::cout.flush();
	Simulation sim{};
	// Use current swapchain extent as world size
	sim.worldWidth = static_cast<float>(sc.extent.width);
	sim.worldHeight = static_cast<float>(sc.extent.height);
	sim.imageManager = &imageManager;
	std::cout << "Loading bias config..." << std::endl; std::cout.flush();
	sim.loadBiasConfig("bias.txt");
	std::cout << "Initializing from assets..." << std::endl; std::cout.flush();
	sim.initializeFromAssets("assets", sim.maxPlayers);
	std::cout << "Simulation initialized!" << std::endl; std::cout.flush();

	uint32_t frameCount = 0;
	uint32_t lastAliveCount = sim.aliveCount();
	std::cout << "Alive count: " << lastAliveCount << std::endl; std::cout.flush();
	std::cout << "Starting battle royale with " << lastAliveCount << " players!\n";
	std::cout.flush();

	std::vector<InstanceLayoutCPU> cpuInstances;
	std::vector<TextVertex> hudVertices;
	hudVertices.reserve(1024);

	uint32_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_C(1'000'000'000));
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		uint32_t imageIndex = 0;
		VkResult acq = vkAcquireNextImageKHR(device, sc.swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
			vkDeviceWaitIdle(device);
			destroySwapchainObjects(sc);
			vkDestroyRenderPass(device, renderPass, nullptr);
			createSwapchainObjects(sc, VK_NULL_HANDLE);
			renderPass = createRenderPass(device, sc.colorFormat);
			// Recreate framebuffers
			sc.framebuffers.resize(sc.imageViews.size());
			for (size_t i = 0; i < sc.imageViews.size(); ++i) {
				VkImageView attachments[] = { sc.imageViews[i] };
				VkFramebufferCreateInfo fci2{};
				fci2.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				fci2.renderPass = renderPass;
				fci2.attachmentCount = 1;
				fci2.pAttachments = attachments;
				fci2.width = sc.extent.width;
				fci2.height = sc.extent.height;
				fci2.layers = 1;
				vkCreateFramebuffer(device, &fci2, nullptr, &sc.framebuffers[i]);
			}
			// Reallocate command buffers if count changed
			vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
			commandBuffers.assign(sc.framebuffers.size(), VK_NULL_HANDLE);
			VkCommandBufferAllocateInfo cbai2{};
			cbai2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cbai2.commandPool = commandPool;
			cbai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cbai2.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
			vkAllocateCommandBuffers(device, &cbai2, commandBuffers.data());
			// Recreate sync objects to match count
			for (auto s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
			for (auto s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
			for (auto f : inFlightFences) vkDestroyFence(device, f, nullptr);
			imageAvailableSemaphores.assign(commandBuffers.size(), VK_NULL_HANDLE);
			renderFinishedSemaphores.assign(commandBuffers.size(), VK_NULL_HANDLE);
			inFlightFences.assign(commandBuffers.size(), VK_NULL_HANDLE);
			VkSemaphoreCreateInfo sci2{}; sci2.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			VkFenceCreateInfo fci3{}; fci3.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fci3.flags = VK_FENCE_CREATE_SIGNALED_BIT;
			for (size_t i = 0; i < commandBuffers.size(); ++i) {
				vkCreateSemaphore(device, &sci2, nullptr, &imageAvailableSemaphores[i]);
				vkCreateSemaphore(device, &sci2, nullptr, &renderFinishedSemaphores[i]);
				vkCreateFence(device, &fci3, nullptr, &inFlightFences[i]);
			}
			currentFrame = 0;
			continue;
		}

		// Record command buffer for this image
		VkCommandBuffer cmd = commandBuffers[imageIndex];
		vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkBeginCommandBuffer(cmd, &begin);

		VkClearValue clearColor{};
		clearColor.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

		VkRenderPassBeginInfo rpbi{};
		rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpbi.renderPass = renderPass;
		rpbi.framebuffer = sc.framebuffers[imageIndex];
		rpbi.renderArea.offset = {0, 0};
		rpbi.renderArea.extent = sc.extent;
		rpbi.clearValueCount = 1;
		rpbi.pClearValues = &clearColor;
		vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

		// Set dynamic viewport/scissor
		VkViewport viewport{}; viewport.x = 0.0f; viewport.y = 0.0f; viewport.width = static_cast<float>(sc.extent.width); viewport.height = static_cast<float>(sc.extent.height); viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
		VkRect2D scissor{}; scissor.offset = {0,0}; scissor.extent = sc.extent;
		vkCmdSetViewport(cmd, 0, 1, &viewport);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, circlePipeline.pipeline);

		// Bind descriptor set for texture atlas
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, circlePipeline.layout, 0, 1, &imageManager.atlas.descriptorSet, 0, nullptr);

		// Push constants: viewport size
		float vp[2] = { static_cast<float>(sc.extent.width), static_cast<float>(sc.extent.height) };
		vkCmdPushConstants(cmd, circlePipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vp), vp);

		// Update simulation and instance buffer
		const float dt = 1.0f / 120.0f; // fixed step
		updateImageManager(imageManager); // Process pending texture uploads
		sim.step(dt);
		sim.updateImageTiers(); // Update image loading tiers based on radius
		sim.writeInstances(cpuInstances);

		// Print status updates periodically
		frameCount++;
		uint32_t currentAlive = sim.aliveCount();
		hudVertices.clear();
		const float hudScale = 2.0f;
		const std::array<float, 4> hudShadowColor{0.0f, 0.0f, 0.0f, 0.6f};
		const std::array<float, 4> hudMainColor{1.0f, 1.0f, 1.0f, 1.0f};
		std::string playersLeftText = "Players left: " + std::to_string(currentAlive);
		appendHudText(hudVertices, 24.0f + 2.0f, 32.0f + 2.0f, playersLeftText, hudScale, hudShadowColor);
		appendHudText(hudVertices, 24.0f, 32.0f, playersLeftText, hudScale, hudMainColor);

		if (sim.inVictory && sim.winnerIndex >= 0) {
			std::string winnerText = "Winner: " + sim.names[sim.winnerIndex];
			const float winnerScale = 3.0f;
			float width = static_cast<float>(hudMeasureWidth(winnerText)) * winnerScale;
			float height = static_cast<float>(hudMeasureHeight(winnerText)) * winnerScale;
			float baseX = 0.5f * (static_cast<float>(sc.extent.width) - width);
			float baseY = 0.5f * (static_cast<float>(sc.extent.height) - height);
			const std::array<float, 4> winnerShadow{0.0f, 0.0f, 0.0f, 0.6f};
			const std::array<float, 4> winnerColor{1.0f, 0.85f, 0.1f, 1.0f};
			appendHudText(hudVertices, baseX + 3.0f, baseY + 3.0f, winnerText, winnerScale, winnerShadow);
			appendHudText(hudVertices, baseX, baseY, winnerText, winnerScale, winnerColor);
		}

		VkDeviceSize hudDataSize = sizeof(TextVertex) * hudVertices.size();
		if (hudDataSize > hud.bufferSize) {
			if (hud.vertexBuffer.buffer != VK_NULL_HANDLE) {
				vkDestroyBuffer(device, hud.vertexBuffer.buffer, nullptr);
			}
			if (hud.vertexBuffer.memory != VK_NULL_HANDLE) {
				vkFreeMemory(device, hud.vertexBuffer.memory, nullptr);
			}
			VkDeviceSize newSize = std::max<VkDeviceSize>(hudDataSize, hud.bufferSize ? hud.bufferSize * 2 : sizeof(TextVertex) * HUD_INITIAL_VERTEX_CAPACITY);
			createBuffer(physical, device, newSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, hud.vertexBuffer);
			hud.bufferSize = newSize;
		}
		if (hudDataSize > 0) {
			void* hudPtr = mapMemory(device, hud.vertexBuffer.memory, hudDataSize);
			std::memcpy(hudPtr, hudVertices.data(), static_cast<size_t>(hudDataSize));
			unmapMemory(device, hud.vertexBuffer.memory);
		}

		if (frameCount % 120 == 0 || currentAlive != lastAliveCount) { // Every second or when count changes
			std::cout << "Players left: " << currentAlive;
			if (sim.inVictory && sim.winnerIndex >= 0) {
				std::cout << " | WINNER: " << sim.names[sim.winnerIndex];
			}
			std::cout << " | Files loaded: " << imageManager.loadSuccessCount.load() 
					  << " success, " << imageManager.loadFailCount.load() << " failed";
			std::cout << "\n";
			lastAliveCount = currentAlive;
		}
		geom.instanceCount = static_cast<uint32_t>(cpuInstances.size());
		VkDeviceSize ibSize = sizeof(InstanceLayoutCPU) * cpuInstances.size();
		if (ibSize > 0) {
			void* ptr = mapMemory(device, geom.instanceBuffer.memory, ibSize);
			std::memcpy(ptr, cpuInstances.data(), static_cast<size_t>(ibSize));
			unmapMemory(device, geom.instanceBuffer.memory);
		}

		// Bind vertex buffers and draw instanced quads
		VkBuffer vbs[] = { geom.quadVertexBuffer.buffer, geom.instanceBuffer.buffer };
		VkDeviceSize offs[] = { 0, 0 };
		vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offs);
		vkCmdDraw(cmd, 6, geom.instanceCount, 0, 0);

		if (!hudVertices.empty()) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline.pipeline);
			vkCmdPushConstants(cmd, textPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vp), vp);
			VkBuffer hudVbs[] = { hud.vertexBuffer.buffer };
			VkDeviceSize hudOffs[] = { 0 };
			vkCmdBindVertexBuffers(cmd, 0, 1, hudVbs, hudOffs);
			vkCmdDraw(cmd, static_cast<uint32_t>(hudVertices.size()), 1, 0, 0);
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);

		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSubmitInfo submit{};
		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.waitSemaphoreCount = 1;
		submit.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
		submit.pWaitDstStageMask = waitStages;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &cmd;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
		vkQueueSubmit(graphicsQueue, 1, &submit, inFlightFences[currentFrame]);

		VkPresentInfoKHR present{};
		present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present.waitSemaphoreCount = 1;
		present.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];
		present.swapchainCount = 1;
		present.pSwapchains = &sc.swapchain;
		present.pImageIndices = &imageIndex;
		VkResult pres = vkQueuePresentKHR(presentQueue, &present);
		if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
			// Trigger recreation on next loop
		}

		currentFrame = (currentFrame + 1) % static_cast<uint32_t>(commandBuffers.size());
	}

	vkDeviceWaitIdle(device);
	destroyImageManager(imageManager);
	vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	for (auto f : inFlightFences) vkDestroyFence(device, f, nullptr);
	for (auto s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
	for (auto s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
	vkDestroyBuffer(device, geom.instanceBuffer.buffer, nullptr);
	vkFreeMemory(device, geom.instanceBuffer.memory, nullptr);
	vkDestroyBuffer(device, geom.quadVertexBuffer.buffer, nullptr);
	vkFreeMemory(device, geom.quadVertexBuffer.memory, nullptr);
	vkDestroyPipeline(device, circlePipeline.pipeline, nullptr);
	vkDestroyPipelineLayout(device, circlePipeline.layout, nullptr);
	vkDestroyDescriptorSetLayout(device, circlePipeline.descriptorSetLayout, nullptr);
	if (textPipeline.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, textPipeline.pipeline, nullptr);
	if (textPipeline.layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, textPipeline.layout, nullptr);
	if (textPipeline.descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, textPipeline.descriptorSetLayout, nullptr);
	if (hud.vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, hud.vertexBuffer.buffer, nullptr);
	if (hud.vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, hud.vertexBuffer.memory, nullptr);
	vkDestroyCommandPool(device, commandPool, nullptr);
	destroySwapchainObjects(sc);
	vkDestroyRenderPass(device, renderPass, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
