#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <thread>

// Fast GPU texture compression for reduced memory bandwidth
struct TextureCompressionSystem {
	enum class CompressionFormat {
		BC1_RGB,   // 4:1 compression ratio (8 bytes per 4x4 block)
		BC3_RGBA,  // 4:1 compression ratio (16 bytes per 4x4 block)  
		BC7_RGBA,  // 4:1 compression ratio (16 bytes per 4x4 block, higher quality)
		ASTC_4x4,  // 4:1 compression ratio, better for mobile
		UNCOMPRESSED
	};
	
	struct CompressedTexture {
		std::vector<uint8_t> data;
		uint32_t width;
		uint32_t height;
		CompressionFormat format;
		uint32_t blockSize;
		VkFormat vkFormat;
	};
	
	// Worker threads for parallel compression
	std::vector<std::thread> compressionThreads;
	std::atomic<bool> shouldStop{false};
	
	// Compression queue
	struct CompressionJob {
		uint32_t imageId;
		std::vector<uint8_t> sourceData;
		uint32_t width;
		uint32_t height;
		CompressionFormat targetFormat;
		std::function<void(CompressedTexture)> callback;
	};
	
	std::queue<CompressionJob> jobQueue;
	std::mutex queueMutex;
	std::condition_variable queueCV;
	
	// Statistics
	std::atomic<uint64_t> bytesCompressed{0};
	std::atomic<uint64_t> bytesUncompressed{0};
};

// Get Vulkan format for compression format
inline VkFormat getVulkanFormat(TextureCompressionSystem::CompressionFormat format) {
	switch (format) {
		case TextureCompressionSystem::CompressionFormat::BC1_RGB:
			return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
		case TextureCompressionSystem::CompressionFormat::BC3_RGBA:
			return VK_FORMAT_BC3_UNORM_BLOCK;
		case TextureCompressionSystem::CompressionFormat::BC7_RGBA:
			return VK_FORMAT_BC7_UNORM_BLOCK;
		case TextureCompressionSystem::CompressionFormat::ASTC_4x4:
			return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
		default:
			return VK_FORMAT_R8G8B8A8_UNORM;
	}
}

// Check if device supports compression format
inline bool supportsCompressionFormat(VkPhysicalDevice physicalDevice, TextureCompressionSystem::CompressionFormat format) {
	VkFormatProperties props;
	vkGetPhysicalDeviceFormatProperties(physicalDevice, getVulkanFormat(format), &props);
	return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
}

// Fast BC1 compression (for RGB textures)
inline void compressBC1Block(const uint8_t* block, uint8_t* output) {
	// Find min/max colors in 4x4 block
	uint8_t minR = 255, minG = 255, minB = 255;
	uint8_t maxR = 0, maxG = 0, maxB = 0;
	
	for (int i = 0; i < 16; i++) {
		minR = std::min(minR, block[i * 4 + 0]);
		minG = std::min(minG, block[i * 4 + 1]);
		minB = std::min(minB, block[i * 4 + 2]);
		maxR = std::max(maxR, block[i * 4 + 0]);
		maxG = std::max(maxG, block[i * 4 + 1]);
		maxB = std::max(maxB, block[i * 4 + 2]);
	}
	
	// Convert to 565 format
	uint16_t color0 = ((maxR >> 3) << 11) | ((maxG >> 2) << 5) | (maxB >> 3);
	uint16_t color1 = ((minR >> 3) << 11) | ((minG >> 2) << 5) | (minB >> 3);
	
	// Ensure color0 > color1 for 4-color mode
	if (color0 < color1) {
		std::swap(color0, color1);
	}
	
	// Write colors
	*reinterpret_cast<uint16_t*>(output + 0) = color0;
	*reinterpret_cast<uint16_t*>(output + 2) = color1;
	
	// Generate indices (simplified - just use closest match)
	uint32_t indices = 0;
	for (int i = 0; i < 16; i++) {
		// Calculate distances to endpoints
		int dr0 = block[i * 4 + 0] - maxR;
		int dg0 = block[i * 4 + 1] - maxG;
		int db0 = block[i * 4 + 2] - maxB;
		int dist0 = dr0 * dr0 + dg0 * dg0 + db0 * db0;
		
		int dr1 = block[i * 4 + 0] - minR;
		int dg1 = block[i * 4 + 1] - minG;
		int db1 = block[i * 4 + 2] - minB;
		int dist1 = dr1 * dr1 + dg1 * dg1 + db1 * db1;
		
		// Choose closest endpoint
		uint32_t index = (dist0 < dist1) ? 0 : 1;
		indices |= (index << (i * 2));
	}
	
	*reinterpret_cast<uint32_t*>(output + 4) = indices;
}

// Compress texture to BC1 format
inline TextureCompressionSystem::CompressedTexture compressTextureBC1(const uint8_t* data, uint32_t width, uint32_t height) {
	TextureCompressionSystem::CompressedTexture result;
	result.width = width;
	result.height = height;
	result.format = TextureCompressionSystem::CompressionFormat::BC1_RGB;
	result.blockSize = 8; // BC1 uses 8 bytes per 4x4 block
	result.vkFormat = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
	
	// Calculate compressed size
	uint32_t blocksX = (width + 3) / 4;
	uint32_t blocksY = (height + 3) / 4;
	result.data.resize(blocksX * blocksY * 8);
	
	// Compress each 4x4 block in parallel
	#pragma omp parallel for
	for (uint32_t by = 0; by < blocksY; by++) {
		for (uint32_t bx = 0; bx < blocksX; bx++) {
			// Extract 4x4 block
			uint8_t block[64]; // 16 pixels * 4 channels
			for (uint32_t y = 0; y < 4; y++) {
				for (uint32_t x = 0; x < 4; x++) {
					uint32_t px = std::min(bx * 4 + x, width - 1);
					uint32_t py = std::min(by * 4 + y, height - 1);
					uint32_t srcIdx = (py * width + px) * 4;
					uint32_t dstIdx = (y * 4 + x) * 4;
					
					block[dstIdx + 0] = data[srcIdx + 0];
					block[dstIdx + 1] = data[srcIdx + 1];
					block[dstIdx + 2] = data[srcIdx + 2];
					block[dstIdx + 3] = data[srcIdx + 3];
				}
			}
			
			// Compress block
			uint32_t blockIdx = by * blocksX + bx;
			compressBC1Block(block, result.data.data() + blockIdx * 8);
		}
	}
	
	return result;
}

// Initialize compression system
inline void initTextureCompression(TextureCompressionSystem& system, uint32_t threadCount = 0) {
	if (threadCount == 0) {
		threadCount = std::thread::hardware_concurrency();
	}
	
	// Start compression worker threads
	for (uint32_t i = 0; i < threadCount; i++) {
		system.compressionThreads.emplace_back([&system]() {
			while (!system.shouldStop) {
				std::unique_lock<std::mutex> lock(system.queueMutex);
				system.queueCV.wait(lock, [&] { return !system.jobQueue.empty() || system.shouldStop; });
				
				if (system.shouldStop) break;
				
				auto job = std::move(system.jobQueue.front());
				system.jobQueue.pop();
				lock.unlock();
				
				// Perform compression
				TextureCompressionSystem::CompressedTexture compressed;
				if (job.targetFormat == TextureCompressionSystem::CompressionFormat::BC1_RGB) {
					compressed = compressTextureBC1(job.sourceData.data(), job.width, job.height);
				}
				
				// Update statistics
				system.bytesUncompressed += job.sourceData.size();
				system.bytesCompressed += compressed.data.size();
				
				// Call callback
				job.callback(std::move(compressed));
			}
		});
	}
}

// Submit texture for compression
inline void submitCompressionJob(TextureCompressionSystem& system, TextureCompressionSystem::CompressionJob job) {
	{
		std::lock_guard<std::mutex> lock(system.queueMutex);
		system.jobQueue.push(std::move(job));
	}
	system.queueCV.notify_one();
}

// Get compression ratio
inline float getCompressionRatio(const TextureCompressionSystem& system) {
	uint64_t uncompressed = system.bytesUncompressed.load();
	uint64_t compressed = system.bytesCompressed.load();
	return (uncompressed > 0) ? static_cast<float>(uncompressed) / static_cast<float>(compressed) : 1.0f;
}
