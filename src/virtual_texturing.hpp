#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <queue>
#include <mutex>

// Helper function for memory type finding
inline uint32_t findVirtualTextureMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	return UINT32_MAX;
}

// Virtual texturing system for massive texture datasets
// Only loads visible texture pages, reducing memory from GB to MB
struct VirtualTextureSystem {
	static constexpr uint32_t PAGE_SIZE = 128; // 128x128 pages
	static constexpr uint32_t CACHE_SIZE = 512; // 512 pages in physical memory
	static constexpr uint32_t INDIRECTION_SIZE = 4096; // 4K indirection texture
	
	struct Page {
		uint32_t virtualX, virtualY;
		uint32_t physicalIndex;
		uint64_t lastAccessFrame;
		std::atomic<bool> locked{false};
	};
	
	struct PhysicalPage {
		VkImage image = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		bool inUse = false;
		uint32_t virtualPageId = UINT32_MAX;
	};
	
	// Indirection texture (maps virtual -> physical pages)
	VkImage indirectionTexture = VK_NULL_HANDLE;
	VkImageView indirectionView = VK_NULL_HANDLE;
	VkDeviceMemory indirectionMemory = VK_NULL_HANDLE;
	
	// Physical page cache
	std::vector<PhysicalPage> physicalPages;
	std::queue<uint32_t> freePages;
	
	// Virtual to physical mapping
	std::unordered_map<uint64_t, uint32_t> virtualToPhysical;
	
	// Page request system
	struct PageRequest {
		uint32_t virtualX, virtualY;
		uint32_t imageId;
		float priority;
	};
	std::priority_queue<PageRequest> requestQueue;
	std::mutex requestMutex;
	
	// Feedback buffer for GPU to request pages
	VkBuffer feedbackBuffer = VK_NULL_HANDLE;
	VkDeviceMemory feedbackMemory = VK_NULL_HANDLE;
	void* mappedFeedback = nullptr;
	
	VkDevice device = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkCommandPool commandPool = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
};

// Shader code for virtual texture sampling
inline const char* getVirtualTextureSamplingCode() {
	return R"(
// Virtual texture sampling function
vec4 sampleVirtualTexture(vec2 uv, uint textureId) {
	// Calculate mip level based on derivatives
	vec2 dx = dFdx(uv);
	vec2 dy = dFdy(uv);
	float maxDerivative = max(length(dx), length(dy));
	float mipLevel = log2(maxDerivative * float(VIRTUAL_TEXTURE_SIZE));
	
	// Sample indirection texture
	vec4 indirection = texture(uIndirectionTexture, uv);
	
	// Decode physical page coordinates
	vec2 physicalUV = fract(uv * float(INDIRECTION_SIZE)) * indirection.zw + indirection.xy;
	
	// Write to feedback buffer for page requests
	if (indirection.a < 0.5) {
		// Page not loaded, request it
		atomicAdd(feedbackBuffer[textureId], 1);
	}
	
	// Sample from physical texture cache
	return texture(uPhysicalTextureArray, vec3(physicalUV, indirection.a));
}
)";
}

// Initialize virtual texture system
inline bool initVirtualTextures(VirtualTextureSystem& vt, VkPhysicalDevice physicalDevice, VkDevice device, VkCommandPool commandPool, VkQueue queue) {
	vt.device = device;
	vt.physicalDevice = physicalDevice;
	vt.commandPool = commandPool;
	vt.queue = queue;
	
	// Create indirection texture
	VkImageCreateInfo indirectionInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	indirectionInfo.imageType = VK_IMAGE_TYPE_2D;
	indirectionInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	indirectionInfo.extent = {VirtualTextureSystem::INDIRECTION_SIZE, VirtualTextureSystem::INDIRECTION_SIZE, 1};
	indirectionInfo.mipLevels = 1;
	indirectionInfo.arrayLayers = 1;
	indirectionInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	indirectionInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	indirectionInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	indirectionInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	indirectionInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	
	if (vkCreateImage(device, &indirectionInfo, nullptr, &vt.indirectionTexture) != VK_SUCCESS) {
		return false;
	}
	
	// Allocate memory for indirection texture
	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(device, vt.indirectionTexture, &memReqs);
	
	VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = findVirtualTextureMemoryType(physicalDevice, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	
	if (vkAllocateMemory(device, &allocInfo, nullptr, &vt.indirectionMemory) != VK_SUCCESS) {
		return false;
	}
	
	vkBindImageMemory(device, vt.indirectionTexture, vt.indirectionMemory, 0);
	
	// Create image view
	VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	viewInfo.image = vt.indirectionTexture;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;
	
	if (vkCreateImageView(device, &viewInfo, nullptr, &vt.indirectionView) != VK_SUCCESS) {
		return false;
	}
	
	// Initialize physical page cache
	vt.physicalPages.resize(VirtualTextureSystem::CACHE_SIZE);
	for (uint32_t i = 0; i < VirtualTextureSystem::CACHE_SIZE; ++i) {
		vt.freePages.push(i);
		
		// Create page texture
		VkImageCreateInfo pageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		pageInfo.imageType = VK_IMAGE_TYPE_2D;
		pageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		pageInfo.extent = {VirtualTextureSystem::PAGE_SIZE, VirtualTextureSystem::PAGE_SIZE, 1};
		pageInfo.mipLevels = 1;
		pageInfo.arrayLayers = 1;
		pageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		pageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		pageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		pageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		pageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		
		vkCreateImage(device, &pageInfo, nullptr, &vt.physicalPages[i].image);
		
		// Allocate and bind memory
		VkMemoryRequirements pageMemReqs;
		vkGetImageMemoryRequirements(device, vt.physicalPages[i].image, &pageMemReqs);
		
		VkMemoryAllocateInfo pageAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		pageAllocInfo.allocationSize = pageMemReqs.size;
		pageAllocInfo.memoryTypeIndex = findVirtualTextureMemoryType(physicalDevice, pageMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		
		vkAllocateMemory(device, &pageAllocInfo, nullptr, &vt.physicalPages[i].memory);
		vkBindImageMemory(device, vt.physicalPages[i].image, vt.physicalPages[i].memory, 0);
		
		// Create view
		VkImageViewCreateInfo pageViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		pageViewInfo.image = vt.physicalPages[i].image;
		pageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		pageViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		pageViewInfo.subresourceRange = viewInfo.subresourceRange;
		
		vkCreateImageView(device, &pageViewInfo, nullptr, &vt.physicalPages[i].view);
	}
	
	// Create feedback buffer
	VkBufferCreateInfo feedbackInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	feedbackInfo.size = sizeof(uint32_t) * 8192; // Track up to 8K textures
	feedbackInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	feedbackInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	vkCreateBuffer(device, &feedbackInfo, nullptr, &vt.feedbackBuffer);
	
	// Allocate and map feedback buffer
	VkMemoryRequirements feedbackMemReqs;
	vkGetBufferMemoryRequirements(device, vt.feedbackBuffer, &feedbackMemReqs);
	
	VkMemoryAllocateInfo feedbackAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	feedbackAllocInfo.allocationSize = feedbackMemReqs.size;
	feedbackAllocInfo.memoryTypeIndex = findVirtualTextureMemoryType(physicalDevice, feedbackMemReqs.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	
	vkAllocateMemory(device, &feedbackAllocInfo, nullptr, &vt.feedbackMemory);
	vkBindBufferMemory(device, vt.feedbackBuffer, vt.feedbackMemory, 0);
	vkMapMemory(device, vt.feedbackMemory, 0, feedbackInfo.size, 0, &vt.mappedFeedback);
	
	return true;
}
