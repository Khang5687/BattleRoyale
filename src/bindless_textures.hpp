#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>

// Bindless texture system for massive performance improvement
// This eliminates texture binding overhead and allows direct GPU indexing
struct BindlessTextureSystem {
	static constexpr uint32_t MAX_BINDLESS_TEXTURES = 16384;
	static constexpr uint32_t INVALID_TEXTURE_INDEX = UINT32_MAX;
	
	VkDevice device = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorSetLayout bindlessLayout = VK_NULL_HANDLE;
	VkDescriptorSet bindlessSet = VK_NULL_HANDLE;
	
	// Individual texture resources
	struct TextureResource {
		VkImage image = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		uint32_t width = 0;
		uint32_t height = 0;
		bool inUse = false;
	};
	
	std::vector<TextureResource> textures;
	std::vector<uint32_t> freeIndices;
	std::unordered_map<uint32_t, uint32_t> imageIdToIndex;
	
	// Streaming system integration
	struct StreamingRequest {
		uint32_t imageId;
		uint32_t textureIndex;
		std::vector<uint8_t> data;
		uint32_t width;
		uint32_t height;
	};
	
	std::queue<StreamingRequest> uploadQueue;
	std::mutex uploadMutex;
	
	// Performance metrics
	struct Metrics {
		uint32_t texturesLoaded = 0;
		uint32_t texturesInUse = 0;
		VkDeviceSize memoryUsed = 0;
		VkDeviceSize memoryBudget = 0;
	} metrics;
};

// Initialize bindless texture system with descriptor indexing
inline bool initBindlessTextures(BindlessTextureSystem& system, VkPhysicalDevice physicalDevice, VkDevice device, VkDescriptorPool pool) {
	system.device = device;
	system.descriptorPool = pool;
	
	// Check for required extensions
	VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
	features2.pNext = &indexingFeatures;
	vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
	
	if (!indexingFeatures.descriptorBindingPartiallyBound || 
		!indexingFeatures.descriptorBindingUpdateUnusedWhilePending ||
		!indexingFeatures.runtimeDescriptorArray) {
		return false; // Device doesn't support bindless textures
	}
	
	// Create bindless descriptor set layout
	VkDescriptorSetLayoutBinding binding{};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	binding.descriptorCount = BindlessTextureSystem::MAX_BINDLESS_TEXTURES;
	binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	
	VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
	std::vector<VkDescriptorBindingFlags> flags = {
		VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
		VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
	};
	bindingFlags.bindingCount = 1;
	bindingFlags.pBindingFlags = flags.data();
	
	VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	layoutInfo.pNext = &bindingFlags;
	layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &binding;
	
	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &system.bindlessLayout) != VK_SUCCESS) {
		return false;
	}
	
	// Allocate descriptor set
	VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &system.bindlessLayout;
	
	if (vkAllocateDescriptorSets(device, &allocInfo, &system.bindlessSet) != VK_SUCCESS) {
		return false;
	}
	
	// Initialize texture slots
	system.textures.resize(BindlessTextureSystem::MAX_BINDLESS_TEXTURES);
	system.freeIndices.reserve(BindlessTextureSystem::MAX_BINDLESS_TEXTURES);
	for (uint32_t i = 0; i < BindlessTextureSystem::MAX_BINDLESS_TEXTURES; ++i) {
		system.freeIndices.push_back(BindlessTextureSystem::MAX_BINDLESS_TEXTURES - 1 - i);
	}
	
	return true;
}

// Allocate a texture slot and return its bindless index
inline uint32_t allocateBindlessTexture(BindlessTextureSystem& system, uint32_t imageId, uint32_t width, uint32_t height) {
	if (system.freeIndices.empty()) {
		return BindlessTextureSystem::INVALID_TEXTURE_INDEX;
	}
	
	uint32_t index = system.freeIndices.back();
	system.freeIndices.pop_back();
	
	system.imageIdToIndex[imageId] = index;
	system.textures[index].width = width;
	system.textures[index].height = height;
	system.textures[index].inUse = true;
	
	system.metrics.texturesInUse++;
	
	return index;
}

// Update bindless descriptor with new texture
inline void updateBindlessTexture(BindlessTextureSystem& system, uint32_t index, VkImageView imageView) {
	VkDescriptorImageInfo imageInfo{};
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	imageInfo.imageView = imageView;
	
	VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
	write.dstSet = system.bindlessSet;
	write.dstBinding = 0;
	write.dstArrayElement = index;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	write.pImageInfo = &imageInfo;
	
	vkUpdateDescriptorSets(system.device, 1, &write, 0, nullptr);
}

// Get bindless index for an image ID
inline uint32_t getBindlessIndex(const BindlessTextureSystem& system, uint32_t imageId) {
	auto it = system.imageIdToIndex.find(imageId);
	return (it != system.imageIdToIndex.end()) ? it->second : BindlessTextureSystem::INVALID_TEXTURE_INDEX;
}
