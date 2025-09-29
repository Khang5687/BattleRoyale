#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>

// GPU-driven rendering system for massive performance gains
// Moves visibility culling and draw call generation to GPU
struct GPUDrivenRenderer {
	static constexpr uint32_t MAX_INSTANCES = 65536;
	static constexpr uint32_t DRAW_INDIRECT_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
	
	// GPU buffers
	VkBuffer instanceDataBuffer = VK_NULL_HANDLE;
	VkDeviceMemory instanceDataMemory = VK_NULL_HANDLE;
	
	VkBuffer drawIndirectBuffer = VK_NULL_HANDLE;
	VkDeviceMemory drawIndirectMemory = VK_NULL_HANDLE;
	
	VkBuffer drawCountBuffer = VK_NULL_HANDLE;
	VkDeviceMemory drawCountMemory = VK_NULL_HANDLE;
	
	VkBuffer visibilityBuffer = VK_NULL_HANDLE;
	VkDeviceMemory visibilityMemory = VK_NULL_HANDLE;
	
	// Compute pipeline for culling
	VkPipeline cullingPipeline = VK_NULL_HANDLE;
	VkPipelineLayout cullingPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout cullingDescriptorLayout = VK_NULL_HANDLE;
	VkDescriptorSet cullingDescriptorSet = VK_NULL_HANDLE;
	
	// Instance data that goes to GPU
	struct GPUInstanceData {
		float center[2];
		float radius;
		float imageLayer;
		float color[4];
		uint32_t textureIndex; // For bindless textures
		uint32_t flags;
		float _padding[2];
	};
	
	// Culling uniforms
	struct CullingUniforms {
		float viewMatrix[16];
		float projMatrix[16];
		float frustumPlanes[6][4];
		float screenSize[2];
		float zoomFactor;
		uint32_t instanceCount;
	};
	
	VkBuffer cullingUniformBuffer = VK_NULL_HANDLE;
	VkDeviceMemory cullingUniformMemory = VK_NULL_HANDLE;
	void* mappedCullingUniforms = nullptr;
	
	VkDevice device = VK_NULL_HANDLE;
	uint32_t maxDrawCount = 0;
};

// Compute shader for GPU culling
inline const char* getGPUCullingComputeShader() {
	return R"(
#version 450
#extension GL_ARB_shader_ballot : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct InstanceData {
	vec2 center;
	float radius;
	float imageLayer;
	vec4 color;
	uint textureIndex;
	uint flags;
	vec2 _padding;
};

struct DrawCommand {
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	int vertexOffset;
	uint firstInstance;
};

layout(set = 0, binding = 0) uniform CullingData {
	mat4 viewMatrix;
	mat4 projMatrix;
	vec4 frustumPlanes[6];
	vec2 screenSize;
	float zoomFactor;
	uint instanceCount;
} culling;

layout(set = 0, binding = 1, std430) readonly buffer InstanceBuffer {
	InstanceData instances[];
} instanceBuffer;

layout(set = 0, binding = 2, std430) writeonly buffer DrawIndirectBuffer {
	DrawCommand commands[];
} drawIndirect;

layout(set = 0, binding = 3, std430) buffer DrawCountBuffer {
	uint count;
} drawCount;

layout(set = 0, binding = 4, std430) writeonly buffer VisibilityBuffer {
	uint visibility[];
} visibilityBuffer;

// Optimized frustum culling
bool isVisible(vec2 center, float radius) {
	vec4 worldPos = vec4(center, 0.0, 1.0);
	
	// Transform to clip space
	vec4 clipPos = culling.projMatrix * culling.viewMatrix * worldPos;
	
	// Early rejection for behind camera
	if (clipPos.w < 0.0) return false;
	
	// Screen space culling
	vec2 screenPos = clipPos.xy / clipPos.w;
	float screenRadius = radius * culling.zoomFactor / clipPos.w;
	
	// Check screen bounds with radius
	if (screenPos.x + screenRadius < -1.0 || screenPos.x - screenRadius > 1.0 ||
		screenPos.y + screenRadius < -1.0 || screenPos.y - screenRadius > 1.0) {
		return false;
	}
	
	// Pixel size culling - skip if too small
	float pixelSize = screenRadius * culling.screenSize.x * 0.5;
	if (pixelSize < 0.5) return false; // Less than half pixel
	
	return true;
}

// Classify LOD level based on screen size
uint getLODLevel(float screenRadius) {
	float pixelRadius = screenRadius * culling.screenSize.x * 0.5;
	
	if (pixelRadius < 2.0) return 0; // Pixel dust
	if (pixelRadius < 10.0) return 1; // Simple shape
	if (pixelRadius < 50.0) return 2; // Basic texture
	return 3; // Full detail
}

void main() {
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= culling.instanceCount) return;
	
	InstanceData instance = instanceBuffer.instances[gid];
	
	// Perform visibility culling
	bool visible = isVisible(instance.center, instance.radius);
	
	if (visible) {
		// Atomically allocate a draw command
		uint drawIndex = atomicAdd(drawCount.count, 1);
		
		// Write draw command
		drawIndirect.commands[drawIndex].indexCount = 6; // Quad indices
		drawIndirect.commands[drawIndex].instanceCount = 1;
		drawIndirect.commands[drawIndex].firstIndex = 0;
		drawIndirect.commands[drawIndex].vertexOffset = 0;
		drawIndirect.commands[drawIndex].firstInstance = gid;
		
		// Mark as visible
		visibilityBuffer.visibility[gid] = 1;
		
		// Could also do LOD selection here
		uint lod = getLODLevel(instance.radius * culling.zoomFactor);
		instance.flags = (instance.flags & 0xFFFFFF00) | lod;
	} else {
		visibilityBuffer.visibility[gid] = 0;
	}
}
)";
}

// Helper function for memory type finding
inline uint32_t findMemoryType(VkPhysicalDevice physical, uint32_t typeBits, VkMemoryPropertyFlags props) {
	VkPhysicalDeviceMemoryProperties memProps{};
	vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}
	throw std::runtime_error("Suitable memory type not found");
}

// Helper function to create compute shader module
inline VkShaderModule createComputeShaderModule(VkDevice device, const char* shaderSource) {
	// For now, we'll assume the shader source is compiled SPIR-V
	// In reality, you'd need glslang or another compiler
	throw std::runtime_error("Compute shader compilation not implemented - requires glslc");
}

// Initialize GPU-driven rendering
inline bool initGPUDrivenRenderer(GPUDrivenRenderer& renderer, VkDevice device, VkPhysicalDevice physicalDevice, 
								  VkDescriptorPool descriptorPool, uint32_t maxInstances) {
	renderer.device = device;
	renderer.maxDrawCount = maxInstances;
	
	// Create instance data buffer
	VkBufferCreateInfo instanceBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	instanceBufferInfo.size = sizeof(GPUDrivenRenderer::GPUInstanceData) * maxInstances;
	instanceBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	instanceBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	if (vkCreateBuffer(device, &instanceBufferInfo, nullptr, &renderer.instanceDataBuffer) != VK_SUCCESS) {
		return false;
	}
	
	// Create draw indirect buffer
	VkBufferCreateInfo indirectBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	indirectBufferInfo.size = sizeof(VkDrawIndexedIndirectCommand) * maxInstances;
	indirectBufferInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	indirectBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	if (vkCreateBuffer(device, &indirectBufferInfo, nullptr, &renderer.drawIndirectBuffer) != VK_SUCCESS) {
		return false;
	}
	
	// Create draw count buffer
	VkBufferCreateInfo countBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	countBufferInfo.size = sizeof(uint32_t);
	countBufferInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	countBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	if (vkCreateBuffer(device, &countBufferInfo, nullptr, &renderer.drawCountBuffer) != VK_SUCCESS) {
		return false;
	}
	
	// Allocate memory for all buffers
	auto allocateBuffer = [&](VkBuffer buffer, VkDeviceMemory& memory, VkMemoryPropertyFlags properties) {
		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(device, buffer, &memReqs);
		
		VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		allocInfo.allocationSize = memReqs.size;
		allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReqs.memoryTypeBits, properties);
		
		vkAllocateMemory(device, &allocInfo, nullptr, &memory);
		vkBindBufferMemory(device, buffer, memory, 0);
	};
	
	allocateBuffer(renderer.instanceDataBuffer, renderer.instanceDataMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	allocateBuffer(renderer.drawIndirectBuffer, renderer.drawIndirectMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	allocateBuffer(renderer.drawCountBuffer, renderer.drawCountMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	
	// Create culling uniform buffer
	VkBufferCreateInfo uniformBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	uniformBufferInfo.size = sizeof(GPUDrivenRenderer::CullingUniforms);
	uniformBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	uniformBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	vkCreateBuffer(device, &uniformBufferInfo, nullptr, &renderer.cullingUniformBuffer);
	allocateBuffer(renderer.cullingUniformBuffer, renderer.cullingUniformMemory, 
				  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	
	// Map uniform buffer
	vkMapMemory(device, renderer.cullingUniformMemory, 0, uniformBufferInfo.size, 0, &renderer.mappedCullingUniforms);
	
	return true;
}

// Execute GPU culling pass
inline void executeGPUCulling(GPUDrivenRenderer& renderer, VkCommandBuffer cmd, uint32_t instanceCount) {
	// Reset draw count to 0
	vkCmdFillBuffer(cmd, renderer.drawCountBuffer, 0, sizeof(uint32_t), 0);
	
	// Memory barrier
	VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 1, &barrier, 0, nullptr, 0, nullptr);
	
	// Bind compute pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderer.cullingPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderer.cullingPipelineLayout,
						   0, 1, &renderer.cullingDescriptorSet, 0, nullptr);
	
	// Dispatch compute threads
	uint32_t groupCount = (instanceCount + 63) / 64;
	vkCmdDispatch(cmd, groupCount, 1, 1);
	
	// Barrier before drawing
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
	
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
						VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
						0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// Execute multi-draw indirect
inline void executeGPUDrivenDraw(GPUDrivenRenderer& renderer, VkCommandBuffer cmd, VkPipeline graphicsPipeline, VkPipelineLayout pipelineLayout) {
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
	
	// Use multi-draw indirect with count from GPU
	vkCmdDrawIndexedIndirectCount(cmd, renderer.drawIndirectBuffer, 0,
								 renderer.drawCountBuffer, 0,
								 renderer.maxDrawCount, DRAW_INDIRECT_STRIDE);
}
