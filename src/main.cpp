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

	std::cout << "Vulkan initialized successfully (MoltenVK)" << std::endl;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}

	vkDeviceWaitIdle(device);
	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
