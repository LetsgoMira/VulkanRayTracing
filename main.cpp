#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "models/bunny.obj";

const int MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t HISTORY_IMAGE_COUNT = 2;
const VkFormat HISTORY_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
// Must remain below the 64-entry traversal stack in shader.frag.
const int MAX_BVH_DEPTH = 48;

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " (VkResult " +
            std::to_string(static_cast<int>(result)) + ")");
    }
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct PushConstants {
    uint32_t sampleCount;
    uint32_t width;
    uint32_t height;
    uint32_t encodeSrgb;
    uint32_t frameSeed;
};
static_assert(sizeof(PushConstants) == 20, "Push constant ABI mismatch");

struct Vertex {
    alignas(16)glm::vec3 pos;
    alignas(16)glm::vec3 color = { 0,0,0 };
    uint32_t emissive = 0;
    float roughness = 1.0f;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
        return {{{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)}}};
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && emissive == other.emissive && roughness == other.roughness;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<uint32_t>()(vertex.emissive) << 1) ^ hash<float>()(vertex.roughness);
        }
    };
}

struct BVHNode {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息
    alignas(16)glm::vec3 AA;
    alignas(16)glm::vec3 BB;        // 碰撞盒
};

// Explicit checks for the std430 structs consumed by the fragment shader.
static_assert(std::is_standard_layout<Vertex>::value, "Vertex must be standard layout");
static_assert(sizeof(Vertex) == 48 && offsetof(Vertex, pos) == 0 &&
    offsetof(Vertex, color) == 16 && offsetof(Vertex, emissive) == 28 &&
    offsetof(Vertex, roughness) == 32, "Vertex/std430 ABI mismatch");
static_assert(sizeof(BVHNode) == 48 && offsetof(BVHNode, AA) == 16 &&
    offsetof(BVHNode, BB) == 32, "BVH/std430 ABI mismatch");

class HelloTriangleApplication {
public:
    void run(const std::string& modelPath, bool modelRequired) {
        try {
            initWindow();
            initVulkan(modelPath, modelRequired);
            mainLoop();
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    // One framebuffer for each (history output image, swapchain image) pair.
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<VkImage, HISTORY_IMAGE_COUNT> historyImages{};
    std::array<VkDeviceMemory, HISTORY_IMAGE_COUNT> historyMemory{};
    std::array<VkImageView, HISTORY_IMAGE_COUNT> historyViews{};
    VkSampler historySampler = VK_NULL_HANDLE;
    uint32_t historyIndex = 0;
    uint32_t sampleCount = 0;
    uint32_t frameSeed = 0;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> triangles;
    std::vector<BVHNode> BVHNodes;
    VkBuffer resourceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory resourceBufferMemory = VK_NULL_HANDLE;
    std::array<VkDeviceSize, 4> resourceOffsets{};
    std::array<VkDeviceSize, 4> resourceSizes{};
    std::vector<Vertex> screenVertices;
    std::vector<uint32_t> screenIndices;
    VkBuffer screenTrianglesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory screenTrianglesBufferMemory = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow() {
        if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed");
        if (glfwVulkanSupported() != GLFW_TRUE) throw std::runtime_error("Vulkan loader/driver unavailable");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Raytracing", nullptr, nullptr);
        if (!window) throw std::runtime_error("Window creation failed");
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan(const std::string& modelPath, bool modelRequired) {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createHistoryResources();
        createFramebuffers();
        loadModel(modelPath, modelRequired);
        if (triangles.empty() || triangles.size() > static_cast<size_t>(std::numeric_limits<int>::max() / 2)) {
            throw std::runtime_error("Scene has no triangles or exceeds BVH index limits");
        }
        createBVH(BVHNodes, 0, static_cast<int>(triangles.size()) - 1, 4, 0);
#ifndef NDEBUG
        validateBVH();
#endif
        createResourceBuffer();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window)) break;
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) vkDestroyFramebuffer(device, framebuffer, nullptr);
        swapChainFramebuffers.clear();
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
        descriptorSets.clear();
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
        vkDestroySampler(device, historySampler, nullptr);
        historySampler = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < HISTORY_IMAGE_COUNT; ++i) {
            vkDestroyImageView(device, historyViews[i], nullptr);
            vkDestroyImage(device, historyImages[i], nullptr);
            vkFreeMemory(device, historyMemory[i], nullptr);
            historyViews[i] = VK_NULL_HANDLE;
            historyImages[i] = VK_NULL_HANDLE;
            historyMemory[i] = VK_NULL_HANDLE;
        }
        for (auto semaphore : renderFinishedSemaphores) vkDestroySemaphore(device, semaphore, nullptr);
        renderFinishedSemaphores.clear();
        for (auto imageView : swapChainImageViews) vkDestroyImageView(device, imageView, nullptr);
        swapChainImageViews.clear();
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        swapChain = VK_NULL_HANDLE;
    }

    void cleanup() noexcept {
        if (device != VK_NULL_HANDLE) {
            // Vulkan 1.0 WSI convention; see README for present-fence limitations.
            vkDeviceWaitIdle(device);
            cleanupSwapChain();
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroyBuffer(device, screenTrianglesBuffer, nullptr);
            vkFreeMemory(device, screenTrianglesBufferMemory, nullptr);
            vkDestroyBuffer(device, resourceBuffer, nullptr);
            vkFreeMemory(device, resourceBufferMemory, nullptr);
            for (auto semaphore : imageAvailableSemaphores) vkDestroySemaphore(device, semaphore, nullptr);
            for (auto fence : inFlightFences) vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            if (debugMessenger != VK_NULL_HANDLE) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
            vkDestroySurfaceKHR(instance, surface, nullptr);
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        if (window) glfwDestroyWindow(window);
        window = nullptr;
        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            if (glfwWindowShouldClose(window)) return;
            glfwWaitEvents();
            glfwGetFramebufferSize(window, &width, &height);
        }
        checkVk(vkDeviceWaitIdle(device), "Wait before swapchain recreation failed");
        checkVk(vkResetCommandPool(device, commandPool, 0), "Command pool reset failed");
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createHistoryResources();
        createFramebuffers();
        createDescriptorPool();
        createDescriptorSets();
        createPresentSemaphores();
        framebufferResized = false;
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Ray Tracing";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "Device count query failed");

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "Device enumeration failed");

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        // Read-only SSBOs and nearest sampling require neither optional feature.

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        const VkCompositeAlphaFlagBitsKHR alphaModes[] = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        };
        for (auto mode : alphaModes) {
            if (swapChainSupport.capabilities.supportedCompositeAlpha & mode) {
                createInfo.compositeAlpha = mode;
                break;
            }
        }
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        checkVk(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr), "Swapchain image count query failed");
        swapChainImages.resize(imageCount);
        checkVk(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data()), "Swapchain image query failed");

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createRenderPass() {
        //changeAttachment:每一帧更新和读取的渲染结果
        VkAttachmentDescription changeAttachment{};
        changeAttachment.format = HISTORY_FORMAT;
        changeAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        changeAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        changeAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        changeAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        changeAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        changeAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        changeAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference changeAttachmentRef{};
        changeAttachmentRef.attachment = 0;
        changeAttachmentRef.layout = VK_IMAGE_LAYOUT_GENERAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 1;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 2> attachmentRefs{ changeAttachmentRef,colorAttachmentRef };

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 2;
        subpass.pColorAttachments = attachmentRefs.data();
        subpass.pDepthStencilAttachment = nullptr;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { changeAttachment, colorAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding vertexLayoutBinding = {
            0, // binding
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr
        };
        VkDescriptorSetLayoutBinding indexLayoutBinding = {
            1, // binding
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           1,
           VK_SHADER_STAGE_FRAGMENT_BIT,
           nullptr
        };
        VkDescriptorSetLayoutBinding triangleLayoutBinding = {
           2, // binding
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           1,
           VK_SHADER_STAGE_FRAGMENT_BIT,
           nullptr
        };
        VkDescriptorSetLayoutBinding BVHLayoutBinding = {
           3, // binding
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           1,
           VK_SHADER_STAGE_FRAGMENT_BIT,
           nullptr
        };

        VkDescriptorSetLayoutBinding samplerLayoutBinding = {
            4,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            nullptr
        };

        std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings{ vertexLayoutBinding ,indexLayoutBinding,triangleLayoutBinding,BVHLayoutBinding ,samplerLayoutBinding};
        VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            5,
            layoutBindings.data()
        };

        if (vkCreateDescriptorSetLayout(device, &descLayoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/pathtrace.vert.spv");
        auto fragShaderCode = readFile("shaders/pathtrace.frag.spv");

        struct ShaderModuleGuard {
            VkDevice device;
            VkShaderModule handle;
            ~ShaderModuleGuard() { vkDestroyShaderModule(device, handle, nullptr); }
        };
        ShaderModuleGuard vertShaderModule{device, createShaderModule(vertShaderCode)};
        ShaderModuleGuard fragShaderModule{device, createShaderModule(fragShaderCode)};

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule.handle;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule.handle;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState changeAttachment{};
        changeAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        changeAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorAttachment.blendEnable = VK_FALSE;

        std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{ changeAttachment ,colorAttachment };

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 2;
        colorBlending.pAttachments = blendAttachments.data();
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; // 适用于顶点和片段着色器
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(HISTORY_IMAGE_COUNT * swapChainImageViews.size(), VK_NULL_HANDLE);
        for (uint32_t history = 0; history < HISTORY_IMAGE_COUNT; ++history) {
            for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
                std::array<VkImageView, 2> attachments = {historyViews[history], swapChainImageViews[i]};
                VkFramebufferCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                info.renderPass = renderPass;
                info.attachmentCount = static_cast<uint32_t>(attachments.size());
                info.pAttachments = attachments.data();
                info.width = swapChainExtent.width;
                info.height = swapChainExtent.height;
                info.layers = 1;
                checkVk(vkCreateFramebuffer(device, &info, nullptr,
                    &swapChainFramebuffers[history * swapChainImageViews.size() + i]), "Framebuffer creation failed");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    void createHistoryResources() {
        for (uint32_t i = 0; i < HISTORY_IMAGE_COUNT; ++i) {
            createImage(swapChainExtent.width, swapChainExtent.height, HISTORY_FORMAT,
                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, historyImages[i], historyMemory[i]);
            historyViews[i] = createImageView(historyImages[i], HISTORY_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        VkCommandBuffer command = beginSingleTimeCommands();
        std::array<VkImageMemoryBarrier, HISTORY_IMAGE_COUNT> barriers{};
        for (uint32_t i = 0; i < HISTORY_IMAGE_COUNT; ++i) {
            auto& barrier = barriers[i];
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = historyImages[i];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
        VkClearColorValue zero{};
        for (uint32_t i = 0; i < HISTORY_IMAGE_COUNT; ++i) {
            vkCmdClearColorImage(command, historyImages[i], VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                &barriers[i].subresourceRange);
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
        endSingleTimeCommands(command);

        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        checkVk(vkCreateSampler(device, &info, nullptr, &historySampler), "History sampler creation failed");
        sampleCount = 0;
        historyIndex = 0;
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        checkVk(vkBindImageMemory(device, image, imageMemory, 0), "Image memory binding failed");
    }

    void loadModel(const std::string& modelPath, bool modelRequired) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        Vertex vertex{};
        vertex.pos = { -1,1,0.5 }; screenVertices.push_back(vertex);
        vertex.pos = { 1,1,0.5 }; screenVertices.push_back(vertex);
        vertex.pos = { 1,-1,0.5 }; screenVertices.push_back(vertex);
        vertex.pos = { -1,-1,0.5 }; screenVertices.push_back(vertex);

        screenIndices = { 0,1,2,0,2,3 };

        const bool modelExists = std::ifstream(modelPath).good();
        if (modelRequired || modelExists) {
            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str(), nullptr, true)) {
                throw std::runtime_error("OBJ load failed: " + modelPath + "\n" + warn + err);
            }
            if (!warn.empty()) std::cerr << warn << '\n';
            size_t modelIndexCount = 0;
            for (const auto& shape : shapes) modelIndexCount += shape.mesh.indices.size();
            if (modelIndexCount == 0) throw std::runtime_error("OBJ contains no triangles: " + modelPath);
        } else {
            std::cerr << "Optional model not found: " << modelPath << "; rendering the built-in room.\n";
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            if (shape.mesh.indices.size() % 3 != 0 ||
                std::any_of(shape.mesh.num_face_vertices.begin(), shape.mesh.num_face_vertices.end(),
                    [](auto count) { return count != 3; })) {
                throw std::runtime_error("OBJ triangulation failed");
            }
            for (const auto& index : shape.mesh.indices) {
                if (index.vertex_index < 0 ||
                    static_cast<size_t>(index.vertex_index) >= attrib.vertices.size() / 3) {
                    throw std::runtime_error("OBJ contains an out-of-range vertex index");
                }
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 0] * 6 + 0.2,
                    attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 1] * 6 - 1,
                    attrib.vertices[3 * static_cast<size_t>(index.vertex_index) + 2] * 6
                };

                if (!std::isfinite(vertex.pos.x) || !std::isfinite(vertex.pos.y) || !std::isfinite(vertex.pos.z) ||
                    glm::any(glm::greaterThan(glm::abs(vertex.pos), glm::vec3(1.0e6f)))) {
                    throw std::runtime_error("OBJ coordinates are non-finite or too large; rescale the model");
                }
                vertex.color = { 1,0.6,0.8 };
                vertex.roughness = 0.5;

                if (uniqueVertices.count(vertex) == 0) {
                    if (vertices.size() >= std::numeric_limits<uint32_t>::max() - 36u) {
                        throw std::runtime_error("Scene exceeds vertex index limits");
                    }
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
        uint32_t index = static_cast<uint32_t>(vertices.size());
        vertex.pos = { 1,-1,1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,-1,-1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,-1,1 };vertex.color = { 1,1,1 };vertices.push_back(vertex);
        vertex.pos = { 1,-1,1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,-1,-1 }; vertex.color = { 1,1,1 };  vertices.push_back(vertex);
        vertex.pos = { -1,-1,-1 }; vertex.color = { 1,1,1 };  vertices.push_back(vertex);

        vertex.pos = { 1,1,1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,-1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,1 }; vertex.color = { 1,1,1 };  vertices.push_back(vertex);
        vertex.pos = { 1,1,1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,1,-1 }; vertex.color = { 1,1,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,-1 }; vertex.color = { 1,1,1 };  vertices.push_back(vertex);

        vertex.pos = { -1,1,-1 }; vertex.color = { 1,0.5,0.5 };  vertices.push_back(vertex);
        vertex.pos = { 1,1,-1 }; vertex.color = { 1,0.5,0.5 }; vertices.push_back(vertex);
        vertex.pos = { 1,-1,-1 }; vertex.color = { 1,0.5,0.5 }; vertices.push_back(vertex);
        vertex.pos = { 1,-1,-1 }; vertex.color = { 1,0.5,0.5 }; vertices.push_back(vertex);
        vertex.pos = { -1,-1,-1 }; vertex.color = { 1,0.5,0.5 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,-1 }; vertex.color = { 1,0.5,0.5 }; vertices.push_back(vertex);

        vertex.pos = { -1,1,1 }; vertex.color = { 0.5,0.5,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,-1 }; vertex.color = { 0.5,0.5,1 };  vertices.push_back(vertex);
        vertex.pos = { -1,-1,-1 }; vertex.color = { 0.5,0.5,1 };  vertices.push_back(vertex);
        vertex.pos = { -1,-1,-1 }; vertex.color = { 0.5,0.5,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,-1,1 }; vertex.color = { 0.5,0.5,1 }; vertices.push_back(vertex);
        vertex.pos = { -1,1,1 }; vertex.color = { 0.5,0.5,1 }; vertices.push_back(vertex);

        vertex.pos = { 1,1,1 }; vertex.color = { 0,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,1,-1 }; vertex.color = { 0,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,-1,-1 }; vertex.color = { 0,1,1 };vertices.push_back(vertex);
        vertex.pos = { 1,-1,-1 }; vertex.color = { 0,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,-1,1 }; vertex.color = { 0,1,1 }; vertices.push_back(vertex);
        vertex.pos = { 1,1,1 }; vertex.color = { 0,1,1 };vertices.push_back(vertex);

        vertex.pos = { 0.4,0.99,0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        vertex.pos = { -0.4,0.99,-0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        vertex.pos = { -0.4,0.99,0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        vertex.pos = { 0.4,0.99,0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        vertex.pos = { 0.4,0.99,-0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        vertex.pos = { -0.4,0.99,-0.4 }; vertex.color = { 1,1,1 }; vertex.emissive = true; vertices.push_back(vertex);
        

        std::vector<uint32_t> v = { index,index + 1,index + 2,index + 3,index + 4,index + 5 ,index + 6,index + 7,index + 8 ,index + 9,index + 10,index + 11,
        index + 12,index + 13,index + 14 ,index + 15,index + 16,index + 17 ,index + 18,index + 19,index + 20 ,index + 21,index + 22,index + 23 ,
        index + 24,index + 25,index + 26 ,index + 27,index + 28,index + 29 ,index + 30,index + 31,index + 32 ,index + 33,index + 34,index + 35 };
        indices.insert(indices.end(), v.begin(), v.end());
        if (indices.size() / 3 > static_cast<size_t>(std::numeric_limits<int>::max() / 2)) {
            throw std::runtime_error("Scene exceeds BVH index limits");
        }
        for (size_t i = 0; i < indices.size(); i += 3)
            triangles.push_back(static_cast<uint32_t>(i / 3));
    }

    bool compareTriangles(uint32_t t1, uint32_t t2, int axis) const {
        // t1/t2 are triangle IDs, not indices into the vector being sorted.
        const glm::vec3 center1 = vertices[indices[3 * t1]].pos +
            vertices[indices[3 * t1 + 1]].pos + vertices[indices[3 * t1 + 2]].pos;
        const glm::vec3 center2 = vertices[indices[3 * t2]].pos +
            vertices[indices[3 * t2 + 1]].pos + vertices[indices[3 * t2 + 2]].pos;
        return center1[axis] == center2[axis] ? t1 < t2 : center1[axis] < center2[axis];
    }
    bool cmpx(uint32_t a, uint32_t b) const { return compareTriangles(a, b, 0); }
    bool cmpy(uint32_t a, uint32_t b) const { return compareTriangles(a, b, 1); }
    bool cmpz(uint32_t a, uint32_t b) const { return compareTriangles(a, b, 2); }

    int createBVH(std::vector<BVHNode>& nodes, int l, int r, int n, int depth) {
        if (l > r) throw std::runtime_error("Invalid BVH range");
        nodes.push_back(BVHNode());
        int id = static_cast<int>(nodes.size()) - 1;
        nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
        glm::vec3 AA(std::numeric_limits<float>::infinity());
        glm::vec3 BB(-std::numeric_limits<float>::infinity());
        for (int i = l; i <= r; ++i) {
            // 最小点 AA
            float minx = fmin(vertices[indices[3 * triangles[i]]].pos.x, fmin(vertices[indices[3 * triangles[i] + 1]].pos.x, vertices[indices[3 * triangles[i] + 2]].pos.x));
            float miny = fmin(vertices[indices[3 * triangles[i]]].pos.y, fmin(vertices[indices[3 * triangles[i] + 1]].pos.y, vertices[indices[3 * triangles[i] + 2]].pos.y));
            float minz = fmin(vertices[indices[3 * triangles[i]]].pos.z, fmin(vertices[indices[3 * triangles[i] + 1]].pos.z, vertices[indices[3 * triangles[i] + 2]].pos.z));
            AA.x = fmin(AA.x, minx);
            AA.y = fmin(AA.y, miny);
            AA.z = fmin(AA.z, minz);
            float maxx = fmax(vertices[indices[3 * triangles[i]]].pos.x, fmax(vertices[indices[3 * triangles[i] + 1]].pos.x, vertices[indices[3 * triangles[i] + 2]].pos.x));
            float maxy = fmax(vertices[indices[3 * triangles[i]]].pos.y, fmax(vertices[indices[3 * triangles[i] + 1]].pos.y, vertices[indices[3 * triangles[i] + 2]].pos.y));
            float maxz = fmax(vertices[indices[3 * triangles[i]]].pos.z, fmax(vertices[indices[3 * triangles[i] + 1]].pos.z, vertices[indices[3 * triangles[i] + 2]].pos.z));
            BB.x = fmax(BB.x, maxx);
            BB.y = fmax(BB.y, maxy);
            BB.z = fmax(BB.z, maxz);
        }
        nodes[id].AA = AA;
        nodes[id].BB = BB;
        // 不多于 n 个三角形 返回叶子节点
        if ((r - l + 1) <= n || depth >= MAX_BVH_DEPTH) {
            nodes[id].n = r - l + 1;
            nodes[id].index = l;
            return id;
        }
        // 否则递归建树(SAH优化)
        float Cost = INFINITY;
        int Axis = 0;
        int Split = l + (r - l) / 2;
        for (int axis = 0; axis < 3; ++axis) {
            // 分别按 x，y，z 轴排序
            if (axis == 0) std::sort(triangles.begin() + l, triangles.begin() + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
                return cmpx(t1, t2);  
                });
            if (axis == 1) std::sort(triangles.begin() + l, triangles.begin() + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
                return cmpy(t1, t2);  
                });
            if (axis == 2) std::sort(triangles.begin() + l, triangles.begin() + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
                return cmpz(t1, t2);  
                });
            // leftMax[i]: [l, i] 中最大的 xyz 值
            // leftMin[i]: [l, i] 中最小的 xyz 值
            std::vector<glm::vec3> leftMax(r - l + 1, glm::vec3(-INFINITY, -INFINITY, -INFINITY));
            std::vector<glm::vec3> leftMin(r - l + 1, glm::vec3(INFINITY, INFINITY, INFINITY));
            // 计算前缀 注意 i-l 以对齐到下标 0
            for (int i = l; i <= r; i++) {
                glm::vec3 p1 = vertices[indices[3 * triangles[i]]].pos;
                glm::vec3 p2 = vertices[indices[3 * triangles[i] + 1]].pos;
                glm::vec3 p3 = vertices[indices[3 * triangles[i] + 2]].pos;

                int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

                leftMax[i - l].x = fmax(leftMax[i - l - bias].x, fmax(p1.x, fmax(p2.x, p3.x)));
                leftMax[i - l].y = fmax(leftMax[i - l - bias].y, fmax(p1.y, fmax(p2.y, p3.y)));
                leftMax[i - l].z = fmax(leftMax[i - l - bias].z, fmax(p1.z, fmax(p2.z, p3.z)));

                leftMin[i - l].x = fmin(leftMin[i - l - bias].x, fmin(p1.x, fmin(p2.x, p3.x)));
                leftMin[i - l].y = fmin(leftMin[i - l - bias].y, fmin(p1.y, fmin(p2.y, p3.y)));
                leftMin[i - l].z = fmin(leftMin[i - l - bias].z, fmin(p1.z, fmin(p2.z, p3.z)));
            }
            std::vector<glm::vec3> rightMax(r - l + 1, glm::vec3(-INFINITY, -INFINITY, -INFINITY));
            std::vector<glm::vec3> rightMin(r - l + 1, glm::vec3(INFINITY, INFINITY, INFINITY));
            // 计算后缀 注意 i-l 以对齐到下标 0
            for (int i = r; i >= l; i--) {
                glm::vec3 p1 = vertices[indices[3 * triangles[i]]].pos;
                glm::vec3 p2 = vertices[indices[3 * triangles[i] + 1]].pos;
                glm::vec3 p3 = vertices[indices[3 * triangles[i] + 2]].pos;
                int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

                rightMax[i - l].x = fmax(rightMax[i - l + bias].x, fmax(p1.x, fmax(p2.x, p3.x)));
                rightMax[i - l].y = fmax(rightMax[i - l + bias].y, fmax(p1.y, fmax(p2.y, p3.y)));
                rightMax[i - l].z = fmax(rightMax[i - l + bias].z, fmax(p1.z, fmax(p2.z, p3.z)));

                rightMin[i - l].x = fmin(rightMin[i - l + bias].x, fmin(p1.x, fmin(p2.x, p3.x)));
                rightMin[i - l].y = fmin(rightMin[i - l + bias].y, fmin(p1.y, fmin(p2.y, p3.y)));
                rightMin[i - l].z = fmin(rightMin[i - l + bias].z, fmin(p1.z, fmin(p2.z, p3.z)));
            }
            // 遍历寻找分割
            float cost = INFINITY;
            int split = l;
            for (int i = l; i <= r - 1; i++) {
                float lenx, leny, lenz;
                // 左侧 [l, i]
                glm::vec3 leftAA = leftMin[i - l];
                glm::vec3 leftBB = leftMax[i - l];
                lenx = leftBB.x - leftAA.x;
                leny = leftBB.y - leftAA.y;
                lenz = leftBB.z - leftAA.z;
                float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
                float leftCost = leftS * (i - l + 1);

                // 右侧 [i+1, r]
                glm::vec3 rightAA = rightMin[i + 1 - l];
                glm::vec3 rightBB = rightMax[i + 1 - l];
                lenx = rightBB.x - rightAA.x;
                leny = rightBB.y - rightAA.y;
                lenz = rightBB.z - rightAA.z;
                float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
                float rightCost = rightS * (r - i);

                // 记录每个分割的最小答案
                float totalCost = leftCost + rightCost;
                if (totalCost < cost) {
                    cost = totalCost;
                    split = i;
                }
            }
            // 记录每个轴的最佳答案
            if (cost < Cost) {
                Cost = cost;
                Axis = axis;
                Split = split;
            }
        }
        // 按最佳轴分割
        if (Axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
            return cmpx(t1, t2);  // 调用成员函数
            });
        if (Axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
            return cmpy(t1, t2);  // 调用成员函数
            });
        if (Axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, [this](const uint32_t& t1, const uint32_t& t2) {
            return cmpz(t1, t2);  // 调用成员函数
            });

        // 递归
        int left = createBVH(nodes, l, Split, n, depth + 1);
        int right = createBVH(nodes, Split + 1, r, n, depth + 1);

        nodes[id].left = left;
        nodes[id].right = right;

        return id;
    }

#ifndef NDEBUG
    void validateBVH() const {
        std::vector<bool> visited(BVHNodes.size(), false);
        std::vector<bool> covered(triangles.size(), false);
        std::vector<bool> triangleIds(triangles.size(), false);
        std::vector<std::pair<int, int>> pending{{0, 0}};
        auto require = [](bool condition) {
            if (!condition) throw std::runtime_error("BVH invariant failed before GPU upload");
        };
        while (!pending.empty()) {
            const auto entry = pending.back();
            pending.pop_back();
            const int id = entry.first;
            const int depth = entry.second;
            require(id >= 0 && static_cast<size_t>(id) < BVHNodes.size());
            require(!visited[id] && depth <= MAX_BVH_DEPTH);
            visited[id] = true;
            const auto& node = BVHNodes[id];
            for (int axis = 0; axis < 3; ++axis) {
                require(std::isfinite(node.AA[axis]) && std::isfinite(node.BB[axis]) && node.AA[axis] <= node.BB[axis]);
            }
            if (node.n > 0) {
                require(node.index >= 0 && static_cast<size_t>(node.index) + node.n <= triangles.size());
                for (int i = node.index; i < node.index + node.n; ++i) {
                    require(!covered[i]);
                    covered[i] = true;
                    const uint32_t triangle = triangles[i];
                    require(triangle < triangleIds.size() && !triangleIds[triangle]);
                    triangleIds[triangle] = true;
                    for (uint32_t corner = 0; corner < 3; ++corner) {
                        const uint32_t vertex = indices[3 * triangle + corner];
                        require(vertex < vertices.size());
                        const auto& point = vertices[vertex].pos;
                        require(glm::all(glm::greaterThanEqual(point, node.AA)) &&
                            glm::all(glm::lessThanEqual(point, node.BB)));
                    }
                }
            } else {
                require(node.n == 0 && depth < MAX_BVH_DEPTH);
                for (int child : {node.left, node.right}) {
                    require(child > id && static_cast<size_t>(child) < BVHNodes.size());
                    require(glm::all(glm::greaterThanEqual(BVHNodes[child].AA, node.AA)) &&
                        glm::all(glm::lessThanEqual(BVHNodes[child].BB, node.BB)));
                    pending.emplace_back(child, depth + 1);
                }
            }
        }
        require(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }));
        require(std::all_of(covered.begin(), covered.end(), [](bool v) { return v; }));
    }
#endif

    void uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
        VkBuffer& destination, VkDeviceMemory& memory) {
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        try {
            createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMemory);
            void* mapped = nullptr;
            checkVk(vkMapMemory(device, stagingMemory, 0, size, 0, &mapped), "Staging map failed");
            std::memcpy(mapped, data, static_cast<size_t>(size));
            vkUnmapMemory(device, stagingMemory);
            createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, destination, memory);
            copyBuffer(staging, destination, size);
        } catch (...) {
            vkDeviceWaitIdle(device);
            vkDestroyBuffer(device, staging, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            throw;
        }
        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    void createResourceBuffer() {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        const VkDeviceSize alignment = std::max<VkDeviceSize>(1, properties.limits.minStorageBufferOffsetAlignment);
        resourceSizes = {sizeof(Vertex) * vertices.size(), sizeof(uint32_t) * indices.size(),
            sizeof(uint32_t) * triangles.size(), sizeof(BVHNode) * BVHNodes.size()};
        VkDeviceSize total = 0;
        for (size_t i = 0; i < resourceSizes.size(); ++i) {
            if (resourceSizes[i] == 0 || resourceSizes[i] > properties.limits.maxStorageBufferRange) {
                throw std::runtime_error("Scene storage buffer exceeds device range limit");
            }
            resourceOffsets[i] = ((total + alignment - 1) / alignment) * alignment;
            total = resourceOffsets[i] + resourceSizes[i];
        }
        if (total > std::numeric_limits<size_t>::max()) throw std::runtime_error("Scene buffer is too large");
        std::vector<char> packed(static_cast<size_t>(total), 0);
        const void* sources[] = {vertices.data(), indices.data(), triangles.data(), BVHNodes.data()};
        for (size_t i = 0; i < resourceSizes.size(); ++i) {
            std::memcpy(packed.data() + resourceOffsets[i], sources[i], static_cast<size_t>(resourceSizes[i]));
        }
        uploadBuffer(packed.data(), total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, resourceBuffer, resourceBufferMemory);
        const size_t vertexBytes = sizeof(Vertex) * screenVertices.size();
        std::vector<char> screen(vertexBytes + sizeof(uint32_t) * screenIndices.size());
        std::memcpy(screen.data(), screenVertices.data(), vertexBytes);
        std::memcpy(screen.data() + vertexBytes, screenIndices.data(), sizeof(uint32_t) * screenIndices.size());
        uploadBuffer(screen.data(), screen.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            screenTrianglesBuffer, screenTrianglesBufferMemory);
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> descPoolSizes{}; 
        descPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descPoolSizes[0].descriptorCount = 4 * HISTORY_IMAGE_COUNT;
        descPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descPoolSizes[1].descriptorCount = HISTORY_IMAGE_COUNT;
        VkDescriptorPoolCreateInfo descPoolInfo{};
        descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descPoolInfo.maxSets = static_cast<uint32_t>(HISTORY_IMAGE_COUNT);
        descPoolInfo.poolSizeCount = 2;
        descPoolInfo.pPoolSizes = descPoolSizes.data();

        if (vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(HISTORY_IMAGE_COUNT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(HISTORY_IMAGE_COUNT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(HISTORY_IMAGE_COUNT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        VkDescriptorBufferInfo vertexBufferInfo{};
        vertexBufferInfo.buffer = resourceBuffer;
        vertexBufferInfo.offset = 0;
        vertexBufferInfo.range = sizeof(Vertex) * vertices.size();

        VkDescriptorBufferInfo indexBufferInfo{};
        indexBufferInfo.buffer = resourceBuffer;
        indexBufferInfo.offset = resourceOffsets[1];
        indexBufferInfo.range = sizeof(uint32_t) * indices.size();

        VkDescriptorBufferInfo triangleBufferInfo{};
        triangleBufferInfo.buffer = resourceBuffer;
        triangleBufferInfo.offset = resourceOffsets[2];
        triangleBufferInfo.range = sizeof(uint32_t) * triangles.size();

        VkDescriptorBufferInfo BVHBufferInfo{};
        BVHBufferInfo.buffer = resourceBuffer;
        BVHBufferInfo.offset = resourceOffsets[3];
        BVHBufferInfo.range = sizeof(BVHNode) * BVHNodes.size();

        VkDescriptorImageInfo changeImgBufferInfo{};
        changeImgBufferInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        changeImgBufferInfo.sampler = historySampler;
        for (size_t i = 0; i < HISTORY_IMAGE_COUNT; i++) {
            changeImgBufferInfo.imageView = historyViews[1 - i];
            std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

            for (int j = 0; j < 5; ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = descriptorSets[i];
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                descriptorWrites[j].descriptorCount = 1;
            }
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

            descriptorWrites[0].pBufferInfo = &vertexBufferInfo;
            descriptorWrites[1].pBufferInfo = &indexBufferInfo;
            descriptorWrites[2].pBufferInfo = &triangleBufferInfo;
            descriptorWrites[3].pBufferInfo = &BVHBufferInfo;
            descriptorWrites[4].pImageInfo = &changeImgBufferInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        checkVk(vkBindBufferMemory(device, buffer, bufferMemory, 0), "Buffer memory binding failed");
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        checkVk(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer), "Command allocation failed");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Command begin failed");

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        checkVk(vkEndCommandBuffer(commandBuffer), "Command end failed");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        checkVk(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "Upload submit failed");
        checkVk(vkQueueWaitIdle(graphicsQueue), "Upload wait failed");

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // RAW: previous color writes -> current history reads.
        // WAR: previous history reads -> reuse of the other image as output.
        // Same graphics queue, including across different submissions.
        VkMemoryBarrier historyBarrier{};
        historyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        historyBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        historyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        const VkPipelineStageFlags historyStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        vkCmdPipelineBarrier(commandBuffer, historyStages, historyStages,
            0, 1, &historyBarrier, 0, nullptr, 0, nullptr);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[historyIndex * swapChainImageViews.size() + imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {1.0f, 1.0f, 1.0f, 1.0f} };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &screenTrianglesBuffer, offsets);

        vkCmdBindIndexBuffer(commandBuffer, screenTrianglesBuffer,sizeof(Vertex)*screenVertices.size(),  VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[historyIndex], 0, nullptr);

        const bool hardwareSrgb = swapChainImageFormat == VK_FORMAT_B8G8R8A8_SRGB ||
            swapChainImageFormat == VK_FORMAT_R8G8B8A8_SRGB;
        PushConstants pushConstants{sampleCount + 1, swapChainExtent.width, swapChainExtent.height,
            hardwareSrgb ? 0u : 1u, frameSeed};

        // 在命令缓冲区中推送常量
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(screenIndices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createPresentSemaphores() {
        renderFinishedSemaphores.resize(swapChainImages.size(), VK_NULL_HANDLE);
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (auto& semaphore : renderFinishedSemaphores) {
            checkVk(vkCreateSemaphore(device, &info, nullptr, &semaphore), "Present semaphore creation failed");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            checkVk(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]), "Acquire semaphore creation failed");
            checkVk(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]), "Frame fence creation failed");
        }
        createPresentSemaphores();
    }

    void drawFrame() {
        if (framebufferResized) {
            recreateSwapChain();
            return;
        }
        checkVk(vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX), "Frame wait failed");
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) checkVk(result, "Image acquire failed");
        const bool suboptimal = result == VK_SUBOPTIMAL_KHR;
        // Keep the counter exactly representable in float; restart only after 2^24 samples.
        if (sampleCount >= (1u << 24)) sampleCount = 0;
        checkVk(vkResetCommandBuffer(commandBuffers[currentFrame], 0), "Command buffer reset failed");
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        // Reset only once recording succeeded and a submission is ready.
        checkVk(vkResetFences(device, 1, &inFlightFences[currentFrame]), "Fence reset failed");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        // Acquire of this image makes its previous presentation semaphore reusable.
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        checkVk(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "Draw submit failed");
        ++sampleCount;
        ++frameSeed;
        historyIndex = 1 - historyIndex;

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        if (result != VK_SUCCESS && result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) {
            checkVk(result, "Presentation failed");
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || suboptimal || framebufferResized) {
            recreateSwapChain();
        }
        static auto lastUpdate = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastUpdate).count() >= 0.5) {
            const std::string title = "Vulkan Raytracing | " + std::to_string(sampleCount) + " samples/pixel";
            glfwSetWindowTitle(window, title.c_str());
            lastUpdate = now;
        }
    }

    VkShaderModule createShaderModule(const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED &&
            availableFormats[0].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        }
        for (VkFormat preferred : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB,
                                  VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
            for (const auto& format : availableFormats) {
                if (format.format == preferred && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return format;
            }
        }
        throw std::runtime_error("No supported SDR swapchain format");
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        checkVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities), "Surface capabilities query failed");

        uint32_t formatCount;
        checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr), "Surface format count query failed");

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data()), "Surface format query failed");
        }

        uint32_t presentModeCount;
        checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr), "Present mode count query failed");

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data()), "Present mode query failed");
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkFormatProperties historyProperties{};
        vkGetPhysicalDeviceFormatProperties(device, HISTORY_FORMAT, &historyProperties);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (swapChainAdequate) {
            const auto support = querySwapChainSupport(device);
            swapChainAdequate = (support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
            try { chooseSwapSurfaceFormat(support.formats); }
            catch (const std::runtime_error&) { swapChainAdequate = false; }
        }
        return indices.isComplete() && extensionsSupported && swapChainAdequate &&
            (historyProperties.optimalTilingFeatures & required) == required &&
            properties.limits.maxColorAttachments >= 2 &&
            properties.limits.maxPerStageDescriptorStorageBuffers >= 4 &&
            properties.limits.maxPushConstantsSize >= sizeof(PushConstants);
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        checkVk(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr), "Extension count query failed");

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        checkVk(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data()), "Extension query failed");

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            checkVk(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport), "Present support query failed");

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions || glfwExtensionCount == 0) throw std::runtime_error("GLFW returned no Vulkan surface extensions");

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        checkVk(vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "Layer count query failed");

        std::vector<VkLayerProperties> availableLayers(layerCount);
        checkVk(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()), "Layer query failed");

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<uint32_t> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open " + filename + "; run shaders/compile.bat manually from the project root first");
        const auto bytes = file.tellg();
        if (bytes < std::streamoff(20) || static_cast<uint64_t>(bytes) % sizeof(uint32_t) != 0) {
            throw std::runtime_error("Invalid SPIR-V size: " + filename);
        }
        std::vector<uint32_t> buffer(static_cast<size_t>(bytes) / sizeof(uint32_t));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(bytes)) ||
            buffer[0] != 0x07230203u) throw std::runtime_error("Invalid or truncated SPIR-V: " + filename);
        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main(int argc, char** argv) {
    HelloTriangleApplication app;

    try {
        if (argc > 2) throw std::runtime_error("Usage: VulkanLearn [model.obj]");
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: VulkanLearn [model.obj]\nRun from the project root. Without an OBJ, the built-in room is rendered.\n";
            return EXIT_SUCCESS;
        }
        app.run(argc == 2 ? argv[1] : MODEL_PATH, argc == 2);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}