#include <assert.h>
#include <chrono>
#include <iostream>
#include <easyvk.h>
#include <vector>
#include <cmath>

#include "json.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION // This line must appear in one .cpp file
#include "stb_image_write.h"

#ifdef __ANDROID__
#define USE_VALIDATION_LAYERS false
#else
#define USE_VALIDATION_LAYERS true
#endif

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;


/**
 * Struct definition for a sphere. 
 * Matches the exact memory layout that the GPU expects.
*/
typedef struct GPUSphere {
    float pos[3];
    float r alignas(16);
} GPUSphere;


int main(int argc, char* argv[]) {
    auto deviceIndex = 0;

    // Query device properties.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    std::cout << "Using device: " << deviceName << "\n";

    // Load shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/megakernel.cinit"
    ;
    auto entry_point = "render";

    // Image dimensions
    int width = 640;
    int height = 480;
    auto image_buf = easyvk::Buffer(device, width * height, sizeof(uint32_t));
    image_buf.clear();
    auto image_buf_width = easyvk::Buffer(device, 1, sizeof(uint32_t));
    image_buf_width.store<uint32_t>(0, width);
    auto image_buf_height = easyvk::Buffer(device, 1, sizeof(uint32_t));
    image_buf_height.store<uint32_t>(0, height);

    // Define scene objects.
    std::vector<GPUSphere> spheres = {
        {{-10.0f, 0.0f, 20.0f}, 1.0f},
        {{0.0f, 0.0f, 20.0f}, 1.0f},
        {{10.0f, 0.0f, 20.0f}, 1.0f}
    };

    auto sphere_buf = easyvk::Buffer(device, spheres.size(), sizeof(GPUSphere));
    for (size_t i = 0; i < spheres.size(); i++) {
        sphere_buf.store<GPUSphere>(i, spheres[i]);
    }
    auto num_spheres_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    num_spheres_buf.store<uint32_t>(0, spheres.size());

    // Init shader program.
    std::vector<easyvk::Buffer> kernelInputs = {image_buf,
                                                image_buf_width,
                                                image_buf_height,
                                                sphere_buf,
                                                num_spheres_buf
                                                };
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    // Divide work so that we launch one thread per pixel.
    auto workgroupSize = 256;
    auto numWorkgroups = std::ceil((double) (width * height) / workgroupSize);
    std::cout << "numWorkgroups: " << numWorkgroups << ", workgroupSize: " << workgroupSize << std::endl;
    std::cout << "Total work size: " << numWorkgroups * workgroupSize << "\n";
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point);

    // Launch kernel.
    auto kernelTime = program.runWithDispatchTiming();
    std::cout << "Kernel time: " << kernelTime / (double) 1000.0 << "ms\n";

    // Copy the buffer to a local vector.
    // TODO: Add a getter method in Buffer returns a reference to the underlying buffer.
    std::vector<uint32_t> imageBuffer(width * height);
    // Loop through the imageBuffer and set the alpha component to 255 (fully opaque)
    for (size_t i = 0; i < imageBuffer.size(); ++i) {
        imageBuffer[i] = image_buf.load<uint32_t>(i);
    }

    // Save the image buffer as a PNG file.
    if (!stbi_write_png("out.png", width, height, 4, imageBuffer.data(), width * sizeof(uint32_t))) {
        std::cerr << "Error saving PNG file.\n";
        return 1;
    }

	return 0;
}