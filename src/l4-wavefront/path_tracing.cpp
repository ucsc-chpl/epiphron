#include <assert.h>
#include <chrono>
#include <iostream>
#include <easyvk.h>
#include <vector>

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

int main(int argc, char* argv[]) {
    auto deviceIndex = 0;

    // Query device properties.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    std::cout << "Using device: " << deviceName << "\n";
    device.teardown();
    instance.teardown();

    // Image dimensions
    int width = 640;
    int height = 480;
    std::vector<uint32_t> imageBuffer(640 * 480, 0);

     // Loop through the imageBuffer and set the alpha component to 255 (fully opaque)
    for (size_t i = 0; i < imageBuffer.size(); ++i) {
        imageBuffer[i] |= 0xFF000000;  // Set the alpha component to 255
    }

    // Save the image buffer as a PNG file.
    if (!stbi_write_png("out.png", width, height, 4, imageBuffer.data(), width * sizeof(uint32_t))) {
        std::cerr << "Error saving PNG file.\n";
        return 1;
    }

	return 0;
}