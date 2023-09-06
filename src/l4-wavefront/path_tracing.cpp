#include <assert.h>
#include <chrono>
#include <iostream>
#include <easyvk.h>

#include "json.h"

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
    device.teardown();
    instance.teardown();

	return 0;
}