#include <iostream>
#include <easyvk.h>
#include "json.h"

using ordered_json = nlohmann::ordered_json;

int main(int argc, char* argv[]) {
	// Initialize 
	auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

	// Cleanup instance
	instance.teardown();
	return 0;
}