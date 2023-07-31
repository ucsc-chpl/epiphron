#include <assert.h>
#include <chrono>
#include <iostream>
#include <easyvk.h>
#include "json.h"

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;

double calculate_average(const std::vector<double>& values) {
    double sum = 0.0;
    int numElements = values.size();

    // Calculate the sum of all elements
    for (const double& value : values) {
        sum += value;
    }

    // Calculate and return the average
    return numElements > 0 ? sum / numElements : 0.0;
}

double calculate_std_dev(const std::vector<double>& values) {
    double mean = calculate_average(values);
    double squaredDifferenceSum = 0.0;
    int numElements = values.size();

    // Calculate the sum of squared differences from the mean
    for (const double& value : values) {
        double difference = value - mean;
        squaredDifferenceSum += difference * difference;
    }

    // Calculate and return the standard deviation
    return numElements > 0 ? std::sqrt(squaredDifferenceSum / numElements) : 0.0;
}


int main(int argc, char* argv[]) {
	// Set up instance.
	auto instance = easyvk::Instance(false);

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(0));

    // Loader shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/ticket_lock.cinit"
    ;
    auto entry_point = "ticket_lock_test";

    auto numWorkgroups = 32;
    auto workgroupSize = 32;

    // Set up buffers.
    auto next_ticket_buf = easyvk::Buffer(device, 1);
    auto now_serving_buf = easyvk::Buffer(device, 1);
    auto counter_buf = easyvk::Buffer(device, 1);
    auto histogram_buf = easyvk::Buffer(device, numWorkgroups * workgroupSize);
    next_ticket_buf.store(0, 0);
    now_serving_buf.store(0, 0);
    counter_buf.store(0, 0);
    for (int i = 0; i < numWorkgroups * workgroupSize; i++) {
        histogram_buf.store(i, 0);
    }

    std::vector<easyvk::Buffer> kernelInputs = {next_ticket_buf, now_serving_buf, histogram_buf, counter_buf};



    // Each thread increments the counter 256 times, so the value of the counter 
    // after the kernel runs should be numWorkgroups * workgroupSize * 256
    assert(counter_buf.load(0) == numWorkgroups * workgroupSize * 256);

	// Cleanup instance
	instance.teardown();
	return 0;
}