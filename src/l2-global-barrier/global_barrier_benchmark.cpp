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

double calculate_coeff_variation(const std::vector<double>& values) {
    double mean = calculate_average(values);
    if (mean == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sd = calculate_std_dev(values);
    return (sd / mean);
}


void ticket_lock_test() {
	// Set up instance.
	auto instance = easyvk::Instance(false);

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(0));
    std::cout << "Device name: " << device.properties.deviceName << "\n";

    // Loader shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/ticket_lock.cinit"
    ;
    auto entry_point = "ticket_lock_test";

    auto numWorkgroups = 32;
    auto workgroupSize = 1;

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

    std::vector<easyvk::Buffer> kernelInputs = {next_ticket_buf, now_serving_buf, counter_buf, histogram_buf};

    // Initialize the kernel.
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point);


    // Launch kernel.
    program.run();

    // Each thread increments the counter 256 times, so the value of the counter 
    // after the kernel runs should be numWorkgroups * workgroupSize * 256
    assert(counter_buf.load(0) == numWorkgroups * workgroupSize * 256);
    std::cout << "numWorkgroups: " << numWorkgroups << "\n";
    std::cout << "workgroupSize: " << workgroupSize << "\n";
    std::cout << "Expected counter:  " << numWorkgroups * workgroupSize * 256 << "\n";
    std::cout << "Global counter: " << counter_buf.load(0) << "\n";

    // Calculate lock fairness (cv should be around 0).
    auto hist_values = std::vector<double>(numWorkgroups * workgroupSize);
    for (int i = 0; i < numWorkgroups * workgroupSize; i++) {
        hist_values.emplace_back(histogram_buf.load(i));
    }
    auto cv = calculate_coeff_variation(hist_values);
    std::cout << "coeff. of variation (fairness): " << cv << "\n";

	// Cleanup.
    program.teardown();
    next_ticket_buf.teardown();
    now_serving_buf.teardown();
    counter_buf.teardown();
    histogram_buf.teardown();
	instance.teardown();
}


void occupancy_discovery_test() {
	// Set up instance.
	auto instance = easyvk::Instance(true);

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(0));
    std::cout << "Device name: " << device.properties.deviceName << "\n";

    // Loader shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/occupancy_discovery.cinit"
    ;
    auto entry_point = "occupancy_discovery";

    auto numWorkgroups = 1000;
    auto workgroupSize = 256;

    // Set up buffers.
    auto count_buf = easyvk::Buffer(device, 1);
    auto poll_open_buf = easyvk::Buffer(device, 1);
    auto M_buf = easyvk::Buffer(device, numWorkgroups);
    auto now_serving_buf = easyvk::Buffer(device, 1);
    auto next_ticket_buf = easyvk::Buffer(device, 1);
    count_buf.store(0, 0);
    poll_open_buf.store(0, 1); // Poll is initially open.
    next_ticket_buf.store(0, 0);
    now_serving_buf.store(0, 0);

    std::vector<easyvk::Buffer> kernelInputs = {count_buf, 
                                                poll_open_buf,
                                                M_buf,
                                                now_serving_buf,
                                                next_ticket_buf};

    // Initialize the kernel.
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point);


    // Launch kernel.
    program.run();


    // Print results.
    std::cout << "numWorkgroups: " << numWorkgroups << "\n";
    std::cout << "workgroupSize: " << workgroupSize << "\n";
    std::cout << "Participating workgroups: " << count_buf.load(0) << "\n";


	// Cleanup.
    program.teardown();
    count_buf.teardown();
    poll_open_buf.teardown();
    M_buf.teardown();
    next_ticket_buf.teardown();
    now_serving_buf.teardown();
    device.teardown();
	instance.teardown();

}

int main(int argc, char* argv[]) {
    occupancy_discovery_test();
	return 0;
}