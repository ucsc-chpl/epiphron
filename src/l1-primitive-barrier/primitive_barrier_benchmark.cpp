#include<chrono>
#include <iostream>
#include <easyvk.h>
#include "json.h"

// TODO: Think about how to parameterize num workgroups.
// Output of 3 benchmarks seem to be same when num workgroups is high??

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

ordered_json no_barrier_benchmark(easyvk::Instance instance, 
								  size_t deviceIndex, 
								  size_t numTrials,
								  size_t numWorkgroups) {
	// Select device to use.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	auto maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];

	// Return benchmark results as JSON.
	ordered_json benchmarkResults;
	benchmarkResults["deviceName"] = device.properties.deviceName;
	benchmarkResults["numTrials"] = numTrials;
	std::vector<ordered_json> results;

	// Load in shader code.
	std::vector<uint32_t> spvCode =
	#include "build/no_barrier.cinit"
	;

	const char *entryPoint = "benchmark";

	std::vector<int> workgroupSizes = {1, 2, 8, 16};
	for (int n = 32; n <= maxWorkgroupSize; n += 32) {
		workgroupSizes.emplace_back(n);
	}

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
		auto numIters = 1024;
		auto buf = easyvk::Buffer(device, bufSize);
		auto buf_size = easyvk::Buffer(device, 1);
		auto num_iters = easyvk::Buffer(device, 1);
		buf_size.store(0, bufSize);
		num_iters.store(0, numIters);

		std::vector<easyvk::Buffer> kernelInputs = {buf, buf_size, num_iters};

		// Initialize kernel.
		auto program = easyvk::Program(device, spvCode, kernelInputs);
		program.setWorkgroups(numWorkgroups);
		program.setWorkgroupSize(n);
		program.initialize(entryPoint);

		// Run benchmark.
		std::vector<double> times(numTrials);

		for (auto i = 0; i < numTrials; i++) {
			// TODO: Seems like times are much slower for the first couple runs??
			auto startTime = high_resolution_clock::now();
			program.run();
			times[i] = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
		}

		auto avgTime = calculate_average(times);
		auto timeStdDev = calculate_std_dev(times);

		// Save test results.
		ordered_json res;
		res["workgroupSize"] = n;
		res["numIters"] = numIters;
		res["avgTime"] = avgTime;
		res["stdDev"] = timeStdDev;
		results.emplace_back(res);

		// Cleanup program.
		program.teardown();
		buf.teardown();
		buf_size.teardown();
		num_iters.teardown();
	}

	// Cleanup device.
	device.teardown();

	benchmarkResults["results"] = results;
	return benchmarkResults;

}


ordered_json workgroup_barrier_local_benchmark(easyvk::Instance instance,
                                               size_t deviceIndex,
											   size_t numTrials,
											   size_t numWorkgroups) {
	// Select device to use from the provided device index.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	auto maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];

	// Return benchmark results as JSON.
	ordered_json benchmarkResults;
	benchmarkResults["deviceName"] = device.properties.deviceName;
	benchmarkResults["numTrials"] = numTrials;
	std::vector<ordered_json> results;

	// Load in shader code.
	std::vector<uint32_t> spvCode =
	#include "build/workgroup-barrier-local.cinit"
	;

	const char *entryPoint = "benchmark";

	std::vector<int> workgroupSizes = {1, 2, 8, 16};
	for (int n = 32; n <= maxWorkgroupSize; n += 32) {
		workgroupSizes.emplace_back(n);
	}

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
		auto numIters = 1024;
		auto buf = easyvk::Buffer(device, bufSize);
		auto buf_size = easyvk::Buffer(device, 1);
		auto num_iters = easyvk::Buffer(device, 1);
		buf_size.store(0, bufSize);
		num_iters.store(0, numIters);

		std::vector<easyvk::Buffer> kernelInputs = {buf, buf_size, num_iters};

		// Initialize kernel.
		auto program = easyvk::Program(device, spvCode, kernelInputs);
		program.setWorkgroups(numWorkgroups);
		program.setWorkgroupSize(n);
		program.initialize(entryPoint);

		// Run benchmark.
		std::vector<double> times(numTrials);

		for (auto i = 0; i < numTrials; i++) {
			// TODO: Seems like times are much slower for the first couple runs??
			auto startTime = high_resolution_clock::now();
			program.run();
			times[i] = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
		}

		auto avgTime = calculate_average(times);
		auto timeStdDev = calculate_std_dev(times);

		// Save test results.
		ordered_json res;
		res["workgroupSize"] = n;
		res["numIters"] = numIters;
		res["avgTime"] = avgTime;
		res["stdDev"] = timeStdDev;
		results.emplace_back(res);

		// Cleanup program.
		program.teardown();
		buf.teardown();
		buf_size.teardown();
		num_iters.teardown();
	}

	// Cleanup device.
	device.teardown();

	benchmarkResults["results"] = results;
	return benchmarkResults;
}


ordered_json workgroup_barrier_global_benchmark(easyvk::Instance instance,
                                                size_t deviceIndex,
											    size_t numTrials,
												size_t numWorkgroups) {
	// Select device to use from the provided device index.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	auto maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];

	// Return benchmark results as JSON.
	ordered_json benchmarkResults;
	benchmarkResults["deviceName"] = device.properties.deviceName;
	benchmarkResults["numTrials"] = numTrials;
	std::vector<ordered_json> results;

	// Load in shader code.
	std::vector<uint32_t> spvCode =
	#include "build/workgroup-barrier-global.cinit"
	;

	const char *entryPoint = "benchmark";

	std::vector<int> workgroupSizes = {1, 2, 8, 16};
	for (int n = 32; n <= maxWorkgroupSize; n += 32) {
		workgroupSizes.emplace_back(n);
	}

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
		auto numIters = 1024;
		auto buf = easyvk::Buffer(device, bufSize);
		auto buf_size = easyvk::Buffer(device, 1);
		auto num_iters = easyvk::Buffer(device, 1);
		buf_size.store(0, bufSize);
		num_iters.store(0, numIters);

		std::vector<easyvk::Buffer> kernelInputs = {buf, buf_size, num_iters};

		// Initialize kernel.
		auto program = easyvk::Program(device, spvCode, kernelInputs);
		program.setWorkgroups(numWorkgroups);
		program.setWorkgroupSize(n);
		program.initialize(entryPoint);

		// Run benchmark.
		std::vector<double> times(numTrials);

		for (auto i = 0; i < numTrials; i++) {
			// TODO: Seems like times are much slower for the first couple runs??
			auto startTime = high_resolution_clock::now();
			program.run();
			times[i] = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
		}

		auto avgTime = calculate_average(times);
		auto timeStdDev = calculate_std_dev(times);

		// Save test results.
		ordered_json res;
		res["workgroupSize"] = n;
		res["numWorkgroups"] = numWorkgroups;
		res["numIters"] = numIters;
		res["avgTime"] = avgTime;
		res["stdDev"] = timeStdDev;
		results.emplace_back(res);

		// Cleanup program.
		program.teardown();
		buf.teardown();
		buf_size.teardown();
		num_iters.teardown();
	}

	// Cleanup device.
	device.teardown();

	benchmarkResults["results"] = results;
	return benchmarkResults;
}


int main(int argc, char* argv[]) {
	// Initialize 
	auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

	auto numTrials = 32;
	auto deviceIndex = 0;
	auto numWorkgroups = 1;
	// NOTE: Submitting some initial work to the GPU to "warm it up" and ensure fair timing results.
	auto _ = no_barrier_benchmark(instance, deviceIndex, 1, numWorkgroups);
	auto noBarrierResults = no_barrier_benchmark(instance, deviceIndex, numTrials, numWorkgroups);
	auto localBarrierResults = workgroup_barrier_local_benchmark(instance, deviceIndex, numTrials, numWorkgroups);
	auto globalBarrierResults = workgroup_barrier_global_benchmark(instance, deviceIndex, numTrials, numWorkgroups);

	ordered_json benchmarkResults;
	benchmarkResults["numWorkgroups"] = numWorkgroups;
	benchmarkResults["noBarrier"] = noBarrierResults;
	benchmarkResults["localWorkgroupBarrier"] = localBarrierResults;
	benchmarkResults["globalWorkgroupBarrier"] = globalBarrierResults;

	// Write results to file.
	// Get current time
    std::time_t currentTime = std::time(nullptr);
    std::tm* currentDateTime = std::localtime(&currentTime);

    // Create file name using current time and date
    char filename[100];
	std::strftime(filename, sizeof(filename), "result%Y-%m-%d_%H-%M-%S.json", currentDateTime);
	std::ofstream outFile(std::string("data/") + std::string(filename));
	if (outFile.is_open()) {
		outFile << benchmarkResults.dump(4) << std::endl;
		outFile.close();
	} else {
		std::cerr << "Failed to write test results to file!\n";
	}

	// Cleanup instance
	instance.teardown();
	return 0;
}