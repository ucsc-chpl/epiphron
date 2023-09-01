#include <chrono>
#include <iostream>
#include <unordered_map>
#include <easyvk.h>
#include "json.h"

#ifdef __ANDROID__
#define USE_VALIDATION_LAYERS false
#else
#define USE_VALIDATION_LAYERS true
#endif

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

std::vector<int> getWorkgroupSizes(size_t maxWorkgroupSize) {
	std::vector<int> workgroupSizes;
	for (int n = 8; n <= 128; n += 8) {
		workgroupSizes.emplace_back(n);
	}

	return workgroupSizes;
}

ordered_json no_barrier_benchmark(easyvk::Instance instance, 
								  size_t deviceIndex, 
								  size_t numTrials,
								  size_t numWorkgroups,
								  size_t numIters) {
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

	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
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
			auto kernelTime = program.runWithDispatchTiming();
			times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
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


ordered_json local_subgroup_barrier_benchmark(easyvk::Instance instance,
                                              size_t deviceIndex,
											  size_t numTrials,
											  size_t numWorkgroups, 
											  size_t numIters) {
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
	#include "build/subgroup-barrier-local.cinit"
	;

	const char *entryPoint = "benchmark";

	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
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
			auto kernelTime = program.runWithDispatchTiming();
			times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
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

ordered_json global_subgroup_barrier_benchmark(easyvk::Instance instance,
                                              size_t deviceIndex,
											  size_t numTrials,
											  size_t numWorkgroups, 
											  size_t numIters) {
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
	#include "build/subgroup-barrier-global.cinit"
	;

	const char *entryPoint = "benchmark";

	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
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
			auto kernelTime = program.runWithDispatchTiming();
			times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
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
											   size_t numWorkgroups, 
											   size_t numIters) {
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

	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
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
			auto kernelTime = program.runWithDispatchTiming();
			times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
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
												size_t numWorkgroups,
												size_t numIters) {
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

	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	for (const auto& n : workgroupSizes) {
		// Set up kernel inputs.
		auto bufSize =  n * numWorkgroups;
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
			auto kernelTime = program.runWithDispatchTiming();
			times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
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


ordered_json primitive_barrier_benchmark(easyvk::Instance instance,
                                         size_t deviceIndex,
									     size_t numTrials,
										 size_t numWorkgroups,
										 size_t numIters) {

	// Select device to use from the provided device index.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	auto maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];

	// Load the kernel.
	std::vector<uint32_t> kernelCode = 
	#include "build/primitive_barrier.cinit"
	;

	// All bencmarks are contained in the single kernel and we can specify which 
	// one to run by selecting the entry point.
	std::vector<const char*> entryPoints = {
        "noBarrier",
        "localSubgroupBarrier",
        "globalSubgroupBarrier",
        "localWorkgroupBarrier",
        "globalWorkgroupBarrier"
    };

	// Define the parameterization across workgroup sizes.
	auto workgroupSizes = getWorkgroupSizes(maxWorkgroupSize);

	// Save all results to a JSON object.
	ordered_json primitiveBarrierBenchmarkResults;

	// Set up and run each primitive barrier benchmark.
    for (const auto& entryPoint : entryPoints) {
		std::cout << "Starting " << entryPoint << " benchmark...\n";
		// Parameterize across workgroup size; all other params stay fixed.
		std::vector<ordered_json> benchmarkResults;
		for (const auto &n : workgroupSizes) {
			// Set up kernel inputs.
			auto bufSize =  n * numWorkgroups;
			auto buf = easyvk::Buffer(device, bufSize);
			auto buf_size = easyvk::Buffer(device, 1);
			auto num_iters = easyvk::Buffer(device, 1);
			buf_size.store(0, bufSize);
			num_iters.store(0, numIters);

			std::vector<easyvk::Buffer> kernelInputs = {buf, buf_size, num_iters};

			// Initialize kernel.
			auto program = easyvk::Program(device, kernelCode, kernelInputs);
			program.setWorkgroups(numWorkgroups);
			program.setWorkgroupSize(n);
			program.initialize(entryPoint);

			// Run the kernel.
			std::vector<double> times(numTrials);
			for (auto i = 0; i < numTrials; i++) {
				auto kernelTime = program.runWithDispatchTiming();
				times[i] = kernelTime / (double) 1000.0; // Convert from nanoseconds to microseconds.
			}

			auto avgTime = calculate_average(times);
			auto timeStdDev = calculate_std_dev(times);

			// Store resutls back in JSON
			ordered_json res;
			res["workgroupSize"] = n;
			res["avgTime"] = avgTime;
			res["stdDev"] = timeStdDev;
			benchmarkResults.emplace_back(res);

			// Cleanup program.
			program.teardown();
			buf.teardown();
			buf_size.teardown();
			num_iters.teardown();

		}
		primitiveBarrierBenchmarkResults[entryPoint] = benchmarkResults;
		std::cout << entryPoint << " benchmark finished!\n";
    }

	device.teardown();

	return primitiveBarrierBenchmarkResults;
}



int main(int argc, char* argv[]) {
	// Select which device to use.
	auto deviceIndex = 1; 

	// Query device properties.
	auto instance = easyvk::Instance(false);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    device.teardown();

	// Benchmark parameters.
	auto numTrials = 16;
	auto numWorkgroups = 64;
	auto numIters = 1024 * 4; // # of iterations to run kernel loop
	auto _ = no_barrier_benchmark(instance, deviceIndex, 16, numWorkgroups, 512); // warmup GPU

	auto primitiveBarrierBenchmarkResult = primitive_barrier_benchmark(instance, deviceIndex, numTrials, numWorkgroups, numIters);

	primitiveBarrierBenchmarkResult["benchmarkName"] = "primitiveBarrier";
	primitiveBarrierBenchmarkResult["deviceName"] = deviceName;
	primitiveBarrierBenchmarkResult["numTrials"] = numTrials;
	primitiveBarrierBenchmarkResult["numIters"] = numIters;
	primitiveBarrierBenchmarkResult["numWorkgroups"] = numWorkgroups;

	// Write results to file.
	// Get current time
    std::time_t currentTime = std::time(nullptr);
    std::tm* currentDateTime = std::localtime(&currentTime);

    // Create file name using current time and date
    char filename[100];
	std::strftime(filename, sizeof(filename), "result%Y-%m-%d_%H-%M-%S.json", currentDateTime);
	#ifdef __ANDROID__
	std::ofstream outFile(filename);
	#else
	std::ofstream outFile(std::string("data/") + std::string(filename));
	#endif
	if (outFile.is_open()) {
		outFile << primitiveBarrierBenchmarkResult.dump(4) << std::endl;
		outFile.close();
	} else {
		std::cerr << "Failed to write test results to file!\n";
	}

	// Cleanup instance
	instance.teardown();
	return 0;
}