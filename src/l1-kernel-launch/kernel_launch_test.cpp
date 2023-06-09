#include <vector>
#include <iostream>
#include <cmath>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <chrono>
#include <ctime>
#include "json.h"

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;

void updateProgressBar(int progress, int total, int barWidth = 50) {
    // Calculate the percentage of completion
    float percentage = static_cast<float>(progress) / total;
    
    // Calculate the number of completed characters in the progress bar
    int completedWidth = static_cast<int>(percentage * barWidth);
    
    // Output the progress bar
    std::cout << "[";
    for (int i = 0; i < completedWidth; ++i) {
        std::cout << "=";
    }
    std::cout << ">";
    for (int i = completedWidth; i < barWidth; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100) << "%\r";
    std::cout.flush();

	if (percentage == 1.0) {
		std::cout << std::endl;
	}
}

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

ordered_json run_vect_add_fixed_dispatch_benchmark(easyvk::Device device, size_t numTrialsPerTest, size_t dispatchSize) {
	auto workGroupSize = 32;
	auto maxThreadWorkload = 1024 * 1024; // Maximum amount of work an individual thread will do.
	auto maxVecSize = maxThreadWorkload / (dispatchSize * workGroupSize);

	// Save results to JSON
	std::vector<ordered_json> testData;

	// Load program.
	std::vector<uint32_t> spvCode =
	#include "build/vect-add-fixed-dispatch.cinit"
	;
	const char *entryPoint = "test";

	// Measure overhead with variety of kernel workloads.
	for (int n = 16; n <= maxVecSize; n *= 2) {
		// Create GPU buffers.
		auto vecSize = n * dispatchSize * workGroupSize;
		auto a = easyvk::Buffer(device, vecSize);
		auto b = easyvk::Buffer(device, vecSize);
		auto c = easyvk::Buffer(device, vecSize);
		auto dispatchSizeBuf = easyvk::Buffer(device, 1);
		dispatchSizeBuf.store(0, dispatchSize);
		auto vecSizeBuf = easyvk::Buffer(device, 1);
		vecSizeBuf.store(0, vecSize);

		// Write initial values to the buffers.
		for (int i = 0; i < vecSize; i++) {
			a.store(i, i);
			b.store(i, i + 1);
			c.store(i, 0);
		}
		std::vector<easyvk::Buffer> bufs = {a, b, c, dispatchSizeBuf, vecSizeBuf};

		auto program = easyvk::Program(device, spvCode, bufs);

		program.setWorkgroups(dispatchSize);
		program.setWorkgroupSize(workGroupSize);

		// Run the kernel.
		program.initialize(entryPoint);

		double totalOverhead = 0.0;
		auto totalGpuTime = 0;
		auto totalCpuTime = 0;
		std::vector<double> cpuTimes(numTrialsPerTest);
		std::vector<double> gpuTimes(numTrialsPerTest);
		std::vector<double> utilizationData(numTrialsPerTest);
		for (auto i = 0; i < numTrialsPerTest; i++) {
			auto startTime = high_resolution_clock::now();
			auto gpuTime = program.runWithDispatchTiming();
			auto cpuTime = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
			cpuTimes[i] = cpuTime;
			gpuTimes[i] = gpuTime;
			utilizationData[i] = (gpuTime / (double)1000.0) / cpuTime;
		}


		auto avgGpuTimeInMicroseconds = calculate_average(gpuTimes) / 1000.0;
		auto avgCpuTimeInMicroseconds = calculate_average(cpuTimes);

		// Calculate average utilization and std dev
		auto avgUtilizationPerTrial = calculate_average(utilizationData);
		auto stdDev = calculate_std_dev(utilizationData);

		// Validate the output.
		for (int i = 0; i < n * dispatchSize; i++) {
			assert(c.load(i) == a.load(i) + b.load(i));
		}

		// Cleanup.
		program.teardown();
		a.teardown();
		b.teardown();
		c.teardown();

		// Save test results to JSON.
		ordered_json res;
		res["kernelWorkload"] = n;
		res["vecSize"] = n * dispatchSize;
		res["avgUtilPerTrial"] = avgUtilizationPerTrial;
		res["stdDev"] = stdDev;
		testData.emplace_back(res);

		updateProgressBar(n, maxVecSize);
	}

	ordered_json vectAddResults;
	vectAddResults["dispatchSize"] = dispatchSize;
	vectAddResults["numTrialsPerTest"] = numTrialsPerTest;
	vectAddResults["results"] = testData;
	return vectAddResults;
}


/**
* @brief Runs the vector addition benchmark on the specified device.
* 
* This benchmark measures kernel launch overhead of a vector addition shader.
* Vector size is parameterized and every thread operates on one element of the vector.
* The maximum vector size will be the product of maxWorkGroupInvocations and whatever work group size is set to.
* 
* @param device The easyvk::Device object for which to run the benchmark on // TODO: Change to a physical device index.
* @param numTrialsPerTest Number of trials to run for each test.
* @param maxWorkGroupInvocations Maximum number of workgroups that will be dispatched.
* @return JSON object containing the benchmark results.
*/
ordered_json run_vect_add_benchmark(easyvk::Device device, size_t numTrialsPerTest, size_t maxWorkGroupInvocations) {
	// Save results to JSON
	std::vector<ordered_json> testData;

	// Load in shader code.
	std::vector<uint32_t> spvCode =
	#include "build/vect-add.cinit"
	;
	const char *entryPoint = "litmus_test";

	// Measure overhead with variety of kernel workloads.
	auto workGroupSize = 32;
	for (int n = 1; n <= maxWorkGroupInvocations; n *= 2) {
		// Create GPU buffers.
		auto kernelWorkloadSize = n * workGroupSize;
		auto a = easyvk::Buffer(device, kernelWorkloadSize);
		auto b = easyvk::Buffer(device, kernelWorkloadSize);
		auto c = easyvk::Buffer(device, kernelWorkloadSize);

		// Write initial values to the buffers.
		for (int i = 0; i < kernelWorkloadSize; i++) {
			a.store(i, i);
			b.store(i, i + 1);
			c.store(i, 0);
		}
		std::vector<easyvk::Buffer> bufs = {a, b, c};

		auto program = easyvk::Program(device, spvCode, bufs);

		program.setWorkgroups(n);
		program.setWorkgroupSize(workGroupSize);

		// Run the kernel.
		program.initialize(entryPoint);

		double totalOverhead = 0.0;
		auto totalGpuTime = 0;
		auto totalCpuTime = 0;
		std::vector<double> cpuTimes(numTrialsPerTest);
		std::vector<double> gpuTimes(numTrialsPerTest);
		std::vector<double> utilizationData(numTrialsPerTest);
		for (auto i = 0; i < numTrialsPerTest; i++) {
			auto startTime = high_resolution_clock::now();
			auto gpuTime = program.runWithDispatchTiming();
			auto cpuTime = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
			cpuTimes[i] = cpuTime;
			gpuTimes[i] = gpuTime;
			utilizationData[i] = (gpuTime / (double)1000.0) / cpuTime;
		}


		auto avgGpuTimeInMicroseconds = calculate_average(gpuTimes) / 1000.0;
		auto avgCpuTimeInMicroseconds = calculate_average(cpuTimes);

		// Calculate average utilization and std dev
		auto avgUtilizationPerTrial = calculate_average(utilizationData);
		auto stdDev = calculate_std_dev(utilizationData);

		// Validate the output.
		for (int i = 0; i < kernelWorkloadSize; i++) {
			// std::cout << "c[" << i << "]: " << c.load(i) << "\n";
			assert(c.load(i) == a.load(i) + b.load(i));
		}

		// Cleanup.
		program.teardown();
		a.teardown();
		b.teardown();
		c.teardown();

		// Save test results to JSON.
		ordered_json res;
		res["vecSize"] = kernelWorkloadSize;
		res["dispatchSize"] = n;
		res["avgUtilPerTrial"] = avgUtilizationPerTrial;
		res["stdDev"] = stdDev;
		testData.emplace_back(res);

		updateProgressBar(n, maxWorkGroupInvocations);
	}

	ordered_json vectAddResults;
	vectAddResults["numTrialsPerTest"] = numTrialsPerTest;
	vectAddResults["results"] = testData;
	return vectAddResults;
}

int main(int argc, char* argv[]) {
	// Initialize 
	auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

	// Select logical device.
	auto device = easyvk::Device(instance, physicalDevices.at(0));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
	auto maxWrkGrpInvocations = device.properties.limits.maxComputeWorkGroupInvocations;
	auto maxWrkGroups = device.properties.limits.maxComputeWorkGroupCount;
	auto maxWrkGrpSize = device.properties.limits. maxComputeWorkGroupSize;
	std::printf(
		"maxComputeWorkgroupInvocations: %d\n", 
		maxWrkGrpInvocations
	);
	std::printf(
		"maxComputeWorkgroupCount: (%d, %d, %d)\n", 
		maxWrkGroups[0],
		maxWrkGroups[1],
		maxWrkGroups[2]
	);
	std::printf(
		"maxWorkGroupSize: (%d, %d, %d)\n", 
		maxWrkGrpSize[0],
		maxWrkGrpSize[1],
		maxWrkGrpSize[2]
	);

	// Save test results to JSON.
	ordered_json testResults;
	testResults["testName"] = "Kernel Launch";
	testResults["deviceName"] = device.properties.deviceName;
	testResults["vendorID"] = device.properties.vendorID;
	testResults["deviceID"] = device.properties.deviceID;
	testResults["driverVersion"] = device.properties.driverVersion;

	// Run vector addition test.
	std::cout << "Running vector addition test...\n";
	auto vectAddResults = run_vect_add_benchmark(device, 64, 1024);
	testResults["vectorAddResults"] = vectAddResults;
	std::cout << "Done!\n";

	// Run fixed dispatch vector addition test.
	std::cout << "Running fixed dispatch test...\n";
	auto fixedDispatchVectAddResults = run_vect_add_fixed_dispatch_benchmark(device, 64, 32);
	testResults["vectorAddFixedDispatchResults"] = fixedDispatchVectAddResults;
	std::cout << "Done!\n";


	// Write results to file.
	// Get current time
    std::time_t currentTime = std::time(nullptr);
    std::tm* currentDateTime = std::localtime(&currentTime);

    // Create file name using current time and date
    char filename[100];
	std::strftime(filename, sizeof(filename), "result%Y-%m-%d_%H-%M-%S.json", currentDateTime);
	std::ofstream outFile(std::string("data/") + std::string(filename));
	if (outFile.is_open()) {
		outFile << testResults.dump(4) << std::endl;
		outFile.close();
	} else {
		std::cerr << "Failed to write test results to file!\n";
	}

	// Cleanup.
	device.teardown();
	instance.teardown();
	return 0;
}