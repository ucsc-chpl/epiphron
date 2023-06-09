#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <easyvk.h>
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

	if (percentage >= 1.0) {
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

/**
 * Rounds a given size_t value up to the nearest power of 2.
 *
 * @param n The input value to be rounded.
 * @return The rounded value, which is the nearest power of 2 greater than or equal to the input.
 */
size_t roundUpToPowerOf2(size_t n) {
    if (n == 0)
        return 1;

    n--;
    for (size_t i = 1; i < sizeof(size_t) * 8; i *= 2)
        n |= n >> i;

    return n + 1;
}

/**
* @brief Runs the varied workload benchmark.
* 
* This benchmark measures kernel launch overhead of a vector addition kernel 
* Every thread is assigned a chunk of the vector to operate on. This workload is parameterized.
* WARNING: The maximum vector size that's passed to the kernel will be numWorkGroups * workGroupSize * maxThreadWorkload
* 
* @param device The easyvk::Device object for which to run the benchmark on // TODO: Change to a physical device index.
* @param numTrialsPerTest Number of trials to run for each test.
* @param numWorkGroups Number of workgroups to dispatch.
* @param workGroupSize Size of the workgroups.
* @param maxThreadWorkload Maximum chunk of the vector a kernel will be assigned to.
* @return JSON object containing the benchmark results.
*/	
ordered_json run_vect_add_fixed_dispatch_benchmark(easyvk::Device device, 
                                                   size_t numTrialsPerTest, 
												   size_t numWorkGroups,
												   size_t workGroupSize,
												   size_t maxThreadWorkload) {
	// Maximum vector size that will be passed to the kernel.
	auto maxVecSize = maxThreadWorkload * numWorkGroups * workGroupSize;

	// Save results to JSON
	std::vector<ordered_json> testData;

	// Load program.
	std::vector<uint32_t> spvCode =
	#include "build/vect-add-fixed-dispatch.cinit"
	;
	const char *entryPoint = "test";

	// Measure overhead with variety of kernel workloads.
	for (int n = numWorkGroups * workGroupSize; n <= maxVecSize; n *= 2) {
		// Create GPU buffers.
		auto vecSize = n;
		auto a = easyvk::Buffer(device, vecSize);
		auto b = easyvk::Buffer(device, vecSize);
		auto c = easyvk::Buffer(device, vecSize);
		auto dispatchSizeBuf = easyvk::Buffer(device, 1);
		dispatchSizeBuf.store(0, numWorkGroups);
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

		program.setWorkgroups(numWorkGroups);
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
		for (int i = 0; i < vecSize; i++) {
			assert(c.load(i) == a.load(i) + b.load(i));
		}

		// Cleanup.
		program.teardown();
		a.teardown();
		b.teardown();
		c.teardown();

		// Save test results to JSON.
		ordered_json res;
		res["threadWorkload"] = vecSize / (workGroupSize * numWorkGroups);
		res["vecSize"] = vecSize;
		res["avgUtilPerTrial"] = avgUtilizationPerTrial;
		res["stdDev"] = stdDev;
		testData.emplace_back(res);

		updateProgressBar(n, maxVecSize);
	}

	ordered_json vectAddResults;
	vectAddResults["numWorkgroups"] = numWorkGroups;
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
* @param workGroupSize Size of the workgroups.
* @return JSON object containing the benchmark results.
*/
ordered_json run_vect_add_benchmark(easyvk::Device device, 
                                   size_t numTrialsPerTest,
								   size_t maxWorkGroupInvocations,
								   size_t workGroupSize) {
	// Save results to JSON
	std::vector<ordered_json> testData;

	// Load in shader code.
	std::vector<uint32_t> spvCode =
	#include "build/vect-add.cinit"
	;
	const char *entryPoint = "litmus_test";

	// Measure overhead with variety of kernel workloads.
	for (int n = 1; n <= roundUpToPowerOf2(maxWorkGroupInvocations); n *= 2) {
		// Ensure that n is less than or equal to maxWorkGroupInvocations
		n = std::min((int) maxWorkGroupInvocations, n);

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
	vectAddResults["workgroupSize"] = workGroupSize;
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

	auto numTrials = 8;

	// Run vector addition test.
	std::cout << "Running vector addition test...\n";
	auto vectAddResults = run_vect_add_benchmark(device, numTrials, maxWrkGroups[0], 32);
	testResults["vectorAddResults"] = vectAddResults;
	std::cout << "Done!\n";

	// Run fixed dispatch vector addition test.
	std::cout << "Running fixed dispatch test...\n";
	auto fixedDispatchVectAddResults = run_vect_add_fixed_dispatch_benchmark(device, numTrials, 8, 32, 1024 * 8);
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