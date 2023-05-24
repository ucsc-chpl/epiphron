#include <vector>
#include <iostream>
#include <cmath>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <chrono>
#include "json.h"

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;

#define DATA_DIR "/home/ssiddens/ucsc-chpl/epiphron-master/src/l1-kernel-launch/data"

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

void run_loop_test(easyvk::Device device) {
	std::cout << "Running loop test...\n";
	std::vector<uint32_t> spvCode =
		#include "loop-test.cinit"
	;	
	const char* entryPoint = "litmus_test";

	auto numTrialsPerTest = 64;
	auto maxKernelWorkload = 1024 * 1024;
	// Measure overhead with variety of kernel workloads.
	for (int n = 2; n <= maxKernelWorkload; n *= 2) {
		std::cout << "Dispatch size: " << n << "\n";

		auto numDispatches = 16;

		// Create GPU buffers.
		auto numLoops = easyvk::Buffer(device, 1);
		auto inputBuf = easyvk::Buffer(device, numDispatches);
		auto outputBuf = easyvk::Buffer(device, numDispatches);
		numLoops.store(0, n);

		for (int i = 0; i < numDispatches; i++) {
			inputBuf.store(i, i);
			outputBuf.store(i, i);
		}

		std::vector<easyvk::Buffer> bufs = {numLoops, inputBuf, outputBuf};

		auto program = easyvk::Program(device, spvCode, bufs);

		// Dispatch a fixed amount
		program.setWorkgroups(numDispatches);
		program.setWorkgroupSize(1);

		// Run the kernel.
		program.initialize(entryPoint);

		double totalOverhead = 0.0;
		auto totalGpuTime = 0;
		auto totalCpuTime = 0;
		for (auto _ = 0; _ < numTrialsPerTest; _++) {
			auto startTime = high_resolution_clock::now();
			auto gpuTime = program.runWithDispatchTiming();
			auto cpuTime = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
			totalGpuTime += gpuTime;
			totalCpuTime += cpuTime;
			totalOverhead += ((gpuTime / (double)1000.0) / cpuTime);
		}


		auto avgGpuTimeInMicroseconds = (totalGpuTime / (double)numTrialsPerTest) / 1000.0;
		std::cout << "Average GPU time: " << avgGpuTimeInMicroseconds << "us" << std::endl;
		auto avgCpuTimeInMicroseconds = totalCpuTime / (double)numTrialsPerTest;
		std::cout << "Average CPU time: " << avgCpuTimeInMicroseconds << "us" << std::endl;

		auto avgOverheadPerTrial = totalOverhead / (double)numTrialsPerTest;
		std::cout << "Average overhead per trial: " <<  avgOverheadPerTrial << "\n\n";

		// Cleanup.
		program.teardown();
		numLoops.teardown();
		inputBuf.teardown();
		outputBuf.teardown();
	}
}

void run_empty_kernel_test(easyvk::Device device) {
	std::vector<uint32_t> spvCode =
		#include "empty-kernel.cinit"
	;	
	const char* entryPoint = "litmus_test";

	auto numTrialsPerTest = 64;
	auto maxKernelWorkload = 1024 * 1024;
	// Measure overhead with variety of kernel workloads.
	for (int n = 2; n <= maxKernelWorkload; n *= 2) {
		std::cout << "Dispatch size: " << n << "\n";

		// Create GPU buffers.
		auto a = easyvk::Buffer(device, 1);
		auto b = easyvk::Buffer(device, 1);
		auto c = easyvk::Buffer(device, 1);

		std::vector<easyvk::Buffer> bufs = {a, b, c};

		auto program = easyvk::Program(device, spvCode, bufs);

		program.setWorkgroups(n);
		program.setWorkgroupSize(1);

		// Run the kernel.
		program.initialize(entryPoint);

		double totalOverhead = 0.0;
		auto totalGpuTime = 0;
		auto totalCpuTime = 0;
		for (auto _ = 0; _ < numTrialsPerTest; _++) {
			auto startTime = high_resolution_clock::now();
			auto gpuTime = program.runWithDispatchTiming();
			auto cpuTime = duration_cast<microseconds>(high_resolution_clock::now() - startTime).count();
			totalGpuTime += gpuTime;
			totalCpuTime += cpuTime;
			totalOverhead += ((gpuTime / (double)1000.0) / cpuTime);
		}


		auto avgGpuTimeInMicroseconds = (totalGpuTime / (double)numTrialsPerTest) / 1000.0;
		std::cout << "Average GPU time: " << avgGpuTimeInMicroseconds << "us" << std::endl;
		auto avgCpuTimeInMicroseconds = totalCpuTime / (double)numTrialsPerTest;
		std::cout << "Average CPU time: " << avgCpuTimeInMicroseconds << "us" << std::endl;

		auto avgOverheadPerTrial = totalOverhead / (double)numTrialsPerTest;
		std::cout << "Average overhead per trial: " <<  avgOverheadPerTrial << "\n\n";

		// Cleanup.
		program.teardown();
		a.teardown();
		b.teardown();
		c.teardown();
	}
}

ordered_json run_vect_add_fixed_dispatch_test(easyvk::Device device, size_t numTrialsPerTest, size_t dispatchSize) {
	// Maximum amount of work each kernel will do.
	auto maxKernelWorkload = (1024 * 1024) / dispatchSize;
	std::cout << "Max kernel workload: " << maxKernelWorkload << "\n";
	std::cout << "Dispatch size: " << dispatchSize << "\n";

	// Save results to JSON
	std::vector<ordered_json> testData;

	// Load program.
	std::vector<uint32_t> spvCode =
	#include "vect-add-fixed-dispatch.cinit"
	;
	const char *entryPoint = "test";

	// Measure overhead with variety of kernel workloads.
	for (int n = 16; n <= maxKernelWorkload; n *= 2) {
		std::cout << "Kernel workload: " << n << "\n";
		std::cout << "Vector size: " << n * dispatchSize << "\n\n";
		// Create GPU buffers.
		auto a = easyvk::Buffer(device, n * dispatchSize);
		auto b = easyvk::Buffer(device, n * dispatchSize);
		auto c = easyvk::Buffer(device, n * dispatchSize);
		auto dispatchSizeBuf = easyvk::Buffer(device, 1);
		dispatchSizeBuf.store(0, dispatchSize);
		auto vecSizeBuf = easyvk::Buffer(device, 1);
		vecSizeBuf.store(0, n * dispatchSize);

		// Write initial values to the buffers.
		for (int i = 0; i < n * dispatchSize; i++) {
			a.store(i, i);
			b.store(i, i + 1);
			c.store(i, 0);
		}
		std::vector<easyvk::Buffer> bufs = {a, b, c, dispatchSizeBuf, vecSizeBuf};

		auto program = easyvk::Program(device, spvCode, bufs);

		program.setWorkgroups(dispatchSize);
		program.setWorkgroupSize(1);

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
		res["kernelWorkload"] = n;
		res["vecSize"] = n * dispatchSize;
		res["avgUtilPerTrial"] = avgUtilizationPerTrial;
		res["stdDev"] = stdDev;
		testData.emplace_back(res);

	}

	ordered_json vectAddResults;
	vectAddResults["dispatchSize"] = dispatchSize;
	vectAddResults["numTrialsPerTest"] = numTrialsPerTest;
	vectAddResults["results"] = testData;
	return vectAddResults;
}


ordered_json run_vect_add_test(easyvk::Device device, size_t numTrialsPerTest, size_t maxKernelWorkload) {
	// Save results to JSON
	std::vector<ordered_json> testData;

	std::vector<uint32_t> spvCode =
	#include "vect-add.cinit"
	;
	const char *entryPoint = "litmus_test";

	// Measure overhead with variety of kernel workloads.
	for (int n = 16; n <= maxKernelWorkload; n *= 2) {
		// Create GPU buffers.
		auto a = easyvk::Buffer(device, n);
		auto b = easyvk::Buffer(device, n);
		auto c = easyvk::Buffer(device, n);

		// Write initial values to the buffers.
		for (int i = 0; i < n; i++) {
			a.store(i, i);
			b.store(i, i + 1);
			c.store(i, 0);
		}
		std::vector<easyvk::Buffer> bufs = {a, b, c};

		auto program = easyvk::Program(device, spvCode, bufs);

		program.setWorkgroups(n);
		program.setWorkgroupSize(1);

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
		for (int i = 0; i < n; i++) {
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
		res["vecSize"] = n;
		res["dispatchSize"] = n;
		res["avgUtilPerTrial"] = avgUtilizationPerTrial;
		res["stdDev"] = stdDev;
		testData.emplace_back(res);

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
	auto device = easyvk::Device(instance, physicalDevices.at(0));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
	auto maxWrkGrpCount = device.properties.limits.maxComputeWorkGroupCount;
	std::printf(
		"maxComputeWorkGroupCount: (%d, %d, %d)\n", 
		maxWrkGrpCount[0], 
		maxWrkGrpCount[1],
		maxWrkGrpCount[2]
	);

	// Save test results to JSON.
	ordered_json testResults;
	testResults["testName"] = "Kernel Launch";
	testResults["deviceName"] = device.properties.deviceName;
	testResults["vendorID"] = device.properties.vendorID;
	testResults["deviceID"] = device.properties.deviceID;
	testResults["driverVersion"] = device.properties.driverVersion;

	// Run vector addition test.
	auto vectAddResults = run_vect_add_test(device, 64, 1024 * 1024);
	testResults["vectorAddResults"] = vectAddResults;

	// Run fixed dispatch vector addition test.
	auto fixedDispatchVectAddResults = run_vect_add_fixed_dispatch_test(device, 16, 8);
	testResults["vectorAddFixedDispatchResults"] = fixedDispatchVectAddResults;

	// Write results to file.
	std::ofstream outFile(std::string(DATA_DIR) + std::string("/results.json"));
	if (outFile.is_open()) {
		outFile << testResults.dump(4) << std::endl;
		outFile.close();
	} else {
		std::cerr << "Failed to write test results to file!\n";
	}

	// run_empty_kernel_test(device);
	// run_loop_test(device);

	// Cleanup.
	device.teardown();
	instance.teardown();
	return 0;
}