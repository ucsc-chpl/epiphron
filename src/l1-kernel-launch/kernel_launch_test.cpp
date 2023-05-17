#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <chrono>

using namespace std::chrono;

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
		program.prepare(entryPoint);

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
		program.prepare(entryPoint);

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


void run_vect_add_test(easyvk::Device device) {
	std::vector<uint32_t> spvCode =
	#include "vect-add.cinit"
	;	
	const char *entryPoint = "litmus_test";

	auto numTrialsPerTest = 64;
	auto maxKernelWorkload = 16;
	// Measure overhead with variety of kernel workloads.
	for (int n = 2; n <= maxKernelWorkload; n *= 2) {
		std::cout << "Vector size: " << n << "\n";

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
		program.prepare(entryPoint);

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
	}
}

int main(int argc, char* argv[]) {
	// Initialize 
	auto instance = easyvk::Instance(true);
	auto device = instance.devices().at(0);
	std::cout << "Using device: " << device.properties.deviceName << "\n";
	auto maxWrkGrpCount = device.properties.limits.maxComputeWorkGroupCount;
	std::printf(
		"maxComputeWorkGroupCount: (%d, %d, %d)\n", 
		maxWrkGrpCount[0], 
		maxWrkGrpCount[1],
		maxWrkGrpCount[2]
	);

	// run_vect_add_test(device);
	// run_empty_kernel_test(device);
	run_loop_test(device);

	device.teardown();
	instance.teardown();
	return 0;
}