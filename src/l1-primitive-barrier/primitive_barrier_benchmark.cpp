#include <chrono>
#include <iostream>
#include <thread>
#include <easyvk.h>
#include "json.h"

#ifdef __ANDROID__
#define USE_VALIDATION_LAYERS false
#else
#define USE_VALIDATION_LAYERS true
#endif

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;

// SPIR-V magic number to verify the binary
const uint32_t SPIRV_MAGIC = 0x07230203;
// NOTE: This function modifies the first OpConstant instruction it finds with a
// value of LOCAL_MEM_SIZE (defined below). It does no semantic check of type or 
// variable name, so ensure the constant you want to modify doens't conflict with
// any previously defined constant (i.e ensure that the first #define constant w/ value
// 1024 is at the top of the OpenCL file).
const uint32_t LOCAL_MEM_SIZE = 1024; 
void modifyLocalMemSize(std::vector<uint32_t>& spirv, uint32_t newValue) {
    if(spirv.size() < 5) {
        std::cerr << "Invalid SPIR-V binary." << std::endl;
        return;
    }

    if(spirv[0] != SPIRV_MAGIC) {
        std::cerr << "Not a SPIR-V binary." << std::endl;
        return;
    }

    // Iterate through SPIR-V instructions
    // https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/parsing_instructions.md
    size_t i = 5;  // skip SPIR-V header
    while(i < spirv.size()) {
        uint32_t instruction = spirv[i];
        uint32_t length = instruction >> 16;
        uint32_t opcode = instruction & 0x0ffffu;

        // Opcode source: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstant
        if(opcode == 43) { // OpConstant
            uint32_t resultType = spirv[i+1];
            uint32_t resultId = spirv[i+2];
            uint32_t constantValue = spirv[i+3];

            // This is a simplistic check.
            // Doesn't verify the type and name (through debug info)
            if(constantValue == LOCAL_MEM_SIZE) {
                spirv[i+3] = newValue;
                return;
            }
        }

        i += length; // move to next instruction
    }

    std::cerr << "Did not modify any instructions when parsing the SPIR-V module!\n";
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


std::vector<int> getWorkgroupSizes() {
	std::vector<int> workgroupSizes;
	for (int n = 8; n <= 128; n += 8) {
		workgroupSizes.emplace_back(n);
	}

	return workgroupSizes;
}

uint32_t getOccupancy(easyvk::Instance instance,
                      easyvk::Device device,
					  size_t numTrials,
					  size_t workgroupSize,
					  size_t localMemSize) {
	// Load the kernel.
	std::vector<uint32_t> kernelCode = 
	#include "build/primitive_barrier_occupancy.cinit"
	;
    modifyLocalMemSize(kernelCode, localMemSize);

	// All bencmarks are contained in the single kernel and we can specify which 
	// one to run by selecting the entry point.
	std::vector<const char*> entryPoints = {
        "noBarrier", // Occupancy should be the same for each kernel?
        // "localSubgroupBarrier",
        // "globalSubgroupBarrier",
        // "localWorkgroupBarrier",
        // "globalWorkgroupBarrier"
    };

	auto numWorkgroups = 1024;

	// Set up kernel inputs.
	auto buf = easyvk::Buffer(device, localMemSize, 1); 
	auto buf_size = easyvk::Buffer(device, 1, sizeof(uint32_t));
	buf_size.store<uint32_t>(0, localMemSize);
	auto num_iters = easyvk::Buffer(device, 1, sizeof(uint32_t));
	num_iters.store<uint32_t>(0, 2);
	auto count_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto poll_open_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto M_buf = easyvk::Buffer(device, numWorkgroups, sizeof(uint32_t));
	auto now_serving_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto next_ticket_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto local_mem_size_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
	count_buf.store<uint32_t>(0, 0);
	poll_open_buf.store<uint32_t>(0, 1); // Poll is initially open.
	next_ticket_buf.store<uint32_t>(0, 0);
	now_serving_buf.store<uint32_t>(0, 0);

	std::vector<easyvk::Buffer> kernelInputs = {buf,
												buf_size,
												num_iters,
												count_buf,
												poll_open_buf,
												M_buf,
												now_serving_buf,
												next_ticket_buf};

	// Initialize kernel.
	auto program = easyvk::Program(device, kernelCode, kernelInputs);
	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	program.initialize(entryPoints[0]);

	// Run the kernel.
	uint32_t maxOccupancyBound = 0;
	for (auto i = 0; i < numTrials; i++) {
		program.run();
        if (count_buf.load<uint32_t>(0) > maxOccupancyBound) {
            maxOccupancyBound = count_buf.load<uint32_t>(0);
        }
	}

	// std::cout << "Occupancy bound: " << maxOccupancyBound << "\n";

	// Cleanup program.
	program.teardown();
	buf.teardown();
	buf_size.teardown();
	num_iters.teardown();
	count_buf.teardown();
	poll_open_buf.teardown();
	next_ticket_buf.teardown();
	now_serving_buf.teardown();

	return maxOccupancyBound;
}

ordered_json primitiveBarrierBenchmark(easyvk::Instance instance,
                                       size_t deviceIndex,
								       size_t numTrials,
								       size_t numIters,
									   size_t bufferSize) {

	// Select device to use from the provided device index.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	std::cout << "Running primitive barrier benchmark on " << device.properties.deviceName << "...\n";
	auto maxLocalSize = device.properties.limits.maxComputeSharedMemorySize;
	std::cout << "Max local memory size: " << device.properties.limits.maxComputeSharedMemorySize << "\n";

	// Load the kernel.
	std::vector<uint32_t> kernelCode = 
	#include "build/primitive_barrier.cinit"
	;
	modifyLocalMemSize(kernelCode, bufferSize);
	
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
	auto workgroupSizes = getWorkgroupSizes();

	// Save all results to a JSON object.
	ordered_json primitiveBarrierBenchmarkResults;

	// Set up and run each primitive barrier benchmark.
    for (size_t i = 0; i < entryPoints.size(); i++) {
		std::cout << "Starting " << entryPoints[i] << " benchmark...\n";
		// Parameterize across workgroup size; all other params stay fixed.
		std::vector<ordered_json> benchmarkResults;
		for (const auto &n : workgroupSizes) {
			// Get occupancy bound.
			uint occupancyBound = getOccupancy(instance, device, 8, n, bufferSize);

			// Set up kernel inputs.
			auto buf = easyvk::Buffer(device, bufferSize, 1);
			auto buf_size = easyvk::Buffer(device, 1, sizeof(uint32_t));
			auto num_iters = easyvk::Buffer(device, 1, sizeof(uint32_t));
			buf_size.store<uint32_t>(0, bufferSize);
			num_iters.store<uint32_t>(0, numIters);

			std::vector<easyvk::Buffer> kernelInputs = {buf, buf_size, num_iters};

			// Initialize kernel.
			auto program = easyvk::Program(device, kernelCode, kernelInputs);
			program.setWorkgroups(occupancyBound);
			program.setWorkgroupSize(n);
			program.initialize(entryPoints[i]);

			if (i == 0) {
				// "warm up" the GPU if we are running the first benchmark.
				for (auto j = 0; j < numTrials; j++) {
					program.run();
				}
			}

			// Run the kernel.
			std::vector<double> times;
			for (auto j = 0; j < numTrials; j++) {
				auto kernelTime = program.runWithDispatchTiming();
				times.push_back(kernelTime / (double) 1000.0); // Convert from nanoseconds to microseconds.
			}

			auto avgTime = calculate_average(times);
			auto timeStdDev = calculate_std_dev(times);

			// Store resutls back in JSON
			ordered_json res;
			res["workgroupSize"] = n;
			res["avgTime"] = avgTime;
			res["stdDev"] = timeStdDev;
			res["numWorkgroups"] = occupancyBound;
			benchmarkResults.emplace_back(res);

			// Cleanup program.
			program.teardown();
			buf.teardown();
			buf_size.teardown();
			num_iters.teardown();
		}
		primitiveBarrierBenchmarkResults[entryPoints[i]] = benchmarkResults;
		std::cout << entryPoints[i] << " benchmark finished!\n";
    }

	device.teardown();
	std::cout << "Primitive barrier benchmark on " << device.properties.deviceName << " finished!\n\n";

	return primitiveBarrierBenchmarkResults;
}


int main(int argc, char* argv[]) {
	// BENCHMARK PARAMETERS
	auto numTrials = 32;
	auto numIters = 1024 * 16; // # of iterations to run kernel loop
	auto deviceIndex = 2;

	// Run benchmark on every availible device.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
	// Query device properties.
	auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
	auto bufferSize = 32768; // Size to set the local and global scratchpad buffers.

	// "warm up" the GPU by giving it some initial work. If this is not done the 
	// results of the first benchmark that is run will skewed for some reason.	
	// auto _ = primitiveBarrierBenchmark(instance, deviceIndex, 32, 1024);

	auto primitiveBarrierBenchmarkResult = primitiveBarrierBenchmark(instance, deviceIndex, numTrials, numIters, bufferSize);

	auto deviceName = device.properties.deviceName;
	primitiveBarrierBenchmarkResult["benchmarkName"] = "primitiveBarrier";
	primitiveBarrierBenchmarkResult["deviceName"] = deviceName;
	primitiveBarrierBenchmarkResult["numTrials"] = numTrials;
	primitiveBarrierBenchmarkResult["numIters"] = numIters;
	primitiveBarrierBenchmarkResult["bufferSize"] = bufferSize;

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
	std::this_thread::sleep_for(std::chrono::seconds(1));

	// Cleanup instance
	device.teardown();
	instance.teardown();
	return 0;
}