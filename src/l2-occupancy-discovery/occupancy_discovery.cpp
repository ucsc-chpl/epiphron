#include <assert.h>
#include <chrono>
#include <iostream>
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
// any previously defined constants.
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

double calculate_coeff_variation(const std::vector<double>& values) {
    double mean = calculate_average(values);
    if (mean == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sd = calculate_std_dev(values);
    return (sd / mean);
}


/**
 * @brief Runs an occupancy discovery protocol.
 * 
 * This test will run an occupancy discovery protocol on the given device.
 * The protocol is parameterized by the number of workgroups to be launched, the 
 * workgroup size, and the amount of local memory for the kernel to use, which all 
 * effect occupancy.
 * @param deviceIndex   The physical device index to use.
 * @param numTrials     How many times the protocol will be repeated.
 * @param numWorkgroups Number of workgroups which are dispatched.
 * @param workgroupSize Size of workgroups to be dispatched.
 * @param localMemSize  Amount of local memory the kernel will use.
 * @return JSON object containing results of the test.
*/
ordered_json occupancy_discovery_test(size_t deviceIndex, 
                                      size_t numTrials,
                                      size_t numWorkgroups, 
                                      size_t workgroupSize, 
                                      size_t localMemSize) {
                            
    if (workgroupSize == 0) {
        workgroupSize = 1;
    }
    // Save test results to JSON.
    ordered_json testResults;

	// Set up instance.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));

    // Load kernel.
    std::vector<uint32_t> spvCode = 
    #include "build/occupancy_discovery.cinit"
    ;
    auto entry_point = "occupancy_discovery";
    modifyLocalMemSize(spvCode, localMemSize);


    uint32_t maxOccupancyBound = 0;
    std::vector<double> trials(numTrials);
    for (int i = 0; i < numTrials; i++) {
        // Set up buffers.
        auto count_buf = easyvk::Buffer(device, 1);
        auto poll_open_buf = easyvk::Buffer(device, 1);
        auto M_buf = easyvk::Buffer(device, numWorkgroups);
        auto now_serving_buf = easyvk::Buffer(device, 1);
        auto next_ticket_buf = easyvk::Buffer(device, 1);
        auto local_mem_size_buf = easyvk::Buffer(device, 1);
        count_buf.store(0, 0);
        poll_open_buf.store(0, 1); // Poll is initially open.
        next_ticket_buf.store(0, 0);
        now_serving_buf.store(0, 0);
        local_mem_size_buf.store(0, localMemSize);

        std::vector<easyvk::Buffer> kernelInputs = {count_buf, 
                                                    poll_open_buf,
                                                    M_buf,
                                                    now_serving_buf,
                                                    next_ticket_buf,
                                                    local_mem_size_buf};

        // Initialize the kernel.
        auto program = easyvk::Program(device, spvCode, kernelInputs);
        program.setWorkgroups(numWorkgroups);
        program.setWorkgroupSize(workgroupSize);
        program.initialize(entry_point);

        // Launch kernel.
        program.run();

        trials[i] = (double) count_buf.load(0);
        if (count_buf.load(0) > maxOccupancyBound) {
            maxOccupancyBound = count_buf.load(0);
        }

        // Cleanup.
        program.teardown();
        count_buf.teardown();
        poll_open_buf.teardown();
        M_buf.teardown();
        next_ticket_buf.teardown();
        now_serving_buf.teardown();
    }
    device.teardown();
    instance.teardown();

    auto avgOccupancyBound = calculate_average(trials);
    auto stdDev = calculate_std_dev(trials);
    auto cv = calculate_coeff_variation(trials);
    testResults["maxOccupancyBound"] = maxOccupancyBound;
    testResults["avgOccupancyBound"] = avgOccupancyBound;
    testResults["stdDev"] = stdDev;
    testResults["CV"] = cv;

    return testResults;
}


int main(int argc, char* argv[]) {
    auto deviceIndex = 0;

    // Query device properties.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    // Divide by four because we are using buffers of uint32_t
    auto maxLocalMemSize = device.properties.limits.maxComputeSharedMemorySize / 4;
    auto maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];
    // std::cout << "maxLocalMemSize: " << maxLocalMemSize << "\n";
    // std::cout << "maxWorkgroupSize: " << maxWorkgroupSize << "\n";
    device.teardown();
    instance.teardown();

    // Save all results to JSON.
    ordered_json testResults;
    testResults["testName"] = "Occupancy Discovery";
    testResults["deviceName"] = deviceName;

    auto numTrials = 4;
    auto numWorkgroups = 1024;
    testResults["numWorkgroups"] = numWorkgroups;
    auto numIters = 1024;
    auto workgroupStepSize = maxWorkgroupSize / 8;
    auto localMemStepSize = maxLocalMemSize / 8;

    std::vector<ordered_json> res;
    for (int localMemSize = 0; 
            localMemSize <= maxLocalMemSize;
            localMemSize += localMemStepSize) {

        for (int workgroupSize = 0;
            workgroupSize <= maxWorkgroupSize; 
            workgroupSize += workgroupStepSize) {

            auto occupancy_res = occupancy_discovery_test(deviceIndex, 
                                                        numTrials, 
                                                        numWorkgroups, 
                                                        workgroupSize,
                                                        localMemSize);
            if (workgroupSize == 0) {
                occupancy_res["workgroupSize"] = 1;
            } else {
                occupancy_res["workgroupSize"] = workgroupSize;
            }
            occupancy_res["localMemSize"] = localMemSize;
            res.emplace_back(occupancy_res);
        }
    }

    testResults["results"] = res;

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
		outFile << testResults.dump(4) << std::endl;
		outFile.close();
	} else {
		std::cerr << "Failed to write test results to file!\n";
	}


	return 0;
}