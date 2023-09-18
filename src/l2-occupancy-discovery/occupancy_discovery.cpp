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
// any previously defined constant (i.e ensure that the #define constant you wish to
// re-define is at the top of the OpenCL file has the value as defined).
const uint32_t LOCAL_MEM_SIZE = 1024;
std::vector<uint32_t> modifyLocalMemSize(const std::vector<uint32_t> &spirv, uint32_t newValue) {
    std::vector<uint32_t> modifiedSpirv = spirv;

    if (spirv.size() < 5) {
        std::cerr << "Invalid SPIR-V binary." << std::endl;
        return modifiedSpirv;
    }

    if (spirv[0] != SPIRV_MAGIC) {
        std::cerr << "Not a SPIR-V binary." << std::endl;
        return modifiedSpirv;
    }
    // https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/parsing_instructions.md
    size_t i = 5;  // skip SPIR-V header
    while (i < spirv.size()) {
        uint32_t instruction = spirv[i];
        uint32_t length = instruction >> 16;
        uint32_t opcode = instruction & 0x0ffffu;

        // Opcode source:
        // https://registry.khronos.org/SPIR-V/specs/unified-1/SPIRV.html#OpConstant
        if (opcode == 43) {  // OpConstant
            uint32_t resultType = spirv[i + 1];
            uint32_t resultId = spirv[i + 2];
            uint32_t constantValue = spirv[i + 3];

            // This is a simplistic check.
            // Doesn't verify the type and name (through debug info)
            if (constantValue == LOCAL_MEM_SIZE) {
                modifiedSpirv[i + 1] = newValue;
                return modifiedSpirv;
            }
        }

        i += length;  // move to next instruction
    }

    std::cerr << "Did not modify any instructions when parsing the SPIR-V module!\n";
    return modifiedSpirv;
}

double calculate_average(const std::vector<double> &values) {
    double sum = 0.0;
    int numElements = values.size();

    // Calculate the sum of all elements
    for (const double &value : values) {
        sum += value;
    }

    // Calculate and return the average
    return numElements > 0 ? sum / numElements : 0.0;
}

double calculate_std_dev(const std::vector<double> &values) {
    double mean = calculate_average(values);
    double squaredDifferenceSum = 0.0;
    int numElements = values.size();

    // Calculate the sum of squared differences from the mean
    for (const double &value : values) {
        double difference = value - mean;
        squaredDifferenceSum += difference * difference;
    }

    // Calculate and return the standard deviation
    return numElements > 0 ? std::sqrt(squaredDifferenceSum / numElements) : 0.0;
}

double calculate_coeff_variation(const std::vector<double> &values) {
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
 * workgroup size, and the amount of local memory for the kernel to use, which
 * all effect occupancy.
 * @param instance
 * @param deviceIndex       The physical device index to use.
 * @param numTrials         How many times the protocol will be repeated.
 * @param maxWorkgroupSize
 * @param maxLocalMemSize
 * @param workgroupStepSize
 * @param localMemStepSize
 * @return JSON object containing results of the test.
 */
std::vector<ordered_json> occupancy_discovery_test(easyvk::Instance instance, 
                                                   size_t deviceIndex,
                                                   size_t numTrials,
                                                   size_t numWorkgroups,
                                                   size_t maxWorkgroupSize,
                                                   size_t maxLocalMemSize,
                                                   size_t stepDivisor) {
    // Save test results to JSON.
    std::vector<ordered_json> testResults;

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    std::cout << "Running occupancy discovery test on " << device.properties.deviceName << "...\n";

    int workgroupStepSize = maxWorkgroupSize / stepDivisor;
    int localMemStepSize = maxLocalMemSize / stepDivisor;

    for (size_t workgroupSize = 1; workgroupSize < maxWorkgroupSize; workgroupSize += workgroupStepSize) {
        for (size_t localMemSize = 1; localMemSize < maxLocalMemSize; localMemSize += localMemStepSize) {

            ordered_json res;
            uint32_t maxOccupancyBound = 0; // Keep track of the max bound we find.

            // Load kernel.
            std::vector<uint32_t> spvCode =
#include "build/occupancy_discovery.cinit"
            ;

            auto entry_point = "occupancy_discovery";
            auto modifiedSpirv = modifyLocalMemSize(spvCode, localMemSize);

            std::vector<double> trials(numTrials);
            for (int i = 0; i < numTrials; i++) {
                // Set up buffers.
                auto count_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
                auto poll_open_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
                auto M_buf = easyvk::Buffer(device, numWorkgroups, sizeof(uint32_t));
                auto now_serving_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
                auto next_ticket_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
                auto local_mem_size_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
                count_buf.store<uint32_t>(0, 0);
                poll_open_buf.store<uint32_t>(0, 1);  // Poll is initially open.
                next_ticket_buf.store<uint32_t>(0, 0);
                now_serving_buf.store<uint32_t>(0, 0);
                local_mem_size_buf.store<uint32_t>(0, localMemSize);

                std::vector<easyvk::Buffer> kernelInputs = {
                    count_buf, poll_open_buf, M_buf, now_serving_buf, next_ticket_buf, local_mem_size_buf};

                // Initialize the kernel.
                auto program = easyvk::Program(device, modifiedSpirv, kernelInputs);
                program.setWorkgroups(numWorkgroups);
                program.setWorkgroupSize(workgroupSize);
                program.initialize(entry_point);

                // Launch kernel.
                program.runWithDispatchTiming();

                trials[i] = (double)count_buf.load<uint32_t>(0);
                if (count_buf.load<uint32_t>(0) > maxOccupancyBound) {
                    maxOccupancyBound = count_buf.load<uint32_t>(0);
                }

                // Cleanup.
                program.teardown();
                count_buf.teardown();
                poll_open_buf.teardown();
                M_buf.teardown();
                next_ticket_buf.teardown();
                now_serving_buf.teardown();
            }

            auto avgOccupancyBound = calculate_average(trials);
            auto stdDev = calculate_std_dev(trials);
            auto cv = calculate_coeff_variation(trials);
            res["maxOccupancyBound"] = maxOccupancyBound;
            res["avgOccupancyBound"] = avgOccupancyBound;
            res["stdDev"] = stdDev;
            res["CV"] = cv;
            res["workgroupSize"] = workgroupSize;
            res["localMemSize"] = localMemSize;
            testResults.emplace_back(res);
        }
    }

    std::cout << "Occupancy discovery test on " << device.properties.deviceName << " finished!\n\n";

    return testResults;
}

int main(int argc, char *argv[]) {
    // BENCHMARK PARAMETERS
    auto numTrials = 8;
    // The maximum workgroup size to test. Set to -1 to go up to the max allowed
    // workgroup size supported by the device.
    auto maxWorkgroupSize = -1;
    // The maximum size of local memory to test. Set to -1 to go up to the max
    // allowed by the device.
    auto maxLocalMemSize = -1;
    auto numWorkgroups = 1024 * 4;
    // The test will be parameterized from 1 to {maxWorkgroupSize,
    // maxLocalMemSize}, the below param will determine w/ what granularity to
    // sample (e.g higher stepDivisor will increase number of samples and significantly 
    // increase benchmark time.
    auto stepDivisor = 8;

    // Run the test on every availible device.
    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    for (size_t deviceIndex = 0; deviceIndex < instance.physicalDevices().size(); deviceIndex++) {
        auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
        auto deviceName = device.properties.deviceName;
        if (maxWorkgroupSize == -1) {
            maxWorkgroupSize = device.properties.limits.maxComputeWorkGroupSize[0];
        }
        if (maxLocalMemSize == -1) {
            maxLocalMemSize = device.properties.limits.maxComputeSharedMemorySize / 4; 
        }
        device.teardown();

        // Save all results to JSON.
        ordered_json testResults;
        testResults["testName"] = "Occupancy Discovery";
        testResults["deviceName"] = deviceName;
        testResults["numWorkgroups"] = numWorkgroups;

        auto occupancyResults = occupancy_discovery_test(instance, 
                                                         deviceIndex,
                                                         numTrials,
                                                         numWorkgroups,
                                                         maxWorkgroupSize, 
                                                         maxLocalMemSize,
                                                         stepDivisor);
        testResults["results"] = occupancyResults;

        // Write results to file.
        // Get current time
        std::time_t currentTime = std::time(nullptr);
        std::tm *currentDateTime = std::localtime(&currentTime);

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
    }
    return 0;
}