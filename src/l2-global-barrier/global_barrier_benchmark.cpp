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


void ticket_lock_test(size_t deviceIndex) {
	// Set up instance.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
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
    // assert(counter_buf.load(0) == numWorkgroups * workgroupSize * 256);
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
    device.teardown();
	instance.teardown();
}

/**
 * @brief Runs the global barrier benchmark.
 * 
 * @param deviceIndex     Index of the physical device to use for the test.
 * @param numWorkgroups   Number of workgroups to launch for the benchmark.
 * @param workgroupSize   Size of each workgroup that will be launched.
 * @param numIters        Number of iterations that the barrier will be invoked in each trial.
 * @param numTrials       Number of times the entire test will be repeated.
 * 
 * @return ordered_json   JSON object containing the benchmark results.
 */
ordered_json global_barrier_benchmark(size_t deviceIndex, 
                                      size_t numWorkgroups,
                                      size_t workgroupSize, 
                                      size_t numIters,
                                      size_t numTrials) {
	// Set up instance.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);

    ordered_json testResults;

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));

    // Loader shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/global_barrier.cinit"
    ;
    auto entry_point = "global_barrier";

    std::vector<double> trials(numTrials);
    for (int i = 0; i < numTrials; i++) {
        // Set up buffers.
        auto count_buf = easyvk::Buffer(device, 1);
        auto poll_open_buf = easyvk::Buffer(device, 1);
        auto M_buf = easyvk::Buffer(device, numWorkgroups);
        auto now_serving_buf = easyvk::Buffer(device, 1);
        auto next_ticket_buf = easyvk::Buffer(device, 1);
        auto flag_buf = easyvk::Buffer(device, numWorkgroups);
        auto output_buf = easyvk::Buffer(device, numWorkgroups);
        auto num_iters_buf = easyvk::Buffer(device, 1);
        count_buf.store(0, 0);
        poll_open_buf.store(0, 1); // Poll is initially open.
        next_ticket_buf.store(0, 0);
        now_serving_buf.store(0, 0);
        num_iters_buf.store(0, numIters);
        for (int j = 0; j < numWorkgroups; j++) {
            flag_buf.store(j, 0);
            output_buf.store(j, 0);
        }

        std::vector<easyvk::Buffer> kernelInputs = {count_buf, 
                                                    poll_open_buf,
                                                    M_buf,
                                                    now_serving_buf,
                                                    next_ticket_buf,
                                                    flag_buf,
                                                    output_buf,
                                                    num_iters_buf};

        // Initialize the kernel.
        auto program = easyvk::Program(device, spvCode, kernelInputs);
        program.setWorkgroups(numWorkgroups);
        program.setWorkgroupSize(workgroupSize);
        program.initialize(entry_point);

        // Launch kernel.
        auto kernelTime = program.runWithDispatchTiming();
        trials[i] = kernelTime / (double) 1000.0; // Convert to us

        // Check the safety and correctness of the barrier.
        for (int j = 0; j < count_buf.load(0); j++) {
            // Each position in the buf corresponding to a participating workgroup 
            // should be incremented exactly numIters times.
            assert(output_buf.load(i) == numIters);
        }

        // Cleanup.
        program.teardown();
        count_buf.teardown();
        poll_open_buf.teardown();
        M_buf.teardown();
        next_ticket_buf.teardown();
        now_serving_buf.teardown();
        flag_buf.teardown();
        output_buf.teardown();
        num_iters_buf.teardown(); 
    }

    // Save benchmark results to JSON.
    auto avgTime = calculate_average(trials);
    auto timeStdDev = calculate_std_dev(trials);
    auto timeCV = calculate_coeff_variation(trials);
    testResults["avgTime"] = avgTime;
    testResults["timeStdDev"] = timeStdDev;
    testResults["timeCV"] = timeCV;

    device.teardown();
    instance.teardown();

    return testResults;
}

/**
 * @brief Runs the kernel barrier benchmark.
 * 
 * @param deviceIndex     Index of the physical device to use for the test.
 * @param numWorkgroups   Number of workgroups to launch for the benchmark.
 * @param workgroupSize   Size of each workgroup that will be launched.
 * @param numIters        Number of iterations that the barrier will be invoked in each trial.
 * @param numTrials       Number of times the entire test will be repeated.
 * 
 * @return ordered_json   JSON object containing the benchmark results.
 */
ordered_json kernel_barrier_benchmark(size_t deviceIndex, 
                                      size_t numWorkgroups,
                                      size_t workgroupSize, 
                                      size_t numIters,
                                      size_t numTrials) {
	// Set up instance.
	auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);

    ordered_json testResults;

    // Select device.
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));

    // Loader shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/global_barrier.cinit"
    ;
    auto entry_point = "global_barrier";

    std::vector<double> trials(numTrials);
    for (int i = 0; i < numTrials; i++) {
        // Set up buffers.
        auto count_buf = easyvk::Buffer(device, 1);
        auto poll_open_buf = easyvk::Buffer(device, 1);
        auto M_buf = easyvk::Buffer(device, numWorkgroups);
        auto now_serving_buf = easyvk::Buffer(device, 1);
        auto next_ticket_buf = easyvk::Buffer(device, 1);
        auto flag_buf = easyvk::Buffer(device, numWorkgroups);
        auto output_buf = easyvk::Buffer(device, numWorkgroups);
        auto num_iters_buf = easyvk::Buffer(device, 1);
        count_buf.store(0, 0);
        poll_open_buf.store(0, 1); // Poll is initially open.
        next_ticket_buf.store(0, 0);
        now_serving_buf.store(0, 0);
        num_iters_buf.store(0, numIters);
        for (int j = 0; j < numWorkgroups; j++) {
            flag_buf.store(j, 0);
            output_buf.store(j, 0);
        }

        std::vector<easyvk::Buffer> kernelInputs = {count_buf, 
                                                    poll_open_buf,
                                                    M_buf,
                                                    now_serving_buf,
                                                    next_ticket_buf,
                                                    flag_buf,
                                                    output_buf,
                                                    num_iters_buf};

        // Initialize the kernel.
        auto program = easyvk::Program(device, spvCode, kernelInputs);
        program.setWorkgroups(numWorkgroups);
        program.setWorkgroupSize(workgroupSize);
        program.initialize(entry_point);

        // Launch kernel.
        auto kernelTime = program.runWithDispatchTiming();
        trials[i] = kernelTime / (double) 1000.0; // Convert to us

        // Check the safety and correctness of the barrier.
        for (int j = 0; j < count_buf.load(0); j++) {
            // Each position in the buf corresponding to a participating workgroup 
            // should be incremented exactly numIters times.
            assert(output_buf.load(i) == numIters);
        }

        // Cleanup.
        program.teardown();
        count_buf.teardown();
        poll_open_buf.teardown();
        M_buf.teardown();
        next_ticket_buf.teardown();
        now_serving_buf.teardown();
        flag_buf.teardown();
        output_buf.teardown();
        num_iters_buf.teardown(); 
    }

    // Save benchmark results to JSON.
    auto avgTime = calculate_average(trials);
    auto timeStdDev = calculate_std_dev(trials);
    auto timeCV = calculate_coeff_variation(trials);
    testResults["avgTime"] = avgTime;
    testResults["timeStdDev"] = timeStdDev;
    testResults["timeCV"] = timeCV;

    device.teardown();
    instance.teardown();

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
    testResults["testName"] = "Global Barrier";
    testResults["deviceName"] = deviceName;

    auto numWorkgroups = 1024;
    auto workgroupSize = 256;
    auto numIters = 1024;
    auto numTrials = 32;

    auto globalBarrierResults = global_barrier_benchmark(deviceIndex, 
                                                         numWorkgroups, 
                                                         workgroupSize,
                                                         numIters,
                                                         numTrials);
    std::cout << globalBarrierResults.dump(4) << "\n";

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