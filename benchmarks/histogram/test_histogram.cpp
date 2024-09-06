#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "easyvk.h"
#include "histogram.h"

#define USE_VALIDATION_LAYERS true

using namespace easyvk;
using histogram::Histogram;

int main() {
    Instance instance = Instance(USE_VALIDATION_LAYERS);
    std::vector<VkPhysicalDevice> devices = instance.physicalDevices();
    printf("Testing GPU histogram...\n\n");

    printf("Select a device: \n");
    for (int i = 0; i < devices.size(); i++) {
        Device device = Device(instance, devices[i]);
        printf("%d - '%s'\n", i, device.properties.deviceName);
        device.teardown();
    }
    printf("Enter device number: ");

    std::string d_s;
    getline(std::cin, d_s);
    printf("\n");
    int d = stoi(d_s);

    if (d < 0 || d >= devices.size()) {
        fprintf(stderr, "Incorrect device number '%d'!", d);
        exit(1);
    }

    Device device = Device(instance, devices[d]);
    printf("Using '%s'...\n", device.properties.deviceName);

    std::vector<uint32_t> data;

    printf("Select input data size:\n");
    printf("0 - '1KB'\n");
    printf("1 - '1MB'\n");
    printf("2 - '1GB'\n");
    printf("3 - '2GB'\n");
    printf("4 - '4GB'\n");
    printf("Enter data size: ");

    std::string ds_s;
    getline(std::cin, ds_s);
    printf("\n");
    int ds = stoi(ds_s);

    uint64_t data_size = 0;
    switch(ds) {
        case 0: 
            data_size = 1llu << 10;
            break;
        case 1:
            data_size = 1llu << 20;
            break;
        case 2:
            data_size = 1llu << 30;
            break;
        case 3:
            data_size = 1llu << 31;
            break;
        case 4:
            data_size = (1llu << 32) - (1llu << 20); // minus one MB to avoid overflow
            break;
    }
    uint32_t data_len = data_size / sizeof(uint32_t);
    data.resize(data_len);
    
    printf("Select input data type:\n");
    printf("0 - 'random'\n");
    printf("1 - 'ascending'\n");
    printf("2 - 'zeroes'\n");
    printf("3 - 'seeded-random'\n");
    printf("Enter data type number: ");

    std::string dt_s;
    getline(std::cin, dt_s);
    printf("\n");
    int dt = stoi(dt_s);

    printf("Generating data...\n\n");
    switch (dt) {
        case 0:
            srand(time(NULL));
            std::generate(data.begin(), data.end(), std::rand);
            break;
        case 1:
            std::iota(data.begin(), data.end(), 0);
            break;
        case 2:
            std::fill(data.begin(), data.end(), 0);
            break;
        case 3:
            srand(0);
            std::generate(data.begin(), data.end(), std::rand);
            break;
    }

    printf("Select histogram implementation:\n");
    printf("0 - 'CPU'\n");
    printf("1 - 'GLOBAL'\n");
    printf("2 - 'SHARED_UINT8'\n");
    printf("3 - 'SHARED_UINT16'\n");
    printf("4 - 'SHARED_UINT32'\n");
    printf("5 - 'SHARED_UINT64'\n");
    printf("6 - 'MULTILEVEL'\n");
    printf("Enter implementation: ");

    std::string impl_s;
    getline(std::cin, impl_s);
    printf("\n");
    histogram::Implementation impl = static_cast<histogram::Implementation>(stoi(impl_s));

    int num_bins = 256;

    printf("Starting histogram with %d bins...\n", num_bins);

    histogram::Histogram histogram = histogram::Histogram(device, data.data(), data.size(), num_bins, impl);

    uint64_t sum = 0;
    printf("Bins: [");
    for (int i = 0; i < num_bins; i++) {
        printf("%llu", histogram.bins[i]);
        if (i < num_bins - 1)
            printf(", ");
        sum += histogram.bins[i];
    }
    printf("]\n");
    printf("Sum: %llu\n", sum);
    printf("Data size: %zu\n", data.size());

    if (impl != histogram::Implementation::CPU) {
        printf("Checking histogram against CPU histogram...\n");
        histogram::Histogram cpu_histogram = histogram::Histogram(device, data.data(), data.size(), num_bins, histogram::Implementation::CPU);
        bool err = false;
        for (int i = 0; i < num_bins; i++)
            if (histogram.bins[i] != cpu_histogram.bins[i]) 
                err = true;
        if (err)
            printf("Error in histogram calculation!\n");
        else
            printf("Histogram bins are correct!\n");
    }


    device.teardown();
    instance.teardown();
    return 0;
}