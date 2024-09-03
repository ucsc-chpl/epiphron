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
    std::srand(std::time(nullptr));
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
            data_size = (1llu << 32) - 1;
            break;
    }
    data.resize(data_size / sizeof(uint32_t));


    printf("Select input data type:\n");
    printf("0 - 'random'\n");
    printf("1 - 'ascending'\n");
    printf("2 - 'zeroes'\n");
    printf("Enter data type number: ");

    std::string dt_s;
    getline(std::cin, dt_s);
    printf("\n");
    int dt = stoi(dt_s);

    printf("Generating data...\n\n");
    switch (dt) {
        case 0:
            std::generate(data.begin(), data.end(), std::rand);
            break;
        case 1:
            std::iota(data.begin(), data.end(), 0);
            break;
        case 2:
            std::fill(data.begin(), data.end(), 0);
            break;
    }

    printf("Enter maximum number of bins (will sweep upwards in powers of 2, starting from 1):\n");
    printf("Bins: ");
    std::string bins_s;
    getline(std::cin, bins_s);
    printf("\n");
    int bins = stoi(bins_s);

    for (int b = 1; b <= bins; b <<= 1) {
        printf("Starting histogram with %d bins...\n", b);

        histogram::Histogram histogram = histogram::Histogram(device, data.data(), data.size(), b);

        printf("Bins: [");
        for (int i = 0; i < b; i++) {
            printf("%d", histogram.bins[i]);
            if (i < b - 1)
                printf(", ");
        }
        printf("]\n");
    }

    device.teardown();
    instance.teardown();
    return 0;
}