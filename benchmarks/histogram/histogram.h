#include <vector>
#include <algorithm>
#include <chrono>
#include "easyvk.h"

namespace histogram {
    enum Implementation : int {
        CPU = 0,
        GLOBAL = 1,
        SHARED_UINT8 = 2,
        SHARED_UINT16 = 3,
        SHARED_UINT32 = 4,
        SHARED_UINT64 = 5,
        MULTILEVEL = 6
    };

    class Histogram {
        public:
            Histogram(easyvk::Device device, uint32_t* data, uint64_t len, uint32_t num_bins, enum Implementation impl = SHARED_UINT32);
            std::vector<uint64_t> bins;
    };
}

