#include <vector>
#include "easyvk.h"

namespace histogram {
    class Histogram {
        public:
            Histogram(easyvk::Device device, uint32_t* data, uint64_t len, uint32_t num_bins);
            std::vector<uint32_t> bins;
    };
}

