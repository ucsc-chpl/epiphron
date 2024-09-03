#include <vector>
#include "easyvk.h"

namespace histogram {
    enum BinType {
        UINT8 = 0,
        UINT16 = 1,
        UINT32 = 2,
        UINT64 = 3
    };

    class Histogram {
        public:
            Histogram(easyvk::Device device, uint32_t* data, uint64_t len, uint32_t num_bins, enum BinType binType = UINT32);
            std::vector<uint64_t> bins;
    };
}

