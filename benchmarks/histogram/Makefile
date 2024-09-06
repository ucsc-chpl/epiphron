CXXFLAGS=-std=c++17
VULKAN_INCLUDES=-L$(VULKAN_SDK)/Lib -lvulkan
RUN_EXT=run
SHADERS=shaders/histogram-global.spv shaders/histogram-shared-uint8.spv shaders/histogram-shared-uint16.spv shaders/histogram-shared-uint32.spv shaders/histogram-shared-uint64.spv shaders/histogram-multilevel.spv
ifeq ($(OS), Windows_NT)
	RUN_EXT = exe
endif

all: easyvk histogram.o $(SHADERS)
	$(CXX) $(CXXFLAGS) $(VULKAN_INCLUDES) -I../../easyvk/src ../../easyvk/build/easyvk.o histogram.o test_histogram.cpp -o histogram.$(RUN_EXT)

easyvk: 
	make -C ../../easyvk

histogram.o: histogram.cpp histogram.h 
	$(CXX) $(CXXFLAGS) -I../../easyvk/src histogram.cpp -c -o histogram.o

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

%.spv: %.comp
	glslc $< -o $@

clean:
	rm shaders/*.spv
	rm histogram.o
	rm histogram.$(RUN_EXT)