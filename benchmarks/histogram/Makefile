CXXFLAGS=-std=c++17
VULKAN_INCLUDES=-L$(VULKAN_SDK)/Lib -lvulkan
RUN_EXT=run
ifeq ($(OS), Windows_NT)
	RUN_EXT = exe
endif

all: easyvk histogram.o
	$(CXX) $(CXXFLAGS) $(VULKAN_INCLUDES) -I../../easyvk/src ../../easyvk/build/easyvk.o histogram.o test_histogram.cpp -o histogram.$(RUN_EXT)

easyvk: 
	make -C ../../easyvk

histogram.o: histogram.cpp histogram.h histogram.spv
	$(CXX) $(CXXFLAGS) -I../../easyvk/src histogram.cpp -c -o histogram.o

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

%.spv: %.comp
	glslc $< -o $@

clean:
	rm histogram.spv
	rm histogram.o
	rm histogram.$(RUN_EXT)