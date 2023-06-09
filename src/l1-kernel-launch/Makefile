CXXFLAGS = -std=c++17

SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))

.PHONY: all build clean easyvk kernel_launch_test

all: build easyvk kernel_launch_test

build:
	mkdir -p build

easyvk:
	make -C ../../easyvk

kernel_launch_test: kernel_launch_test.cpp $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o kernel_launch_test.cpp -lvulkan -o build/kernel_launch_test.run

build/%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

build/%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf build
