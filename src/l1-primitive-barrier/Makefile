CXXFLAGS = -std=c++17

SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
BENCHMARK = primitive_barrier_benchmark

.PHONY: all build clean easyvk kernel_launch_test

all: build easyvk kernel_launch_test

build:
	mkdir -p build

easyvk:
	make -C ../../easyvk

kernel_launch_test: $(BENCHMARK).cpp $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o  $(BENCHMARK).cpp -lvulkan -o build/$(BENCHMARK).run

build/%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

build/%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf build
