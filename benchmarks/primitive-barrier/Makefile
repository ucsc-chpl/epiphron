CXXFLAGS = -std=c++17
CLSPVFLAGS = --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points

SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
BENCHMARK = primitive_barrier_benchmark

.PHONY: all build clean easyvk kernel_launch_test

all: build easyvk kernel_launch_test

build:
	mkdir -p build
	mkdir -p build/android

easyvk:
	make -C ../../easyvk

kernel_launch_test: $(BENCHMARK).cpp $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o  $(BENCHMARK).cpp -lvulkan -o build/$(BENCHMARK).run

android: build easyvk kernel_launch_test
	ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

build/%.spv: %.cl
	clspv $(CLSPVFLAGS) $< -o $@

build/%.cinit: %.cl
	clspv $(CLSPVFLAGS) --output-format=c  $< -o $@

clean:
	rm -rf build
