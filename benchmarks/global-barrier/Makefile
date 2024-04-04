CXXFLAGS = -std=c++17
CLSPVFLAGS = --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points -O0

SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
BENCHMARK = global_barrier_benchmark

.PHONY: all build clean easyvk $(BENCHMARK)

all: build easyvk $(BENCHMARK) 

build:
	mkdir -p build

easyvk:
	make -C ../../easyvk

$(BENCHMARK): $(BENCHMARK).cpp $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o  $(BENCHMARK).cpp -lvulkan -o build/$(BENCHMARK).run

android: build easyvk $(BENCHMARK)
	ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

build/%.spv: %.cl
	clspv $(CLSPVFLAGS) $< -o $@

build/%.cinit: %.cl
	clspv $(CLSPVFLAGS) --output-format=c  $< -o $@

clean:
	rm -rf build
