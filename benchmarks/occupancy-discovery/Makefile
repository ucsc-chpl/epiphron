CXXFLAGS = -std=c++17
CLSPVFLAGS = --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points -O0
SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
BENCHMARK = occupancy_discovery

.PHONY: all build clean easyvk

all: build easyvk build/$(BENCHMARK).run

easyvk: 
	make -C ../../easyvk

build/$(BENCHMARK).run: $(BENCHMARK).cpp $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o $< -lvulkan -o $@

android: build easyvk build/$(BENCHMARK).run
	ndk-build APP_BUILD_SCRIPT=./Android.mk NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

build/%.spv: %.cl
	clspv $(CLSPVFLAGS) $< -o $@

build/%.cinit: %.cl
	clspv $(CLSPVFLAGS) --output-format=c  $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build
