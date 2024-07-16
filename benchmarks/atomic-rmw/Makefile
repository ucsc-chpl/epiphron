CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard branched/*.cl contiguous_access/*.cl random_access/*.cl cross_warp/*.cl *.cl))
BENCHMARK = atomic_rmw_test
SOURCES = $(wildcard *cpp)
OS := $(shell uname)
ifeq ($(OS),Darwin)
$(info OS is $(OS))
CXXFLAGS += -rpath /usr/local/lib
endif

all: results easyvk $(BENCHMARK)

results:
	mkdir -p results

easyvk:
	make -C ../../easyvk

$(BENCHMARK): $(SOURCES) $(CINITS)
	$(CXX) $(CXXFLAGS) -I../../easyvk/src -I. ../../easyvk/build/easyvk.o $(SOURCES) -lvulkan -o $(BENCHMARK).run

android: easyvk atomic_rmw_test
	ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf branched/*.cinit
	rm -rf branched/*.spv
	rm -rf contiguous_access/*.cinit
	rm -rf contiguous_access/*.spv
	rm -rf random_access/*.cinit
	rm -rf random_access/*.spv
	rm -rf cross_warp/*.cinit
	rm -rf cross_warp/*.spv
	rm -rf *.cinit
	rm -rf *.spv
	rm -rf build
	rm *.run
	rm *.o
