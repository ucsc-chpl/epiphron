CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard branched/*.cl contiguous_access/*.cl random_access/*.cl cross_warp/*.cl *.cl))
BENCHMARK = atomic_rmw_test
SOURCES = $(wildcard *cpp)

ifeq ($(OS),Darwin)
	CXXFLAGS += -rpath /usr/local/lib
endif

RUN_EXT = run
ifeq ($(OS), Windows_NT)
	RUN_EXT = exe
endif

all: results easyvk $(BENCHMARK)

results:
ifeq ($(OS), Windows_NT)
	if not exist "results" mkdir results
else
	mkdir -p results
endif

easyvk:
	make -C ../../easyvk 

$(BENCHMARK): $(SOURCES) $(CINITS)
	$(CXX) $(CXXFLAGS) -L$(VULKAN_SDK)/Lib -I../../easyvk/src -I. ../../easyvk/build/easyvk.o $(SOURCES) -lvulkan -o $(BENCHMARK).$(RUN_EXT)

android: easyvk atomic_rmw_test
	ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -r branched/*.cinit \
	branched/*.spv \
	contiguous_access/*.cinit \
	contiguous_access/*.spv \
	random_access/*.cinit \
	random_access/*.spv \
	cross_warp/*.cinit \
	cross_warp/*.spv \
	*.cinit \
	*.spv \
	build \
	*.run \
	*.o 
