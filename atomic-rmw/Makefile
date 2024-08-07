CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard branched/*.cl NVIDIA_instance_access/*.cl contiguous_access/*.cl random_access/*.cl cross_warp/*.cl *.cl))
BENCHMARK = atomic_rmw_test
SOURCES = $(wildcard *cpp)
RUN_EXT = run
RM_CMD = rm -r
SEP = /
ifeq ($(OS), Windows_NT)
	RUN_EXT = exe
	RM_CMD = del /q /s
	SEP = \\
endif

all: results easyvk $(BENCHMARK)

results:
ifeq ($(OS), Windows_NT)
	if not exist "results" mkdir results
else
	mkdir -p results
endif

easyvk:
	make -C ../easyvk 

$(BENCHMARK): $(SOURCES) $(CINITS)
	$(CXX) $(CXXFLAGS) -L$(VULKAN_SDK)/Lib -I../../easyvk/src -I. ../../easyvk/build/easyvk.o $(SOURCES) -lvulkan -o $(BENCHMARK).$(RUN_EXT)

android: easyvk atomic_rmw_test
	ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

%.spv: %.cl
	../clspv/clspv.exe -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	../clspv/clspv.exe -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	$(RM_CMD) branched$(SEP)*.cinit \
	branched$(SEP)*.spv \
	NVIDIA_instance_access$(SEP)*.cinit \
	NVIDIA_instance_access$(SEP)*.spv \
	contiguous_access$(SEP)*.cinit \
	contiguous_access$(SEP)*.spv \
	random_access$(SEP)*.cinit \
	random_access$(SEP)*.spv \
	cross_warp$(SEP)*.cinit \
	cross_warp$(SEP)*.spv \
	*.cinit \
	*.spv \
	build \
	*.$(RUN_EXT) \
	*.o \
