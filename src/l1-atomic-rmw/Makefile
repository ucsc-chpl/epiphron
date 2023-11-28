CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard branched/*.cl contiguous_access/*.cl random_access/*.cl cross_warp/*.cl *.cl))
all: easyvk atomic_rmw_test

easyvk:
	make -C ../../easyvk

atomic_rmw_test: atomic_rmw_test.cpp $(CINITS)
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o atomic_rmw_test.cpp -lvulkan -o atomic_rmw_test.run

android: easyvk atomic_rmw_test
	/home/gdhillon/Research/android-ndk-r26/ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

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
