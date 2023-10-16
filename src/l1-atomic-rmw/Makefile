CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard chunking/*.cl striding/*.cl))
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
	rm -rf chunking/*.cinit
	rm -rf chunking/*.spv
	rm -rf striding/*.cinit
	rm -rf striding/*.spv
	rm -rf build
	rm *.run
