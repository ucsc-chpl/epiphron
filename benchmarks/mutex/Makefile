CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard *.cl))
all: easyvk mutex_test

easyvk:
	make -C ../../easyvk

mutex_test: mutex_test.cpp $(CINITS)
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o mutex_test.cpp -lvulkan -o mutex_test.run

android: easyvk mutex_test
	/home/gdhillon/Research/android-ndk-r26/ndk-build APP_BUILD_SCRIPT=./Android.mk  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk NDK_LIBS_OUT=./build/android/libs NDK_OUT=./build/android/obj

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf *.cinit
	rm -rf *.spv
	rm -rf build
	rm *.run
