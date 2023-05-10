CXXFLAGS = -std=c++17

all: easyvk kernel_launch_test

easyvk:
	make -C ../../easyvk

kernel_launch_test: kernel_launch_test.cpp vect-add.spv vect-add.cinit
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o kernel_launch_test.cpp -lvulkan -o kernel_launch_test.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm *.run
	rm *.cinit
	rm *.spv
