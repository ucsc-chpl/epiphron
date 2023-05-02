CXXFLAGS = -std=c++17

all: easyvk vk_lock_test

easyvk:
	make -C ../../easyvk

vk_lock_test: vk_lock_test.cpp tas_lock.cinit ttas_lock.cinit cas_lock.cinit
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o vk_lock_test.cpp -lvulkan -o vk_lock_test.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm *.run
	rm *.cinit
	rm *.spv
