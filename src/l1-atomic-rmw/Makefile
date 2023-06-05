CXXFLAGS = -std=c++17

all: easyvk atomic_rmw_test

easyvk:
	make -C ../../easyvk

atomic_rmw_test: atomic_rmw_test.cpp atomic_cas_succeed_store.cinit atomic_cas_succeed_no_store.cinit atomic_cas_fail_no_store.cinit atomic_fa.cinit atomic_fa_relaxed.cinit atomic_ex.cinit atomic_ex_relaxed.cinit
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o atomic_rmw_test.cpp -lvulkan -o atomic_rmw_test.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm *.run
	rm *.cinit
	rm *.spv
