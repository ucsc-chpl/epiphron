CXXFLAGS = -std=c++17

all: easyvk atomic_rmw_test

easyvk:
	make -C ../../easyvk

atomic_rmw_test: atomic_rmw_test.cpp chunking/atomic_cas_succeed_store.cinit chunking/atomic_cas_succeed_no_store.cinit chunking/atomic_cas_fail_no_store.cinit chunking/atomic_fa.cinit chunking/atomic_fa_relaxed.cinit chunking/atomic_ex.cinit chunking/atomic_ex_relaxed.cinit striding/atomic_cas_succeed_store.cinit striding/atomic_cas_succeed_no_store.cinit striding/atomic_cas_fail_no_store.cinit striding/atomic_fa.cinit striding/atomic_fa_relaxed.cinit striding/atomic_ex.cinit striding/atomic_ex_relaxed.cinit
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o atomic_rmw_test.cpp -lvulkan -o atomic_rmw_test.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf chunking/*.cinit
	rm -rf chunking/*.spv
	rm -rf striding/*.cinit
	rm -rf striding/*.spv
	rm *.run
