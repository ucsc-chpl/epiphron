all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_86 atomic_rmw_test.cu -o atomic_rmw_test

clean:
	rm -f atomic_rmw_test
