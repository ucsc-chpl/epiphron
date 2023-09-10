CXXFLAGS = -std=c++17
CINITS = $(patsubst %.cl,%.cinit,$(wildcard *.cl))
all: easyvk atomic_rmw_test

easyvk:
	make -C ../../easyvk

atomic_rmw_test: mutex_test.cpp $(CINITS)
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o mutex_test.cpp -lvulkan -o mutex_test.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o $@

clean:
	rm -rf *.cinit
	rm -rf *.spv
	rm *.run
