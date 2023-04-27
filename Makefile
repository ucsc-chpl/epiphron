CXXFLAGS = -std=c++17
CLSPVFLAGS = -cl-std=CL2.0 -inline-entry-points

.PHONY: clean

all: build easyvk l1 l2 l3

build:
	mkdir -p build

clean:
	rm -r build

easyvk: easyvk/src/easyvk.cpp easyvk/src/easyvk.h
	$(CXX) $(CXXFLAGS) -Ieasyvk/src -c easyvk/src/easyvk.cpp -o build/easyvk.o

l1: l1-atomic-load-store l1-atomic-rmw l1-kernel-launch l1-shared-memory l1-subgroup l1-workgroup-barrier

l2: l2-barrier l2-concurrent-queue l2-mutex l2-producer-consumer

l3: l3-workstealing


