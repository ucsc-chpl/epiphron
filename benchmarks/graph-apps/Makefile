CXXFLAGS = -std=c++17
CLSPVFLAGS = --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points -O3
SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
BENCHMARK = graph_apps

.PHONY: all build clean easyvk

all: build easyvk build/$(BENCHMARK).run

easyvk: 
	make -C ../../easyvk

build/$(BENCHMARK).run: graph_apps.cpp $(SPVS) $(CINITS) common.h
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o $< -lvulkan -o $@

build/%.spv: %.cl common.h
	clspv $(CLSPVFLAGS) $< -o $@

build/%.cinit: %.cl common.h
	clspv $(CLSPVFLAGS) --output-format=c  $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build
