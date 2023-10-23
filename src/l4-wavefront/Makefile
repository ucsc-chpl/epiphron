CXXFLAGS = -std=c++17
CLSPVFLAGS = --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points
SHADERS = $(wildcard *.cl)
SPVS = $(patsubst %.cl,build/%.spv,$(SHADERS))
CINITS = $(patsubst %.cl,build/%.cinit,$(SHADERS))
SOURCES = $(wildcard *cpp)
BENCHMARK = path_tracing

.PHONY: all build clean easyvk

all: build easyvk build/$(BENCHMARK).run

easyvk: 
	make -C ../../easyvk

build/$(BENCHMARK).run: $(SOURCES) $(SPVS) $(CINITS)
	mkdir -p data
	$(CXX) $(CXXFLAGS) -I../../easyvk/src ../../easyvk/build/easyvk.o $(SOURCES) -lvulkan -o $@

build/%.spv: %.cl
	clspv $(CLSPVFLAGS) $< -o $@

build/%.cinit: %.cl
	clspv $(CLSPVFLAGS) --output-format=c  $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build
