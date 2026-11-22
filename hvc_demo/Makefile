CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude -ffast-math
LDFLAGS ?=

LIB_OBJ = build/synth.o
EVAL_OBJ = build/eval.o
DEMO_OBJ = build/demo.o
INTEG_OBJ = build/integration_example.o

all: build/eval build/demo build/integration_example

build:
	mkdir -p build

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/eval: $(EVAL_OBJ) $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/demo: $(DEMO_OBJ) $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/integration_example: $(INTEG_OBJ) $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: all clean
