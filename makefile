CC = gcc
CXX = g++
CCU = nvcc
CFLAGS = -g -Wall -Iinclude/ 
CXXFLAGS = $(CFLAGS)
CUFLAGS = -std=c++17 -Iinclude/ -Xcompiler -Wall
LDFLAGS = -lGLEW -lGL -lGLU -lSDL2 -lcudart

EXEC = fluidsim
C_SRC = $(wildcard src/*.c)
CU_SRC = $(wildcard src/cuda/*.cu)
CXX_SRC = $(wildcard src/*.cpp)
OBJ = $(C_SRC:src/%.c=bin/%.o) $(CXX_SRC:src/%.cpp=bin/%.o) $(CU_SRC:src/cuda/%.cu=bin/%.o)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

bin/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: src/cuda/%.cu
	$(CCU) $(CUFLAGS) -c $< -o $@

clean:
	rm -f bin/*.o $(EXEC)

.PHONY: all clean