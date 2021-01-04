CC = gcc
CXX = g++
CFLAGS = -lm -O3 
hw3: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw3

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
