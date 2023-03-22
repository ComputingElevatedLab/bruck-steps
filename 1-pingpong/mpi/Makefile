MPICXX ?= mpic++
COBJFLAGS := $(CFLAGS) -c

TARGET_NAME := pingpong-mpi
TARGET := $(TARGET_NAME)

SRC := $(foreach x, ./, $(wildcard $(addprefix $(x)/*,.c*)))

NODES := 2
NPROC := $(NODES) * 8

default: build

build:
	$(MPICXX) -o $(TARGET) $(TARGET).cpp $(CFLAGS)

test:
	mpirun -n 2 ./$(TARGET)

run: build test

clean:
	@rm -f $(TARGET)

.PHONY: build