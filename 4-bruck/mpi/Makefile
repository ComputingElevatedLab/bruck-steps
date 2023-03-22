MPICXX ?= mpic++
COBJFLAGS := $(CFLAGS) -c

TARGET_NAME := bruck-mpi
TARGET := $(TARGET_NAME)

SRC := $(foreach x, ./, $(wildcard $(addprefix $(x)/*,.c*)))

default: build

build:
	$(MPICXX) -o $(TARGET) $(TARGET).cpp $(CFLAGS)

test:
	mpirun -n 2 ./$(TARGET)

run: build test

clean:
	@rm -f $(TARGET)

.PHONY: build