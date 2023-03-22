export NCCL_DEBUG=WARN

TARGET = bruck-nccl
COMMON = ../common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib
THETA_FLAGS = -L/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.12.12-1_gcc-9.4.0-1ubuntu1-20.04/lib -L/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib
SRC = $(TARGET).cu

build:
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(THETA_FLAGS)

test:
	mpirun -n 2 ./$(TARGET)

run: build test

debug:
	mpiexec -n 2 konsole -e gdb ./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.log

.PHONY.: build