# =============================================================================
# PSCKangaroo v56C — A fork of RCKangaroo by RetiredCoder
# Pollard's Kangaroo for ECDLP on secp256k1 (GPU-accelerated)
#
# Original: https://github.com/RetiredC/RCKangaroo
# License:  GPLv3
# =============================================================================
#
# Build options:
#   make                          # Default build (recommended settings)
#   make GPU_ARCH="-gencode=arch=compute_89,code=sm_89"   # For RTX 4090
#   make V45_TABLE_BITS=33        # Larger table (128GB+ RAM systems)
#

CUDA_HOME = /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CC = g++

# GPU Architecture — change this for your GPU:
#   sm_86  = RTX 3060/3070/3080/3090 (Ampere)
#   sm_89  = RTX 4060/4070/4080/4090 (Ada Lovelace)
#   sm_120 = RTX 5070/5080/5090 (Blackwell)
GPU_ARCH ?= -gencode=arch=compute_120,code=sm_120

# Stratified jumps (disabled by default — uniform proved 19% faster)
V46_ENABLE ?= 0
V46_EXPLORER_PCT ?= 20
V46_EXPLORER_SHIFT ?= 2

# GPU kernel tuning
V45_OCCUPANCY ?= 1
V45_TABLE_BITS ?= 33
V45_PNT_GROUP_CNT ?= 24
V45_STEP_CNT ?= 1000

V46_DEFS = -DV46_ENABLE=$(V46_ENABLE) \
           -DV46_EXPLORER_PCT=$(V46_EXPLORER_PCT) \
           -DV46_EXPLORER_SHIFT=$(V46_EXPLORER_SHIFT)

V45_DEFS = -DV45_OCCUPANCY=$(V45_OCCUPANCY) \
           -DV45_TABLE_BITS=$(V45_TABLE_BITS) \
           -DV45_PNT_GROUP_CNT=$(V45_PNT_GROUP_CNT) \
           -DV45_STEP_CNT=$(V45_STEP_CNT)

ALL_DEFS = $(V46_DEFS) $(V45_DEFS)

NVCC_FLAGS = -O3 $(GPU_ARCH) $(ALL_DEFS) -Xcompiler -O3,-march=native,-pthread --ptxas-options=-v
CC_FLAGS = -O3 -march=native -pthread $(ALL_DEFS)

LIBS = -lcuda -lcudart -lpthread

OBJS = RCKangaroo_hunt_v2.o Ec.o GpuKang.o RCGpuCore.o utils.o

TARGET = psckangaroo

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(LIBS)
	@echo ""
	@echo "=== Build complete: $(TARGET) ==="
	@echo "  TABLE_BITS=$(V45_TABLE_BITS)  OCCUPANCY=$(V45_OCCUPANCY)"
	@echo ""

RCKangaroo_hunt_v2.o: RCKangaroo_hunt_v2.cpp defs.h utils.h GpuKang.h TameStore.h
	$(CC) $(CC_FLAGS) -I$(CUDA_HOME)/include -c RCKangaroo_hunt_v2.cpp -o $@

Ec.o: Ec.cpp Ec.h defs.h
	$(CC) $(CC_FLAGS) -c Ec.cpp -o $@

GpuKang.o: GpuKang.cpp GpuKang.h defs.h utils.h Ec.h TameStore.h
	$(CC) $(CC_FLAGS) -I$(CUDA_HOME)/include -c GpuKang.cpp -o $@

RCGpuCore.o: RCGpuCore.cu defs.h RCGpuUtils.h
	$(NVCC) $(NVCC_FLAGS) -c RCGpuCore.cu -o $@

utils.o: utils.cpp utils.h defs.h
	$(CC) $(CC_FLAGS) -c utils.cpp -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
