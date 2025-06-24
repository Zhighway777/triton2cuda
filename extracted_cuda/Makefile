# CUDA编译配置
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60
INCLUDES = -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -lcudart

# 目标文件
TARGETS = vecadd

all: $(TARGETS)

vecadd: vecadd.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean