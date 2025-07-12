# CUDA编译配置
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60
INCLUDES = -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -lcudart

# 目标文件
TARGETS = matmul_temp vecadd_temp layer_norm_temp mat_transpose_temp softmax_temp

all: $(TARGETS)

matmul_temp: matmul_temp.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

vecadd_temp: vecadd_temp.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

layer_norm_temp: layer_norm_temp.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

mat_transpose_temp: mat_transpose_temp.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

softmax_temp: softmax_temp.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean