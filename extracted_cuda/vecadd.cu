#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    return out;
}

// 错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_result = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)i * 2;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_result;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_result, size));
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // 配置执行参数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // 启动kernel: vector_add_kernel
    vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));
    
    // 验证结果
    printf("验证前10个结果:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_result[%d] = %.2f (expected: %.2f)\n", i, h_result[i], h_a[i] + h_b[i]);
    }
    
    // 清理内存
    free(h_a); free(h_b); free(h_result);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    
    printf("CUDA程序执行完成！\n");
    return 0;
}