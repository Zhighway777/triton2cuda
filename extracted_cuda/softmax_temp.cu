#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda.h>
#include <cuda_runtime.h>
__global__ void softmax_kernel(
    const float* input_ptr, float* output_ptr,
    int N, int BLOCK_SIZE
) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = pid * BLOCK_SIZE + tid;
    __shared__ float shared_max;
    __shared__ float shared_sum;

    if (tid == 0) {
        shared_max = -INFINITY;
        shared_sum = 0.0f;
    }
    __syncthreads();
    float max_val = -INFINITY;
    for (int off = 0; off < N; off += BLOCK_SIZE) {
        int idx = off + tid;
        if (idx < N) {
            max_val = fmax(max_val, input_ptr[idx]);
        }
    }
    atomicMax(&shared_max, max_val);
    __syncthreads();
    max_val = shared_max;
    float sum_val = 0.0f;
    for (int off = 0; off < N; off += BLOCK_SIZE) {
        int idx = off + tid;
        if (idx < N) {
            sum_val += exp(input_ptr[idx] - max_val);
        }
    }
    atomicAdd(&shared_sum, sum_val);
    __syncthreads();
    float sum_val_final = shared_sum;
    if (offset < N) {
        output_ptr[offset] = exp(input_ptr[offset] - max_val) / sum_val_final;
    }

    int N = input.size(0);
    int BLOCK_SIZE = 256;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_kernel<<<num_blocks, BLOCK_SIZE>>>(
        N, BLOCK_SIZE
    );
    return output;
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
    
    // 启动kernel: softmax_kernel
    softmax_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
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