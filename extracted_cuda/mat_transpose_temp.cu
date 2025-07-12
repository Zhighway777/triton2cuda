#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
__global__ void matrix_transpose_kernel(
    const float* input_ptr, float* output_ptr,
    int M, int N,
    int stride_ir, int stride_ic,
    int stride_or, int stride_oc,
    int BLOCK_SIZE
) {
    int pid_m = blockIdx.x;  // Block index in M direction
    int pid_n = blockIdx.y;  // Block index in N direction

    int offs_m = pid_m * BLOCK_SIZE + threadIdx.x;
    int offs_n = pid_n * BLOCK_SIZE + threadIdx.y;

    if (offs_m < M && offs_n < N) {
        int input_idx = offs_m * stride_ir + offs_n * stride_ic;
        int output_idx = offs_n * stride_or + offs_m * stride_oc;
        output_ptr[output_idx] = input_ptr[input_idx];
    }
}
    int M = input.size(0);
    int N = input.size(1);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    matrix_transpose_kernel<<<grid, block>>>(
        M, N,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE
    );
    return output;


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
    
    // 启动kernel: matrix_transpose_kernel
    matrix_transpose_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
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