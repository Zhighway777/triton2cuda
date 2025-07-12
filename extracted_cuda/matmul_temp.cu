#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda.h>
#include <cuda_runtime.h>
__global__ void softmax_kernel(float* output, const float* input, int input_row_stride, int output_row_stride, int n_rows, int n_cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= n_rows) return;

    const float* row_start_ptr = input + row_idx * input_row_stride;
    float* output_row_start_ptr = output + row_idx * output_row_stride;

    float max_val = -INFINITY;
    for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
        max_val = fmax(max_val, row_start_ptr[col_idx]);
    }
    float sum_exp = 0.0f;
    for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
        float val = exp(row_start_ptr[col_idx] - max_val);
        sum_exp += val;
        output_row_start_ptr[col_idx] = val;
    }
    for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
        output_row_start_ptr[col_idx] /= sum_exp;
    }

    int n_rows = input.size(0);
    int n_cols = input.size(1);
    auto options = input.options();
    int input_row_stride = input.stride(0);
    int output_row_stride = output.stride(0);
    int threads_per_block = 256;
    int num_blocks = (n_rows + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<num_blocks, threads_per_block>>>(
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols
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