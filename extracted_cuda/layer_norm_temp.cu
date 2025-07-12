#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda.h>
#include <cuda_runtime.h>
__global__ void layer_norm_fwd_fused(
    const float* X,
    float* Y,
    const float* W,
    const float* B,
    float* Mean,
    float* Rstd,
    int stride,
    int N,
    float eps,
    int BLOCK_SIZE
) {
    int row = blockIdx.x;
    X += row * stride;
    Y += row * stride;

    float mean = 0.0f;
    float var = 0.0f;

    for (int off = 0; off < N; off += BLOCK_SIZE) {
        float a = 0.0f;
        if (off + threadIdx.x < N) {
            a = X[off + threadIdx.x];
        }
        __shared__ float shared_mean[BLOCK_SIZE];
        shared_mean[threadIdx.x] = a;
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            mean += shared_mean[i];
        }
    }
    mean /= N;
    for (int off = 0; off < N; off += BLOCK_SIZE) {
        float x = 0.0f;
        if (off + threadIdx.x < N) {
            x = X[off + threadIdx.x] - mean;
        }
        __shared__ float shared_var[BLOCK_SIZE];
        shared_var[threadIdx.x] = x * x;
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            var += shared_var[i];
        }
    }
    var /= N;
    float rstd = 1.0f / sqrt(var + eps);
    Mean[row] = mean;
    Rstd[row] = rstd;
    for (int off = 0; off < N; off += BLOCK_SIZE) {
        if (off + threadIdx.x < N) {
            float x = X[off + threadIdx.x];
            float x_hat = (x - mean) * rstd;
            Y[off + threadIdx.x] = x_hat * W[off + threadIdx.x] + B[off + threadIdx.x];
        }
    }
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
    
    // 启动kernel: layer_norm_fwd_fused
    layer_norm_fwd_fused<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
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