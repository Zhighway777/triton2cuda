#include <torch/extension.h>
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