import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Define the custom CUDA kernel for vector addition
vector_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    vector_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

vector_add_cpp_source = (
    "torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
vector_add = load_inline( 
    name="vector_add",
    cpp_sources=vector_add_cpp_source,
    cuda_sources=vector_add_source,
    functions=["vector_add_cuda"],
    verbose=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_add = vector_add

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.vector_add.vector_add_cuda(x, y)