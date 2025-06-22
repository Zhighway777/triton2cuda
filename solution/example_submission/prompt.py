def get_full_prompt(triton_code):
    """
    生成优化后的prompt，结合比赛要求和最佳实践
    """
    base_prompt = """
你是专业的Triton到CUDA转换专家。请将以下Triton代码转换为功能等效的CUDA代码。
## 转换要求：
1. 生成的代码必须能通过nvcc编译
2. 数值结果与原代码一致（torch.allclose精度：atol=1e-3, rtol=1e-3）
3. 必须包含继承自torch.nn.Module的ModelNew类
4. 使用torch.utils.cpp_extension.load_inline进行动态编译

## 输出格式要求：
- 直接输出完整的Python代码，无需其他解释
- 代码必须包含所有必要的import语句
- 确保ModelNew类的forward方法参数与原Model类一致

## 关键映射规则：
- tl.program_id(axis=0) → blockIdx.x * blockDim.x + threadIdx.x
- tl.load(ptr + offsets, mask) → if (idx < size) data = ptr[idx]
- tl.store(ptr + offsets, data, mask) → if (idx < size) ptr[idx] = data
- BLOCK_SIZE → 256或其他2的幂
- grid计算 → (size + block_size - 1) / block_size

## 参考示例结构：
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kernel_name(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 内核逻辑
        output[idx] = input[idx]; // 示例
    }
}

torch::Tensor wrapper_function(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    kernel_name<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    return output;
}
\"\"\"

cpp_source = "torch::Tensor wrapper_function(torch::Tensor input);"

module = load_inline(
    name="module_name",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["wrapper_function"],
    verbose=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return self.module.wrapper_function(*args)
```

请将以下Triton代码转换为CUDA代码：

```python
"""
    
    suffix = """
```

请直接输出完整的CUDA代码实现，确保：
1. 代码结构完整，可以直接运行
2. ModelNew类的forward方法能够正确处理输入参数
3. 数值计算结果与原Triton代码一致
4. 所有必要的错误检查和内存管理都已包含"""
    
    return base_prompt + triton_code + suffix


def get_prompt_through_function(triton_code):
    # 生成优化后的prompt，结合比赛要求和最佳实践
    base_prompt = """
你是专业的Triton到CUDA转换专家。请将以下Triton代码转换为功能等效的CUDA代码。

## 转换要求：
1. 生成的代码必须能通过nvcc编译
2. 数值结果与原代码一致（torch.allclose精度：atol=1e-3, rtol=1e-3）
3. 必须包含继承自torch.nn.Module的ModelNew类
4. 使用torch.utils.cpp_extension.load_inline进行动态编译

## 输出格式要求：
- 直接输出完整的Python代码，无需其他解释
- 代码必须包含所有必要的import语句
- 确保ModelNew类的forward方法参数与原Model类一致

## 关键映射规则：
- tl.program_id(axis=0) → blockIdx.x * blockDim.x + threadIdx.x
- tl.load(ptr + offsets, mask) → if (idx < size) data = ptr[idx]
- tl.store(ptr + offsets, data, mask) → if (idx < size) ptr[idx] = data
- BLOCK_SIZE → 256或其他2的幂
- grid计算 → (size + block_size - 1) / block_size

## 参考示例 (输入代码)
```python
import torch
import triton
import triton.language as tl
torch.manual_seed(46)

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def solve(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and output.dtype == torch.float32
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return solve(x, y)
```

## 参考示例（输出代码）
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kernel_name(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 内核逻辑
        output[idx] = input[idx]; // 示例
    }
}

torch::Tensor wrapper_function(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    kernel_name<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    return output;
}
\"\"\"

cpp_source = "torch::Tensor wrapper_function(torch::Tensor input);"

module = load_inline(
    name="module_name",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["wrapper_function"],
    verbose=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return self.module.wrapper_function(*args)
```

请将以下Triton代码转换为CUDA代码：

```python
"""
    suffix = """
```
    请直接输出完整的CUDA代码实现，确保：
    1. 代码结构完整，可以直接运行
    2. ModelNew类的forward方法能够正确处理输入参数
    3. 数值计算结果与原Triton代码一致
    4. 所有必要的错误检查和内存管理都已包含
"""
    
    return base_prompt + triton_code + suffix



def get_simple_prompt(triton_code):
    """
    简化版本的prompt，适合快速测试
    """
    simple_prompt = """Convert the following Triton code to equivalent CUDA code.

Requirements:
- Must compile with nvcc
- Must produce same numerical results
- Must include ModelNew class inheriting from torch.nn.Module
- Use torch.utils.cpp_extension.load_inline

Triton code:
```python
"""
    
    suffix = """
```

Output complete Python code with CUDA implementation:"""
    
    return simple_prompt + triton_code + suffix 