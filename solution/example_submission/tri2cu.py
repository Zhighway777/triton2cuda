from zhipuai import ZhipuAI

# ATTENTION Please!!
# Ensure all modification in tri2cu.py
# 保证 submission.zip 仅有 tri2cu.py
def get_full_prompt(triton_code):
    """
    生成严格的prompt，专门避免编译和运行时错误
    """
    base_prompt = """你是专业的Triton到CUDA转换专家。请将以下Triton代码转换为功能等效的CUDA代码。

## 严格要求（必须遵守）：

### 1. 编译要求
- 必须包含所有必要的头文件：#include <torch/extension.h>, #include <cuda_runtime.h>
- 所有CUDA kernel必须使用__global__修饰符
- 所有device函数必须使用__device__修饰符
- 变量声明必须符合C++语法
- 避免使用未定义的变量或函数

### 2. 内存安全
- 所有数组访问必须进行边界检查：if (idx < size)
- 使用正确的数据类型指针：float*, int*, double*等
- 确保shared memory声明正确：__shared__ float sdata[BLOCK_SIZE]
- 避免内存越界访问

### 3. CUDA语法规范
- 线程索引计算：int idx = blockIdx.x * blockDim.x + threadIdx.x;
- Grid和Block尺寸计算：dim3 grid((size + block_size - 1) / block_size);
- 同步操作：__syncthreads()（仅在需要时使用）
- 原子操作：atomicAdd, atomicMax等（仅在需要时使用）

### 4. PyTorch集成
- 使用torch::Tensor作为参数类型
- 正确获取tensor属性：input.numel(), input.size(0)等
- 正确的数据指针获取：input.data_ptr<float>()
- 创建输出tensor：torch::empty_like(input)或torch::zeros_like(input)

### 5. 关键映射规则（严格遵守）：
```
Triton                          →  CUDA
tl.program_id(axis=0)          →  blockIdx.x
tl.program_id(axis=1)          →  blockIdx.y
tl.arange(0, BLOCK_SIZE)       →  threadIdx.x + blockIdx.x * blockDim.x
tl.load(ptr + offsets, mask)   →  if (idx < size) value = ptr[idx]
tl.store(ptr + offsets, val, mask) → if (idx < size) ptr[idx] = value
tl.sum(x, axis=0)              →  使用reduction操作或shared memory
tl.max(x, axis=0)              →  使用reduction操作或shared memory
BLOCK_SIZE                     →  256, 512, 1024等2的幂
```

### 6. 错误检查
- 添加CUDA错误检查（可选但推荐）
- 验证输入tensor维度和类型
- 确保计算结果正确

## 标准模板（必须使用此结构）：

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void your_kernel_name(
    const float* input_ptr,
    float* output_ptr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 在这里实现核心逻辑
        // 确保所有计算都有边界检查
        output_ptr[idx] = input_ptr[idx]; // 示例操作
    }
}

torch::Tensor your_wrapper_function(torch::Tensor input) {
    // 获取输入尺寸
    auto size = input.numel();
    
    // 创建输出tensor
    auto output = torch::empty_like(input);
    
    // 配置kernel启动参数
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // 启动kernel
    your_kernel_name<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    // 等待kernel完成（可选）
    cudaDeviceSynchronize();
    
    return output;
}
\"\"\"

cpp_source = \"\"\"
torch::Tensor your_wrapper_function(torch::Tensor input);
\"\"\"

# 编译模块
module = load_inline(
    name="cuda_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["your_wrapper_function"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuda_module = module
    
          def forward(self, *args):
         # 确保参数处理正确
         if len(args) == 1:
             return self.cuda_module.your_wrapper_function(args[0])
         else:
             # 处理多个参数的情况
             return self.cuda_module.your_wrapper_function(*args)
```

## 特殊注意事项：

1. **数据类型一致性**：确保CUDA kernel中的数据类型与PyTorch tensor一致
2. **维度处理**：正确处理多维tensor，考虑stride和内存布局
3. **Reduction操作**：如果涉及sum/max等reduction，使用proper shared memory pattern
4. **命名规范**：kernel名称和wrapper函数名称要清晰且一致
5. **性能考虑**：使用合适的block size（通常256或512）

## 转换步骤：
1. 分析Triton代码的核心逻辑
2. 识别所有的tl.*操作并映射到CUDA等价操作
3. 确定正确的线程索引和内存访问模式
4. 实现边界检查和错误处理
5. 验证数据类型和维度匹配

请将以下Triton代码转换为CUDA代码：

```python
"""
    
    suffix = """
```

## 输出要求：
1. 直接输出完整的Python代码，不要任何解释文字
2. 代码必须能够通过nvcc编译
3. 数值结果必须与原Triton代码一致
4. 包含所有必要的import和环境设置
5. ModelNew类必须正确处理forward方法的参数
6. 确保所有CUDA操作都有适当的错误检查和边界检查

请严格按照上述模板和要求进行转换。"""
    
    return base_prompt + triton_code + suffix

def get_robust_prompt(triton_code):
    """
    极其严格的prompt，专门避免所有常见的编译和运行时错误
    """
    robust_prompt = """你是专业的Triton到CUDA转换专家。以下是一个ZERO-ERROR转换任务。

## 🚨 CRITICAL ERROR PREVENTION 🚨

### 常见编译错误及解决方案：
1. **未定义符号错误** → 确保所有函数都有正确的声明和定义
2. **类型不匹配** → 严格使用float*, const float*, int等正确类型
3. **语法错误** → 遵循标准C++/CUDA语法，正确使用分号、括号
4. **头文件缺失** → 必须包含 #include <torch/extension.h> 和 #include <cuda_runtime.h>
5. **kernel修饰符错误** → 所有kernel必须用 __global__ 修饰

### 常见运行时错误及解决方案：
1. **内存访问越界** → 每个数组访问都必须有 if (idx < size) 检查
2. **空指针访问** → 确保所有指针都有效
3. **维度不匹配** → 正确计算grid和block尺寸
4. **同步问题** → 适当使用 cudaDeviceSynchronize()
5. **数据类型错误** → 确保tensor数据类型与kernel参数匹配

## 🔒 MANDATORY TEMPLATE (严格遵守)：

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triton_to_cuda_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查 - 防止内存越界
    if (idx >= total_elements) return;
    
    // 在这里实现转换逻辑
    // 所有操作都在边界检查内
    output[idx] = input[idx];  // 替换为实际逻辑
}

torch::Tensor cuda_wrapper(torch::Tensor input_tensor) {
    // 检查输入tensor是否在CUDA上
    TORCH_CHECK(input_tensor.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input_tensor.is_contiguous(), "Input tensor must be contiguous");
    
    // 获取tensor信息
    auto total_size = input_tensor.numel();
    auto output_tensor = torch::empty_like(input_tensor);
    
    // 配置kernel启动参数
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_size + threads_per_block - 1) / threads_per_block;
    
    // 启动kernel
    triton_to_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        total_size
    );
    
    // 检查kernel启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed");
    }
    
    // 等待kernel完成
    cudaDeviceSynchronize();
    
    return output_tensor;
}
\"\"\"

cpp_source = \"\"\"
torch::Tensor cuda_wrapper(torch::Tensor input_tensor);
\"\"\"

# 编译扩展模块
try:
    cuda_module = load_inline(
        name="triton_cuda_converter",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["cuda_wrapper"],
                 verbose=False,  # 设为True可看到编译详情
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--use_fast_math"]
    )
except Exception as e:
    print(f"编译失败: {e}")
    raise

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_module = cuda_module
    
    def forward(self, *args, **kwargs):
        // 处理单个参数情况
        if len(args) == 1 and len(kwargs) == 0:
            input_tensor = args[0]
            // 确保输入在正确设备上
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            return self.cuda_module.cuda_wrapper(input_tensor)
        
        // 处理多个参数情况
        elif len(args) > 1:
            // 将所有参数移到CUDA并调用
            cuda_args = [arg.cuda() if hasattr(arg, 'cuda') and not arg.is_cuda else arg for arg in args]
            return self.cuda_module.cuda_wrapper(*cuda_args)
        
        else:
            raise ValueError("Invalid arguments for ModelNew.forward()")
```

## 🎯 TRITON TO CUDA MAPPING RULES:

```
# 基础映射
tl.program_id(0) → blockIdx.x
tl.program_id(1) → blockIdx.y
threadIdx.x → threadIdx.x (保持不变)

# 内存操作
tl.load(ptr + offset, mask) → if (idx < size) { value = ptr[idx]; }
tl.store(ptr + offset, value, mask) → if (idx < size) { ptr[idx] = value; }

# 数组操作
tl.arange(0, BLOCK_SIZE) → 使用 threadIdx.x + blockIdx.x * blockDim.x
tl.sum(x) → 使用reduction pattern或shared memory
tl.max(x) → 使用reduction pattern或shared memory

# 常量
BLOCK_SIZE → 256, 512, 1024 (2的幂)
```

## 🧪 DEBUGGING CHECKLIST:
- [ ] 所有变量都已声明
- [ ] 所有数组访问都有边界检查
- [ ] kernel函数有__global__修饰符
- [ ] 数据类型匹配 (float*, int*, etc.)
- [ ] Grid/Block尺寸计算正确
- [ ] 包含必要的头文件
- [ ] 错误检查代码已添加

请转换以下Triton代码：

```python
"""
    
    suffix = """
```

## 🚀 OUTPUT REQUIREMENTS:
1. 输出完整可运行的Python代码
2. 不要包含任何解释文字或注释
3. 代码必须通过nvcc编译
4. 必须产生正确的数值结果
5. 包含完整的错误处理
6. ModelNew类必须正确实现

⚠️ 如果不确定某个转换，选择最保守和安全的实现方式！"""
    
    return robust_prompt + triton_code + suffix

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

def get_model_configs():
    """
    获取所有可用的模型配置
    """
    return {
        "glm-4": {
            "model": "glm-4",
            "api_key": "NaN",
            "platform": "zhipuai",
            "enabled": False,
            "description": "GLM-4 基础版本"
        },
        "glm-4-plus": {
            "model": "glm-4-plus",
            "api_key": "5bf98ea765f642aeb720420e522592f7.DWMrwJ2rfsWPYhHJ",
            "platform": "zhipuai",
            "enabled": True,
            "description": "GLM-4 增强版本，推荐使用"
        },
        "glm-4-0520": {
            "model": "glm-4-0520",
            "api_key": "NaN",
            "platform": "zhipuai",
            "enabled": True,
            "description": "GLM-4 特定版本"
        }
    }

def list_available_models():
    """
    列出所有可用的模型
    """
    configs = get_model_configs()
    available_models = []
    for model_name, config in configs.items():
        if config.get("enabled", True):
            available_models.append({
                "name": model_name,
                "model": config["model"],
                "platform": config["platform"],
                "description": config.get("description", "")
            })
    return available_models

def create_api_client(platform, api_key):
    """
    根据平台创建API客户端
    """
    if platform == "zhipuai":
        return ZhipuAI(api_key=api_key)
    # elif platform == "claudeai":
    #     return ClaudeAI(api_key=api_key)
    else:
        raise ValueError(f"不支持的平台: {platform}")

def triton2cuda(triton_code, model_type="glm-4-plus", prompt_type="robust"):
    """
    将Triton代码转换为CUDA代码
    
    Args:
        triton_code: Triton源代码
        model_type: 要使用的模型类型，默认为 glm-4-plus
        prompt_type: prompt类型，可选 "full", "robust", "simple"
    
    Returns:
        转换后的CUDA代码
    """
    # 获取模型配置
    model_configs = get_model_configs()
    
    # 验证模型类型是否存在
    if model_type not in model_configs:
        available = list(model_configs.keys())
        raise ValueError(f"不支持的模型类型: {model_type}，可用选项: {available}")
    
    selected_config = model_configs[model_type]
    
    # 检查模型是否启用
    if not selected_config.get("enabled", True):
        raise ValueError(f"模型 {model_type} 已被禁用")
    
    # 选择prompt策略
    if prompt_type == "robust":
        prompt_content = get_robust_prompt(triton_code)
    elif prompt_type == "full":
        prompt_content = get_full_prompt(triton_code)
    elif prompt_type == "simple":
        prompt_content = get_simple_prompt(triton_code)
    else:
        raise ValueError(f"不支持的prompt类型: {prompt_type}，可选: 'robust', 'full', 'simple'")
    
    # 创建客户端并发送请求
    try:
        client = create_api_client(selected_config["platform"], selected_config["api_key"])
        response = client.chat.completions.create(
            model=selected_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Triton to CUDA conversion expert. Focus on generating error-free, compilable code.",
                },
                {"role": "user", "content": prompt_content},
            ],
            temperature=0.1,  # 降低随机性，提高一致性
            max_tokens=4000,  # 确保有足够空间生成完整代码
        )
        content = response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"API调用失败: {str(e)}")
    
    # 提取代码块中的内容
    if "```python" in content:
        start = content.find("```python") + len("```python")
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()

    return content
