# overview
## 赛题背景
随着深度学习算子的定制化需求不断提升，Triton作为一种高层次GPU编程语言被广泛用于编写灵活而高效的张量程序，逐渐成为AI编译生态的重要组成部分。然而，Triton仍处于快速演进阶段，在一些场景中仍需与底层CUDA程序互通或转换以获得更强的可控性与兼容性。

传统的代码翻译依赖人工实现或基于规则的转换器，效率低下且难以泛化。近年来，大语言模型（LLM）在程序理解与跨语言生成方面展现出强大能力，为自动化的程序迁移与代码转换带来新机遇。基于LLM实现Triton程序到CUDA的自动转换，不仅可提升底层系统编程效率，也为构建智能化编译基础设施奠定基础。

## 赛题描述
本赛题聚焦于利用大语言模型（LLM）完成Triton程序到CUDA程序的自动转换。参赛者需要在程序框架中设计一个转换函数，接收一段Triton程序（字符串形式），输出功能等价的CUDA程序（字符串形式）。转换过程中可调用LLM，也可引入额外的代码处理逻辑。若转换后的CUDA程序能够通过测试用例，则该程序转换成功。参赛者最后提交所实现的转换函数进行评测。

## 解题思路
利用LLM的能力完成Triton到CUDA的代码转换
可以考虑使用Chain-of-Thoughts或者Test-Time Scaling的思想，对问题进行拆解

## 比赛时间及评价标准
比赛分为两个阶段：

第一阶段（线上），2025年6月13日00:00-2025年7月1日23:59
第二阶段（线下），2025年7月1日00:00-2025年7月13日10:00
第一阶段和第二阶段分数各占比50%，最后计算总分。单个题目的评分规则见【Evaluation】页面。

# Task & Data
## 任务要求
本赛题第一阶段包含10个由Triton Kernel及其PyTorch Wrapper的代码集（非公开）。参赛者需要在一个名为tri2cu.py的python脚本中实现一个名为triton2cuda的转换函数，该函数的类型签名为triton2cuda(triton_code: str) -> str，该函数的目的是将该代码集中的代码转换成相应等价的CUDA Kernel及其PyTorch Wrapper的代码。

如果转换后的代码可以通过nvcc编译、成功运行并且输出结果和转换前的代码在精度允许的范围内保持一致，则该代码转换成功。由于triton2cuda是评测系统开始评测的入口函数，用户最后需要提交的代码中需要包含tri2cu.py模块。

请注意：参考所提供的solution的模板代码提供继承自torch.nn.Module的名为ModelNew的类作为CUDA代码功能的python侧入口。评测时会查找ModelNew类并严格按照本地评测的类似逻辑进行线上评测，可以参考solution中提供的本地评测逻辑。

## 环境说明
评测将在如下环境进行：

python版本：3.10.12
torch版本：2.4.0a0+07cecf4168.nv24.05
triton版本：3.0.0
cuda版本：12.4
示例代码（本地调试）
本赛题提供了一份示例代码用于本地调试，参赛者可以在Getting Started/Files页面选择Solution下载获取，其目录结构及文件功能如下：
```shell
solution
├── data
│   ├── cuda
│   │   └── vecadd.py        # 供参考的可运行的正确cuda实现
│   ├── ref
│   │   └── vecadd.py        # 作为reference的正确triton kernel（with pytorch wrapper）
│   └── triton
│       └── vecadd.py        # 评测时会提供给triton2cuda的代码
├── eval
│   └── eval.py              # 可在本地运行当前example_submission中的tri2cu的方案
├── example_submission
│   └── tri2cu.py            # 解决方案参考结构
├── example_submission.zip   # 提交到submission页面的参考压缩包
└── script
    └── bundle.sh            # 在ubuntu系统下打包的脚本参考
```
## 测试数据说明
在data目录下，包含了1个待转换Triton程序（Kernel及其PyTorch Wrapper），实际运行的测试用例和该程序的实现方式在结构上保持一致。

## 导入pytorch和triton相关的模块
```python
import torch
import triton
import triton.language as tl

# 实现向量加法的triton kernel
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

# triton kernel的wrapper
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
每个测试样例（接口的triton_code字符串）包括以下两部分内容：
1. Triton Kernel实现
2. Triton Kernel的Wrapper实现（输入输出一般为torch.Tensor），函数名为solve

## LLM模型接口使用说明
以智谱大模型为例，参赛者可以通过以下API调用的方式使用LLM：

```python
from zhipuai import ZhipuAI
from prompt import get_full_prompt

def triton2cuda(triton_code):
    client = ZhipuAI(api_key="Your Key")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 填写需要调用的模型编码
        messages=[
            {
                "role": "system",
                "content": "You are a helpful llm compiler assistant.",
            },
            {"role": "user", "content": get_full_prompt(triton_code)},
        ],
    )
    content = response.choices[0].message.content
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
```
注：

智谱AI为参赛者提供了一定数量的免费 tokens 及优惠资源包，请发送邮件至赛题联系人邮箱获取
example_submission的tri2cu.py中给出了参考的智谱AI平台API Key的使用方式

# Evalution
## 提交流程
请在 tri2cu.py 中完成对 triton2cuda 函数的实现，其余逻辑均须实现在该文件中（只保证 submission.zip 仅有 tri2cu.py 的评测流程稳定性，包含其他文件或复杂文件结构压缩包的情况请谨慎使用，不保证能正常评测）
将该文件打包为 submission.zip
将压缩包提交至本赛题在 codabench 上的 submission 页面进行在线评测
## 评分与错误反馈
单个测试样例的评分规则
本赛题对于每个测试样例的评分规则如下，对于单个测试样例，满分为 10分：

|内容|	分值|	要求|
|---|---|---|
程序编译|	1分|	每个测试用例的转换结果是否通过编译|
结果验证|	9分|	所有测试用例的转换结果是否正确|

注意：对于结果正确性，本赛题所有程序使用 float32 作为计算精度，并且使用 torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3) 方法判断转换前后结果的正确性。

评测结果反馈
用户可以通过查看评测平台的评测程序的标准输出日志查看每个测试样例的错误反馈，每个测试样例可能的错误类型主要包括：

|错误类型|说明
|---|---|
RuntimeError| CUDA代码生成失败或未遵循赛题的输入文件规范，或其他运行时错误
CompilationError|	生成的CUDA代码未能通过nvcc编译器编译
OutputMismatchError	|生成的CUDA代码的运行结果和标准程序不匹配

## 其他说明
如果在LOGS/Prediction Logs中的Ingestion stderr中出现docker pull失败之类的提示，通常由评测机网络波动造成，请等待一小段时间然后重新提交。其他情况类似，如果不是持续稳定重复的报错（如评测机的某种明确异常）则重新提交即可。