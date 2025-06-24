# triton2cuda
A tool based LLM to translate Triton to CUDA

# 主要目标
## 快速完成题目基本要求，保证正确性

## 环境配置流程
### 1. 配置环境
```bash
./setup.sh
```

### 2. 激活环境
```bash
./activate_env.sh
```
### 3. 提取cuda程序的方法：
1. 首先运行prompt,生成py的包裹文件，在solution/data/cuda下
2. 运行 python3 extract_cuda.py solution/data/cuda -o extracted_cuda
## 环境要求

- Ubuntu 22.04+
- Python 3.10+
- NVIDIA驱动

## 安装内容

- CUDA 12.4工具包
- PyTorch 2.4.0 (CUDA 12.4)
- Triton 3.0.0
- requirements.txt中的所有依赖

## 常见问题

**检查环境**
```bash
python3 quick_check.py
```

**清理环境**
```bash
rm -rf llm_env activate_env.sh
```

**手动安装**
```bash
# 1. 系统依赖
sudo apt-get install -y build-essential ninja-build cuda-toolkit-12-4

# 2. 虚拟环境
python3 -m venv llm_env
source llm_env/bin/activate

# 3. Python包
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.0.0
pip install -r requirements.txt
```


## git-协作方法
我们使用github平台进行工作的同步，注意以下几点：
1. 首先拉取main分支，然后每个人需要新建自己的独立的开发分支，命名为“dev-xxx”
2. 不要直接将代码推送到主分支，先推送到自己的独立分支，等待每一个阶段的功能完善后再提交PR请求合并到main中
3. 每次进行开发前，应该首先拉取最新的main与本地分支进行合并，以免confict堆积
4. 在进行commit的时候 尽量把描述写的清楚一点，以便于其他协作成员查看
5. 在github设置了CI来对基本的正确性进行检查，再github仓库页面点击“Actions”可以查看通过情况。（但是这方面需要进一步丰富检查内容）

## 任务分工

## 相关资料（我们可以一起补充）

### Triton
[⭐|技术博客|Triton执行流程](https://www.cnblogs.com/BobHuang/p/18324040)  


### LLM/Prompt
[教程|提示词工程指南](https://www.promptingguide.ai/zh)  
[DeepSeek 提示词样例](https://api-docs.deepseek.com/prompt-library)  
[Test-Time Scaling](https://zhuanlan.zhihu.com/p/773907223)

### 编译器

### CUDA
[⭐|技术博客|CUDA极简教程](https://zhuanlan.zhihu.com/p/34587739)  
[技术博客|CUDA教程](https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89)  
[技术博客|将PTX代码逆向反汇编为CUDA C++](https://forums.developer.nvidia.com/t/is-there-a-reverse-engineering-tool-which-gives-approximate-cuda-c-code-from-ptx-code/305665)  
[知乎专栏|CUDA实现及优化](https://www.zhihu.com/column/c_1681252213014466560)  

### 其他杂项/论文等
[⭐|题库|leetGPU](https://leetgpu.com/)
## 注意
1. 原赛题的文件夹为triton2cuda/solution，solution之外的文件均为我后来添加用于测试或者git操作的文件
