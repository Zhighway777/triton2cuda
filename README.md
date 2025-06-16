# triton2cuda
A tool based LLM to translate Triton to CUDA

# 主要目标

## git-协作方法
我们使用github平台进行工作的同步，注意以下几点：
1. 首先拉取main分支，然后每个人需要新建自己的独立的开发分支，命名为“dev-xxx”
2. 不要直接将代码推送到主分支，先推送到自己的独立分支，等待每一个阶段的功能完善后再提交PR请求合并到main中
3. 每次进行开发前，应该首先拉取最新的main与本地分支进行合并，以免confict堆积
4. 在进行commit的时候 尽量把描述写的清楚一点，以便于其他协作成员查看
5. 在github设置了CI来对基本的正确性进行检查，再github仓库页面点击“Actions”可以查看通过情况。（但是这方面需要进一步丰富检查内容）

## 任务分工

## 相关资料（我们可以一起补充）

### triton

### LLM/Prompt

### 编译器

### CUDA
[技术博客|将PTX代码逆向反汇编为CUDA C++](https://forums.developer.nvidia.com/t/is-there-a-reverse-engineering-tool-which-gives-approximate-cuda-c-code-from-ptx-code/305665)

### 其他杂项/论文等

## 注意
1. 原赛题的文件夹为triton2cuda/solution，solution之外的文件均为我后来添加用于测试或者git操作的文件
