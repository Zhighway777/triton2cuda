1. 首先运行eval.py进行基础测试 会将data/triton/local_test_list 下的所有triton文件 进行转换为CUDA并分析原因
    可调参数：在eval.py文件最后可以选择针对不同的文件进行测试。
    ```python
    if __name__ == "__main__":
    try:
        # # 评测所有文件（详细模式）
        eval_all_files(verbose=True)
        
        # Golden标准评测
        print("\n" + "="*50)
        eval_golden()
        
        # 可选：只评测特定文件
        # eval_single_file("flashatt.py", verbose=True)
        
        # 原始函数保留兼容性
        # eval_simple()
    ```

    如果正确 不会有文件产生，如果出现错误，相关错误信息会放在/data/debug_output，会包含生成的CUDA信息和相关的报错类型

2. 在初步了解报错信息后 筛除已经正确的kernel，只留下错误的kernel，然后使用eval_enhanced.py脚本继续分析，该脚本的使用方法：
> 测试单个文件
> python eval/eval_enhanced.py --file vecadd.py --verbose

> 测试所有文件
> python eval/eval_enhanced.py --all --verbose

测试结果会以report的形式保存在debug_output文件夹下

3. eval.py新增功能

命令行参数:
```python
python eval.py --file vecadd.py        # 评测单个文件
python eval.py --all                   # 评测所有文件
python eval.py --all --verbose         # 详细模式评测所有文件
python eval.py --network              # 测试API端点可用性
python eval.py --golden               # Golden标准评测
python eval.py --simple               # 简单vecadd评测
```

API连通性测试功能:
- 自动检测API相关错误并分类
- 提供API端点可用性检查
- 支持智谱AI、OpenRouter、DeepSeek等多个API
- 当转换失败时可先运行--network测试API是否可用

4. eval_enhanced.py增强功能

主要特性:
- 多模型和prompt策略自动尝试
- 详细的API调用监控和记录
- 完整的错误分类和诊断
- 评分系统(编译1分+准确性9分=总分10分)
- 生成详细的评测报告

命令行参数:
```python
python eval_enhanced.py --file vecadd.py --verbose    # 评测单个文件
python eval_enhanced.py --all --verbose               # 评测所有文件
python eval_enhanced.py --network                     # 测试API端点可用性
```

评测报告包含:
- 总体统计和成功率
- 错误类型分析
- API调用详情和成功率
- 每个文件的详细测试结果
- 转换过程和API调用历史

注意事项:
1. API相关错误时使用--network参数测试连通性
2. 详细模式(--verbose)提供更多调试信息
3. 所有调试输出和报告保存在debug_output目录
