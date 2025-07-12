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
