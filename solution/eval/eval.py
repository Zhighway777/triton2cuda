import os
import sys
import tempfile
import torch
import importlib.util
import sys
import glob

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(folder_path)
sys.path.append(os.path.join(folder_path, "example_submission"))
sys.path.append(os.path.join(folder_path, "data", "ref"))

TEST_NN_MODEL_NAME = 'ModelNew'

from tri2cu import triton2cuda

def get_inputs_for_file(file_name):
    """为不同的文件提供相应的输入数据（当没有参考实现时使用）"""
    if file_name == "vecadd.py":
        # 两个相同形状的张量
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32), torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(8432, device='cuda', dtype=torch.float32), torch.randn(8432, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["constant_add.py", "constant_add_block.py"]:
        # 一个张量
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(200, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["outer_vecadd.py", "outer_vecadd_block.py"]:
        # 两个1D张量
        return [
            [torch.randn(32, device='cuda', dtype=torch.float32), torch.randn(32, device='cuda', dtype=torch.float32)],
            [torch.randn(100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["longsum.py", "longsoftmax.py", "softmax.py"]:
        # 一个2D张量
        return [
            [torch.randn(4, 200, device='cuda', dtype=torch.float32)],
            [torch.randn(8, 128, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "conv2d.py":
        # 输入张量和卷积核
        return [
            [torch.randn(4, 8, 8, device='cuda', dtype=torch.float32), torch.randn(4, 4, device='cuda', dtype=torch.float32)],
            [torch.randn(2, 6, 6, device='cuda', dtype=torch.float32), torch.randn(3, 3, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "flashatt.py":
        # q, k, v三个张量
        return [
            [torch.randn(200, device='cuda', dtype=torch.float32), torch.randn(200, device='cuda', dtype=torch.float32), torch.randn(200, device='cuda', dtype=torch.float32)],
            [torch.randn(128, device='cuda', dtype=torch.float32), torch.randn(128, device='cuda', dtype=torch.float32), torch.randn(128, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "matmul.py":
        # 两个3D张量（批量矩阵乘法）
        return [
            [torch.randn(4, 32, 32, device='cuda', dtype=torch.float32), torch.randn(4, 32, 32, device='cuda', dtype=torch.float32)],
            [torch.randn(2, 16, 24, device='cuda', dtype=torch.float32), torch.randn(2, 24, 20, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "mul_relu_fused_block.py":
        # 两个1D张量
        return [
            [torch.randn(100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32)],
            [torch.randn(64, device='cuda', dtype=torch.float32), torch.randn(48, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "outer_mul_relu_fused_block.py":
        # 三个张量（x, y, dz）
        return [
            [torch.randn(90, 100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32), torch.randn(90, 100, device='cuda', dtype=torch.float32)],
            [torch.randn(64, 48, device='cuda', dtype=torch.float32), torch.randn(64, device='cuda', dtype=torch.float32), torch.randn(64, 48, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "quant_matmul.py":
        # 四个张量（scale, offset, weight, activation）
        return [
            [
                torch.randn(32, 8, device='cuda', dtype=torch.float32),
                torch.randint(0, 15, (32,), device='cuda', dtype=torch.int32),
                torch.randint(0, 15, (32, 8), device='cuda', dtype=torch.int32),
                torch.randn(64, 32, device='cuda', dtype=torch.float32)
            ],
            [
                torch.randn(16, 8, device='cuda', dtype=torch.float32),
                torch.randint(0, 15, (16,), device='cuda', dtype=torch.int32),
                torch.randint(0, 15, (16, 8), device='cuda', dtype=torch.int32),
                torch.randn(64, 16, device='cuda', dtype=torch.float32)
            ]
        ]
    else:
        # 默认返回一个张量
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(512, device='cuda', dtype=torch.float32)]
        ]

def get_reference_model_and_inputs(file_name):
    """获取参考模型和输入数据"""
    # 首先检查是否有对应的参考实现
    ref_file_path = os.path.join(folder_path, "data", "ref", file_name)
    
    if os.path.exists(ref_file_path):
        # 使用参考实现
        print(f"  使用参考实现: data/ref/{file_name}")
        module_name = file_name.replace('.py', '_ref')
        spec = importlib.util.spec_from_file_location(module_name, ref_file_path)
        ref_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_module)
        
        # 获取参考Model类和get_inputs函数
        RefModel = getattr(ref_module, 'Model', None)
        get_inputs_func = getattr(ref_module, 'get_inputs', None)
        
        if RefModel is None:
            raise ValueError(f"无法在参考实现 {file_name} 中找到 Model 类")
        
        if get_inputs_func is None:
            print(f"  警告: 参考实现 {file_name} 中没有 get_inputs 函数，使用默认输入")
            input_tensors = get_inputs_for_file(file_name)
        else:
            input_tensors = get_inputs_func()
        
        return RefModel, input_tensors
    else:
        # 使用local_test_list中的文件作为参考
        print(f"  使用本地测试文件作为参考: data/triton/local_test_list/{file_name}")
        triton_file_path = os.path.join(folder_path, "data", "triton", "local_test_list", file_name)
        
        module_name = file_name.replace('.py', '_triton')
        spec = importlib.util.spec_from_file_location(module_name, triton_file_path)
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        
        # 获取Triton Model类
        TritonModel = getattr(triton_module, 'Model', None)
        if TritonModel is None:
            raise ValueError(f"无法在 {file_name} 中找到 Model 类")
        
        # 使用默认输入生成函数
        input_tensors = get_inputs_for_file(file_name)
        
        return TritonModel, input_tensors

def eval_single_file(file_name):
    """评测单个文件的Triton到CUDA转换"""
    print(f"\n=== 评测 {file_name} ===")
    
    try:
        # 1. 读取Triton代码
        triton_file_path = os.path.join("data", "triton", "local_test_list", file_name)
        with open(triton_file_path, "r") as f:
            triton_code = f.read()
        
        # 2. 转换为CUDA代码
        print("  正在转换Triton代码为CUDA代码...")
        cuda_code = triton2cuda(triton_code)
        
        # 3. 获取参考模型和输入数据
        RefModel, input_tensors = get_reference_model_and_inputs(file_name)
        
        # 4. 创建临时文件并加载转换后的CUDA模块
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "cuda_code.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)
            
            # 动态加载转换后的CUDA模块
            spec = importlib.util.spec_from_file_location("cuda_module", temp_file)
            cuda_module = importlib.util.module_from_spec(spec)
            
            # 临时添加到sys.modules以支持相对导入
            sys.modules["cuda_module"] = cuda_module
            
            try:
                spec.loader.exec_module(cuda_module)
            except Exception as e:
                print(f"  错误: 加载转换后的CUDA代码失败: {str(e)}")
                return False
            
            # 获取转换后的ModelNew类
            CudaModel = getattr(cuda_module, TEST_NN_MODEL_NAME, None)
            if CudaModel is None:
                print(f"  错误: 无法在转换后的代码中找到 {TEST_NN_MODEL_NAME} 类")
                return False
        
        # 5. 对比测试
        success_count = 0
        total_count = len(input_tensors)
        
        for i, input_tensor in enumerate(input_tensors):
            try:
                # 确保输入在CUDA上
                input_tensor_cuda = []
                for inp in input_tensor:
                    if not inp.is_cuda:
                        inp = inp.cuda()
                    input_tensor_cuda.append(inp.detach().clone())
                
                # 计算参考输出
                ref_output = RefModel()(*input_tensor_cuda)
                
                # 计算转换后的CUDA输出
                cuda_output = CudaModel()(*input_tensor_cuda)
                
                # 比较结果
                if torch.allclose(cuda_output, ref_output, atol=1e-3, rtol=1e-3):
                    print(f"  测试 {i+1}: 通过")
                    success_count += 1
                else:
                    print(f"  测试 {i+1}: 失败 - 输出不匹配")
                    print(f"    参考输出形状: {ref_output.shape}")
                    print(f"    CUDA输出形状: {cuda_output.shape}")
                    max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
                    print(f"    最大差异: {max_diff}")
                    
            except Exception as e:
                print(f"  测试 {i+1}: 失败 - 异常: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"结果: {success_count}/{total_count} 测试通过")
        return success_count == total_count
        
    except Exception as e:
        print(f"错误: 评测 {file_name} 时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def eval_all_files():
    """评测所有文件"""
    os.chdir(folder_path)
    
    # 获取所有triton文件
    test_dir = os.path.join("data", "triton", "local_test_list")
    triton_files = [f for f in os.listdir(test_dir) if f.endswith('.py')]
    triton_files.sort()  # 按字母顺序排序
    
    print(f"找到 {len(triton_files)} 个triton文件需要评测")
    
    success_files = []
    failed_files = []
    
    for file_name in triton_files:
        if eval_single_file(file_name):
            success_files.append(file_name)
        else:
            failed_files.append(file_name)
    
    print(f"\n=== 总体结果 ===")
    print(f"成功: {len(success_files)}/{len(triton_files)} 个文件")
    
    if success_files:
        print("\n成功的文件:")
        for file_name in success_files:
            print(f"  ✓ {file_name}")
    
    if failed_files:
        print("\n失败的文件:")
        for file_name in failed_files:
            print(f"  ✗ {file_name}")
    
    return len(failed_files) == 0

def eval_simple():
    """原始的vecadd评测函数（保留兼容性）"""
    return eval_single_file("vecadd.py")

def eval_golden():
    """使用golden标准评测vecadd"""
    os.chdir(folder_path)
    
    # 读取Triton代码
    triton_code = open("data/triton/local_test_list/vecadd.py", "r").read()
    
    # 转换为CUDA代码
    cuda_code = triton2cuda(triton_code)
    print("转换后的CUDA代码:")
    print(cuda_code)
    
    # 加载参考实现
    from vecadd import Model, get_inputs
    input_tensors = get_inputs()
    
    # 加载golden CUDA实现
    golden_file_path = os.path.join(folder_path, "data", "cuda", "vecadd.py")
    spec = importlib.util.spec_from_file_location("golden_cuda", golden_file_path)
    golden_module = importlib.util.module_from_spec(spec)
    sys.modules["golden_cuda"] = golden_module
    spec.loader.exec_module(golden_module)
    
    GoldenModel = getattr(golden_module, TEST_NN_MODEL_NAME, None)
    if GoldenModel is None:
        raise ValueError(f"无法在golden实现中找到 {TEST_NN_MODEL_NAME} 类")
    
    # 对比测试
    for i, input_tensor in enumerate(input_tensors):
        ref_output = Model()(*input_tensor)
        golden_output = GoldenModel()(*input_tensor)
        
        if torch.allclose(golden_output, ref_output, atol=1e-3, rtol=1e-3):
            print(f"Test {i+1} passed")
        else:
            print(f"Test {i+1} failed")
            return False
    
    print("All tests passed")
    return True

if __name__ == "__main__":
    # 评测所有文件
    eval_all_files()
    
    # 可选：只评测特定文件
    # eval_single_file("vecadd.py")
    
    # 原始函数保留兼容性
    # eval_simple()
    # eval_golden()
    