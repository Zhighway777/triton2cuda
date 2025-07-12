import os
import sys
import tempfile
import torch
import importlib.util
import subprocess
import time
import argparse
from typing import Tuple, List, Dict, Any
from enum import Enum
import traceback

# 错误类型枚举
class ErrorType(Enum):
    RUNTIME_ERROR = "RuntimeError"
    COMPILATION_ERROR = "CompilationError"
    OUTPUT_MISMATCH_ERROR = "OutputMismatchError"
    SUCCESS = "Success"

class TestResult:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.error_type = ErrorType.SUCCESS
        self.error_message = ""
        self.compilation_score = 0  # 0 or 1
        self.accuracy_score = 0     # 0-9
        self.total_score = 0        # 0-10
        self.test_details = []      # 每个测试用例的详细信息
        self.cuda_code = ""         # 生成的CUDA代码
        self.compilation_output = ""  # 编译器输出
        
    def set_error(self, error_type: ErrorType, message: str):
        self.error_type = error_type
        self.error_message = message
        
    def set_compilation_result(self, success: bool, output: str = ""):
        self.compilation_score = 1 if success else 0
        self.compilation_output = output
        
    def set_accuracy_result(self, passed_tests: int, total_tests: int):
        if total_tests > 0:
            self.accuracy_score = int(9 * passed_tests / total_tests)
        else:
            self.accuracy_score = 0
            
    def calculate_total_score(self):
        self.total_score = self.compilation_score + self.accuracy_score
        
    def get_summary(self) -> str:
        return f"{self.file_name}: {self.total_score}/10 (编译:{self.compilation_score}/1, 准确性:{self.accuracy_score}/9)"

# 修复路径问题
current_file_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.dirname(current_file_dir)
original_cwd = os.getcwd()
os.chdir(folder_path)

sys.path.append(folder_path)
sys.path.append(os.path.join(folder_path, "example_submission"))
sys.path.append(os.path.join(folder_path, "data", "ref"))

TEST_NN_MODEL_NAME = 'ModelNew'

from tri2cu import triton2cuda

def get_inputs_for_file(file_name):
    """为不同的文件提供相应的输入数据"""
    if file_name == "vecadd.py":
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32), torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(8432, device='cuda', dtype=torch.float32), torch.randn(8432, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["constant_add.py", "constant_add_block.py"]:
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(200, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["outer_vecadd.py", "outer_vecadd_block.py"]:
        return [
            [torch.randn(32, device='cuda', dtype=torch.float32), torch.randn(32, device='cuda', dtype=torch.float32)],
            [torch.randn(100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32)]
        ]
    elif file_name in ["longsum.py", "longsoftmax.py", "softmax.py"]:
        return [
            [torch.randn(4, 200, device='cuda', dtype=torch.float32)],
            [torch.randn(8, 128, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "conv2d.py":
        return [
            [torch.randn(4, 8, 8, device='cuda', dtype=torch.float32), torch.randn(4, 4, device='cuda', dtype=torch.float32)],
            [torch.randn(2, 6, 6, device='cuda', dtype=torch.float32), torch.randn(3, 3, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "flashatt.py":
        return [
            [torch.randn(200, device='cuda', dtype=torch.float32), torch.randn(200, device='cuda', dtype=torch.float32), torch.randn(200, device='cuda', dtype=torch.float32)],
            [torch.randn(128, device='cuda', dtype=torch.float32), torch.randn(128, device='cuda', dtype=torch.float32), torch.randn(128, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "matmul.py":
        return [
            [torch.randn(4, 32, 32, device='cuda', dtype=torch.float32), torch.randn(4, 32, 32, device='cuda', dtype=torch.float32)],
            [torch.randn(2, 16, 24, device='cuda', dtype=torch.float32), torch.randn(2, 24, 20, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "mul_relu_fused_block.py":
        return [
            [torch.randn(100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32)],
            [torch.randn(64, device='cuda', dtype=torch.float32), torch.randn(48, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "outer_mul_relu_fused_block.py":
        return [
            [torch.randn(90, 100, device='cuda', dtype=torch.float32), torch.randn(90, device='cuda', dtype=torch.float32), torch.randn(90, 100, device='cuda', dtype=torch.float32)],
            [torch.randn(64, 48, device='cuda', dtype=torch.float32), torch.randn(64, device='cuda', dtype=torch.float32), torch.randn(64, 48, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "quant_matmul.py":
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
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32)],
            [torch.randn(512, device='cuda', dtype=torch.float32)]
        ]

def get_reference_model_and_inputs(file_name):
    """获取参考模型和输入数据"""
    # 首先检查是否有对应的参考实现
    ref_file_path = os.path.join("data", "ref", file_name)
    
    if os.path.exists(ref_file_path):
        module_name = file_name.replace('.py', '_ref')
        spec = importlib.util.spec_from_file_location(module_name, ref_file_path)
        ref_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_module)
        
        RefModel = getattr(ref_module, 'Model', None)
        get_inputs_func = getattr(ref_module, 'get_inputs', None)
        
        if RefModel is None:
            raise ValueError(f"无法在参考实现 {file_name} 中找到 Model 类")
        
        input_tensors = get_inputs_func() if get_inputs_func else get_inputs_for_file(file_name)
        return RefModel, input_tensors
    else:
        # 使用本地测试文件作为参考
        triton_file_path = os.path.join("data", "triton", "local_test_list", file_name)
        
        if not os.path.exists(triton_file_path):
            raise FileNotFoundError(f"无法找到Triton文件: {triton_file_path}")
        
        module_name = file_name.replace('.py', '_triton')
        spec = importlib.util.spec_from_file_location(module_name, triton_file_path)
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        
        TritonModel = getattr(triton_module, 'Model', None)
        if TritonModel is None:
            raise ValueError(f"无法在 {file_name} 中找到 Model 类")
        
        input_tensors = get_inputs_for_file(file_name)
        return TritonModel, input_tensors

def check_cuda_compilation(cuda_code: str, file_name: str) -> Tuple[bool, str]:
    """检查CUDA代码是否能够编译通过"""
    if not cuda_code.strip():
        return False, "生成的CUDA代码为空"
    
    try:
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_cuda.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)
            
            # 尝试用Python语法检查
            try:
                with open(temp_file, "r") as f:
                    compile(f.read(), temp_file, "exec")
                return True, "Python语法检查通过"
            except SyntaxError as e:
                return False, f"Python语法错误: {str(e)}"
            except Exception as e:
                return False, f"Python编译错误: {str(e)}"
                
    except Exception as e:
        return False, f"编译检查失败: {str(e)}"

def run_cuda_code_test(cuda_code: str, input_tensors: List[List[torch.Tensor]], file_name: str) -> Tuple[List[torch.Tensor], str]:
    """运行CUDA代码并返回结果"""
    if not cuda_code.strip():
        raise RuntimeError("生成的CUDA代码为空")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "cuda_code.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location("cuda_module", temp_file)
            cuda_module = importlib.util.module_from_spec(spec)
            sys.modules["cuda_module"] = cuda_module
            
            try:
                spec.loader.exec_module(cuda_module)
            except Exception as e:
                raise RuntimeError(f"CUDA代码加载失败: {str(e)}")
            
            # 获取ModelNew类
            CudaModel = getattr(cuda_module, TEST_NN_MODEL_NAME, None)
            if CudaModel is None:
                available_attrs = [attr for attr in dir(cuda_module) if not attr.startswith('_')]
                raise RuntimeError(f"无法找到 {TEST_NN_MODEL_NAME} 类。可用属性: {available_attrs}")
            
            # 运行测试
            results = []
            for i, input_tensor in enumerate(input_tensors):
                try:
                    # 确保输入在CUDA上
                    input_tensor_cuda = []
                    for inp in input_tensor:
                        if not inp.is_cuda:
                            inp = inp.cuda()
                        input_tensor_cuda.append(inp.detach().clone())
                    
                    # 执行CUDA模型
                    cuda_output = CudaModel()(*input_tensor_cuda)
                    results.append(cuda_output)
                    
                except Exception as e:
                    raise RuntimeError(f"测试用例 {i+1} 执行失败: {str(e)}")
            
            return results, "CUDA代码执行成功"
            
    except Exception as e:
        raise RuntimeError(f"CUDA代码执行失败: {str(e)}")

def eval_single_file_enhanced(file_name: str, verbose: bool = False) -> TestResult:
    """增强版单文件评测"""
    print(f"\n=== 评测 {file_name} ===")
    result = TestResult(file_name)
    
    try:
        # 第一步：读取Triton代码
        triton_file_path = os.path.join("data", "triton", "local_test_list", file_name)
        if not os.path.exists(triton_file_path):
            result.set_error(ErrorType.RUNTIME_ERROR, f"无法找到Triton文件: {triton_file_path}")
            return result
        
        with open(triton_file_path, "r") as f:
            triton_code = f.read()
        
        # 第二步：转换为CUDA代码
        print("  正在转换Triton代码为CUDA代码...")
        try:
            cuda_code = triton2cuda(triton_code)
            result.cuda_code = cuda_code
            
            if not cuda_code.strip():
                result.set_error(ErrorType.RUNTIME_ERROR, "triton2cuda返回空代码")
                return result
                
            print(f"  转换成功，代码长度: {len(cuda_code)} 字符")
            
        except Exception as e:
            result.set_error(ErrorType.RUNTIME_ERROR, f"CUDA代码生成失败: {str(e)}")
            if verbose:
                result.error_message += f"\n详细错误:\n{traceback.format_exc()}"
            # 保存调试信息
            save_debug_info(file_name, result.cuda_code, f"CUDA代码生成失败: {str(e)}\n\n{traceback.format_exc()}")
            return result
        
        # 第三步：编译检查
        print("  正在检查CUDA代码编译...")
        compilation_success, compilation_message = check_cuda_compilation(cuda_code, file_name)
        result.set_compilation_result(compilation_success, compilation_message)
        
        if not compilation_success:
            result.set_error(ErrorType.COMPILATION_ERROR, compilation_message)
            result.calculate_total_score()
            return result
        
        print(f"  编译检查通过: {compilation_message}")
        
        # 第四步：获取参考模型和输入数据
        try:
            RefModel, input_tensors = get_reference_model_and_inputs(file_name)
            print(f"  准备了 {len(input_tensors)} 组测试数据")
        except Exception as e:
            result.set_error(ErrorType.RUNTIME_ERROR, f"获取参考模型失败: {str(e)}")
            return result
        
        # 第五步：运行CUDA代码
        print("  正在运行CUDA代码测试...")
        try:
            cuda_results, run_message = run_cuda_code_test(cuda_code, input_tensors, file_name)
            print(f"  CUDA代码执行成功")
        except Exception as e:
            result.set_error(ErrorType.RUNTIME_ERROR, f"CUDA代码执行失败: {str(e)}")
            if verbose:
                result.error_message += f"\n详细错误:\n{traceback.format_exc()}"
            result.calculate_total_score()
            return result
        
        # 第六步：计算参考结果并比较
        print("  正在比较结果...")
        passed_tests = 0
        total_tests = len(input_tensors)
        
        for i, (input_tensor, cuda_output) in enumerate(zip(input_tensors, cuda_results)):
            try:
                # 确保输入在CUDA上
                input_tensor_cuda = []
                for inp in input_tensor:
                    if not inp.is_cuda:
                        inp = inp.cuda()
                    input_tensor_cuda.append(inp.detach().clone())
                
                # 计算参考输出
                ref_output = RefModel()(*input_tensor_cuda)
                
                # 比较结果
                if torch.allclose(cuda_output, ref_output, atol=1e-3, rtol=1e-3):
                    print(f"  测试 {i+1}: 通过")
                    passed_tests += 1
                    result.test_details.append(f"测试 {i+1}: 通过")
                else:
                    max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
                    error_msg = f"测试 {i+1}: 失败 - 输出不匹配，最大差异: {max_diff:.6f}"
                    print(f"  {error_msg}")
                    result.test_details.append(error_msg)
                    
                    if verbose:
                        print(f"    参考输出形状: {ref_output.shape}, 统计: min={ref_output.min().item():.6f}, max={ref_output.max().item():.6f}")
                        print(f"    CUDA输出形状: {cuda_output.shape}, 统计: min={cuda_output.min().item():.6f}, max={cuda_output.max().item():.6f}")
                        
            except Exception as e:
                error_msg = f"测试 {i+1}: 失败 - 比较过程异常: {str(e)}"
                print(f"  {error_msg}")
                result.test_details.append(error_msg)
                if verbose:
                    print(f"    详细错误: {traceback.format_exc()}")
        
        # 设置准确性分数
        result.set_accuracy_result(passed_tests, total_tests)
        
        if passed_tests == total_tests:
            print(f"  所有测试通过: {passed_tests}/{total_tests}")
            result.error_type = ErrorType.SUCCESS
        else:
            print(f"  部分测试失败: {passed_tests}/{total_tests}")
            result.set_error(ErrorType.OUTPUT_MISMATCH_ERROR, f"结果不匹配: {passed_tests}/{total_tests} 测试通过")
        
        result.calculate_total_score()
        return result
        
    except Exception as e:
        result.set_error(ErrorType.RUNTIME_ERROR, f"评测过程发生异常: {str(e)}")
        if verbose:
            result.error_message += f"\n详细错误:\n{traceback.format_exc()}"
        result.calculate_total_score()
        return result

def save_debug_info(file_name, cuda_code, error_info):
    """保存调试信息到debug_output文件夹"""
    debug_dir = os.path.join(folder_path, "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 保存转换后的CUDA代码
    cuda_file = os.path.join(debug_dir, f"{file_name}_cuda.py")
    with open(cuda_file, "w") as f:
        f.write(cuda_code)
    
    # 保存错误信息
    error_file = os.path.join(debug_dir, f"{file_name}_error.txt")
    with open(error_file, "w") as f:
        f.write(error_info)
    
    print(f"  调试信息已保存到: {debug_dir}/")

def save_result_report(results: List[TestResult], output_file: str = "evaluation_report.txt"):
    """保存详细的评测报告到debug_output文件夹"""
    # 确保debug_output文件夹存在
    debug_dir = os.path.join(folder_path, "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 如果没有指定完整路径，则保存到debug_output文件夹
    if not os.path.isabs(output_file):
        output_file = os.path.join(debug_dir, output_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Triton到CUDA转换评测报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 总体统计
        total_files = len(results)
        successful_files = sum(1 for r in results if r.error_type == ErrorType.SUCCESS)
        total_score = sum(r.total_score for r in results)
        max_score = total_files * 10
        
        f.write(f"总体统计:\n")
        f.write(f"  测试文件数: {total_files}\n")
        f.write(f"  成功文件数: {successful_files}\n")
        f.write(f"  总分: {total_score}/{max_score}\n")
        f.write(f"  平均分: {total_score/total_files:.2f}/10\n")
        f.write(f"  成功率: {successful_files/total_files*100:.1f}%\n\n")
        
        # 错误类型统计
        error_stats = {}
        for result in results:
            error_type = result.error_type.value
            error_stats[error_type] = error_stats.get(error_type, 0) + 1
        
        f.write(f"错误类型统计:\n")
        for error_type, count in error_stats.items():
            f.write(f"  {error_type}: {count}\n")
        f.write("\n")
        
        # 按错误类型分类显示文件
        f.write("按错误类型分类:\n")
        for error_type in ErrorType:
            files_with_error = [r for r in results if r.error_type == error_type]
            if files_with_error:
                f.write(f"  {error_type.value} ({len(files_with_error)} 个文件):\n")
                for result in files_with_error:
                    f.write(f"    - {result.file_name}: {result.total_score}/10\n")
        f.write("\n")
        
        # 编译和准确性统计
        compilation_success = sum(1 for r in results if r.compilation_score > 0)
        accuracy_success = sum(1 for r in results if r.accuracy_score > 0)
        
        f.write(f"分项统计:\n")
        f.write(f"  编译通过: {compilation_success}/{total_files} ({compilation_success/total_files*100:.1f}%)\n")
        f.write(f"  有准确性分数: {accuracy_success}/{total_files} ({accuracy_success/total_files*100:.1f}%)\n\n")
        
        # 详细结果
        f.write("详细结果:\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            f.write(f"\n文件: {result.file_name}\n")
            f.write(f"总分: {result.total_score}/10 (编译: {result.compilation_score}/1, 准确性: {result.accuracy_score}/9)\n")
            f.write(f"错误类型: {result.error_type.value}\n")
            
            if result.error_message:
                f.write(f"错误信息: {result.error_message}\n")
            
            if result.compilation_output:
                f.write(f"编译输出: {result.compilation_output}\n")
            
            if result.test_details:
                f.write("测试详情:\n")
                for detail in result.test_details:
                    f.write(f"  {detail}\n")
            
            # 保存转换后的CUDA代码（如果有的话）
            if result.cuda_code:
                cuda_file = os.path.join(debug_dir, f"{result.file_name}_cuda.py")
                with open(cuda_file, "w") as cuda_f:
                    cuda_f.write(result.cuda_code)
                f.write(f"CUDA代码已保存到: {cuda_file}\n")
            
            f.write("-" * 40 + "\n")
    
    print(f"详细报告已保存到: {output_file}")

def cleanup():
    """清理函数"""
    os.chdir(original_cwd)

def eval_all_files_enhanced(verbose: bool = False) -> List[TestResult]:
    """增强版批量评测所有文件"""
    print(f"当前工作目录: {os.getcwd()}")
    
    # 获取所有triton文件
    test_dir = os.path.join("data", "triton", "local_test_list")
    
    if not os.path.exists(test_dir):
        print("❌ 无法找到测试目录，终止评测")
        return []
    
    try:
        triton_files = [f for f in os.listdir(test_dir) if f.endswith('.py')]
        triton_files.sort()  # 按字母顺序排序
    except Exception as e:
        print(f"❌ 无法列出测试目录内容: {e}")
        return []
    
    print(f"找到 {len(triton_files)} 个triton文件需要评测")
    
    results = []
    
    for i, file_name in enumerate(triton_files):
        print(f"\n进度: {i+1}/{len(triton_files)}")
        result = eval_single_file_enhanced(file_name, verbose)
        results.append(result)
        
        # 显示简要结果
        print(f"结果: {result.get_summary()}")
        if result.error_type != ErrorType.SUCCESS:
            print(f"错误类型: {result.error_type.value}")
            if result.error_message:
                print(f"错误信息: {result.error_message}")
    
    return results

def print_final_summary(results: List[TestResult]):
    """打印最终汇总"""
    print(f"\n{'='*60}")
    print("最终评测结果汇总")
    print(f"{'='*60}")
    
    # 总体统计
    total_files = len(results)
    successful_files = sum(1 for r in results if r.error_type == ErrorType.SUCCESS)
    total_score = sum(r.total_score for r in results)
    max_score = total_files * 10
    
    print(f"总体统计:")
    print(f"  测试文件数: {total_files}")
    print(f"  成功文件数: {successful_files}")
    print(f"  总分: {total_score}/{max_score}")
    print(f"  平均分: {total_score/total_files:.2f}/10")
    print(f"  成功率: {successful_files/total_files*100:.1f}%")
    
    # 错误类型统计
    error_stats = {}
    for result in results:
        error_type = result.error_type.value
        error_stats[error_type] = error_stats.get(error_type, 0) + 1
    
    print(f"\n错误类型统计:")
    for error_type, count in error_stats.items():
        print(f"  {error_type}: {count}")
    
    # 按错误类型分类显示文件
    print(f"\n按错误类型分类:")
    for error_type in ErrorType:
        files_with_error = [r for r in results if r.error_type == error_type]
        if files_with_error:
            print(f"  {error_type.value} ({len(files_with_error)} 个文件):")
            for result in files_with_error:
                print(f"    - {result.file_name}: {result.total_score}/10")
    
    # 编译和准确性统计
    compilation_success = sum(1 for r in results if r.compilation_score > 0)
    accuracy_success = sum(1 for r in results if r.accuracy_score > 0)
    
    print(f"\n分项统计:")
    print(f"  编译通过: {compilation_success}/{total_files} ({compilation_success/total_files*100:.1f}%)")
    print(f"  有准确性分数: {accuracy_success}/{total_files} ({accuracy_success/total_files*100:.1f}%)")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Triton到CUDA转换评测系统")
        parser.add_argument("--file", type=str, help="评测单个文件")
        parser.add_argument("--all", action="store_true", help="评测所有文件")
        parser.add_argument("--verbose", action="store_true", help="详细输出")
        parser.add_argument("--output", type=str, default="evaluation_report.txt", help="报告输出文件名（将保存到debug_output文件夹）")
        
        args = parser.parse_args()
        
        if args.file:
            # 单文件评测
            result = eval_single_file_enhanced(args.file, args.verbose)
            print(f"\n{result.get_summary()}")
            save_result_report([result], f"{args.file}_evaluation_report.txt")
        elif args.all:
            # 批量评测
            results = eval_all_files_enhanced(args.verbose)
            print_final_summary(results)
            save_result_report(results, args.output)
        else:
            # 默认测试flashatt.py
            result = eval_single_file_enhanced("flashatt.py", verbose=True)
            print(f"\n{result.get_summary()}")
            save_result_report([result], "flashatt_evaluation_report.txt")
        
    except Exception as e:
        print(f"评测过程发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup() 