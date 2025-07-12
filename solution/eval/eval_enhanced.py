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
import requests
from datetime import datetime
import json

# 错误类型枚举
class ErrorType(Enum):
    RUNTIME_ERROR = "RuntimeError"
    COMPILATION_ERROR = "CompilationError"
    OUTPUT_MISMATCH_ERROR = "OutputMismatchError"
    NETWORK_ERROR = "NetworkError"        # 新增：API连接相关错误
    API_ERROR = "APIError"               # 新增：API调用错误
    SUCCESS = "Success"

class APICallMonitor:
    """API调用监控类"""
    
    def __init__(self):
        self.call_history = []
        self.current_call = None
    
    def start_call(self, model_type: str, prompt_type: str, triton_code_length: int):
        """开始监控API调用"""
        self.current_call = {
            "model_type": model_type,
            "prompt_type": prompt_type,
            "triton_code_length": triton_code_length,
            "start_time": datetime.now(),
            "start_timestamp": time.time(),
            "request_details": {},
            "response_details": {},
            "error_details": {},
            "success": False,
            "duration": 0
        }
    
    def record_request(self, **kwargs):
        """记录请求详情"""
        if self.current_call:
            self.current_call["request_details"].update(kwargs)
    
    def record_response(self, **kwargs):
        """记录响应详情"""
        if self.current_call:
            self.current_call["response_details"].update(kwargs)
    
    def record_error(self, error_type: str, error_message: str, full_traceback: str = None):
        """记录错误详情"""
        if self.current_call:
            self.current_call["error_details"] = {
                "error_type": error_type,
                "error_message": error_message,
                "full_traceback": full_traceback,
                "timestamp": datetime.now().isoformat()
            }
    
    def end_call(self, success: bool, response_content: str = None):
        """结束API调用监控"""
        if self.current_call:
            self.current_call["success"] = success
            self.current_call["duration"] = time.time() - self.current_call["start_timestamp"]
            self.current_call["end_time"] = datetime.now()
            
            if response_content:
                self.current_call["response_details"]["content_length"] = len(response_content)
                self.current_call["response_details"]["has_code_block"] = "```" in response_content
            
            self.call_history.append(self.current_call.copy())
            self.current_call = None
    
    def get_summary(self):
        """获取调用摘要"""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for call in self.call_history if call["success"])
        total_duration = sum(call["duration"] for call in self.call_history)
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "total_duration": total_duration,
            "average_duration": total_duration / total_calls if total_calls > 0 else 0,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0
        }

class NetworkMonitor:
    """API连通性监控类"""
    
    @staticmethod
    def check_api_endpoints():
        """检查API端点可用性"""
        endpoints = {
            "智谱AI": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
            "DeepSeek": "https://api.deepseek.com/v1/chat/completions"
        }
        
        results = {}
        for name, url in endpoints.items():
            try:
                # 只检查端点是否可达，不发送实际请求
                response = requests.head(url, timeout=10)
                results[name] = {
                    "available": response.status_code != 404,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                results[name] = {
                    "available": False,
                    "error": str(e)
                }
        
        return results

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
        self.conversion_details = []  # 转换过程详情
        self.api_call_history = []  # API调用历史
        
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

def enhanced_triton2cuda_with_monitoring(triton_code, file_name="unknown"):
    """增强版triton2cuda调用，包含详细的API调用监控"""
    
    monitor = APICallMonitor()
    
    # 1. API端点检查
    print("  正在检查API端点...")
    api_status = NetworkMonitor.check_api_endpoints()
    available_apis = [name for name, status in api_status.items() if status.get("available", False)]
    
    if not available_apis:
        error_msg = "所有API端点都不可用:\n"
        for name, status in api_status.items():
            error_msg += f"  - {name}: {status.get('error', '不可用')}\n"
        raise Exception(error_msg)
    
    print(f"  可用的API: {', '.join(available_apis)}")
    
    # 2. 尝试不同的模型和prompt策略
    models_to_try = [
        ("deepseek-R1", "robust"),
        ("claude-sonnet-4", "full"),
        ("glm-4-plus", "function"),
        ("deepseek-R1", "simple")
    ]
    
    last_error = None
    attempt_details = []
    
    for model_type, prompt_type in models_to_try:
        # 开始监控这次API调用
        monitor.start_call(model_type, prompt_type, len(triton_code))
        
        try:
            print(f"  尝试使用模型: {model_type} (prompt: {prompt_type})")
            
            # 记录请求详情
            monitor.record_request(
                model=model_type,
                prompt_strategy=prompt_type,
                input_code_length=len(triton_code),
                timestamp=datetime.now().isoformat()
            )
            
            # 调用原始的triton2cuda函数，但我们需要包装它来捕获更多信息
            cuda_code = monitored_triton2cuda_call(triton_code, model_type, prompt_type, monitor)
            
            # 记录成功的响应
            monitor.record_response(
                api_call_end=datetime.now().isoformat(),
                response_type="text",
                contains_python_code="```python" in cuda_code if cuda_code else False,
                contains_any_code="```" in cuda_code if cuda_code else False
            )
            
            attempt_details.append({
                "model": model_type,
                "prompt": prompt_type,
                "success": True,
                "duration": monitor.current_call["duration"] if monitor.current_call else 0,
                "code_length": len(cuda_code) if cuda_code else 0
            })
            
            # 检查返回的代码是否为空
            if not cuda_code or not cuda_code.strip():
                print(f"    警告: 返回的代码为空，尝试下一个模型")
                last_error = f"模型 {model_type} 返回空代码"
                monitor.record_error("EmptyResponse", last_error)
                monitor.end_call(False, cuda_code)
                continue
            
            print(f"    成功! 代码长度: {len(cuda_code)} 字符")
            
            # 结束成功的监控
            monitor.end_call(True, cuda_code)
            
            # 保存成功的转换记录
            save_conversion_log_with_api_details(file_name, model_type, prompt_type, cuda_code, attempt_details, monitor)
            
            return cuda_code, attempt_details, monitor.call_history
            
        except Exception as e:
            error_msg = str(e)
            
            # 分析错误类型
            error_type = classify_api_error(error_msg)
            monitor.record_error(error_type, error_msg, traceback.format_exc())
            
            attempt_details.append({
                "model": model_type,
                "prompt": prompt_type,
                "success": False,
                "duration": time.time() - monitor.current_call["start_timestamp"] if monitor.current_call else 0,
                "error": error_msg,
                "error_type": error_type
            })
            
            print(f"    失败: {error_msg}")
            last_error = error_msg
            
            # 结束失败的监控
            monitor.end_call(False)
            
            # 检查是否是API连接相关错误
            if error_type in ["NetworkError", "TimeoutError", "ConnectionError"]:
                print(f"    检测到API连接错误，等待5秒后重试...")
                time.sleep(5)
            
            continue
    
    # 所有尝试都失败了
    error_summary = f"所有模型都失败了。最后错误: {last_error}\n"
    error_summary += "尝试详情:\n"
    for attempt in attempt_details:
        if attempt["success"]:
            error_summary += f"  ✓ {attempt['model']}: 成功 ({attempt['duration']:.2f}s)\n"
        else:
            error_summary += f"  ✗ {attempt['model']}: {attempt.get('error_type', 'Unknown')} - {attempt['error']}\n"
    
    # 添加API调用统计
    summary = monitor.get_summary()
    error_summary += f"\nAPI调用统计:\n"
    error_summary += f"  总调用次数: {summary['total_calls']}\n"
    error_summary += f"  成功次数: {summary['successful_calls']}\n"
    error_summary += f"  失败次数: {summary['failed_calls']}\n"
    error_summary += f"  总耗时: {summary['total_duration']:.2f}秒\n"
    error_summary += f"  平均耗时: {summary['average_duration']:.2f}秒\n"
    
    raise Exception(error_summary)

def monitored_triton2cuda_call(triton_code, model_type, prompt_type, monitor):
    """包装的triton2cuda调用，用于记录更详细的信息"""
    
    try:
        # 记录更多请求参数
        monitor.record_request(
            api_call_start=datetime.now().isoformat(),
            temperature=0.1,
            max_tokens=4000
        )
        
        # 调用实际的triton2cuda函数
        result = triton2cuda(triton_code, model_type=model_type, prompt_type=prompt_type)
        
        # 记录响应信息
        monitor.record_response(
            api_call_end=datetime.now().isoformat(),
            response_type="text",
            contains_python_code="```python" in result if result else False,
            contains_any_code="```" in result if result else False
        )
        
        return result
        
    except Exception as e:
        # 记录API调用失败的详细信息
        error_type = classify_api_error(str(e))
        monitor.record_error(error_type, str(e), traceback.format_exc())
        raise

def classify_api_error(error_message):
    """分类API错误类型"""
    error_msg_lower = error_message.lower()
    
    # API连接超时
    if any(keyword in error_msg_lower for keyword in ["timeout", "timed out"]):
        return "TimeoutError"
    
    # 连接错误
    if any(keyword in error_msg_lower for keyword in ["connection", "unreachable", "dns"]):
        return "ConnectionError"
    
    # API认证错误
    if any(keyword in error_msg_lower for keyword in ["unauthorized", "authentication", "api key", "forbidden"]):
        return "AuthenticationError"
    
    # 速率限制
    if any(keyword in error_msg_lower for keyword in ["rate limit", "quota", "too many requests"]):
        return "RateLimitError"
    
    # API服务器错误
    if any(keyword in error_msg_lower for keyword in ["500", "502", "503", "504", "server error"]):
        return "ServerError"
    
    # SSL/TLS错误
    if any(keyword in error_msg_lower for keyword in ["ssl", "certificate", "tls"]):
        return "SSLError"
    
    # 一般API连接错误
    if any(keyword in error_msg_lower for keyword in ["network", "socket", "http"]):
        return "NetworkError"
    
    return "UnknownError"

def save_conversion_log_with_api_details(file_name, model_type, prompt_type, cuda_code, attempts, monitor):
    """保存转换日志，包含API调用详情"""
    debug_dir = os.path.join(folder_path, "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 保存详细的转换日志
    log_file = os.path.join(debug_dir, f"{file_name}_conversion_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"转换日志 - {file_name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"成功模型: {model_type}\n")
        f.write(f"成功prompt: {prompt_type}\n")
        f.write(f"生成代码长度: {len(cuda_code)}\n\n")
        
        # API调用统计
        summary = monitor.get_summary()
        f.write("API调用统计:\n")
        f.write(f"  总调用次数: {summary['total_calls']}\n")
        f.write(f"  成功次数: {summary['successful_calls']}\n")
        f.write(f"  失败次数: {summary['failed_calls']}\n")
        f.write(f"  总耗时: {summary['total_duration']:.2f}秒\n")
        f.write(f"  平均耗时: {summary['average_duration']:.2f}秒\n")
        f.write(f"  成功率: {summary['success_rate']*100:.1f}%\n\n")
        
        f.write("所有尝试:\n")
        for i, attempt in enumerate(attempts, 1):
            f.write(f"  {i}. {attempt['model']} ({attempt['prompt']}): ")
            if attempt["success"]:
                f.write(f"成功 ({attempt['duration']:.2f}s)\n")
            else:
                error_type = attempt.get("error_type", "Unknown")
                f.write(f"失败 - {error_type}: {attempt['error']}\n")
    
    # 保存详细的API调用历史（JSON格式）
    api_log_file = os.path.join(debug_dir, f"{file_name}_api_calls.json")
    with open(api_log_file, "w", encoding="utf-8") as f:
        json.dump(monitor.call_history, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"  API调用详情已保存到: {api_log_file}")

def classify_error_type(error_message):
    """根据错误消息分类错误类型"""
    error_msg_lower = error_message.lower()
    
    # API连接相关错误
    network_keywords = ["timeout", "connection", "network", "dns", "ssl", "certificate", "unreachable"]
    if any(keyword in error_msg_lower for keyword in network_keywords):
        return ErrorType.NETWORK_ERROR
    
    # API相关错误
    api_keywords = ["api", "authentication", "unauthorized", "forbidden", "rate limit", "quota"]
    if any(keyword in error_msg_lower for keyword in api_keywords):
        return ErrorType.API_ERROR
    
    # 编译错误
    compilation_keywords = ["syntax error", "compilation error", "compile", "nvcc", "undefined", "redefinition"]
    if any(keyword in error_msg_lower for keyword in compilation_keywords):
        return ErrorType.COMPILATION_ERROR
    
    # 运行时错误
    runtime_keywords = ["runtime error", "execution", "cuda", "kernel", "memory", "device"]
    if any(keyword in error_msg_lower for keyword in runtime_keywords):
        return ErrorType.RUNTIME_ERROR
    
    # 默认为运行时错误
    return ErrorType.RUNTIME_ERROR

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
        
        # 第二步：转换为CUDA代码（使用增强版）
        print("  正在转换Triton代码为CUDA代码...")
        try:
            cuda_code, conversion_attempts, api_call_history = enhanced_triton2cuda_with_monitoring(triton_code, file_name)
            result.cuda_code = cuda_code
            
            # 记录转换详情和API调用历史
            result.conversion_details = conversion_attempts
            result.api_call_history = api_call_history
            
            if not cuda_code.strip():
                result.set_error(ErrorType.RUNTIME_ERROR, "转换后的代码为空")
                return result
                
            print(f"  转换成功，代码长度: {len(cuda_code)} 字符")
            
        except Exception as e:
            error_type = classify_error_type(str(e))
            result.set_error(error_type, f"CUDA代码生成失败: {str(e)}")
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
        error_type = classify_error_type(str(e))
        result.set_error(error_type, f"评测过程发生异常: {str(e)}")
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
        
        # API调用统计汇总
        total_api_calls = 0
        total_api_duration = 0
        successful_api_calls = 0
        api_error_types = {}
        
        for result in results:
            if hasattr(result, 'api_call_history') and result.api_call_history:
                for call in result.api_call_history:
                    total_api_calls += 1
                    total_api_duration += call.get('duration', 0)
                    if call.get('success', False):
                        successful_api_calls += 1
                    else:
                        error_type = call.get('error_details', {}).get('error_type', 'Unknown')
                        api_error_types[error_type] = api_error_types.get(error_type, 0) + 1
        
        f.write(f"API调用统计汇总:\n")
        f.write(f"  总API调用次数: {total_api_calls}\n")
        f.write(f"  成功API调用: {successful_api_calls}\n")
        f.write(f"  失败API调用: {total_api_calls - successful_api_calls}\n")
        f.write(f"  API成功率: {successful_api_calls/total_api_calls*100:.1f}%\n" if total_api_calls > 0 else "  API成功率: N/A\n")
        f.write(f"  总API耗时: {total_api_duration:.2f}秒\n")
        f.write(f"  平均API耗时: {total_api_duration/total_api_calls:.2f}秒\n" if total_api_calls > 0 else "  平均API耗时: N/A\n")
        
        if api_error_types:
            f.write(f"  API错误类型分布:\n")
            for error_type, count in api_error_types.items():
                f.write(f"    {error_type}: {count}\n")
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
        
        # API连接和调用统计
        network_errors = sum(1 for r in results if r.error_type == ErrorType.NETWORK_ERROR)
        api_errors = sum(1 for r in results if r.error_type == ErrorType.API_ERROR)
        
        f.write(f"API相关统计:\n")
        f.write(f"  API连接错误: {network_errors}/{total_files} ({network_errors/total_files*100:.1f}%)\n")
        f.write(f"  API调用错误: {api_errors}/{total_files} ({api_errors/total_files*100:.1f}%)\n\n")
        
        # 详细结果
        f.write("详细结果:\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            f.write(f"\n文件: {result.file_name}\n")
            f.write(f"总分: {result.total_score}/10 (编译: {result.compilation_score}/1, 准确性: {result.accuracy_score}/9)\n")
            f.write(f"错误类型: {result.error_type.value}\n")
            
            # API调用详情
            if hasattr(result, 'api_call_history') and result.api_call_history:
                f.write("API调用详情:\n")
                for i, call in enumerate(result.api_call_history, 1):
                    status = "成功" if call.get('success', False) else "失败"
                    duration = call.get('duration', 0)
                    model = call.get('model_type', 'Unknown')
                    f.write(f"  调用{i}: {model} - {status} ({duration:.2f}s)\n")
                    
                    if not call.get('success', False) and call.get('error_details'):
                        error_info = call['error_details']
                        f.write(f"    错误类型: {error_info.get('error_type', 'Unknown')}\n")
                        f.write(f"    错误信息: {error_info.get('error_message', '')[:100]}...\n")
            
            # 转换详情
            if hasattr(result, 'conversion_details') and result.conversion_details:
                f.write("转换尝试详情:\n")
                for i, attempt in enumerate(result.conversion_details, 1):
                    if attempt["success"]:
                        f.write(f"  {i}. {attempt['model']} ({attempt['prompt']}): 成功 ({attempt['duration']:.2f}s, {attempt['code_length']} 字符)\n")
                    else:
                        error_type = attempt.get('error_type', 'Unknown')
                        f.write(f"  {i}. {attempt['model']} ({attempt['prompt']}): 失败 ({attempt['duration']:.2f}s) - {error_type}\n")
            
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
            
            # 保存API调用详情（如果有的话）
            if hasattr(result, 'api_call_history') and result.api_call_history:
                api_file = os.path.join(debug_dir, f"{result.file_name}_api_details.json")
                with open(api_file, "w", encoding="utf-8") as api_f:
                    json.dump(result.api_call_history, api_f, indent=2, default=str, ensure_ascii=False)
                f.write(f"API调用详情已保存到: {api_file}\n")
            
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
    
    # 特别关注API相关错误
    network_errors = error_stats.get("NetworkError", 0)
    api_errors = error_stats.get("APIError", 0)
    
    if network_errors > 0 or api_errors > 0:
        print(f"\n⚠️  API相关问题:")
        if network_errors > 0:
            print(f"  API连接错误: {network_errors} 个文件")
        if api_errors > 0:
            print(f"  API调用错误: {api_errors} 个文件")
        print(f"  建议: 检查API配置和端点可用性")
    
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

# 新增：API连通性测试函数
def test_api_connectivity():
    """测试API端点可用性"""
    print("=== API连通性测试 ===")
    
    # API端点测试
    print("API端点测试:")
    api_status = NetworkMonitor.check_api_endpoints()
    
    for name, status in api_status.items():
        if status.get("available", False):
            response_time = status.get("response_time", 0)
            print(f"  {name}: ✓ 可用 (响应时间: {response_time:.2f}s)")
        else:
            error = status.get("error", "不可用")
            print(f"  {name}: ✗ 不可用 ({error})")
    
    available_apis = [name for name, status in api_status.items() if status.get("available", False)]
    
    if available_apis:
        print(f"\n可用API: {', '.join(available_apis)}")
        return True
    else:
        print("\n❌ 所有API都不可用")
        return False

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Triton到CUDA转换评测系统")
        parser.add_argument("--file", type=str, help="评测单个文件")
        parser.add_argument("--all", action="store_true", help="评测所有文件")
        parser.add_argument("--verbose", action="store_true", help="详细输出")
        parser.add_argument("--network", action="store_true", help="测试API端点可用性")
        parser.add_argument("--output", type=str, default="evaluation_report.txt", help="报告输出文件名（将保存到debug_output文件夹）")
        
        args = parser.parse_args()
        
        # 检查是否需要安装requests库
        try:
            import requests
        except ImportError:
            print("❌ 需要安装requests库")
            print("请运行: pip install requests")
            exit(1)
        
        if args.network:
            # API连通性测试
            success = test_api_connectivity()
            if not success:
                print("\n建议:")
                print("1. 验证API密钥配置")
                print("2. 确认防火墙设置")
                print("3. 尝试使用VPN或代理")
                print("4. 检查API服务是否正常运行")
            exit(0)
        
        elif args.file:
            # 单文件评测
            result = eval_single_file_enhanced(args.file, args.verbose)
            print(f"\n{result.get_summary()}")
            
            # 显示API相关提示
            if result.error_type in [ErrorType.NETWORK_ERROR, ErrorType.API_ERROR]:
                print("\n💡 API相关错误解决建议:")
                print("1. 运行 --network 参数测试API端点可用性")
                print("2. 检查API密钥是否正确配置")
                print("3. 确认API服务是否正常运行")
                print("4. 尝试切换网络环境或使用代理")
            
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
            
            # 显示API相关提示
            if result.error_type in [ErrorType.NETWORK_ERROR, ErrorType.API_ERROR]:
                print("\n💡 API相关错误解决建议:")
                print("1. 运行 --network 参数测试API端点可用性")
                print("2. 检查API密钥是否正确配置")
                print("3. 确认API服务是否正常运行")
                print("4. 尝试切换网络环境或使用代理")
            
            save_result_report([result], "flashatt_evaluation_report.txt")
        
    except Exception as e:
        print(f"评测过程发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup() 