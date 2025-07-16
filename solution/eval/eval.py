import os
import sys
import tempfile
import torch
import importlib.util
import sys
import glob
import argparse
import requests
import traceback

# 修复路径问题：确保路径始终正确
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # eval目录
folder_path = os.path.dirname(current_file_dir)  # solution目录

# 确保工作目录正确
original_cwd = os.getcwd()
os.chdir(folder_path)

sys.path.append(folder_path)
sys.path.append(os.path.join(folder_path, "example_submission"))
sys.path.append(os.path.join(folder_path, "data", "ref"))

TEST_NN_MODEL_NAME = 'ModelNew'

from tri2cu import triton2cuda

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

def check_file_exists(file_path, description="文件"):
    """检查文件是否存在，如果不存在则提供详细的调试信息"""
    if os.path.exists(file_path):
        return True
    
    print(f"❌ {description}不存在: {file_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"绝对路径: {os.path.abspath(file_path)}")
    
    # 检查父目录是否存在
    parent_dir = os.path.dirname(file_path)
    if os.path.exists(parent_dir):
        print(f"父目录存在，包含文件:")
        try:
            files = os.listdir(parent_dir)
            for f in files[:10]:  # 只显示前10个文件
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
        except Exception as e:
            print(f"  无法列出父目录内容: {e}")
    else:
        print(f"父目录也不存在: {parent_dir}")
    
    return False

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
    elif file_name == "02-fused-softmax.py":
        # 一个2D张量 (n_rows, n_cols)
        return [
            [torch.randn(32, 128, device='cuda', dtype=torch.float32)],
            [torch.randn(64, 256, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "03-matrix-multiplication.py":
        # 两个2D张量 a (M, K) 和 b (K, N)
        return [
            [torch.randn(128, 64, device='cuda', dtype=torch.float32), torch.randn(64, 32, device='cuda', dtype=torch.float32)],
            [torch.randn(256, 128, device='cuda', dtype=torch.float32), torch.randn(128, 64, device='cuda', dtype=torch.float32)]
        ]
    elif file_name == "04-low-memory-dropout.py":
        # 一个张量 x、一个float值 p 和一个int值 seed
        return [
            [torch.randn(1024, device='cuda', dtype=torch.float32), 0.1, 42],
            [torch.randn(2048, device='cuda', dtype=torch.float32), 0.2, 123]
        ]
    elif file_name == "05-layer-norm.py":
        # 三个张量 x、weight、bias，和一个float值 eps
        return [
            [torch.randn(32, 256, device='cuda', dtype=torch.float32), torch.randn(256, device='cuda', dtype=torch.float32), torch.randn(256, device='cuda', dtype=torch.float32), 1e-5],
            [torch.randn(64, 512, device='cuda', dtype=torch.float32), torch.randn(512, device='cuda', dtype=torch.float32), torch.randn(512, device='cuda', dtype=torch.float32), 1e-5]
        ]
    elif file_name == "06-fused-attention.py":
        # 三个4D张量 q, k, v (BATCH, N_HEAD, N_CTX, HEAD_DIM)
        return [
            [torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16), torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16), torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16), True, 1.0],
            [torch.randn(4, 16, 256, 32, device='cuda', dtype=torch.float16), torch.randn(4, 16, 256, 32, device='cuda', dtype=torch.float16), torch.randn(4, 16, 256, 32, device='cuda', dtype=torch.float16), True, 1.0]
        ]
    elif file_name == "08-grouped-gemm.py":
        # 两个列表 group_A 和 group_B，每个包含多个2D张量
        return [
            [
                [torch.randn(32, 64, device='cuda', dtype=torch.float16), torch.randn(64, 32, device='cuda', dtype=torch.float16), torch.randn(48, 96, device='cuda', dtype=torch.float16)],
                [torch.randn(64, 32, device='cuda', dtype=torch.float16), torch.randn(32, 48, device='cuda', dtype=torch.float16), torch.randn(96, 64, device='cuda', dtype=torch.float16)]
            ],
            [
                [torch.randn(16, 32, device='cuda', dtype=torch.float16), torch.randn(24, 48, device='cuda', dtype=torch.float16)],
                [torch.randn(32, 16, device='cuda', dtype=torch.float16), torch.randn(48, 24, device='cuda', dtype=torch.float16)]
            ]
        ]
    elif file_name == "09-persistent-matmul.py":
        # 两个2D张量 a (M, K) 和 b (K, N)
        return [
            [torch.randn(128, 64, device='cuda', dtype=torch.float16), torch.randn(64, 32, device='cuda', dtype=torch.float16)],
            [torch.randn(256, 128, device='cuda', dtype=torch.float16), torch.randn(128, 64, device='cuda', dtype=torch.float16)]
        ]
    elif file_name == "10-block-scaled-matmul.py":
        # 四个张量 a, b, a_scale, b_scale
        return [
            [
                torch.randn(128, 64, device='cuda', dtype=torch.float16),
                torch.randn(64, 32, device='cuda', dtype=torch.float16),
                torch.randn(32, 8, device='cuda', dtype=torch.float32),
                torch.randn(8, 8, device='cuda', dtype=torch.float32)
            ],
            [
                torch.randn(256, 128, device='cuda', dtype=torch.float16),
                torch.randn(128, 64, device='cuda', dtype=torch.float16),
                torch.randn(64, 16, device='cuda', dtype=torch.float32),
                torch.randn(16, 16, device='cuda', dtype=torch.float32)
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
    ref_file_path = os.path.join("data", "ref", file_name)
    
    if check_file_exists(ref_file_path, f"参考实现 {file_name}"):
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
        triton_file_path = os.path.join("data", "triton", "local_test_list", file_name)
        
        if not check_file_exists(triton_file_path, f"Triton文件 {file_name}"):
            raise FileNotFoundError(f"无法找到Triton文件: {triton_file_path}")
        
        print(f"  使用本地测试文件作为参考: {triton_file_path}")
        
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

def save_debug_info(file_name, cuda_code, error_info):
    """保存调试信息到文件"""
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

def eval_single_file(file_name, verbose=False):
    """评测单个文件的Triton到CUDA转换"""
    print(f"\n=== 评测 {file_name} ===")
    error_messages = []
    
    try:
        # 1. 读取Triton代码
        triton_file_path = os.path.join("data", "triton", "local_test_list", file_name)
        
        if not check_file_exists(triton_file_path, f"Triton文件 {file_name}"):
            error_msg = f"无法找到Triton文件: {triton_file_path}"
            error_messages.append(error_msg)
            return False, error_messages
        
        with open(triton_file_path, "r") as f:
            triton_code = f.read()
        
        # 2. 转换为CUDA代码
        print("  正在转换Triton代码为CUDA代码...")
        try:
            cuda_code = triton2cuda(triton_code)
            if verbose:
                print(f"  转换成功，代码长度: {len(cuda_code)} 字符")
        except Exception as e:
            error_msg = f"转换失败: {str(e)}"
            error_messages.append(error_msg)
            
            # 分类API错误类型
            api_error_type = classify_api_error(str(e))
            if api_error_type in ["TimeoutError", "ConnectionError", "AuthenticationError", "NetworkError"]:
                print(f"  API错误 [{api_error_type}]: {error_msg}")
                if verbose:
                    print("  💡 建议: 运行 --network 参数测试API端点可用性")
            else:
                print(f"  错误: {error_msg}")
            
            return False, error_messages
        
        # 3. 获取参考模型和输入数据
        try:
            RefModel, input_tensors = get_reference_model_and_inputs(file_name)
            print(f"  准备了 {len(input_tensors)} 组测试数据")
        except Exception as e:
            error_msg = f"获取参考模型失败: {str(e)}"
            error_messages.append(error_msg)
            print(f"  错误: {error_msg}")
            return False, error_messages
        
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
                print("  CUDA模块加载成功")
            except Exception as e:
                error_msg = f"加载转换后的CUDA代码失败: {str(e)}"
                error_messages.append(error_msg)
                print(f"  错误: {error_msg}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                    save_debug_info(file_name, cuda_code, f"{error_msg}\n\n{traceback.format_exc()}")
                return False, error_messages
            
            # 获取转换后的ModelNew类
            CudaModel = getattr(cuda_module, TEST_NN_MODEL_NAME, None)
            if CudaModel is None:
                error_msg = f"无法在转换后的代码中找到 {TEST_NN_MODEL_NAME} 类"
                error_messages.append(error_msg)
                print(f"  错误: {error_msg}")
                if verbose:
                    available_attrs = [attr for attr in dir(cuda_module) if not attr.startswith('_')]
                    print(f"  可用的类/函数: {available_attrs}")
                    save_debug_info(file_name, cuda_code, f"{error_msg}\n可用属性: {available_attrs}")
                return False, error_messages
        
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
                try:
                    ref_output = RefModel()(*input_tensor_cuda)
                except Exception as e:
                    error_msg = f"参考模型计算失败: {str(e)}"
                    print(f"  测试 {i+1}: 失败 - {error_msg}")
                    error_messages.append(f"测试 {i+1}: {error_msg}")
                    continue
                
                # 计算转换后的CUDA输出
                try:
                    cuda_output = CudaModel()(*input_tensor_cuda)
                except Exception as e:
                    error_msg = f"CUDA模型计算失败: {str(e)}"
                    print(f"  测试 {i+1}: 失败 - {error_msg}")
                    error_messages.append(f"测试 {i+1}: {error_msg}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                
                # 比较结果
                try:
                    if torch.allclose(cuda_output, ref_output, atol=1e-3, rtol=1e-3):
                        print(f"  测试 {i+1}: 通过")
                        success_count += 1
                    else:
                        error_msg = f"输出不匹配 - 参考输出形状: {ref_output.shape}, CUDA输出形状: {cuda_output.shape}"
                        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
                        error_msg += f", 最大差异: {max_diff}"
                        print(f"  测试 {i+1}: 失败 - {error_msg}")
                        error_messages.append(f"测试 {i+1}: {error_msg}")
                        
                        if verbose:
                            print(f"    参考输出统计: min={ref_output.min().item():.6f}, max={ref_output.max().item():.6f}, mean={ref_output.mean().item():.6f}")
                            print(f"    CUDA输出统计: min={cuda_output.min().item():.6f}, max={cuda_output.max().item():.6f}, mean={cuda_output.mean().item():.6f}")
                            
                except Exception as e:
                    error_msg = f"结果比较失败: {str(e)}"
                    print(f"  测试 {i+1}: 失败 - {error_msg}")
                    error_messages.append(f"测试 {i+1}: {error_msg}")
                    
            except Exception as e:
                error_msg = f"测试执行异常: {str(e)}"
                print(f"  测试 {i+1}: 失败 - {error_msg}")
                error_messages.append(f"测试 {i+1}: {error_msg}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"结果: {success_count}/{total_count} 测试通过")
        return success_count == total_count, error_messages
        
    except Exception as e:
        error_msg = f"评测过程发生异常: {str(e)}"
        error_messages.append(error_msg)
        print(f"错误: {error_msg}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False, error_messages

def eval_all_files(verbose=False):
    """评测所有文件"""
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Solution目录: {folder_path}")
    
    # 获取所有triton文件
    test_dir = os.path.join("data", "triton", "local_test_list")
    
    if not check_file_exists(test_dir, "测试目录"):
        print("❌ 无法找到测试目录，终止评测")
        return False
    
    try:
        triton_files = [f for f in os.listdir(test_dir) if f.endswith('.py')]
        triton_files.sort()  # 按字母顺序排序
    except Exception as e:
        print(f"❌ 无法列出测试目录内容: {e}")
        return False
    
    print(f"找到 {len(triton_files)} 个triton文件需要评测")
    
    success_files = []
    failed_files = []
    all_errors = {}
    
    for file_name in triton_files:
        success, error_messages = eval_single_file(file_name, verbose)
        if success:
            success_files.append(file_name)
        else:
            failed_files.append(file_name)
            all_errors[file_name] = error_messages
    
    print(f"\n=== 总体结果 ===")
    print(f"成功: {len(success_files)}/{len(triton_files)} 个文件")
    
    if success_files:
        print("\n成功的文件:")
        for file_name in success_files:
            print(f"  ✓ {file_name}")
    
    if failed_files:
        print("\n失败的文件:")
        api_error_files = []
        for file_name in failed_files:
            print(f"  ✗ {file_name}")
            if verbose and file_name in all_errors:
                for error in all_errors[file_name][:3]:  # 只显示前3个错误
                    print(f"    - {error}")
                if len(all_errors[file_name]) > 3:
                    print(f"    - ... 还有 {len(all_errors[file_name]) - 3} 个错误")
            
            # 检查是否有API相关错误
            if file_name in all_errors:
                for error in all_errors[file_name]:
                    error_type = classify_api_error(error)
                    if error_type in ["TimeoutError", "ConnectionError", "AuthenticationError", "NetworkError"]:
                        api_error_files.append(file_name)
                        break
        
        # 提供API相关错误建议
        if api_error_files:
            print(f"\n⚠️ 检测到 {len(api_error_files)} 个文件可能存在API相关问题:")
            for file_name in api_error_files:
                print(f"  - {file_name}")
            print("\n💡 API相关错误解决建议:")
            print("1. 运行 --network 参数测试API端点可用性")
            print("2. 检查API密钥是否正确配置")
            print("3. 确认API服务是否正常运行")
            print("4. 尝试切换网络环境或使用代理")
    
    return len(failed_files) == 0

def eval_simple():
    """原始的vecadd评测函数（保留兼容性）"""
    success, errors = eval_single_file("vecadd.py")
    return success

def eval_golden():
    """使用golden标准评测vecadd - 比较转换后的CUDA代码与golden CUDA实现"""
    print("\n=== Golden标准评测 ===")
    print("比较: 转换后的CUDA代码 vs Golden CUDA实现")
    
    try:
        # 1. 读取Triton代码并转换为CUDA
        triton_file_path = os.path.join("data", "triton", "local_test_list", "vecadd.py")
        
        if not check_file_exists(triton_file_path, "Triton vecadd文件"):
            return False
        
        with open(triton_file_path, "r") as f:
            triton_code = f.read()
        
        cuda_code = triton2cuda(triton_code)
        print("✓ Triton代码转换完成")
        
        # 2. 加载参考实现获取输入数据
        from vecadd import Model as RefModel, get_inputs
        input_tensors = get_inputs()
        print(f"✓ 获取了 {len(input_tensors)} 组测试数据")
        
        # 3. 加载golden CUDA实现
        golden_file_path = os.path.join("data", "cuda", "vecadd.py")
        
        if not check_file_exists(golden_file_path, "Golden CUDA文件"):
            return False
        
        spec = importlib.util.spec_from_file_location("golden_cuda", golden_file_path)
        golden_module = importlib.util.module_from_spec(spec)
        sys.modules["golden_cuda"] = golden_module
        spec.loader.exec_module(golden_module)
        
        GoldenModel = getattr(golden_module, TEST_NN_MODEL_NAME, None)
        if GoldenModel is None:
            raise ValueError(f"无法在golden实现中找到 {TEST_NN_MODEL_NAME} 类")
        print("✓ Golden CUDA实现加载完成")
        
        # 4. 加载转换后的CUDA代码
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "converted_cuda.py")
            with open(temp_file, "w") as f:
                f.write(cuda_code)
            
            spec = importlib.util.spec_from_file_location("converted_cuda", temp_file)
            converted_module = importlib.util.module_from_spec(spec)
            sys.modules["converted_cuda"] = converted_module
            spec.loader.exec_module(converted_module)
            
            ConvertedModel = getattr(converted_module, TEST_NN_MODEL_NAME, None)
            if ConvertedModel is None:
                raise ValueError(f"无法在转换后的代码中找到 {TEST_NN_MODEL_NAME} 类")
            print("✓ 转换后的CUDA代码加载完成")
        
        # 5. 三方对比测试：参考实现 vs Golden实现 vs 转换后的实现
        print("\n开始三方对比测试...")
        all_passed = True
        
        for i, input_tensor in enumerate(input_tensors):
            # 参考输出（Triton）
            ref_output = RefModel()(*input_tensor)
            
            # Golden输出（手写CUDA）
            golden_output = GoldenModel()(*input_tensor)
            
            # 转换输出（转换后的CUDA）
            converted_output = ConvertedModel()(*input_tensor)
            
            # 检查参考 vs Golden
            ref_golden_match = torch.allclose(golden_output, ref_output, atol=1e-3, rtol=1e-3)
            
            # 检查转换 vs 参考
            converted_ref_match = torch.allclose(converted_output, ref_output, atol=1e-3, rtol=1e-3)
            
            # 检查转换 vs Golden
            converted_golden_match = torch.allclose(converted_output, golden_output, atol=1e-3, rtol=1e-3)
            
            print(f"  测试 {i+1}:")
            print(f"    参考 vs Golden: {'✓' if ref_golden_match else '✗'}")
            print(f"    转换 vs 参考:   {'✓' if converted_ref_match else '✗'}")
            print(f"    转换 vs Golden: {'✓' if converted_golden_match else '✗'}")
            
            if not (ref_golden_match and converted_ref_match and converted_golden_match):
                all_passed = False
                if not ref_golden_match:
                    max_diff = torch.max(torch.abs(golden_output - ref_output)).item()
                    print(f"      参考vs Golden最大差异: {max_diff}")
                if not converted_ref_match:
                    max_diff = torch.max(torch.abs(converted_output - ref_output)).item()
                    print(f"      转换vs参考最大差异: {max_diff}")
                if not converted_golden_match:
                    max_diff = torch.max(torch.abs(converted_output - golden_output)).item()
                    print(f"      转换vs Golden最大差异: {max_diff}")
        
        if all_passed:
            print("\n🎉 Golden标准评测通过！所有实现结果一致")
        else:
            print("\n❌ Golden标准评测失败！存在实现差异")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Golden评测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_usage():
    """显示使用说明"""
    print("Triton到CUDA转换评测系统使用说明:")
    print("")
    print("基本用法:")
    print("  python eval.py --file vecadd.py        # 评测单个文件")
    print("  python eval.py --all                   # 评测所有文件")
    print("  python eval.py --all --verbose         # 详细模式评测所有文件")
    print("")
    print("特殊功能:")
    print("  python eval.py --network              # 测试API端点可用性")
    print("  python eval.py --golden               # Golden标准评测")
    print("  python eval.py --simple               # 简单vecadd评测")
    print("")
    print("API连通性测试:")
    print("  当遇到转换失败时，可以先运行 --network 测试API是否可用")
    print("  系统会自动检测API相关错误并提供解决建议")
    print("")

def cleanup():
    """恢复原始工作目录"""
    os.chdir(original_cwd)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Triton到CUDA转换评测系统")
        parser.add_argument("--file", type=str, help="评测单个文件")
        parser.add_argument("--all", action="store_true", help="评测所有文件")
        parser.add_argument("--verbose", action="store_true", help="详细输出")
        parser.add_argument("--network", action="store_true", help="测试API端点可用性")
        parser.add_argument("--golden", action="store_true", help="Golden标准评测")
        parser.add_argument("--simple", action="store_true", help="简单vecadd评测")
        parser.add_argument("--help-usage", action="store_true", help="显示详细使用说明")
        
        args = parser.parse_args()
        
        # 检查是否需要安装requests库
        try:
            import requests
        except ImportError:
            print("❌ 需要安装requests库")
            print("请运行: pip install requests")
            exit(1)
        
        if args.help_usage:
            # 显示使用说明
            show_usage()
            exit(0)
        
        elif args.network:
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
            success, error_messages = eval_single_file(args.file, args.verbose)
            if success:
                print(f"\n✅ {args.file} 评测通过")
            else:
                print(f"\n❌ {args.file} 评测失败")
                if error_messages:
                    # 检查是否有API相关错误
                    api_errors = []
                    for error in error_messages:
                        error_type = classify_api_error(error)
                        if error_type in ["TimeoutError", "ConnectionError", "AuthenticationError", "NetworkError"]:
                            api_errors.append(error_type)
                    
                    if api_errors:
                        print("\n💡 API相关错误解决建议:")
                        print("1. 运行 --network 参数测试API端点可用性")
                        print("2. 检查API密钥是否正确配置")
                        print("3. 确认API服务是否正常运行")
                        print("4. 尝试切换网络环境或使用代理")
        
        elif args.all:
            # 批量评测
            success = eval_all_files(args.verbose)
            if success:
                print("\n🎉 所有文件评测通过!")
            else:
                print("\n❌ 部分文件评测失败")
        
        elif args.golden:
            # Golden标准评测
            success = eval_golden()
            if success:
                print("\n🎉 Golden标准评测通过!")
            else:
                print("\n❌ Golden标准评测失败")
        
        elif args.simple:
            # 简单vecadd评测
            success = eval_simple()
            if success:
                print("\n✅ 简单vecadd评测通过")
            else:
                print("\n❌ 简单vecadd评测失败")
        
        else:
            # 默认评测单个文件
            success, error_messages = eval_single_file("mat_transpose.py", verbose=True)
            if success:
                print("\n✅ 默认评测通过")
            else:
                print("\n❌ 默认评测失败")
                if error_messages:
                    # 检查是否有API相关错误
                    api_errors = []
                    for error in error_messages:
                        error_type = classify_api_error(error)
                        if error_type in ["TimeoutError", "ConnectionError", "AuthenticationError", "NetworkError"]:
                            api_errors.append(error_type)
                    
                    if api_errors:
                        print("\n💡 API相关错误解决建议:")
                        print("1. 运行 --network 参数测试API端点可用性")
                        print("2. 检查API密钥是否正确配置")
                        print("3. 确认API服务是否正常运行")
                        print("4. 尝试切换网络环境或使用代理")
        
    except Exception as e:
        print(f"评测过程发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保恢复原始工作目录
        cleanup()
    