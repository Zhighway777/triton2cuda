#!/usr/bin/env python3
"""
独立的CUDA Kernel测试文件
专门用于测试 vecadd_temp.py 中的CUDA kernel部分
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import time
import numpy as np

# 设置环境变量
os.environ["TORCH_USE_CUDA_DSA"] = "1"

class CUDAKernelExtractor:
    """
    CUDA Kernel 提取器和测试器
    """
    
    def __init__(self):
        self.cuda_module = None
        self.compilation_success = False
        
    def extract_and_compile_kernel(self):
        """
        提取并编译CUDA kernel
        """
        print("🔧 提取并编译CUDA Kernel...")
        
        # 提取的纯CUDA kernel代码
        cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// ============ CUDA KERNEL 核心部分 ============
__global__ void add_kernel(
    const float* x_ptr,        // 输入张量1的指针
    const float* y_ptr,        // 输入张量2的指针  
    float* output_ptr,         // 输出张量的指针
    int n_elements,            // 总元素数量
    int BLOCK_SIZE             // 线程块大小
) {
    // 1. 计算当前线程的全局索引
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 2. 边界检查，防止内存越界访问
    if (idx < n_elements) {
        // 3. 执行向量加法运算
        output_ptr[idx] = x_ptr[idx] + y_ptr[idx];
        
        // 可选：调试信息（仅在小数据集时启用）
        // if (idx < 10) {
        //     printf("Thread %d: %.2f + %.2f = %.2f\\n", 
        //            idx, x_ptr[idx], y_ptr[idx], output_ptr[idx]);
        // }
    }
}

// ============ C++ 包装函数 ============
torch::Tensor cuda_add_wrapper(torch::Tensor x_tensor, torch::Tensor y_tensor) {
    // 1. 输入验证
    TORCH_CHECK(x_tensor.is_cuda(), "输入张量x必须在CUDA设备上");
    TORCH_CHECK(y_tensor.is_cuda(), "输入张量y必须在CUDA设备上");
    TORCH_CHECK(x_tensor.is_contiguous(), "输入张量x必须是连续的");
    TORCH_CHECK(y_tensor.is_contiguous(), "输入张量y必须是连续的");
    TORCH_CHECK(x_tensor.sizes() == y_tensor.sizes(), "输入张量尺寸必须相同");
    TORCH_CHECK(x_tensor.dtype() == torch::kFloat32, "仅支持float32类型");
    TORCH_CHECK(y_tensor.dtype() == torch::kFloat32, "仅支持float32类型");

    // 2. 获取张量信息
    auto total_size = x_tensor.numel();
    auto output_tensor = torch::empty_like(x_tensor);

    // 3. 配置CUDA kernel启动参数
    const int BLOCK_SIZE = 256;  // 每个线程块256个线程
    const int blocks_per_grid = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 4. 调试信息
    // printf("启动kernel: blocks=%d, threads_per_block=%d, total_elements=%ld\\n", 
    //        blocks_per_grid, BLOCK_SIZE, total_size);

    // 5. 启动CUDA kernel
    add_kernel<<<blocks_per_grid, BLOCK_SIZE>>>(
        x_tensor.data_ptr<float>(),
        y_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        total_size,
        BLOCK_SIZE
    );

    // 6. 检查kernel启动错误
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel启动失败: ") + cudaGetErrorString(launch_err)
        );
    }

    // 7. 等待kernel执行完成
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel执行失败: ") + cudaGetErrorString(sync_err)
        );
    }

    return output_tensor;
}

// ============ 额外的测试函数 ============
void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("=== CUDA设备信息 ===\\n");
    printf("设备名称: %s\\n", prop.name);
    printf("计算能力: %d.%d\\n", prop.major, prop.minor);
    printf("多处理器数量: %d\\n", prop.multiProcessorCount);
    printf("每个线程块最大线程数: %d\\n", prop.maxThreadsPerBlock);
    printf("全局内存大小: %.2f GB\\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("===================\\n");
}

int test_kernel_basic() {
    print_device_info();
    
    // 创建测试数据
    const int N = 1000;
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_x[i] = i * 1.0f;
        h_y[i] = i * 2.0f;
    }
    
    // 分配GPU内存
    float *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    
    // 复制数据到GPU
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动kernel
    const int BLOCK_SIZE = 256;
    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_y, d_result, N, BLOCK_SIZE);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel启动失败: %s\\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 等待完成
    cudaDeviceSynchronize();
    
    // 复制结果回CPU
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_x[i] + h_y[i];
        if (abs(h_result[i] - expected) > 1e-5) {
            printf("错误位置 %d: 得到 %.2f, 期望 %.2f\\n", i, h_result[i], expected);
            correct = false;
            break;
        }
    }
    
    printf("基础kernel测试: %s\\n", correct ? "通过" : "失败");
    
    // 清理内存
    free(h_x); free(h_y); free(h_result);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_result);
    
    return correct ? 0 : -1;
}
"""

        # C++接口声明
        cpp_source = """
torch::Tensor cuda_add_wrapper(torch::Tensor x_tensor, torch::Tensor y_tensor);
int test_kernel_basic();
void print_device_info();
"""

        try:
            print("  编译CUDA代码...")
            self.cuda_module = load_inline(
                name="cuda_kernel_test",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["cuda_add_wrapper", "test_kernel_basic", "print_device_info"],
                verbose=True,  # 显示编译详情
                extra_cflags=["-O2"],
                extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_75"]  # 根据您的GPU调整架构
            )
            
            print("  ✅ CUDA Kernel编译成功!")
            self.compilation_success = True
            return True
            
        except Exception as e:
            print(f"  ❌ 编译失败: {e}")
            self.compilation_success = False
            return False
    
    def test_compilation(self):
        """
        测试编译正确性
        """
        print("\n📝 测试1: 编译正确性")
        print("-" * 30)
        
        success = self.extract_and_compile_kernel()
        if success:
            print("✅ 编译测试通过")
        else:
            print("❌ 编译测试失败")
        return success
    
    def test_device_info(self):
        """
        测试设备信息获取
        """
        if not self.compilation_success:
            return False
            
        print("\n🖥️  获取CUDA设备信息")
        print("-" * 30)
        
        try:
            self.cuda_module.print_device_info()
            return True
        except Exception as e:
            print(f"❌ 获取设备信息失败: {e}")
            return False
    
    def test_basic_kernel(self):
        """
        测试基础kernel功能
        """
        if not self.compilation_success:
            return False
            
        print("\n🧪 测试2: 基础Kernel功能")
        print("-" * 30)
        
        try:
            result = self.cuda_module.test_kernel_basic()
            if result == 0:
                print("✅ 基础kernel测试通过")
                return True
            else:
                print("❌ 基础kernel测试失败")
                return False
        except Exception as e:
            print(f"❌ 基础kernel测试异常: {e}")
            return False
    
    def test_pytorch_integration(self):
        """
        测试PyTorch集成
        """
        if not self.compilation_success:
            return False
            
        print("\n🔗 测试3: PyTorch集成")
        print("-" * 30)
        
        test_cases = [
            {"name": "小张量", "size": (100,)},
            {"name": "中等张量", "size": (10000,)},
            {"name": "大张量", "size": (1000000,)},
            {"name": "2D张量", "size": (1000, 100)},
            {"name": "3D张量", "size": (10, 100, 100)},
        ]
        
        all_passed = True
        
        for case in test_cases:
            try:
                print(f"  测试 {case['name']} {case['size']}...")
                
                # 创建测试数据
                x = torch.randn(case['size'], dtype=torch.float32, device='cuda')
                y = torch.randn(case['size'], dtype=torch.float32, device='cuda')
                
                # 使用CUDA kernel
                start_time = time.time()
                cuda_result = self.cuda_module.cuda_add_wrapper(x, y)
                cuda_time = time.time() - start_time
                
                # 使用PyTorch原生
                start_time = time.time()
                pytorch_result = x + y
                pytorch_time = time.time() - start_time
                
                # 检查结果正确性
                max_diff = torch.max(torch.abs(cuda_result - pytorch_result)).item()
                
                print(f"    最大差异: {max_diff:.2e}")
                print(f"    CUDA时间: {cuda_time*1000:.3f}ms")
                print(f"    PyTorch时间: {pytorch_time*1000:.3f}ms")
                print(f"    加速比: {pytorch_time/cuda_time:.2f}x")
                
                if max_diff > 1e-5:
                    print(f"    ❌ 数值精度测试失败")
                    all_passed = False
                else:
                    print(f"    ✅ 测试通过")
                    
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                all_passed = False
        
        return all_passed
    
    def test_error_handling(self):
        """
        测试错误处理
        """
        if not self.compilation_success:
            return False
            
        print("\n⚠️  测试4: 错误处理")
        print("-" * 30)
        
        error_cases = [
            {
                "name": "CPU张量输入",
                "setup": lambda: (torch.randn(100), torch.randn(100, device='cuda')),
                "should_fail": True
            },
            {
                "name": "不同尺寸张量",
                "setup": lambda: (torch.randn(100, device='cuda'), torch.randn(200, device='cuda')),
                "should_fail": True
            },
            {
                "name": "错误数据类型",
                "setup": lambda: (torch.randn(100, device='cuda', dtype=torch.float64), 
                                torch.randn(100, device='cuda', dtype=torch.float32)),
                "should_fail": True
            },
            {
                "name": "正常情况",
                "setup": lambda: (torch.randn(100, device='cuda'), torch.randn(100, device='cuda')),
                "should_fail": False
            }
        ]
        
        all_passed = True
        
        for case in error_cases:
            print(f"  测试 {case['name']}...")
            try:
                x, y = case['setup']()
                result = self.cuda_module.cuda_add_wrapper(x, y)
                
                if case['should_fail']:
                    print(f"    ❌ 应该失败但成功了")
                    all_passed = False
                else:
                    print(f"    ✅ 按预期成功")
                    
            except Exception as e:
                if case['should_fail']:
                    print(f"    ✅ 按预期失败: {str(e)[:50]}...")
                else:
                    print(f"    ❌ 不应该失败: {e}")
                    all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        print("🚀 开始CUDA Kernel独立测试")
        print("="*50)
        
        tests = [
            ("编译正确性", self.test_compilation),
            ("设备信息", self.test_device_info),
            ("基础Kernel", self.test_basic_kernel),
            ("PyTorch集成", self.test_pytorch_integration),
            ("错误处理", self.test_error_handling),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name}测试异常: {e}")
                results.append((test_name, False))
        
        # 输出总结
        print("\n" + "="*50)
        print("📊 测试总结")
        print("="*50)
        
        for test_name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name}: {status}")
        
        overall_success = all(result[1] for result in results)
        print(f"\n总体结果: {'🎉 全部通过' if overall_success else '💥 存在问题'}")
        
        return overall_success

def main():
    """
    主函数
    """
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查CUDA安装")
        return
    
    print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name()}")
    print()
    
    tester = CUDAKernelExtractor()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 建议: CUDA Kernel工作正常，可以安全使用")
    else:
        print("\n🔧 建议: 请检查CUDA安装和GPU驱动")

if __name__ == "__main__":
    main() 