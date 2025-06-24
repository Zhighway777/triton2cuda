#!/usr/bin/env python3
"""
CUDA代码提取工具
从Python文件中提取CUDA kernel代码并生成独立的可运行CUDA文件
"""

import os
import re
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class CudaExtractor:
    def __init__(self, input_dir: str, output_dir: str = "extracted_cuda"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_cuda_from_python(self, py_file: Path) -> Optional[Dict]:
        """从Python文件中提取CUDA相关代码"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析AST
            tree = ast.parse(content)
            
            cuda_info = {
                'kernel_source': None,
                'cpp_source': None,
                'function_names': [],
                'includes': [],
                'kernel_functions': []
            }
            
            # 查找CUDA源码字符串
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # 检查变量值是否包含CUDA代码
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                source_code = node.value.value
                                # 更宽松的CUDA代码检测：检查是否包含__global__关键字
                                if '__global__' in source_code:
                                    cuda_info['kernel_source'] = source_code
                                    cuda_info['kernel_functions'].extend(
                                        self._extract_kernel_functions(source_code)
                                    )
                                    print(f"    找到CUDA kernel源码在变量: {target.id}")
                                # 检查是否是CPP源码（通常包含函数声明）
                                elif ('torch::Tensor' in source_code or 
                                      target.id.lower().endswith('cpp_source') or 
                                      target.id.lower().endswith('cpp')):
                                    cuda_info['cpp_source'] = source_code
                                    print(f"    找到CPP源码在变量: {target.id}")
                                    
                # 查找load_inline调用
                elif isinstance(node, ast.Call):
                    if (hasattr(node.func, 'attr') and node.func.attr == 'load_inline') or \
                       (hasattr(node.func, 'id') and node.func.id == 'load_inline'):
                        for keyword in node.keywords:
                            if keyword.arg == 'functions':
                                if isinstance(keyword.value, ast.List):
                                    cuda_info['function_names'] = [
                                        elt.value for elt in keyword.value.elts 
                                        if isinstance(elt, ast.Constant)
                                    ]
                                    print(f"    找到函数列表: {cuda_info['function_names']}")
                            elif keyword.arg == 'cuda_sources':
                                # 处理cuda_sources参数，可能是变量引用
                                if isinstance(keyword.value, ast.Name):
                                    print(f"    找到cuda_sources参数引用变量: {keyword.value.id}")
                            elif keyword.arg == 'cpp_sources':
                                # 处理cpp_sources参数
                                if isinstance(keyword.value, ast.Name):
                                    print(f"    找到cpp_sources参数引用变量: {keyword.value.id}")
            
            return cuda_info if cuda_info['kernel_source'] else None
            
        except Exception as e:
            print(f"解析文件 {py_file} 时出错: {e}")
            return None
    
    def _extract_kernel_functions(self, cuda_source: str) -> List[str]:
        """从CUDA源码中提取kernel函数名"""
        kernel_pattern = r'__global__\s+\w+\s+(\w+)\s*\('
        return re.findall(kernel_pattern, cuda_source)
    
    def _extract_includes(self, cuda_source: str) -> List[str]:
        """从CUDA源码中提取include语句"""
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        return re.findall(include_pattern, cuda_source)
    
    def generate_cuda_file(self, cuda_info: Dict, output_name: str) -> str:
        """生成独立的CUDA文件"""
        cuda_content = []
        
        # 添加标准includes
        standard_includes = [
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <cuda_runtime.h>',
            '#include <device_launch_parameters.h>'
        ]
        
        cuda_content.extend(standard_includes)
        cuda_content.append('')
        
        # 提取并清理CUDA kernel代码
        if cuda_info['kernel_source']:
            kernel_code = self._extract_kernel_only(cuda_info['kernel_source'])
            cuda_content.append(kernel_code)
            cuda_content.append('')
        
        # 生成主函数
        main_function = self._generate_main_function(cuda_info)
        cuda_content.append(main_function)
        
        # 写入文件
        output_file = self.output_dir / f"{output_name}.cu"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cuda_content))
            
        return str(output_file)
    
    def _extract_kernel_only(self, source: str) -> str:
        """仅提取CUDA kernel函数，移除torch相关代码"""
        lines = source.split('\n')
        kernel_lines = []
        in_kernel = False
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过torch相关include
            if '#include <torch/extension.h>' in line:
                continue
                
            # 保留其他include
            if stripped.startswith('#include'):
                kernel_lines.append(line)
                continue
                
            # 检测kernel函数开始
            if '__global__' in stripped:
                in_kernel = True
                kernel_lines.append(line)
                continue
                
            # 如果在kernel函数内，保留所有行直到函数结束
            if in_kernel:
                kernel_lines.append(line)
                # 简单的大括号计数来检测函数结束
                if stripped == '}' and not any(c in stripped for c in ['if', 'for', 'while']):
                    in_kernel = False
                continue
                
            # 跳过torch相关的wrapper函数
            if any(keyword in stripped for keyword in ['torch::', 'Tensor', 'data_ptr', 'numel', 'empty_like']):
                continue
                
            # 保留其他代码
            if stripped and not stripped.startswith('//'):
                kernel_lines.append(line)
        
        return '\n'.join(kernel_lines)
    
    def _clean_cuda_source(self, source: str) -> str:
        """清理CUDA源码，移除torch相关代码"""
        lines = source.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过torch相关的include
            if '#include <torch/extension.h>' in line:
                continue
                
            # 替换torch::Tensor函数参数为float*
            line = re.sub(r'torch::Tensor\s+(\w+)', r'float* \1', line)
            
            # 替换torch相关函数调用
            line = re.sub(r'(\w+)\.data_ptr<float>\(\)', r'\1', line)
            line = re.sub(r'(\w+)\.numel\(\)', r'size', line)
            line = re.sub(r'auto\s+(\w+)\s*=\s*torch::empty_like\([^)]+\);', r'float* \1 = d_result;', line)
            line = re.sub(r'auto\s+size\s*=.*\.numel\(\);', r'// size is passed as parameter', line)
            
            # 修复函数返回类型
            line = re.sub(r'torch::Tensor\s+(\w+)\s*\(', r'void \1(', line)
            
            # 移除return语句（如果返回的是tensor）
            if 'return out;' in line and 'torch::' in source:
                line = '    // result is stored in out parameter'
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _generate_main_function(self, cuda_info: Dict) -> str:
        """生成主函数用于测试CUDA kernel"""
        kernel_functions = cuda_info['kernel_functions']
        if not kernel_functions:
            return self._generate_default_main()
            
        main_code = [
            "// 错误检查宏",
            "#define CHECK_CUDA(call) do { \\",
            "    cudaError_t err = call; \\",
            "    if (err != cudaSuccess) { \\",
            "        printf(\"CUDA error at %s:%d - %s\\n\", __FILE__, __LINE__, cudaGetErrorString(err)); \\",
            "        exit(1); \\",
            "    } \\",
            "} while(0)",
            "",
            "int main() {",
            "    const int N = 1024;",
            "    const int size = N * sizeof(float);",
            "    ",
            "    // 分配主机内存",
            "    float *h_a = (float*)malloc(size);",
            "    float *h_b = (float*)malloc(size);",
            "    float *h_result = (float*)malloc(size);",
            "    ",
            "    // 初始化数据",
            "    for (int i = 0; i < N; i++) {",
            "        h_a[i] = (float)i;",
            "        h_b[i] = (float)i * 2;",
            "    }",
            "    ",
            "    // 分配设备内存",
            "    float *d_a, *d_b, *d_result;",
            "    CHECK_CUDA(cudaMalloc(&d_a, size));",
            "    CHECK_CUDA(cudaMalloc(&d_b, size));",
            "    CHECK_CUDA(cudaMalloc(&d_result, size));",
            "    ",
            "    // 复制数据到设备",
            "    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));",
            "    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));",
            "    ",
            "    // 配置执行参数",
            "    int blockSize = 256;",
            "    int numBlocks = (N + blockSize - 1) / blockSize;",
            "    ",
            f"    // 启动kernel: {kernel_functions[0] if kernel_functions else 'kernel'}",
            f"    {kernel_functions[0] if kernel_functions else 'vector_add_kernel'}<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);",
            "    CHECK_CUDA(cudaDeviceSynchronize());",
            "    ",
            "    // 复制结果回主机",
            "    CHECK_CUDA(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));",
            "    ",
            "    // 验证结果",
            "    printf(\"验证前10个结果:\\n\");",
            "    for (int i = 0; i < 10; i++) {",
            "        printf(\"h_result[%d] = %.2f (expected: %.2f)\\n\", i, h_result[i], h_a[i] + h_b[i]);",
            "    }",
            "    ",
            "    // 清理内存",
            "    free(h_a); free(h_b); free(h_result);",
            "    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);",
            "    ",
            "    printf(\"CUDA程序执行完成！\\n\");",
            "    return 0;",
            "}"
        ]
        
        return '\n'.join(main_code)
    
    def _generate_default_main(self) -> str:
        """生成默认的主函数"""
        return """
int main() {
    printf("CUDA kernel extracted successfully!\\n");
    return 0;
}
"""
    
    def generate_makefile(self, cuda_files: List[str]) -> str:
        """生成Makefile用于编译CUDA程序"""
        makefile_content = [
            "# CUDA编译配置",
            "NVCC = nvcc",
            "CUDA_FLAGS = -O3 -arch=sm_60",
            "INCLUDES = -I/usr/local/cuda/include",
            "LIBS = -L/usr/local/cuda/lib64 -lcudart",
            "",
            "# 目标文件",
            "TARGETS = " + " ".join([Path(f).stem for f in cuda_files]),
            "",
            "all: $(TARGETS)",
            "",
        ]
        
        # 为每个CUDA文件生成编译规则
        for cuda_file in cuda_files:
            stem = Path(cuda_file).stem
            makefile_content.extend([
                f"{stem}: {Path(cuda_file).name}",
                f"\t$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)",
                ""
            ])
        
        makefile_content.extend([
            "clean:",
            "\trm -f $(TARGETS)",
            "",
            ".PHONY: all clean"
        ])
        
        makefile_path = self.output_dir / "Makefile"
        with open(makefile_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(makefile_content))
            
        return str(makefile_path)
    
    def process_directory(self) -> None:
        """处理整个目录中的Python文件"""
        py_files = list(self.input_dir.glob("*.py"))
        if not py_files:
            print(f"在目录 {self.input_dir} 中没有找到Python文件")
            return
        
        cuda_files = []
        
        for py_file in py_files:
            print(f"处理文件: {py_file.name}")
            cuda_info = self.extract_cuda_from_python(py_file)
            
            if cuda_info:
                output_name = py_file.stem
                cuda_file = self.generate_cuda_file(cuda_info, output_name)
                cuda_files.append(cuda_file)
                print(f"  -> 生成CUDA文件: {cuda_file}")
            else:
                print(f"  -> 未找到CUDA代码")
        
        if cuda_files:
            makefile = self.generate_makefile(cuda_files)
            print(f"\n生成Makefile: {makefile}")
            
            # 生成编译脚本
            self._generate_build_script(cuda_files)
            
            print(f"\n提取完成！生成了 {len(cuda_files)} 个CUDA文件")
            print(f"输出目录: {self.output_dir}")
            print("\n编译说明:")
            print("1. 使用Makefile: make")
            print("2. 使用编译脚本: ./build.sh")
            print("3. 手动编译: nvcc -O3 -arch=sm_60 -o <output> <input.cu> -lcudart")
        else:
            print("没有找到任何CUDA代码")
    
    def _generate_build_script(self, cuda_files: List[str]) -> str:
        """生成编译脚本"""
        script_content = [
            "#!/bin/bash",
            "# CUDA编译脚本",
            "",
            "echo \"编译CUDA程序...\"",
            ""
        ]
        
        for cuda_file in cuda_files:
            stem = Path(cuda_file).stem
            filename = Path(cuda_file).name
            script_content.extend([
                f"echo \"编译 {filename}...\"",
                f"nvcc -O3 -arch=sm_60 -o {stem} {filename} -lcudart",
                f"if [ $? -eq 0 ]; then",
                f"    echo \"  -> {stem} 编译成功\"",
                f"else",
                f"    echo \"  -> {stem} 编译失败\"",
                f"fi",
                ""
            ])
        
        script_content.append("echo \"编译完成！\"")
        
        script_path = self.output_dir / "build.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_content))
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        
        return str(script_path)


def main():
    parser = argparse.ArgumentParser(description="从Python文件中提取CUDA代码并生成独立的CUDA文件")
    parser.add_argument("input_dir", help="包含Python文件的输入目录")
    parser.add_argument("-o", "--output", default="extracted_cuda", help="输出目录 (默认: extracted_cuda)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return
    
    extractor = CudaExtractor(args.input_dir, args.output)
    extractor.process_directory()


if __name__ == "__main__":
    main() 