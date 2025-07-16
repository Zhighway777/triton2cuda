#!/usr/bin/env python3
"""
测试新增的kernel文件的测试数据是否正确
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eval import get_inputs_for_file

def test_kernel_inputs():
    """测试新增的kernel文件的测试数据"""
    
    new_kernels = [
        "02-fused-softmax.py",
        "03-matrix-multiplication.py", 
        "04-low-memory-dropout.py",
        "05-layer-norm.py",
        "06-fused-attention.py",
        "08-grouped-gemm.py",
        "09-persistent-matmul.py",
        "10-block-scaled-matmul.py"
    ]
    
    for kernel_file in new_kernels:
        print(f"\n=== 测试 {kernel_file} ===")
        
        try:
            inputs = get_inputs_for_file(kernel_file)
            print(f"✓ 成功生成了 {len(inputs)} 组测试数据")
            
            for i, test_case in enumerate(inputs):
                print(f"  测试组 {i+1}:")
                for j, input_item in enumerate(test_case):
                    if hasattr(input_item, 'shape'):
                        print(f"    参数 {j+1}: torch.Tensor 形状 {input_item.shape}, 类型 {input_item.dtype}, 设备 {input_item.device}")
                    elif isinstance(input_item, list):
                        print(f"    参数 {j+1}: list 长度 {len(input_item)}")
                        for k, tensor in enumerate(input_item):
                            if hasattr(tensor, 'shape'):
                                print(f"      张量 {k+1}: 形状 {tensor.shape}, 类型 {tensor.dtype}, 设备 {tensor.device}")
                    else:
                        print(f"    参数 {j+1}: {type(input_item).__name__} 值 {input_item}")
                        
        except Exception as e:
            print(f"✗ 失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_kernel_inputs() 