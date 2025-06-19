#!/usr/bin/env python3
"""快速环境检查脚本"""

import sys
import os

def main():
    print("🔍 环境检查")
    
    try:
        # Python
        v = sys.version_info
        print(f"Python: {v.major}.{v.minor}.{v.micro}")
        
        # PyTorch
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        
        # Triton
        import triton
        print(f"Triton: {triton.__version__}")
        
        # CUDA_HOME
        cuda_home = os.environ.get('CUDA_HOME', 'Not Set')
        print(f"CUDA_HOME: {cuda_home}")
        
        print("✅ 检查完成")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 