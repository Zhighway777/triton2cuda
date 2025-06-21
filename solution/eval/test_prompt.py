import os
import sys

# 添加tri2cu模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'example_submission'))
import tri2cu

# 定义目录路径
cuda_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cuda', 'output_cuda')
tri_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'triton', 'test_list')
ref_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'ref', 'ref_cuda')

# 确保输出目录存在
os.makedirs(cuda_dir, exist_ok=True)

# 遍历triton目录中的所有Python文件，然后调用tri2cu中的转换函数，将转换完毕的cuda程序依次输出到cuda/output_cuda文件夹下
for filename in os.listdir(tri_dir):
    if filename.endswith('.py'):
        triton_file_path = os.path.join(tri_dir, filename)
        output_file_path = os.path.join(cuda_dir, filename.replace('.py', '_temp.py'))
        
        print(f"处理文件: {filename}")
        
        # 读取triton代码
        with open(triton_file_path, 'r', encoding='utf-8') as f:
            triton_code = f.read()
        
        try:
            # 调用转换函数
            cuda_code = tri2cu.triton2cuda(triton_code)
            
            # 保存转换后的CUDA代码
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cuda_code)
            
            print(f"✅ 成功转换 {filename}")
            
        except Exception as e:
            print(f"❌ 转换失败 {filename}: {str(e)}")

print("转换完成！")


