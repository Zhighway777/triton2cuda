import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def softmax_spec(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)

@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504
    
    # 边界检查
    if pid_0 >= N0:
        return
    
    # 循环1：找到当前batch的最大值
    max_val = -float('inf')
    for t_start in range(0, T, B1):
        t_offsets = t_start + tl.arange(0, B1)
        mask = t_offsets < T
        
        idx = pid_0 * T + t_offsets
        x_chunk = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        
        chunk_max = tl.max(x_chunk)
        max_val = tl.maximum(max_val, chunk_max)
    
    # 循环2：计算指数值的和
    sum_exp = 0.0
    for t_start in range(0, T, B1):
        t_offsets = t_start + tl.arange(0, B1)
        mask = t_offsets < T
        
        idx = pid_0 * T + t_offsets
        x_chunk = tl.load(x_ptr + idx, mask=mask, other=0.0)
        
        # 数值稳定的指数计算
        stable_x = (x_chunk - max_val) * log2_e
        exp_chunk = tl.exp2(stable_x)
        
        # 只对有效元素求和
        exp_chunk = tl.where(mask, exp_chunk, 0.0)
        sum_exp += tl.sum(exp_chunk)
    
    # 循环3：计算并存储最终的softmax值
    for t_start in range(0, T, B1):
        t_offsets = t_start + tl.arange(0, B1)
        mask = t_offsets < T
        
        idx = pid_0 * T + t_offsets
        x_chunk = tl.load(x_ptr + idx, mask=mask, other=0.0)
        
        # 计算 softmax
        stable_x = (x_chunk - max_val) * log2_e
        exp_chunk = tl.exp2(stable_x)
        softmax_chunk = exp_chunk / sum_exp
        
        # 存储结果
        tl.store(z_ptr + idx, softmax_chunk, mask=mask)
    return

def solve(x: torch.Tensor):
    # 输入形状为 [N0, N1]，输出形状为 [N0, N1]
    N0, N1 = x.shape
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置block size
    B0 = 1  # 每个程序处理一行
    B1 = 32  # 每次处理32个元素
    
    # 设置grid
    grid = lambda meta: (triton.cdiv(N0, meta['B0']),)
    
    # 调用kernel
    softmax_kernel[grid](
        x_ptr=x,
        z_ptr=output,
        N0=N0,
        N1=N1,
        T=N1,  # T等于N1，表示每行的元素个数
        B0=B0,
        B1=B1
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return solve(x)

# test(softmax_kernel, softmax_spec, B={"B0": 1, "B1":32}, nelem={"N0": 4, "N1": 32, "T": 200})