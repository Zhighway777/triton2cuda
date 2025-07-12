import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def sum_spec(x: torch.Tensor) -> torch.Tensor:
    return x.sum(1)

@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # 1. 获取当前处理的batch索引
    pid = tl.program_id(0)
    batch_idx = pid * B0 + tl.arange(0, B0)
    
    # 2. 边界检查
    mask_batch = batch_idx < N0
    
    # 3. 初始化累加器
    acc = tl.zeros((B0,), dtype=tl.float32)
    
    # 4. 循环处理每个batch的所有元素
    for t_start in range(0, T, B1):
        # 计算当前循环的元素偏移
        t_offsets = t_start + tl.arange(0, B1)
        
        # 边界检查
        mask_t = t_offsets < T
        
        # 计算二维索引：batch_idx * T + t_offsets
        # 使用广播：[B0, 1] * T + [1, B1] = [B0, B1]
        idx_2d = batch_idx[:, None] * T + t_offsets[None, :]
        
        # 组合掩码
        mask_2d = mask_batch[:, None] & mask_t[None, :]
        
        # 加载数据
        x_chunk = tl.load(x_ptr + idx_2d, mask=mask_2d, other=0.0)
        
        # 对每个batch的当前chunk求和并累加
        acc += tl.sum(x_chunk, axis=1)
    
    # 5. 存储结果
    tl.store(z_ptr + batch_idx, acc, mask=mask_batch)
    return

def solve(x: torch.Tensor):
    # 输入形状为 [N0, N1]，输出形状为 [N0]
    N0, N1 = x.shape
    output = torch.empty((N0,), dtype=x.dtype, device=x.device)
    assert x.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置block size
    B0 = 1  # 每个程序处理一个batch
    B1 = 32  # 每次处理32个元素
    
    # 设置grid
    grid = lambda meta: (triton.cdiv(N0, meta['B0']),)
    
    # 调用kernel
    sum_kernel[grid](
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

# test(sum_kernel, sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})