import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def dot_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y

@triton.jit
def dot_kernel(x_ptr, y_ptr, z_ptr, N0, N1, N2, MID, B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B_MID: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    
    # 当前处理的批次
    batch_idx = pid_2
    if batch_idx >= N2:
        return
    
    # 当前处理的行和列块
    row_start = pid_0 * B0
    col_start = pid_1 * B1
    
    # 生成行和列偏移
    rows = row_start + tl.arange(0, B0)
    cols = col_start + tl.arange(0, B1)
    
    # 边界掩码
    row_mask = rows < N0
    col_mask = cols < N1
    
    # 初始化累加器
    acc = tl.zeros((B0, B1), dtype=tl.float32)
    
    # 在中间维度上分块处理
    for k in range(0, MID, B_MID):
        k_offsets = k + tl.arange(0, B_MID)
        k_mask = k_offsets < MID
        
        # 加载 x 块：[B0, B_MID]
        x_ptrs = (x_ptr + batch_idx * N0 * MID + 
                  rows[:, None] * MID + k_offsets[None, :])
        x_mask = row_mask[:, None] & k_mask[None, :]
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # 加载 y 块：[B_MID, B1]
        y_ptrs = (y_ptr + batch_idx * MID * N1 + 
                  k_offsets[:, None] * N1 + cols[None, :])
        y_mask = k_mask[:, None] & col_mask[None, :]
        y_block = tl.load(y_ptrs, mask=y_mask, other=0.0)
        
        # 矩阵乘法累加
        acc += tl.dot(x_block, y_block)
    
    # 存储结果
    z_ptrs = (z_ptr + batch_idx * N0 * N1 + 
              rows[:, None] * N1 + cols[None, :])
    z_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(z_ptrs, acc, mask=z_mask)

def solve(x: torch.Tensor, y: torch.Tensor):
    # 输入: x [N2, N0, MID], y [N2, MID, N1]
    # 输出: z [N2, N0, N1]
    N2, N0, MID = x.shape
    _, _, N1 = y.shape
    output = torch.empty((N2, N0, N1), dtype=x.dtype, device=x.device)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置block size
    B0 = 16
    B1 = 16
    B2 = 1
    B_MID = 16
    
    # 设置3D grid
    grid = lambda meta: (
        triton.cdiv(N0, meta['B0']),
        triton.cdiv(N1, meta['B1']),
        triton.cdiv(N2, meta['B2'])
    )
    
    # 调用kernel
    dot_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=output,
        N0=N0,
        N1=N1,
        N2=N2,
        MID=MID,
        B0=B0,
        B1=B1,
        B2=B2,
        B_MID=B_MID
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return solve(x, y)

# test(dot_kernel, dot_spec, B={"B0": 16, "B1": 16, "B2": 1, "B_MID": 16}, nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32})
