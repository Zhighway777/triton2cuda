import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def add_vec_block_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    offs_i = pid_i * B0 + tl.arange(0, B0)
    offs_j = pid_j * B1 + tl.arange(0, B1)
    
    mask_i = offs_i < N0
    mask_j = offs_j < N1
    
    x = tl.load(x_ptr + offs_i, mask=mask_i)
    y = tl.load(y_ptr + offs_j, mask=mask_j)
    
    # 使用expand操作进行广播
    x_2d = tl.expand_dims(x, 0)  # [1, B0]
    y_2d = tl.expand_dims(y, 1)  # [B1, 1]
    
    z = y_2d + x_2d  # [B1, B0]
    
    # 计算输出索引
    offs_i_2d = tl.expand_dims(offs_i, 0)  # [1, B0]
    offs_j_2d = tl.expand_dims(offs_j, 1)  # [B1, 1]
    out_idx = offs_j_2d * N0 + offs_i_2d
    
    mask_2d = tl.expand_dims(mask_j, 1) & tl.expand_dims(mask_i, 0)
    
    tl.store(z_ptr + out_idx, z, mask=mask_2d)
    return

def solve(x: torch.Tensor, y: torch.Tensor):
    # 输出形状为 [y.shape[0], x.shape[0]]
    output = torch.empty((y.shape[0], x.shape[0]), dtype=x.dtype, device=x.device)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and output.dtype == torch.float32
    
    N0, N1 = x.shape[0], y.shape[0]
    
    # 设置block size
    B0 = 32
    B1 = 32
    
    # 设置2D grid
    grid = lambda meta: (
        triton.cdiv(N0, meta['B0']),
        triton.cdiv(N1, meta['B1'])
    )
    
    # 调用kernel
    add_vec_block_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=output,
        N0=N0,
        N1=N1,
        B0=B0,
        B1=B1
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return solve(x, y)

# test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90})