import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def mul_relu_block_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x[None, :] * y[:, None])

@triton.jit
def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    # 1. 计算偏移和掩码
    start_i = pid_0 * B0
    start_j = pid_1 * B1
    offset_i = start_i + tl.arange(0, B0)
    offset_j = start_j + tl.arange(0, B1)
    
    mask_i = offset_i < N0
    mask_j = offset_j < N1
    
    # 2. 加载数据
    x = tl.load(x_ptr + offset_i, mask=mask_i) 
    y = tl.load(y_ptr + offset_j, mask=mask_j)
    
    # 3. 计算外积
    product = x[None, :] * y[:, None]  # [B1, B0]
    
    # 4. 手写 ReLU - 方法1：使用 tl.where
    # z = tl.where(product > 0, product, 0)
    
    # 或者方法2：使用 tl.maximum
    z = tl.maximum(product, 0)
    
    # 或者方法3：使用掩码
    # relu_mask = product > 0
    # z = product * relu_mask
    
    # 5. 计算输出索引
    out_idx = offset_j[:, None] * N0 + offset_i[None, :]
    print("Output indices shape:", out_idx.shape)
    # 6. 存储结果
    mask_2d = mask_j[:, None] & mask_i[None, :]
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
    mul_relu_block_kernel[grid](
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

# test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})