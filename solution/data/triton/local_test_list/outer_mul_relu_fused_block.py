import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def mul_relu_block_back_spec(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx

@triton.jit
def mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # 计算偏移
    start_i = pid_0 * B0
    start_j = pid_1 * B1
    offset_i = start_i + tl.arange(0, B0)
    offset_j = start_j + tl.arange(0, B1)

    # 创建掩码
    mask_i = offset_i < N0
    mask_j = offset_j < N1
    mask_2d = mask_j[:, None] & mask_i[None, :]

    # 计算二维索引
    idx_2d = offset_j[:, None] * N0 + offset_i[None, :]

    # 加载数据
    x = tl.load(x_ptr + idx_2d, mask=mask_2d)   # [B1, B0]
    y = tl.load(y_ptr + offset_j, mask=mask_j)  # [B1]
    dz = tl.load(dz_ptr + idx_2d, mask=mask_2d) # [B1, B0]

    # 计算梯度
    product = x * y[:, None]                    # [B1, B0]
    relu_mask = product > 0                     # [B1, B0]
    dx = dz * relu_mask * y[:, None]           # [B1, B0]

    # 存储结果
    tl.store(dx_ptr + idx_2d, dx, mask=mask_2d)

    return

def solve(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor):
    # 输入: x [N1, N0], y [N1], dz [N1, N0]
    # 输出: dx [N1, N0]
    N1, N0 = x.shape
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and dz.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and dz.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置block size
    B0 = 32
    B1 = 32
    
    # 设置2D grid
    grid = lambda meta: (
        triton.cdiv(N0, meta['B0']),
        triton.cdiv(N1, meta['B1'])
    )
    
    # 调用kernel
    mul_relu_block_back_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        dz_ptr=dz,
        dx_ptr=output,
        N0=N0,
        N1=N1,
        B0=B0,
        B1=B1
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor):
        return solve(x, y, dz)

# test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90})