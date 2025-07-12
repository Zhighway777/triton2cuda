import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def add_vec_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    x_range = tl.arange(0, B0)[None, :]
    y_range = tl.arange(0, B1)[:, None]

    mask_x = x_range < N0
    mask_y = y_range < N1   
    x = tl.load(x_ptr + x_range, mask=mask_x)
    y = tl.load(y_ptr + y_range, mask=mask_y)
    out_idx = y_range * N0 + x_range
    z = x + y
    tl.store(z_ptr + out_idx, z, mask=mask_x * mask_y)
    return

def solve(x: torch.Tensor, y: torch.Tensor):
    # 输出形状为 [y.shape[0], x.shape[0]]
    output = torch.empty((y.shape[0], x.shape[0]), dtype=x.dtype, device=x.device)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and output.dtype == torch.float32
    
    N0, N1 = x.shape[0], y.shape[0]
    
    # 设置block size，使其能够处理完整的向量
    B0 = triton.next_power_of_2(N0)
    B1 = triton.next_power_of_2(N1)
    
    # 只需要一个程序块
    grid = lambda meta: (1,)
    
    # 调用kernel
    add_vec_kernel[grid](
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

# test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32})