import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def add2_spec(x: torch.Tensor) -> torch.Tensor:
    return x + 10.

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    range = tl.arange(0, B0)
    pid = tl.program_id(0)
    offset = pid * B0 + range
    mask = offset < N0
    x = tl.load(x_ptr + offset, mask=mask)
    z = x + 10
    tl.store(z_ptr + offset, z, mask=mask)
    return

def solve(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and output.dtype == torch.float32
    n_elements = output.numel()
    
    # 设置grid和block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['B0']),)
    
    # 调用kernel
    add_mask2_kernel[grid](
        x_ptr=x,
        z_ptr=output,
        N0=n_elements,
        B0=BLOCK_SIZE
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return solve(x)

# test(add_mask2_kernel, add2_spec, nelem={"N0": 200})