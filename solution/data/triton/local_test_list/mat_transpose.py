import torch
import triton
import triton.language as tl

torch.manual_seed(46)

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_ir, stride_ic,
    stride_or, stride_oc,
    BLOCK_SIZE : tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(0)  # 分块在 M 方向的索引
    pid_n = tl.program_id(1)  # 分块在 N 方向的索引

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_ptrs = input_ptr + offs_m[:, None] * stride_ir + offs_n[None, :] * stride_ic

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    block = tl.load(input_ptrs, mask=mask, other=0)

    transposed_block = tl.trans(block)  # Triton 内置转置函数

    output_ptrs = output_ptr + offs_n[:, None] * M + offs_m[None, :]  # 注意 M 是转置后的行步长

    tl.store(output_ptrs, transposed_block, mask=mask.T)  # mask 也需要转置

    return

def solve(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and output.dtype == torch.float32
    M, N = x.shape

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))

    matrix_transpose_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        M=M,
        N=N,
        stride_ir=x.stride(0),
        stride_ic=x.stride(1),
        stride_or=output.stride(0),
        stride_oc=output.stride(1),
        BLOCK_SIZE=32
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return solve(x)