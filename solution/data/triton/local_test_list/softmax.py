import torch
import triton
import triton.language as tl

torch.manual_seed(46)

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):

        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def solve(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and output.dtype == torch.float32
    n_rows, n_cols = x.shape
    # 计算每行的stride（以元素为单位）
    input_row_stride = x.stride(0)
    output_row_stride = output.stride(0)
    
    # 设置grid - 使用足够的程序来并行处理所有行
    grid = lambda meta: (min(n_rows, 1024),)
    
    # 调用kernel
    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        input_row_stride=input_row_stride,
        output_row_stride=output_row_stride,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_stages=4
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return solve(x)