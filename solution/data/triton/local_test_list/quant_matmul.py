import torch
import triton
import triton.language as tl

torch.manual_seed(46)

FPINT = 32 // 4
GROUP = 8

def quant_dot_spec(scale: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    offset = offset.view(32, 1)
    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation

@triton.jit
def quant_dot_kernel(scale_ptr, offset_ptr, weight_ptr, activation_ptr,
                     z_ptr, N0, N1, MID, B0: tl.constexpr, B1: tl.constexpr, B_MID: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    row_start = pid_0 * B0
    col_start = pid_1 * B1
    
    row_offsets = row_start + tl.arange(0, B0)
    col_offsets = col_start + tl.arange(0, B1)
    
    row_mask = row_offsets < N0
    col_mask = col_offsets < N1
    
    acc = tl.zeros((B0, B1), dtype=tl.float32)
    
    # 在MID维度上循环，但减少嵌套
    for current_mid in range(MID):
        group_idx = current_mid // 8
        in_group_pos = current_mid % 8
        
        # 向量化加载激活值
        act_ptrs = activation_ptr + current_mid * N1 + col_offsets
        act_vals = tl.load(act_ptrs, mask=col_mask, other=0.0)  # [B1]
        
        # 向量化加载scale
        scale_ptrs = scale_ptr + row_offsets * 8 + group_idx
        scale_vals = tl.load(scale_ptrs, mask=row_mask, other=0.0)  # [B0]
        
        # 向量化加载offset
        offset_ptrs = offset_ptr + row_offsets
        offset_packed = tl.load(offset_ptrs, mask=row_mask, other=0)  # [B0]
        offset_vals = (offset_packed >> (group_idx * 4)) & 0xF
        offset_vals = offset_vals.to(tl.float32)
        
        # 向量化加载weight
        weight_ptrs = weight_ptr + row_offsets * 8 + current_mid // 8
        weight_packed = tl.load(weight_ptrs, mask=row_mask, other=0)  # [B0]
        weight_vals = (weight_packed >> (in_group_pos * 4)) & 0xF
        weight_vals = weight_vals.to(tl.float32)
        
        # 向量化计算反量化权重
        dequant_weights = scale_vals * (weight_vals - offset_vals)  # [B0]
        
        # 向量化累加：外积操作
        acc += dequant_weights[:, None] * act_vals[None, :]  # [B0, B1]
    
    # 存储结果
    z_ptrs = z_ptr + row_offsets[:, None] * N1 + col_offsets[None, :]
    z_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(z_ptrs, acc, mask=z_mask)

def solve(scale: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor, activation: torch.Tensor):
    # 输入: scale [N0, 8], offset [N0], weight [N0, 8], activation [MID, N1]
    # 输出: z [N0, N1]
    N0 = scale.shape[0]
    MID, N1 = activation.shape
    output = torch.empty((N0, N1), dtype=scale.dtype, device=scale.device)
    assert scale.is_cuda and offset.is_cuda and weight.is_cuda and activation.is_cuda and output.is_cuda
    assert scale.dtype == torch.float32 and activation.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置block size
    B0 = 16
    B1 = 16
    B_MID = 64
    
    # 设置2D grid
    grid = lambda meta: (
        triton.cdiv(N0, meta['B0']),
        triton.cdiv(N1, meta['B1'])
    )
    
    # 调用kernel
    quant_dot_kernel[grid](
        scale_ptr=scale,
        offset_ptr=offset,
        weight_ptr=weight,
        activation_ptr=activation,
        z_ptr=output,
        N0=N0,
        N1=N1,
        MID=MID,
        B0=B0,
        B1=B1,
        B_MID=B_MID
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scale: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor, activation: torch.Tensor):
        return solve(scale, offset, weight, activation)

# test(quant_dot_kernel, quant_dot_spec, B={"B0": 16, "B1": 16, "B_MID": 64}, nelem={"N0": 32, "N1": 32, "MID": 64})