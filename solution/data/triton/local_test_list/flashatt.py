import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def flashatt_spec(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr):
    log2_e = 1.44269504
    
    # 根据不同的 B0 值使用不同的 arange
    if B0 <= 128:
        offsets = tl.arange(0, 128)
    elif B0 <= 256:
        offsets = tl.arange(0, 256)
    elif B0 <= 512:
        offsets = tl.arange(0, 512)
    else:
        offsets = tl.arange(0, 1024)
    
    mask = offsets < B0
    
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    
    # 计算注意力分数
    scores = q[:, None] * k[None, :]
    
    # 数值稳定的 softmax
    row_max = tl.max(scores, axis=1, keep_dims=True)
    stable_scores = scores - row_max
    exp_scores = tl.exp2(stable_scores * log2_e)
    
    # 应用掩码
    valid_mask = mask[:, None] & mask[None, :]
    exp_scores = tl.where(valid_mask, exp_scores, 0.0)
    
    # 归一化
    row_sum = tl.sum(exp_scores, axis=1, keep_dims=True)
    row_sum = tl.where(row_sum > 0, row_sum, 1.0)
    attention_weights = exp_scores / row_sum
    
    # 计算输出
    output = tl.sum(attention_weights * v[None, :], axis=1)
    
    # 存储结果
    tl.store(z_ptr + offsets, output, mask=mask)
    return

def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    output = torch.empty_like(q)
    assert q.is_cuda and k.is_cuda and v.is_cuda and output.is_cuda
    assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32 and output.dtype == torch.float32
    
    seq_len = q.shape[0]
    B0 = triton.next_power_of_2(seq_len)
    
    # 只需要一个程序块
    grid = lambda meta: (1,)
    
    # 调用kernel
    flashatt_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        z_ptr=output,
        N0=seq_len,
        T=seq_len,
        B0=B0
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return solve(q, k, v)

# test(flashatt_kernel, flashatt_spec, B={"B0":200}, nelem={"N0": 200, "T": 200})