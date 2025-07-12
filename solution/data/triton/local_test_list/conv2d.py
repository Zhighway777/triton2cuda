import torch
import triton
import triton.language as tl

torch.manual_seed(46)

def conv2d_spec(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z

@triton.jit
def conv2d_kernel(x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr):
    pid_0 = tl.program_id(0)
    
    batch_start = pid_0 * B0
    batch_offsets = batch_start + tl.arange(0, B0)
    batch_mask = batch_offsets < N0
    
    for out_h in range(H):
        for out_w in range(W):
            conv_sum = tl.zeros((B0,), dtype=tl.float32)
            
            for kh in range(KH):
                for kw in range(KW):
                    in_h = out_h + kh
                    in_w = out_w + kw
                    
                    if in_h < H and in_w < W:
                        in_idx = batch_offsets * H * W + in_h * W + in_w
                        x_vals = tl.load(x_ptr + in_idx, mask=batch_mask, other=0.0)
                        
                        k_idx = kh * KW + kw
                        k_val = tl.load(k_ptr + k_idx)
                        
                        conv_sum += x_vals * k_val
            
            out_idx = batch_offsets * H * W + out_h * W + out_w
            tl.store(z_ptr + out_idx, conv_sum, mask=batch_mask)
    return

def solve(x: torch.Tensor, k: torch.Tensor):
    # 输入: x [N0, H, W], k [KH, KW]
    # 输出: z [N0, H, W]
    N0, H, W = x.shape
    KH, KW = k.shape
    output = torch.zeros_like(x)
    assert x.is_cuda and k.is_cuda and output.is_cuda
    assert x.dtype == torch.float32 and k.dtype == torch.float32 and output.dtype == torch.float32
    
    # 设置grid和block size
    B0 = 1
    grid = lambda meta: (triton.cdiv(N0, meta['B0']),)
    
    # 调用kernel
    conv2d_kernel[grid](
        x_ptr=x,
        k_ptr=k,
        z_ptr=output,
        N0=N0,
        H=H,
        W=W,
        KH=KH,
        KW=KW,
        B0=B0
    )
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, k: torch.Tensor):
        return solve(x, k)

# test(conv2d_kernel, conv2d_spec, B={"B0": 1}, nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4})