import torch
import triton
import triton.language as tl

torch.manual_seed(46)

@triton.jit
def block_scaled_matmul_kernel(
        a_ptr, b_ptr, c_ptr,  #
        a_scale_ptr, b_scale_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        stride_a_scale_m, stride_a_scale_k,  #
        stride_b_scale_n, stride_b_scale_k,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        SCALE_BLOCK_M: tl.constexpr,  #
        SCALE_BLOCK_N: tl.constexpr,  #
        SCALE_BLOCK_K: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Scale offsets
    scale_offs_m = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    scale_offs_n = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    scale_offs_k = tl.arange(0, SCALE_BLOCK_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load data blocks
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Load scale blocks
        a_scale_ptrs = a_scale_ptr + (scale_offs_m[:, None] * stride_a_scale_m + scale_offs_k[None, :] * stride_a_scale_k)
        b_scale_ptrs = b_scale_ptr + (scale_offs_n[:, None] * stride_b_scale_n + scale_offs_k[None, :] * stride_b_scale_k)
        
        a_scale = tl.load(a_scale_ptrs, mask=scale_offs_k[None, :] < K - k * SCALE_BLOCK_K, other=1.0)
        b_scale = tl.load(b_scale_ptrs, mask=scale_offs_k[:, None] < K - k * SCALE_BLOCK_K, other=1.0)
        
        # Apply scaling
        a_scaled = a.to(tl.float32) * a_scale
        b_scaled = b.to(tl.float32) * b_scale
        
        # Accumulate
        accumulator += tl.dot(a_scaled, b_scaled)
        
        # Update offsets
        offs_k += BLOCK_SIZE_K
        scale_offs_k += SCALE_BLOCK_K
    
    c = accumulator.to(tl.float16)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def solve(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor):
    assert a.is_cuda and b.is_cuda and a_scale.is_cuda and b_scale.is_cuda
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    
    M, K = a.shape
    K, N = b.shape
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Simple configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    SCALE_BLOCK_M = 32
    SCALE_BLOCK_N = 32
    SCALE_BLOCK_K = 8
    
    # Grid configuration
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    block_scaled_matmul_kernel[grid](
        a, b, c,  #
        a_scale, b_scale,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        a_scale.stride(0), a_scale.stride(1),  #
        b_scale.stride(0), b_scale.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SCALE_BLOCK_M=SCALE_BLOCK_M,
        SCALE_BLOCK_N=SCALE_BLOCK_N,
        SCALE_BLOCK_K=SCALE_BLOCK_K,
    )
    
    return c

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor):
        return solve(a, b, a_scale, b_scale) 