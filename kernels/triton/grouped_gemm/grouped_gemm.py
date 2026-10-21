"""MoE expert FFN with one grouped-GEMM launch per linear layer (varlen-M), in Triton.

Why a for-loop over experts is slow: E small GEMMs = E launches, many with tiny M
(a cold expert may get 3 tokens), each leaving most SMs idle. On the CPU side the loop
needs every expert's token count, which forces a device->host sync (.item()/.tolist()).

What grouped GEMM does instead:

  * Weights: every expert's weight has the SAME shape. A tile finds its expert's
    weight by **indexing**: w_ptr + e * stride_e.
  * Tokens: experts get DIFFERENT numbers of tokens. After sorting tokens by expert
    (the permute step), expert e owns rows [m_off[e], m_off[e+1]) -> found by a
    **prefix sum** of the counts, computed on the GPU.
  * Tiles: all experts' M-tiles form one pool. Program `pid` finds its expert by
    comparing pid against the prefix sum of tiles per expert. The launch grid uses a
    static upper bound (ceil(total_rows / BM) + E), so the host never reads the counts:
    no sync, and the whole thing is CUDA-graph friendly. Extra programs just exit.

Flow:  route -> permute (sort by expert) -> grouped GEMM (gate,up) -> SwiGLU
       -> grouped GEMM (down) -> unpermute -> weighted combine

    TRITON_INTERPRET=1 python grouped_gemm.py   # runs on CPU through the Triton interpreter
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    x_ptr, w_ptr, y_ptr, m_off_ptr, tile_off_ptr,
    E, N, K,
    stride_xm, stride_xk, stride_we, stride_wk, stride_wn, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, E_POW2: tl.constexpr,
):
    pid = tl.program_id(0)    # index into the pool of M-tiles of all experts
    pid_n = tl.program_id(1)

    # Which expert owns this tile? Count how many tile-prefix-sums are <= pid.
    e_idx = tl.arange(0, E_POW2)
    ends = tl.load(tile_off_ptr + 1 + e_idx, mask=e_idx < E, other=2**30)
    e = tl.sum((ends <= pid).to(tl.int32), axis=0)
    total_tiles = tl.load(tile_off_ptr + E)
    live = pid < total_tiles
    e = tl.minimum(e, E - 1)

    tile_start = tl.load(tile_off_ptr + e)
    m_start = tl.load(m_off_ptr + e)
    m_end = tl.load(m_off_ptr + e + 1)

    rows = m_start + (pid - tile_start) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = (rows < m_end) & live
    col_ok = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * stride_we          # same shape for every expert: just index
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_ok = ks < K
        a = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk,
                    mask=row_ok[:, None] & k_ok[None, :], other=0.0)
        b = tl.load(w_base + ks[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_ok[:, None] & col_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + rows[:, None] * stride_ym + cols[None, :] * stride_yn,
             acc.to(y_ptr.dtype.element_ty), mask=row_ok[:, None] & col_ok[None, :])


def grouped_gemm(x_sorted: torch.Tensor, w: torch.Tensor, m_off: torch.Tensor,
                 BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    """y[m_off[e]:m_off[e+1]] = x_sorted[m_off[e]:m_off[e+1]] @ w[e]   for every expert e.

    x_sorted: [M_total, K] rows grouped by expert;  w: [E, K, N];  m_off: [E+1] int32 on device.
    """
    M, K = x_sorted.shape
    E, K2, N = w.shape
    assert K == K2
    counts = m_off[1:] - m_off[:-1]
    tiles = (counts + BLOCK_M - 1) // BLOCK_M
    tile_off = torch.zeros(E + 1, dtype=torch.int32, device=x_sorted.device)
    tile_off[1:] = torch.cumsum(tiles, 0)
    y = torch.empty(M, N, dtype=x_sorted.dtype, device=x_sorted.device)
    grid = (triton.cdiv(M, BLOCK_M) + E, triton.cdiv(N, BLOCK_N))  # static bound: no host sync
    _grouped_gemm_kernel[grid](
        x_sorted, w, y, m_off, tile_off, E, N, K,
        x_sorted.stride(0), x_sorted.stride(1), w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, E_POW2=triton.next_power_of_2(E),
    )
    return y


# ----------------------------------------------------------------------------
# The full MoE FFN around it
# ----------------------------------------------------------------------------

def moe_ffn_grouped(x, router_w, w_gate_up, w_down, top_k):
    """x: [T, H]; router_w: [H, E]; w_gate_up: [E, H, 2I]; w_down: [E, I, H]."""
    T, H = x.shape
    E = router_w.shape[1]
    # 1. route
    probs = torch.softmax(x.float() @ router_w.float(), dim=-1)
    topw, tope = probs.topk(top_k, dim=-1)                     # [T, k]
    topw = topw / topw.sum(-1, keepdim=True)
    # 2. permute: group the T*k (token, expert) assignments by expert
    flat_e = tope.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    token_of = order // top_k                                   # which token each sorted row came from
    x_sorted = x[token_of]
    counts = torch.bincount(flat_e, minlength=E)
    m_off = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    m_off[1:] = torch.cumsum(counts, 0)                         # prefix sum, stays on the GPU
    # 3. grouped GEMM gate+up, SwiGLU, grouped GEMM down
    gu = grouped_gemm(x_sorted, w_gate_up, m_off)
    g, u = gu.chunk(2, dim=-1)
    h = torch.nn.functional.silu(g) * u
    y_sorted = grouped_gemm(h.contiguous(), w_down, m_off)
    # 4. unpermute + combine with routing weights
    wts = topw.reshape(-1)[order].to(y_sorted.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, token_of, y_sorted * wts[:, None])
    return out


def moe_ffn_loop(x, router_w, w_gate_up, w_down, top_k):
    """Reference: one small GEMM per expert (what grouped GEMM replaces)."""
    probs = torch.softmax(x.float() @ router_w.float(), dim=-1)
    topw, tope = probs.topk(top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for e in range(router_w.shape[1]):
        tok, slot = (tope == e).nonzero(as_tuple=True)          # host sync happens here
        if tok.numel() == 0:
            continue
        g, u = (x[tok] @ w_gate_up[e]).chunk(2, dim=-1)
        y = (torch.nn.functional.silu(g) * u) @ w_down[e]
        out.index_add_(0, tok, y * topw[tok, slot, None].to(y.dtype))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    T, H, I, E, k = 96, 64, 32, 8, 2
    x = torch.randn(T, H, device=dev, dtype=dt)
    rw = torch.randn(H, E, device=dev, dtype=dt)
    wgu = torch.randn(E, H, 2 * I, device=dev, dtype=dt) / H ** 0.5
    wd = torch.randn(E, I, H, device=dev, dtype=dt) / I ** 0.5

    # raw grouped GEMM vs per-expert matmul, with an empty expert and a 1-row expert
    counts = torch.tensor([5, 0, 1, 64, 17, 3, 0, 6], device=dev)
    m_off = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    m_off[1:] = torch.cumsum(counts, 0)
    xs = torch.randn(int(counts.sum()), H, device=dev, dtype=dt)
    y = grouped_gemm(xs, wgu, m_off)
    ref = torch.cat([xs[m_off[e]:m_off[e + 1]] @ wgu[e] for e in range(E)])
    err = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"grouped_gemm vs per-expert matmul: normalized L2 = {err:.2e}")
    assert err < 1e-2

    a = moe_ffn_grouped(x, rw, wgu, wd, k)
    b = moe_ffn_loop(x, rw, wgu, wd, k)
    err = ((a.float() - b.float()).norm() / b.float().norm()).item()
    print(f"MoE FFN grouped vs loop:            normalized L2 = {err:.2e}")
    assert err < 1e-2
    print("OK")
