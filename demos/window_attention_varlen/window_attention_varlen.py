"""Launch-bound window attention, and the varlen fix.

Vision towers such as Qwen2.5-VL's use *window attention*: each image's patches are
cut into windows of a few dozen tokens, and attention runs inside each window only.
A naive adapter calls the attention kernel once per window:

    for (s, e) in windows: out[s:e] = attn(q[s:e], k[s:e], v[s:e])

With 64-token windows, each call does a few microseconds of GPU work, and the CPU
cannot issue launches that fast. Signs in an Nsight Systems timeline:
  1. total attention GPU time barely moves when you switch to a faster kernel,
  2. each kernel lasts only a few microseconds,
  3. the GPU row is full of gaps while the CPU row is a solid wall of cudaLaunchKernel.
The GPU is not slow. It is waiting for the CPU: launch-bound.

The fix is not a faster kernel. It is the right *call granularity*: concatenate all
windows into one flat sequence and describe the boundaries with cu_seqlens. One varlen
launch computes every window, and never computes (not just masks) cross-window pairs.

This script reproduces the effect with PyTorch only. On a GPU, swap `batched_*` for
flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len).

    python window_attention_varlen.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def make_windows(n_images: int, patches_per_image: int, window: int):
    lens = []
    for _ in range(n_images):
        full, rem = divmod(patches_per_image, window)
        lens += [window] * full + ([rem] if rem else [])
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    return lens, cu


def per_window(q, k, v, cu):
    """One launch per window (the slow adapter)."""
    out = torch.empty_like(q)
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        out[s:e] = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k[s:e].transpose(0, 1), v[s:e].transpose(0, 1)).transpose(0, 1)
    return out


def batched_equal_windows(q, k, v, lens, window):
    """Same work, few launches: full windows are reshaped into a batch; the tail windows
    (one per image at most) are handled separately. A real varlen kernel needs no split."""
    H, D = q.shape[1], q.shape[2]
    out = torch.empty_like(q)
    starts = torch.tensor([0] + lens[:-1]).cumsum(0)
    full = [i for i, L in enumerate(lens) if L == window]
    tail = [i for i, L in enumerate(lens) if L != window]
    if full:
        idx = (starts[full][:, None] + torch.arange(window)[None, :]).reshape(-1).to(q.device)
        qb, kb, vb = (t[idx].view(len(full), window, H, D).transpose(1, 2) for t in (q, k, v))
        out[idx] = F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2).reshape(-1, H, D)
    for i in tail:
        s, e = int(starts[i]), int(starts[i]) + lens[i]
        out[s:e] = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k[s:e].transpose(0, 1), v[s:e].transpose(0, 1)).transpose(0, 1)
    return out


def _time(fn, *a, reps=5):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    fn(*a); sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn(*a)
    sync()
    return (time.perf_counter() - t) / reps


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    torch.manual_seed(0)
    window, H, D = 64, 16, 80
    lens, cu = make_windows(n_images=4, patches_per_image=4000, window=window)
    n = int(cu[-1])
    q, k, v = (torch.randn(n, H, D, device=dev, dtype=dt) for _ in range(3))

    a = per_window(q, k, v, cu)
    b = batched_equal_windows(q, k, v, lens, window)
    err = ((a.float() - b.float()).norm() / a.float().norm()).item()
    print(f"{len(lens)} windows, {n} tokens; outputs match: normalized L2 = {err:.1e}")

    t_loop = _time(per_window, q, k, v, cu)
    t_batch = _time(batched_equal_windows, q, k, v, lens, window)
    print(f"per-window launches : {len(lens):5d} calls  {t_loop*1e3:8.2f} ms")
    print(f"batched             : {1 + sum(L != window for L in lens):5d} calls  {t_batch*1e3:8.2f} ms"
          f"   ({t_loop / t_batch:.1f}x)")
