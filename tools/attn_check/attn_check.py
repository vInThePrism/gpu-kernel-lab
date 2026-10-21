"""Correctness matrix for attention kernels (forward + dq/dk/dv), varlen-first.

A speedup only counts if it is correct and real. This file is the "correct" half.

Two ideas:
  1. Sample shapes by *bug probability*, not only by the production distribution.
     Bugs live at edges: tail blocks (length not a multiple of the tile), empty
     sequences inside a varlen batch, causal masks with q_len != k_len, GQA head mapping.
  2. Check three things together. Normalized L2 alone hides a single bad element:
       * normalized L2 error  ||out - ref|| / ||ref||   <= tol (default 1e-2)
       * max absolute error, reported; bounded by zero_atol for zero references
       * any NaN/Inf          -> hard fail
     dQ is usually the hardest to match: it is accumulated across CTAs with atomics,
     so the summation order (and the rounding) changes from run to run.

Any candidate with the flash_attn_varlen_func signature can be tested:

    fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=..., softmax_scale=...)
    q: [total_q, Hq, D], k/v: [total_k, Hkv, D]  ->  out: [total_q, Hq, D]

    python attn_check.py                 # SDPA-per-segment candidate on CPU, quick matrix
    python attn_check.py --impl flash    # flash_attn.flash_attn_varlen_func on GPU
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    q_lens: Sequence[int]
    k_lens: Sequence[int]
    hq: int
    hkv: int
    d: int
    causal: bool


# ----------------------------------------------------------------------------
# Reference: fp32 math, explicit mask, per segment. Slow on purpose: obviously right.
# ----------------------------------------------------------------------------

def reference(q, k, v, cu_q, cu_k, causal: bool, scale: float):
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    hq, hkv = q.shape[1], k.shape[1]
    rep = hq // hkv
    for i in range(len(cu_q) - 1):
        qs, qe, ks, ke = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        if qe == qs:
            continue
        qi = q[qs:qe].float().transpose(0, 1)                                  # [H, Lq, D]
        ki = k[ks:ke].float().transpose(0, 1).repeat_interleave(rep, 0)       # [H, Lk, D]
        vi = v[ks:ke].float().transpose(0, 1).repeat_interleave(rep, 0)
        if ke == ks:
            continue  # no keys: output defined as 0
        s = qi @ ki.transpose(-1, -2) * scale
        if causal:  # bottom-right aligned (flash-attn convention): q row r sees keys <= r + (Lk - Lq)
            Lq, Lk = qe - qs, ke - ks
            r = torch.arange(Lq, device=q.device)[:, None]
            c = torch.arange(Lk, device=q.device)[None, :]
            s = s.masked_fill(c > r + (Lk - Lq), float("-inf"))
        p = torch.softmax(s, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # rows with no visible key
        out[qs:qe] = (p @ vi).transpose(0, 1)
    return out


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------

def sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, softmax_scale=None):
    """One SDPA call per segment. A correct but launch-heavy baseline
    (exactly the pattern that makes short-window attention launch-bound)."""
    outs = []
    rep = q.shape[1] // k.shape[1]
    for i in range(len(cu_q) - 1):
        qs, qe, ks, ke = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        Lq, Lk = qe - qs, ke - ks
        if Lq == 0:
            continue
        if Lk == 0:
            outs.append(torch.zeros_like(q[qs:qe]))
            continue
        qi = q[qs:qe].transpose(0, 1)[None]
        ki = k[ks:ke].transpose(0, 1).repeat_interleave(rep, 0)[None]
        vi = v[ks:ke].transpose(0, 1).repeat_interleave(rep, 0)[None]
        mask = None
        if causal:
            r = torch.arange(Lq, device=q.device)[:, None]
            c = torch.arange(Lk, device=q.device)[None, :]
            mask = c <= r + (Lk - Lq)
        o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=softmax_scale)
        o = torch.nan_to_num(o, nan=0.0)
        outs.append(o[0].transpose(0, 1))
    return torch.cat(outs) if outs else torch.zeros_like(q)


def flash_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, softmax_scale=None):
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=causal, softmax_scale=softmax_scale)


IMPLS = {"sdpa": sdpa_varlen, "flash": flash_varlen}


# ----------------------------------------------------------------------------
# Shape matrix
# ----------------------------------------------------------------------------

def edge_cases(d_list=(64, 128)) -> List[Case]:
    """Sampled by bug probability."""
    cases = []
    for d in d_list:
        cases += [
            Case(f"tail-block d{d}", [129, 255, 1], [129, 255, 1], 8, 8, d, True),
            Case(f"empty-seq d{d}", [64, 0, 33], [64, 0, 33], 8, 8, d, True),
            Case(f"q<k causal d{d}", [17, 1], [100, 64], 8, 2, d, True),     # decode-like, bottom-right
            Case(f"q>k noncausal d{d}", [100], [7], 8, 8, d, False),
            Case(f"gqa 8:1 d{d}", [300, 77], [300, 77], 16, 2, d, True),
            Case(f"single token d{d}", [1, 1, 1], [1, 1, 1], 4, 4, d, False),
        ]
    return cases


def production_matrix(seq_lens=(8192, 16384, 32768, 65536), d_list=(64, 128)) -> List[Case]:
    """The 4-axis grid: attention type x mask x seq len x head dim = 2 x 3 x 4 x 2 = 48 shapes.
    (ViT full MHA vs LLM GQA; varlen / causal / varlen+causal.)"""
    out = []
    for (kind, hq, hkv), mask, L, d in itertools.product(
            [("vit-mha", 16, 16), ("llm-gqa", 32, 8)], ["varlen", "causal", "varlen+causal"], seq_lens, d_list):
        if "varlen" in mask:  # several documents packed into L tokens, uneven lengths
            lens = [L // 2, L // 4, L // 8, L - L // 2 - L // 4 - L // 8]
        else:
            lens = [L]
        out.append(Case(f"{kind} {mask} L{L} d{d}", lens, lens, hq, hkv, d, "causal" in mask))
    return out


# ----------------------------------------------------------------------------
# Checking
# ----------------------------------------------------------------------------

def _cu(lens, device):
    return torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=device)


def _metrics(x: torch.Tensor, ref: torch.Tensor) -> dict:
    x, ref = x.detach().float(), ref.detach().float()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(ref).all())
    ref_norm = float(ref.norm())
    return {"nl2": float((x - ref).norm()) / max(ref_norm, 1e-12),
            "max_abs": float((x - ref).abs().max()) if x.numel() else 0.0,
            "reference_zero": ref_norm <= 1e-12, "finite": finite}


def check(fn: Callable, case: Case, dtype=torch.bfloat16, device="cpu", tol=1e-2, seed=0,
          zero_atol=1e-5) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq, tk = sum(case.q_lens), sum(case.k_lens)
    mk = lambda n, h: torch.randn(n, h, case.d, generator=g).to(device=device, dtype=dtype).requires_grad_(True)
    q, k, v = mk(tq, case.hq), mk(tk, case.hkv), mk(tk, case.hkv)
    cu_q, cu_k = _cu(case.q_lens, device), _cu(case.k_lens, device)
    scale = 1.0 / math.sqrt(case.d)
    dout = torch.randn(tq, case.hq, case.d, generator=g).to(device=device, dtype=dtype)

    out = fn(q, k, v, cu_q, cu_k, max(case.q_lens), max(case.k_lens), causal=case.causal, softmax_scale=scale)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout, allow_unused=True)

    q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    ref = reference(q32, k32, v32, cu_q, cu_k, case.causal, scale)
    rq, rk, rv = torch.autograd.grad(ref, (q32, k32, v32), dout.float(), allow_unused=True)

    res = {"case": case.name}
    ok = True
    for name, a, b in (("out", out, ref), ("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        a = torch.zeros_like(b) if a is None else a
        m = _metrics(a, b)
        res[name] = m
        # A one-token attention row has exactly zero q/k reference gradients.
        # Relative error is undefined there; bound the absolute residual instead.
        error_ok = m["max_abs"] <= zero_atol if m["reference_zero"] else m["nl2"] <= tol
        ok &= m["finite"] and error_ok
    res["pass"] = ok
    return res


def run(fn, cases, **kw):
    print(f"{'case':<28}{'out':>10}{'dq':>10}{'dk':>10}{'dv':>10}  result")
    fails = 0
    for c in cases:
        r = check(fn, c, **kw)
        cell = lambda m: ("NaN/Inf" if not m["finite"] else
                          f"a{m['max_abs']:.1e}" if m["reference_zero"] else f"{m['nl2']:.1e}")
        print(f"{c.name:<28}{cell(r['out']):>10}{cell(r['dq']):>10}{cell(r['dk']):>10}{cell(r['dv']):>10}  "
              f"{'PASS' if r['pass'] else 'FAIL  <- fall back for this config'}")
        fails += not r["pass"]
    print(f"\n{len(cases) - fails}/{len(cases)} passed")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="sdpa", choices=sorted(IMPLS))
    ap.add_argument("--full", action="store_true", help="also run the 48-shape production grid (GPU)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = edge_cases()
    if a.full:
        cases += production_matrix()
    raise SystemExit(run(IMPLS[a.impl], cases, device=dev))
