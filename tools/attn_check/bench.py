"""Timing that does not lie (the "real" half of "correct and real").

Pitfalls this handles:
  * Warm-up: first calls include JIT compilation (Triton, CuTe DSL), autotuning,
    allocator growth and clock ramp-up. Default 50 warm-up iterations.
  * L2 cache: re-running on the same buffers lets small shapes hit L2 and look
    faster than they will be in training. Rotate through enough input copies to
    exceed L2 (H100: 50 MB, B200: ~126 MB).
  * Noise: 200 timed iterations x 3 repeated runs; report the median and the
    coefficient of variation across runs. If CV > 2%, do not trust the number.
  * Clocks: lock them outside Python, e.g. `sudo nvidia-smi -lgc <freq>`, and
    unlock with `nvidia-smi -rgc`. Boost clocks drift with temperature.
  * Microbenchmark != end-to-end: after picking a config here, re-measure inside a
    fixed multi-GPU training job with the same seed, data order and token budget.

    python bench.py   # needs a CUDA GPU
"""
from __future__ import annotations

import statistics
from typing import Callable, List, Sequence

import torch


def l2_bytes() -> int:
    try:
        return torch.cuda.get_device_properties(0).L2_cache_size
    except Exception:
        return 128 << 20


def bench(make_inputs: Callable[[], Sequence[torch.Tensor]], fn: Callable, warmup: int = 50,
          iters: int = 200, runs: int = 3, rotate: bool = True) -> dict:
    """make_inputs() returns one fresh tuple of inputs; fn(*inputs) runs the kernel."""
    sample = make_inputs()
    per_set = sum(t.numel() * t.element_size() for t in sample)
    n_sets = max(1, -(-2 * l2_bytes() // per_set)) if rotate else 1  # 2x L2 so nothing survives
    n_sets = min(n_sets, 64)
    sets: List[Sequence[torch.Tensor]] = [sample] + [make_inputs() for _ in range(n_sets - 1)]

    for i in range(warmup):
        fn(*sets[i % n_sets])
    torch.cuda.synchronize()

    run_ms = []
    for _ in range(runs):
        start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            start[i].record()
            fn(*sets[i % n_sets])
            end[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start, end)]
        run_ms.append(statistics.median(times))
    med = statistics.median(run_ms)
    cv = statistics.pstdev(run_ms) / statistics.mean(run_ms) if len(run_ms) > 1 else 0.0
    return {"median_ms": med, "runs_ms": run_ms, "cv": cv, "input_sets": n_sets,
            "trustworthy": cv < 0.02}


def geomean_speedup(base_ms: Sequence[float], new_ms: Sequence[float]) -> float:
    """Report a geometric mean over shapes: one huge win on one shape can't dominate."""
    import math
    r = [b / n for b, n in zip(base_ms, new_ms)]
    return math.exp(sum(math.log(x) for x in r) / len(r))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    from attn_check import IMPLS
    lens = [4096, 2048, 1024, 1024]
    cu = torch.tensor([0, 4096, 6144, 7168, 8192], dtype=torch.int32, device="cuda")

    def mk():
        q = torch.randn(8192, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(8192, 8, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(8192, 8, 128, device="cuda", dtype=torch.bfloat16)
        return q, k, v

    for name, f in IMPLS.items():
        try:
            r = bench(mk, lambda q, k, v: f(q, k, v, cu, cu, 4096, 4096, causal=True))
            print(f"{name:>6}: {r['median_ms']:.3f} ms  (CV {r['cv']:.1%}, {r['input_sets']} input sets)")
        except Exception as e:  # e.g. flash-attn not installed
            print(f"{name:>6}: skipped ({e.__class__.__name__}: {e})")
