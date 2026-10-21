"""Shared-memory budget for pipelined Triton GEMM tiles, before you hit OutOfResources.

A pipelined matmul keeps several (A, B) tile pairs in shared memory at once:

    smem = buffers * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * bytes_per_el  (+ extra operands)

``buffers`` is num_stages or num_stages - 1 depending on the Triton version and pipeliner.
Do not guess: paste the "Required: N" number from an OutOfResources error into --calibrate
and the script tells you which convention your build uses.

The lesson this tool encodes: tiles tuned on one GPU can silently fail on a sibling with
less shared memory (H100 tiles on H800 with a different opt-in limit, B200 tiles on a
consumer card). A global fallback such as "halve BLOCK_N everywhere" fixes the crash but
slows every shape. Search BLOCK_N and num_stages *together*, per shape, under the real limit.

    python smem_budget.py --device H100 --M 8192 --N 3584 --K 18944
    python smem_budget.py --calibrate 245760 --BM 64 --BN 256 --BK 128 --stages 4
"""
from __future__ import annotations

import argparse
import itertools

# Max dynamic shared memory per block with opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize).
SMEM_LIMIT = {
    "V100": 98_304,      # 96 KB
    "A100": 166_912,     # 163 KB
    "A10": 101_376,      # 99 KB (sm_86)
    "RTX4090": 101_376,  # 99 KB (sm_89)
    "L40S": 101_376,
    "H100": 232_448,     # 227 KB (sm_90)
    "H800": 232_448,
    "H200": 232_448,
    "B200": 232_448,     # 227 KB (sm_100)
}


def smem_bytes(bm: int, bn: int, bk: int, stages: int, el: int = 2, buffers_offset: int = -1,
               extra: int = 0) -> int:
    buffers = max(1, stages + buffers_offset)
    return buffers * (bm * bk + bk * bn) * el + extra


def calibrate(required: int, bm: int, bn: int, bk: int, stages: int, el: int = 2) -> None:
    for off, name in ((0, "num_stages"), (-1, "num_stages - 1")):
        b = smem_bytes(bm, bn, bk, stages, el, off)
        tag = "MATCH" if b == required else f"diff {required - b:+,d} B (extra operands?)"
        print(f"  buffers = {name:<15} -> {b:>9,d} B   {tag}")


def search(device: str, M: int, N: int, K: int, el: int = 2, buffers_offset: int = -1, extra: int = 0):
    """For every (BM, BN, BK): the deepest pipeline that fits. Registers are checked too:
    the fp32 accumulator is BM*BN/(32*num_warps) values per thread; above ~128 it spills."""
    limit = SMEM_LIMIT[device]
    rows = []
    for bm, bn, bk in itertools.product((64, 128, 256), (64, 128, 256), (32, 64, 128)):
        if bm > max(64, M) or bn > max(64, N) or bk > max(32, K):
            continue
        if bm * bn / (32 * 8) > 128:  # even with 8 warps the accumulator would not fit
            continue
        fits = [st for st in (2, 3, 4, 5, 6) if smem_bytes(bm, bn, bk, st, el, buffers_offset, extra) <= limit]
        if not fits:
            continue
        st = max(fits)
        need = smem_bytes(bm, bn, bk, st, el, buffers_offset, extra)
        reuse = bm * bn / (bm + bn)          # FLOPs per byte loaded, up to a constant
        tiles = -(-M // bm) * -(-N // bn)
        rows.append((reuse, st, bm, bn, bk, need, tiles))
    rows.sort(reverse=True)  # crude prior; the autotuner makes the real decision
    return limit, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="H100", choices=sorted(SMEM_LIMIT))
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=3584)
    ap.add_argument("--K", type=int, default=18944)
    ap.add_argument("--bytes", type=int, default=2, help="2 for bf16/fp16, 1 for fp8")
    ap.add_argument("--buffers-offset", type=int, default=-1, help="-1: num_stages-1 buffers, 0: num_stages")
    ap.add_argument("--extra", type=int, default=0, help="extra smem bytes (epilogue, LoRA tiles, ...)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--calibrate", type=int, help="'Required' bytes from an OutOfResources message")
    ap.add_argument("--BM", type=int)
    ap.add_argument("--BN", type=int)
    ap.add_argument("--BK", type=int)
    ap.add_argument("--stages", type=int)
    a = ap.parse_args()

    if a.calibrate:
        print(f"calibrating against Required = {a.calibrate:,d} B")
        calibrate(a.calibrate, a.BM, a.BN, a.BK, a.stages, a.bytes)
        for dev, lim in SMEM_LIMIT.items():
            if a.calibrate > lim:
                print(f"  does not fit on {dev} ({lim:,d} B)")
        raise SystemExit

    limit, rows = search(a.device, a.M, a.N, a.K, a.bytes, a.buffers_offset, a.extra)
    print(f"{a.device}: limit {limit:,d} B per block; shape M={a.M} N={a.N} K={a.K}")
    print(f"{'BM':>4}{'BN':>5}{'BK':>5}{'stages':>8}{'smem (B)':>11}{'tiles':>8}")
    for reuse, st, bm, bn, bk, need, tiles in rows[: a.top]:
        print(f"{bm:>4}{bn:>5}{bk:>5}{st:>8}{need:>11,d}{tiles:>8}")
    print("\nPaste into @triton.autotune(configs=[...]) and let it time them on the real shape:")
    for reuse, st, bm, bn, bk, need, tiles in rows[: min(a.top, 6)]:
        print(f"  triton.Config({{'BLOCK_M': {bm}, 'BLOCK_N': {bn}, 'BLOCK_K': {bk}}}, num_stages={st}, num_warps={8 if bm*bn >= 128*128 else 4}),")
