# Asynchrony, streams and pinned memory: removing false dependencies, one layer at a time

These are my notes on Part 2 of NVIDIA's *Fundamentals of Accelerated Computing with Modern CUDA C++* (asynchrony, streams, pinned memory). The tutorial teaches three APIs. What made the chapter make sense to me was a single question underneath all three:

> **How do we keep several pieces of hardware — CPU, SMs, copy engines, disk — busy at the same time?**

Two operations can overlap unless one needs the other's result. That gives a vocabulary:

- **True dependency:** B reads data that A produces. Must be respected.
- **False dependency:** B waits for A only because of *how the code or system is arranged* — same queue, blocking API, memory type — not because of data.

Each section of the chapter removes one class of false dependency, and each removal exposes a true dependency that something else had been protecting by accident.

## The running example

A simulation loop: copy state D2H, run N GPU steps, write the host copy to disk.

```
CPU : |issue copy, wait| issue compute, wait ...... wait |  write to disk  |
GPU : |      D2H       |========== compute ==========|                   |
```

The GPU idles while the CPU writes; the CPU idles while the GPU computes. But the disk write uses the *previous* step's data and the GPU uses the *current* state. The data sets don't intersect. That waiting is false.

## Layer 1 — separate "issue" from "wait" (asynchronous APIs)

The CPU and GPU are independent processors. A kernel launch is the CPU writing a command into a queue the GPU front-end consumes; it takes microseconds and could return immediately. **Asynchrony is the hardware's natural state; synchronization is a wait added by software.** A synchronous API (e.g. Thrust algorithms) bundles "issue" with "wait until done".

With an async API (e.g. CUB), the program decides where to wait. **Rule: wait only where a GPU result is actually needed, and as late as possible.** The disk write doesn't need the new result, so the wait moves after it:

```
CPU : |D2H| issue x N |    write to disk    |wait|D2H| ...
GPU : |D2H|========= compute =========|          |D2H| ...
```

Per step, `copy + compute + write` becomes `copy + max(compute, write)`. Overlap can only hide the *shorter* side: if the write is slower than the compute, it stays on the critical path.

Two side effects to know:

- **Timing trap.** Host-side timers around an async call measure the launch, not the work. Use CUDA events, or synchronize before reading the clock.
- **Errors are reported late.** A failing kernel surfaces at some later API call. `CUDA_LAUNCH_BLOCKING=1` makes everything synchronous for debugging.

## (Interlude) — see the timeline before believing it

Everything above is a *predicted* timeline. Nsight Systems shows the real one: CPU threads, CUDA API calls, kernels, memcpys and OS calls on one time axis. NVTX ranges (`nvtx3::scoped_range r{"compute"};`, RAII-scoped) label phases so a timeline with thousands of kernels maps back to code. The next two layers are both stories of "I thought it overlapped; the timeline said no."

Division of labour: **nsys** answers *where is time going across the system* (gaps, serialization, launch-bound stretches); **ncu** answers *why is this one kernel slow* (bandwidth, occupancy, stall reasons). nsys first, ncu second.

## Layer 2 — separate unrelated work into different queues (streams)

Switching the D2H copy to `cudaMemcpyAsync` alone changes nothing on the GPU, because every operation without an explicit stream lands in the same default stream, a FIFO. **Same stream: strictly ordered. Different streams: no ordering guarantee.** The copy of step k and the compute of step k+1 have no data dependency; they were only ordered because they shared a queue.

Put them in different streams and a **true** dependency appears that the single queue had been protecting by accident: the copy *reads* a buffer the next compute *writes*. That's a data race now.

Fix used in the tutorial: snapshot the buffer with a device-to-device copy, then copy the snapshot to the host asynchronously. D2D runs at HBM/GDDR bandwidth (~300 GB/s on a T4); D2H runs over PCIe (~16 GB/s Gen3 x16). Spending a short synchronous D2D to make a long D2H asynchronous is a good trade — one extra buffer for a ~20× shorter critical-path copy.

Expressing cross-stream true dependencies without blocking the CPU: `cudaEventRecord(e, A); cudaStreamWaitEvent(B, e);` — B waits for A *on the GPU*.

A false dependency hiding in the default stream: the **legacy default stream** synchronizes with every blocking stream. A library issuing one small op on stream 0 silently serializes your pipeline. Fixes: `--default-stream per-thread`, or create streams with `cudaStreamNonBlocking`.

## Layer 3 — separate the CPU from the DMA (pinned memory)

Streams separated, code correct — and the timeline still shows copy and compute in series. The code has no dependency left, so it must come from below the code: memory.

1. Programs use virtual addresses. The OS can remap a page at any time: swap it out, move it, allocate it lazily. That memory is **pageable**.
2. The GPU's copy engine does DMA on **physical** addresses, without the CPU's MMU and without handling page faults. During a transfer the physical pages must exist and must not move.
3. Pageable memory doesn't guarantee that, so the driver stages: the CPU copies chunks into a driver-owned pinned buffer and the DMA engine moves them from there. The CPU is in the loop, so the "async" copy is effectively synchronous for the host — `cudaMemcpyAsync` to pageable memory doesn't return until the copy is done, and every launch after it is delayed.

The false dependency: using ordinary host memory chained the CPU to the DMA engine. **Pinned (page-locked) memory** (`cudaMallocHost`) removes it: the DMA engine reads it directly, and the CPU is free.

Costs: pinned pages can't be swapped, so over-allocating squeezes the rest of the system; allocation itself is slow, so allocate once and reuse.

The PyTorch version of this whole chapter is one line: `DataLoader(pin_memory=True)` plus `.to(device, non_blocking=True)`. Each is useless without the other: `non_blocking` from pageable memory is not actually asynchronous, and pinned memory with a blocking copy still makes the CPU wait.

## Hardware that makes overlap possible

Copies and kernels can run concurrently because **copy engines are DMA units separate from the SMs** (`cudaDeviceProp::asyncEngineCount`). By contrast, NCCL collectives run as *kernels* on the SMs, so communication/compute overlap in distributed training competes for SMs in a way that H2D/compute overlap does not ([note 06](06-cuda-is-distributed-training-one-level-down.md)).

Don't confuse the two "async copies":

| | `cudaMemcpyAsync` | `cp.async` (Ampere+) / TMA (Hopper+) |
|---|---|---|
| Path | host ↔ device over PCIe/NVLink | global → shared memory, on chip |
| Engine | copy engine (DMA) | issued by the SM's own threads / TMA unit |
| Hides | transfer time behind compute | global-memory latency inside a kernel (multi-stage GEMM/attention pipelines) |
| T4 (sm_75) | yes | no |

## Summary

| Layer | False dependency removed | True dependency it exposed | Tool |
|---|---|---|---|
| async API | "issue" bundled with "wait" | sync must precede the first use of the GPU result | CUB / async launches, late sync |
| streams | unrelated ops sharing one FIFO | a copy reads what the next compute writes | multiple streams, D2D snapshot, events |
| pinned memory | CPU chained to the DMA engine | — | `cudaMallocHost` |

**Synchronize only where there is a true dependency, with the narrowest scope that covers it** (event < stream < device).

A runnable version of the whole chapter — serial vs. multi-stream pinned vs. multi-stream pageable, with NVTX ranges — is [11_async_pipeline.cu](../kernels/cuda/11_async_pipeline.cu).
