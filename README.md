# GPU Kernel Lab

**GPU kernels explained through indexing, synchronization, and numerical correctness.**

[Start learning](#start-learning) · [Run a kernel](#run-a-kernel) · [Explore the lab](#explore-the-lab) · [Status](#status)

This lab pairs small CUDA and Triton implementations with the reasoning behind them: who owns each output, which threads must cooperate, and what happens at the boundaries. The bug log is part of the learning path—failed assumptions often explain a kernel better than the final code alone.

## Start learning

| Your question | Start here |
|---|---|
| How do threads map to data? | [Three coordinates](notes/01-three-coordinates.md) |
| Where does a barrier belong? | [Every barrier names a hazard](notes/02-every-barrier-names-a-hazard.md) |
| What should an out-of-range thread contribute? | [Reductions are monoids](notes/03-reductions-are-monoids.md) |
| Why does my kernel fail only on awkward sizes? | [The bug log](notes/04-bug-log.md) |
| What is the GEMM K loop doing? | [Tiled GEMM](notes/08-tiled-gemm-the-k-loop.md) |
| Why does XOR help both reductions and layouts? | [Why XOR](notes/10-why-xor.md) |

A useful first path: **coordinates → barriers → reductions → transpose → tiled GEMM**. Read a note, inspect its kernel, then change a boundary size and predict what should happen.

## Run a kernel

Use a CUDA development environment with `nvcc`, a C++17 host compiler, and an NVIDIA GPU. Choose the architecture for your device; the example below targets a T4.

```bash
git clone https://github.com/vInThePrism/gpu-kernel-lab.git
cd gpu-kernel-lab/kernels/cuda
make 02_reduce ARCH=sm_75
./02_reduce
```

The program compares GPU results with host references and prints `PASS` or exits on a failure. The CUDA examples share helpers in `common/`; keep that directory alongside them.

`make test` builds and runs the full collection. The FP8 conversion example requires `cuda_fp8.h`; the pipeline example requires NVTX headers. Check those toolchain requirements before building every target.

## Explore the lab

| Area | Contents |
|---|---|
| [CUDA kernels](kernels/cuda/) | Bias + GeLU, reduction, histogram, transpose, scan, online softmax, RMSNorm forward/backward, GEMV, tiled SGEMM, row-wise quantization, and an asynchronous transfer pipeline |
| [Cooperative primitives](kernels/cuda/common/primitives.cuh) | Warp/block reductions, online-softmax merging, and inclusive scan |
| [Triton grouped GEMM](kernels/triton/grouped_gemm/) | A variable-row-count teaching implementation with a forward MoE FFN example |
| [Window-attention demo](demos/window_attention_varlen/) | PyTorch SDPA batching that illustrates the cost of many small calls |
| [Attention checks](tools/attn_check/) | Forward/gradient comparison against a reference and a CUDA timing helper |
| [Shared-memory budget](tools/triton_smem_budget/) | A CPU-side estimate to narrow a Triton tile search before measuring |

## Status

These are teaching implementations. Included checks describe how to validate them; they do not establish a tested hardware matrix by themselves.

- The CPU SDPA attention matrix passes **12/12 cases**; the PyTorch window demo matches its reference on CPU.
- The Triton grouped-GEMM implementation and GPU attention/benchmark paths remain **unverified on NVIDIA hardware**.
- The grouped-GEMM example does not provide a custom backward implementation.
- The window-attention demo uses PyTorch batching; it is not a FlashAttention varlen implementation.
- Register-tiled GEMM, Tensor Core GEMM, and CuTe implementations are future work.

Record the GPU, CUDA/compiler versions, shapes, tolerances, and commands alongside new results. Keep correctness results separate from performance measurements.

## Related reading

[AI Infra Notebook](https://github.com/vInThePrism/ai-infra-notebook) develops the system-level ideas. [Training Performance Field Notes](https://github.com/vInThePrism/training-perf-field-notes) follows optimizations into complete training jobs.

## Reuse and contributions

Corrections, small reproductions, and explanations of failed assumptions are welcome. For a kernel change, describe its supported shapes and how you checked it.

Source code is MIT-licensed. Notes retain their CC BY 4.0 attribution requirements; see the repository license files and source notices.

## Continue learning and contributing

See [where new work belongs](docs/content-routing.md), [contribution guidance](CONTRIBUTING.md), [release validation](VALIDATION.md), and the [license and source map](LICENSES.md).
