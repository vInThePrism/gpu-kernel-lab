# Window attention: launch-bound vs one varlen call

Reproduces, with PyTorch only, why a vision tower's window attention can ignore a faster attention
kernel: one launch per 64-token window makes the job **launch-bound**. Batching all windows into one
call (on GPU: `flash_attn_varlen_func` with `cu_seqlens`) fixes the call granularity.

```bash
python window_attention_varlen.py
```

Status: CPU output equivalence checked in this release; GPU launch-bound behavior has not been reproduced. It needs PyTorch. CPU timing is not evidence for GPU acceleration.

How to recognize it in Nsight Systems:

1. attention GPU time does not change when the kernel gets faster,
2. each attention kernel lasts a few microseconds,
3. the GPU row has gaps while the CPU row is a solid block of `cudaLaunchKernel`.

In the original job: ~1M → 10K launches, attention GPU time 8.8 s → 4.0 s over a fixed 10-step replay.
