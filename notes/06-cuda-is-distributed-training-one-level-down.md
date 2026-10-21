# Distributed training is the same set of problems, one level up

I came to CUDA from the other direction: I had tuned multi-GPU training jobs (data/tensor/expert parallelism, all-to-all, compute–communication overlap) before I could write a reduction kernel. Learning single-GPU CUDA afterwards, I kept recognizing the same problems. This note is the mapping, because it is a useful way to reason about either side.

## The hierarchy

| Level | Workers | Fast shared storage | Communication | Barrier |
|---|---|---|---|---|
| warp | 32 lanes | registers | shuffles | implicit / `__syncwarp` |
| block | ≤ 32 warps | shared memory | smem reads/writes | `__syncthreads` |
| grid | blocks on all SMs | L2 / HBM | global memory, atomics | kernel boundary |
| node | 8 GPUs | HBM per GPU | NVLink / NVSwitch | collective calls |
| cluster | nodes | — | InfiniBand / RoCE | collective calls |

Bandwidth drops at every step outward (registers ≫ smem ≫ HBM ≫ NVLink ≫ PCIe/IB), and latency rises. The design rule is the same at every level: **do as much as possible at the inner level, and cross the slow boundary as rarely and as coarsely as possible.**

## Technique by technique

| Single-GPU technique | Distributed counterpart | Shared idea |
|---|---|---|
| warp → block → grid reduction | hierarchical all-reduce (intra-node NVLink, then inter-node IB) | reduce inside the fast domain first; send one value per domain over the slow link |
| histogram privatization (per-block smem histogram, one global atomic per bin) | gradient accumulation over micro-batches, one all-reduce per step | accumulate locally, communicate once |
| atomic contention on one address | stragglers / hotspots on one link or one expert | the problem is contention, not the operation |
| histogram → scan → scatter (counting sort) | MoE token dispatch: count tokens per expert → prefix sum for offsets → all-to-all | data-dependent placement needs a global scan before the scatter |
| `__syncthreads` waits for the slowest warp | a collective waits for the slowest rank | barrier cost = slowest participant, so load balance matters as much as speed |
| streams + events for copy/compute overlap | a dedicated communication stream; bucketed all-reduce overlapped with backward | issue early, wait late, wait only on true dependencies ([note 05](05-async-is-removing-false-dependencies.md)) |
| tiled GEMM: split C into tiles, K loop | tensor parallelism: split the weight matrix by columns or rows | partition the matrix so most work is local; communicate the reduction dimension |
| split-K (partial sums + reduce) | row-parallel linear: partial outputs + all-reduce | splitting the reduction dimension always costs a reduction |
| online softmax merge of `(m, l)` | ring attention / context parallelism: merge per-block `(m, l, O)` as K/V shards arrive | an associative merge lets partial results arrive in any order and any place |
| roofline (compute vs. bandwidth bound) | α-β model: `time = α + bytes / β` per message | know which term dominates before optimizing |

## One difference that matters

Inside a GPU, H2D/D2H copies run on **copy engines** that are separate from the SMs, so they overlap with kernels almost for free. **NCCL collectives are kernels**: they occupy SMs and compete with compute for them. Overlapping communication with computation in distributed training therefore isn't free in the same way. It costs SM time and can slow the overlapped GEMMs, which is why frameworks limit the number of SMs/channels NCCL uses.

## How it looks in code

Kernels know nothing about other GPUs. Parallelism lives entirely in host code as library calls enqueued on streams:

```cpp
// Data parallel: every rank computes full gradients on its own batch, then averages
backward(model, batch);                                   // kernels on compute stream
ncclAllReduce(grads, grads, n, ncclFloat, ncclSum, comm, comm_stream);

// ZeRO-3 / FSDP: parameters and grads are sharded
ncclAllGather(param_shard, full_param, shard_n, ncclBf16, comm, s);      // before use
ncclReduceScatter(full_grad, grad_shard, shard_n, ncclFloat, ncclSum, comm, s); // after backward

// Tensor parallel MLP (Megatron): column-split W1, row-split W2
h  = gelu(x @ W1_shard);        // no communication: GeLU is element-wise, stays local
y_partial = h @ W2_shard;       // each rank holds a partial sum over its slice of the hidden dim
ncclAllReduce(y_partial, y, n, ncclBf16, ncclSum, comm, s);   // one all-reduce per MLP forward
```

The column-then-row order is chosen precisely so the element-wise activation in the middle needs no communication: column split shards the *outputs* of W1, row split shards the *reduction dimension* of W2, and only the end of the block has a reduction to communicate.

## Communication volume, derived the same way as memory traffic

Ring all-reduce over `p` ranks on a message of `S` bytes = reduce-scatter (`p − 1` steps) + all-gather (`p − 1` steps), each step moving `S/p`:

```
bytes sent per rank = 2 · (p − 1) / p · S  ≈ 2S
```

Independent of `p` — the reason ring all-reduce scales in bandwidth — but `2(p − 1)` latency terms, which is why small messages prefer tree algorithms. This is the distributed analogue of counting HBM bytes for a kernel before profiling it: predict, then measure.

## Why this helped me

On the job I could profile a training run, find the gap on the timeline, and fix it. What I couldn't do well was explain *why* from first principles. Single-GPU CUDA turned out to be the smallest setting in which every one of those distributed problems appears, and where the feedback loop is minutes instead of a cluster reservation.
