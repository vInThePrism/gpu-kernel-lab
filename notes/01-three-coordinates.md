# Three coordinates: a way to design any CUDA kernel's indexing

Most indexing bugs I made while learning CUDA came from one confusion: treating `threadIdx`/`blockIdx` as if they were data positions. In `c[i] = a[i] + b[i]` they happen to coincide, so the habit forms early and then breaks on the first kernel where they don't (transpose, GEMV, scan).

The fix is to keep **three kinds of coordinates** apart and connect them with **two functions**:

```
hardware coordinate  --f (you design)-->  output coordinate  --g (the math)-->  input coordinates
   "who am I"                              "what do I produce"                  "what must I read"
```

- **f: hardware → output.** Which output(s) this thread, warp or block is responsible for. This is the *only* design decision, and it decides coalescing and performance.
- **g: output → input.** Which inputs that output depends on. This is the operator's mathematical definition. It already existed when you wrote the CPU version; CUDA does not change it.

Outputs are data coordinates too. Once a kernel has computed its data coordinates, `threadIdx` and `blockIdx` should not appear again.

## The four-step template

```cuda
__global__ void kernel(...) {
    // 1. hardware coordinates: who am I
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. f: which output I own                 (design)
    int out_i = tid;
    // 3. g: which inputs that output reads     (math)
    int in_i = out_i;
    // 4. work, using data coordinates only
    if (out_i < n) c[out_i] = a[in_i] + b[in_i];
}
```

**Self-check:** if `threadIdx` or `blockIdx` shows up in step 3 or 4, the separation isn't clean yet. Every kernel in [`kernels/`](../kernels/cuda) is commented with these four steps.

## When many threads share one output

For reductions (sum, softmax, RMSNorm, GEMV), f has to say three things:

1. which output I *participate* in,
2. which slice of g's input set I handle, and
3. **who writes** — exactly one thread per output.

```cuda
// GEMV, warp per row: y[r] = sum_k A[r][k] * x[k]
int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
int r = blockIdx.x * (blockDim.x / 32) + warp;   // f: my warp owns row r
float acc = 0;
for (int k = lane; k < K; k += 32)               // f: I own k = lane, lane+32, ...
    acc += A[(size_t)r * K + k] * x[k];          // g: row r of A, all of x
acc = warp_allreduce_sum(acc);
if (lane == 0) y[r] = acc;                       // who writes: one thread
```

There is a second, separate kind of index in such kernels: the *hardware* index used for cooperation (`lane`, `warp`, shared-memory slot `partial[warp]`). I wrap all cooperation in reusable primitives (`warp_allreduce`, `block_allreduce`, `block_inclusive_scan`), so that the kernel body only contains data indices and the primitives only contain hardware indices. Global memory is addressed by data coordinates; threads talk to each other by hardware coordinates. Keeping those in different functions removed a whole class of bugs for me (see the [bug log](04-bug-log.md): writing `smem[lane]` where `smem[warp]` was meant).

## Six base classes and four modifiers

Once f is the thing you design, kernels sort themselves by *what one output depends on*. Classify first, and the mapping is mostly decided.

| Class | One output depends on | Mapping skeleton | Cooperation |
|---|---|---|---|
| ① element-wise | one (or a fixed few) inputs | thread ↔ element | none |
| ② global reduce / scan | all inputs | block ↔ segment, thread ↔ element | shuffle + smem in a block; a 2nd kernel, atomics or look-back across blocks |
| ③ row reduction | a whole row | row ↔ warp or block, columns strided over lanes | warp or block reduce |
| ④ fixed permutation | one input, at a position known at compile time | block ↔ tile, separate read and write mappings | stage through smem |
| ⑤ blocked accumulation | a row × a column, summed over K | block ↔ output tile, thread ↔ output sub-tile, loop over K | smem holds shared input tiles |
| ⑥ data-dependent placement | an input whose destination depends on its *value* | block ↔ segment; local count → global scan → scatter | relies on ②'s scan |

④ vs ⑥: in a transpose, element `(i, j)` always goes to `(j, i)`; you know that when writing the code. In a sort or histogram, where an element goes depends on all the data, so ⑥ usually needs a scan (②) to compute destinations first.

| Modifier | What it does | Effect on the mapping |
|---|---|---|
| a. many elements per thread | `float4`, grid-stride loops, register tiling | thread ↔ a group of elements |
| b. split | cut the reduced dimension across blocks, merge later | turns ③ or ⑤ into "③/⑤ + ②" (split-K, flash-decoding) |
| c. fusion | several classes in one kernel | each stage keeps its own mapping; intermediates stay in registers/smem |
| d. low precision | change element type; dequantize on load, quantize on store | skeleton unchanged; bytes per element and instructions change |

### Common kernels, classified

| Kernel | Class | Notes |
|---|---|---|
| fused bias + GeLU | ① + a + c | thread ↔ one `float4`; `bias[col]` with `col = i % N` |
| matrix transpose | ④ | smem tile; padding or XOR swizzle for the column read |
| reduce sum / max | ② | the classic 7-step ladder only changes the "thread ↔ element" layer |
| warp/block reduce | — | not a kernel: the cooperation primitive used by ②, ③, ⑥ |
| prefix sum | ② | the same two-level shape as block reduce, then lifted to a grid |
| softmax, LayerNorm, RMSNorm | ③ | one block (or warp) per row; online softmax makes the reduction a pair |
| RoPE | ① | pairs of elements; the pairing rule is g |
| naive attention | ⑤ → ③ → ⑤ | QKᵀ, softmax, PV as three kernels |
| FlashAttention | ⑤ + ③ fused (c) | online softmax lets ③ run inside ⑤'s K/V loop |
| decode attention / flash-decoding | ③ + b | split the KV length across blocks, then merge |
| SGEMM ladder | ⑤ + a | naive → smem tile → register tile → vectorized → double buffer → Tensor Cores |
| GEMV | ③ (+ b + d for decode) | GEMM with N = 1, no reuse of W |
| top-k / argmax | ③ or ② | argmax is a reduction over (value, index) pairs |
| histogram, sort | ⑥ | histogram = local count + merge; radix sort = histogram + scan + scatter |
| quantization | ③ + ① + d | absmax per row, then element-wise scale |

## How I actually design one

1. Write the CPU loop. That is g.
2. Classify the kernel (table above).
3. Draw a small table of **what every thread holds after every step** (8 or 32 threads is enough). This is where cooperation bugs show up before any code exists.
4. Choose f so that consecutive `threadIdx.x` touch consecutive addresses on the side that dominates traffic.
5. Push every piece of cooperation into a primitive; the kernel body should read like the CPU loop plus f.
