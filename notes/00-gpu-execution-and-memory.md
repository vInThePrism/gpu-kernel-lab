# How a GPU runs your code, from one root fact

**Root fact: memory is slow, so the GPU keeps many warps ready and switches between them.**
Everything below follows from that.

## Execution model

- A grid has many blocks; a block has warps of **32 threads**. Each block is assigned to **one SM**
  and stays there until it finishes. Warp schedulers issue one instruction per warp; the threads of a
  warp run in lockstep (SIMT).
- When a warp waits for global memory, the scheduler issues from another **eligible warp** in the next
  cycle. Switching is almost free because every warp's registers stay in the register file: there is no
  context switch. This is **latency hiding**, and it is why GPUs want many resident warps (occupancy).
- **Launch config:** start with 128 or 256 threads per block (a multiple of 32, so no warp is half
  empty), then tune for occupancy. Grid = `ceil(N / block)`, or a **grid-stride loop** with a few blocks
  per SM for very large N, which decouples the grid from N and amortizes setup.
- `if (idx < n)` is needed because the grid is rounded up.
- **Divergence:** threads of one warp take different branches → each path runs in turn with the other
  threads masked off; cost = sum of both paths. It only matters *inside* a warp: make branches
  warp-uniform.
- **Since Volta:** each thread has its own program counter, so a warp is not guaranteed to be converged.
  That is why `__shfl_sync` and friends take a member mask.
- **Synchronization:** `__syncwarp` (warp), `__syncthreads` (block), cooperative groups (tile → grid,
  grid needs a cooperative launch); atomics make a read-modify-write one step; fences only order memory
  visibility and make nobody wait. `__syncthreads()` inside a branch that some threads skip is undefined
  behavior and usually a deadlock.
- **CUDA Cores vs Tensor Cores:** scalar ops per thread vs a whole matrix-tile multiply-accumulate per
  warp(-group) instruction, low precision in, FP32 accumulate. Tensor Cores for GEMM/conv/attention
  matmuls; CUDA Cores for element-wise ops, reductions, softmax, norms.

## Memory hierarchy, level by level

**Fast memory is small; big memory is slow.**

| Level | Scope | Rough size | Notes |
|---|---|---|---|
| Registers | thread | 64K × 32-bit per SM | fastest; too many per thread → spills to *local memory* (which lives in HBM) |
| Shared memory / L1 | block / SM | up to ~228 KB per SM (Hopper/Blackwell) | programmable L1; 32 banks × 4 bytes |
| L2 | whole GPU | 40–50 MB (A100/H100), ~126 MB (B200) | handles reuse across blocks for free |
| HBM | whole GPU | 80–192 GB | ~3.35 TB/s on H100, hundreds of cycles latency |

- **Coalescing** is the rule for HBM: a warp reading 32 consecutive floats moves 128 bytes = four 32-byte
  sectors. Strided or scattered access touches more sectors for the same useful bytes.
- **Bank conflicts** are the rule for shared memory: threads of a warp hitting different addresses in
  the same bank serialize (the same address is a broadcast, not a conflict). Fix with padding
  (`[32][33]`) or an XOR swizzle.
- **Transpose:** do every HBM access by rows (coalesced read and write) and move the column access into
  shared memory, where only bank conflicts matter, then pad.
- **float4:** 16-byte loads (`LDG.128`) don't reduce bytes moved; they cut the number of load/store
  instructions 4×, so more bytes are in flight per thread. Needs 16-byte alignment.
- **Double buffering:** two *input* buffers in shared memory; compute on one while the next tile loads
  into the other (`cp.async` on Ampere, TMA on Hopper), then swap roles.

## A systematic answer: improving memory efficiency

First confirm it is memory-bound (Nsight Compute, roofline). Then three layers:

1. **Make each access efficient:** coalesced, vectorized, no bank conflicts, no spills.
2. **Move fewer bytes:** tiling for reuse, kernel fusion, lower precision.
3. **Hide latency:** enough bytes in flight (Little's law), double buffering / async copies.

## Recipe: scalar → vectorized access

1. Check preconditions: contiguous elements, 16-byte alignment (including every row offset), length
   divisible by the vector width (or handle a tail).
2. Pick a 16-byte type: `float4` for fp32, `uint4` (8 × bf16) for half types.
3. `reinterpret_cast` every pointer you vectorize, after computing its base address.
4. Rescale indices to vector units: loop bound `n` → `n / 4`; anything indexed by column
   (like a bias) must be cast and indexed the same way.
5. One vector load → compute on `.x .y .z .w` → one vector store.
6. Leave the thread → work mapping unchanged; adjacent threads now hit adjacent 16-byte chunks.

Common bugs: keeping the scalar loop bound (runs 4× too far), and indexing a broadcast operand with the
vector index but scalar type (adds the same bias to all four lanes).
