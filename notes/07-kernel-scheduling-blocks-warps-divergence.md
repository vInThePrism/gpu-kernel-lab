# Kernel scheduling: blocks, warps, and divergence

**Root question: you launch tens of thousands of blocks onto ~100 SMs. Who decides what runs where, and when?**

Two schedulers answer it, at two granularities:

| Level | Unit scheduled | Decision | Who controls it |
|---|---|---|---|
| Device level ("GigaThread engine") | thread block | which SM a block runs on | hardware |
| SM level (warp scheduler) | warp | which warp issues an instruction this cycle | hardware |

The programmer controls neither. What you *do* control—block size, registers, shared memory, branch structure—decides how much freedom those schedulers have. Everything below follows from that.

---

## 1. Device level: blocks → SMs

- A grid's blocks are enumerated and handed to SMs that signal they have room. An A100 has 108 SMs, so a large grid runs in **waves**.
- An SM can host **more than one block** at a time, up to its residency limits (§4).
- **A block never leaves its SM.** Once placed, it stays until *every* thread in it has finished. Only then are its resources (block slot, registers, shared memory) released and the next block dispatched there.
- Consequences:
  - Blocks are independent; the order in which they run is unspecified. Never make one block wait on another.
  - If some warps of a block finish early, the block's resources stay claimed until the slowest warp retires. Idle threads from boundary conditions (`if (idx < n)` false) still hold their block's resources.
  - At grid scale, the last partial wave leaves SMs idle: the **tail effect** (wave quantization).

## 2. Inside a block: threads → warps

- A block is split into **warps of 32 threads**. The number 32 is fixed by NVIDIA; it has never changed and the programmer has no say.
- Warps are formed from the **linearized thread ID**. In multidimensional blocks, `x` varies fastest, then `y`, then `z`:

  ```
  tid  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
  warp = tid / 32,   lane = tid % 32
  ```

  Threads 0–31 are warp 0, threads 32–63 are warp 1, and so on. The partition is always the same.
- **Ordering:** the mapping of threads to lanes is deterministic. The ordering *among warps* does not exist—warp scheduling is not under user control.
- The warp is a **hardware unit**. A block whose size is not a multiple of 32 rounds up: a 50-thread block occupies 2 warps (64 lanes), and 14 lanes do nothing. A 1-thread block still costs a full warp slot.

## 3. SM level: warp scheduling

### What issues each cycle

- Each warp scheduler picks one **eligible** warp per cycle and issues its next instruction. All active threads of that warp execute it in lockstep (SIMT).
- A **scoreboard** tracks which warps have their operands ready.
- Number of schedulers per SM:
  - Fermi: 2 schedulers. Even warps go to one, odd warps to the other, and warps never move between them. At most 2 warp-instructions issue per cycle.
  - Volta / Ampere / Hopper: the SM is split into **4 sub-partitions**, each with its own scheduler, register file slice, and execution units. Up to 4 warp-instructions issue per cycle.
- "Only 2 (or 4) warps issue per cycle" is not "only 2 warps run". Issued instructions occupy pipelines for many cycles, so instructions from many warps are in flight across the FP32, INT32, SFU, and LD/ST units at once.
- The pipeline is simple: in-order issue, no out-of-order execution, no speculation, no prefetching, limited ILP. **The GPU hides latency with thread-level parallelism—many warps—not with a clever single thread.**

### Why switching warps is (almost) free

Every resident warp keeps its registers permanently allocated in the register file. Moving from warp A to warp B saves and restores nothing; the scheduler simply issues from a different warp next cycle. A CPU context switch, by contrast, must spill and reload architectural state.

### Warp states

- **Active**: resident on an SM. At any moment an active warp is one of:
  - **Eligible**: next instruction's inputs are ready.
  - **Issued**: eligible and selected this cycle.
  - **Ineligible (stalled)**: waiting.
- **Retired**: no instructions left.

An active warp becomes ineligible because of:

| Cause | Example | Nsight Compute stall reason (roughly) |
|---|---|---|
| Instruction fetch | next instruction not in the instruction cache | no instruction / branch resolving |
| Memory dependency | a global load's result is not back yet | long scoreboard |
| Execution dependency | previous ALU result not ready | wait / short scoreboard |
| Synchronization barrier | waiting at `__syncthreads()` | barrier |

**Latency hiding:** while warp A waits hundreds of cycles for HBM, the scheduler issues from warps B, C, D. Keeping many warps in flight—resident and with instructions outstanding—is what keeps the functional units busy.

## 4. Resources decide how many warps can be resident

### Residency limits per SM

| Architecture | Max blocks / SM | Max warps / SM | Max threads / SM | Registers / SM |
|---|---|---|---|---|
| Fermi | 8 | 48 | 1536 | 32K |
| Turing (T4) | 16 | 32 | 1024 | 64K |
| Pascal, Volta, A100, H100 | 32 | 64 | 2048 | 64K |

Other per-architecture constants are stable: 32 threads per warp, max 1024 threads per block, max 255 registers per thread.

### Registers

- Every variable a thread operates on lives in a register at some point. What matters is the **peak** number live at once, and the **compiler** decides the allocation.
- The register file is partitioned among resident blocks. A block's registers are private to it; within a block a thread reads only its own registers (except through warp-level primitives such as `__shfl_sync`).
- Above 255 registers per thread—or above a cap you set—the compiler **spills** to *local memory*, which is physically in global memory and slow. Check with `nvcc -Xptxas -v`.

### Shared memory

Shared memory per SM (up to 164 KB on A100, ~228 KB on H100) is split among resident blocks. A block requesting a lot of it limits how many blocks fit.

### Memory spaces, for reference

| Memory | Location | Cached | Access | Visible to |
|---|---|---|---|---|
| Register | on-chip | — | R/W | one thread |
| Shared | on-chip | — | R/W | all threads in a block |
| Global | off-chip (device HBM) | yes | R/W | all threads + host |
| Constant | off-chip | yes | R | all threads + host |
| Texture | off-chip | yes | R | all threads + host |

"Off-chip" means on the GPU but outside the SM, hence higher latency. See [GPU execution and memory](00-gpu-execution-and-memory.md) for the full hierarchy.

### Occupancy

**Occupancy = resident warps / maximum resident warps per SM.** The number of resident blocks is the minimum over four limits:

```
blocks_per_SM = min( max_blocks_per_SM,
                     max_warps_per_SM   / warps_per_block,
                     regs_per_SM        / (regs_per_thread * threads_per_block),
                     smem_per_SM        / smem_per_block )
```

Worked example (A100, 256-thread blocks = 8 warps):

- 32 registers/thread → 65536 / (32 × 256) = 8 blocks = 64 warps → **100%**.
- 64 registers/thread → 65536 / (64 × 256) = 4 blocks = 32 warps → **50%**. Registers are now the limiter.

## 5. Launch configuration

For a 1D problem of N elements:

```cpp
kernel<<<N, 1>>>(...);               // bad: 31 of 32 lanes idle per block; hits the
                                     //      blocks/SM limit long before the warps/SM limit
kernel<<<(N + 31) / 32, 32>>>(...);  // okay: full warps, but one warp per block
kernel<<<(N + 127) / 128, 128>>>(...); // better: 4 warps per block
```

- Use a block size that is a **multiple of 32** so no warp is partly empty. 128 or 256 is a good starting point; tune from there.
- More warps per block gives each SM more warps to switch between, and the switch costs almost nothing.
- Shared memory and register use per thread may cap how large a block can be.
- The grid is rounded up, so the kernel needs `if (idx < n)`. A grid-stride loop decouples the grid size from N (see [Three coordinates](01-three-coordinates.md)).

## 6. Thread divergence

### What happens

All threads of a warp issue the *same* instruction. When lanes of one warp take different sides of a branch:

```cpp
__global__ void odd_even(int n, int* x) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if ((i & 1) == 0) x[i] = x[i] + 1;   // even lanes
    else              x[i] = x[i] + 2;   // odd lanes
}
```

1. One subset of lanes executes its path while the others are masked off.
2. Then the other subset executes its path.
3. After the branch, the warp reconverges and continues in lockstep.

The two paths do **not** run in parallel, even with no data dependency between them. A two-way branch halves the useful parallelism; nested branches mask more lanes; a `switch (i % 32)` with 32 cases serializes the warp completely.

Divergence only exists **within a warp**. A branch whose condition is uniform across the warp—for example, on `warp_id`—costs nothing extra.

### Correctness vs performance

- For **performance**, divergence always matters.
- For **correctness**, usually not—except that which side runs first is **not guaranteed**. Code that depends on the order is a data race:

  ```cpp
  // x[] initialized to all 1
  if ((i & 1) == 0) x[i]   = x[i] + 1;   // even thread 2k writes x[2k]
  else              x[i-1] = x[i] * 3;   // odd thread 2k+1 also writes x[2k]
  ```

  Both threads write `x[2k]`. If the `if` side runs first, `x[2k]` becomes 2, then 3. If the `else` side runs first, it becomes 3, then 4. The result is undefined.
- Pre-Volta, spin-waiting on a lock held by another lane of the same warp could **deadlock**: the waiting path might never yield to the path holding the lock.
- NVIDIA calls the model SIMT rather than SIMD because each thread has its own logical control flow, even though a warp issues one instruction at a time.

### Branchless code

Rewrite so every lane executes the same instruction sequence and the data selects the result. The building block is a sign mask: an arithmetic right shift of a signed 32-bit int by 31 copies the sign bit into every position, giving 0 for non-negative values and -1 (all 1s) for negative ones.

```cpp
int abs_branchless(int x) {
    int mask = x >> (sizeof(int) * 8 - 1);  // 0 if x >= 0, -1 if x < 0
    return (x + mask) ^ mask;               // x < 0: (x - 1) ^ ~0 = ~(x - 1) = -x
}

int max_branchless(int x, int y) {
    int diff = x - y;
    int mask = diff >> (sizeof(int) * 8 - 1); // 0 if x >= y, -1 if x < y
    return x - (diff & mask);                 // x < y: x - (x - y) = y
}
```

Caveats:

- `abs(INT_MIN)` overflows, and so does `x - y` for operands of opposite sign and large magnitude.
- The compiler already turns short branches into **predicated** or select instructions (`SEL`, `IMNMX`), which do not diverge. Prefer `max()`, `fabsf()`, and similar intrinsics; hand-written branchless code pays off when the branch bodies are substantial.

### Volta and later: independent thread scheduling

- Starting with Volta (GV100), each thread keeps its own **program counter and call stack**. Pre-Volta, a warp shared one PC and stack.
- Threads can diverge and reconverge at sub-warp granularity, and the scheduler can interleave the divergent paths. This removes the intra-warp spin-lock deadlock and enables finer-grained synchronization.
- It does **not** make divergent paths execute in parallel: the cost of divergence remains.
- Because a warp is no longer guaranteed to be converged, warp-level primitives take an explicit **member mask** (`__shfl_sync(mask, …)`, `__ballot_sync`), and `__syncwarp()` forces reconvergence. Code must not assume implicit lockstep.

---

## Interview questions

### 1. What is the relationship between grid, block, warp, and thread? How does an SM schedule threads?

A grid contains blocks; a block contains threads; the hardware groups each block's threads into warps of 32 consecutive thread IDs (`x` fastest, then `y`, then `z`). Scheduling happens at two levels. The device-level scheduler assigns whole blocks to SMs that have room; a block stays on its SM until all its threads finish, then its resources are freed and the next block is dispatched there. The SM-level scheduler works on warps: the warp is the basic scheduling unit, and each cycle the scheduler picks an eligible warp—one whose next instruction has its operands ready—and issues that instruction to all its active threads in lockstep.

**Follow-up: how many warp schedulers does an SM have, and how many instructions can it issue per cycle?**
Fermi has 2 schedulers per SM, each issuing one warp-instruction per cycle, so at most 2 warps issue per cycle; even warps go to one scheduler and odd warps to the other. Volta, Ampere, and Hopper split the SM into 4 sub-partitions with one scheduler each, so up to 4 warp-instructions issue per cycle. More warps than that are in flight, because issued instructions occupy the functional units for many cycles.

**Follow-up: when a warp waits on memory, how does the scheduler switch, and why is it almost free?**
The pending load marks the warp ineligible (a memory dependency, tracked by the scoreboard). On the next cycle the scheduler simply issues from another eligible warp. Nothing is saved or restored, because every resident warp's registers stay in the register file the whole time. That is why the SM implements almost zero-overhead warp scheduling, and why GPUs hide memory latency by keeping many warps in flight rather than with out-of-order execution or prefetching.

### 2. How do you choose grid and block sizes? How does block = 256 relate to warps? Why multiples of 32?

Choose the block size first: a multiple of 32, typically 128 or 256, then tune. Then grid = `ceil(N / blockSize)` (with an `if (idx < n)` guard), or a grid-stride loop with a few blocks per SM. A 256-thread block is exactly 8 warps. Multiples of 32 matter because the warp is the hardware unit: a block that is not a multiple of 32 rounds up to whole warps, and the unused lanes are wasted—a 50-thread block occupies 64 lanes. Larger blocks also give the SM more warps to switch between, which is how it hides latency. Register and shared-memory use can cap the block size.

### 3. Can one block be scheduled on several SMs? How many blocks can be resident on one SM, and what limits it?

No. A block is assigned to exactly one SM and does not leave until all its threads finish; that is what lets its threads share shared memory and synchronize with `__syncthreads()`. An SM can host several blocks at once. The number is the minimum of four limits: the maximum blocks per SM (32 on Pascal–Ampere, 16 on Turing, 8 on Fermi), the maximum warps or threads per SM (64 warps / 2048 threads on Pascal–Ampere, 32 / 1024 on Turing), the register file (64K 32-bit registers split among resident blocks), and shared memory per SM.

### 4. What is warp divergence? How does the hardware execute if/else? What does it cost?

Divergence is when threads of the same warp take different branches. Since a warp issues one instruction for all its threads, the hardware runs one path with the other lanes masked off, then the other path, then reconverges. The paths do not run in parallel even when they are independent, so a two-way branch cuts parallelism by half and deeper nesting masks off more lanes; a 32-way `switch` fully serializes the warp. Which side runs first is not guaranteed, so code whose result depends on the order is a data race. Divergence only occurs inside a warp, so making conditions warp-uniform, or using predication/branchless code for short paths, avoids it.

**Follow-up: what did Volta's independent thread scheduling change? Why do warp primitives take a mask?**
From Volta on, each thread has its own program counter and call stack, so threads can diverge and reconverge at sub-warp granularity and the scheduler can interleave divergent paths. That fixes deadlocks such as lanes spin-waiting on a lock held by another lane of the same warp, and allows finer-grained synchronization. Divergent paths still execute one after another, so the performance cost remains. Because a warp is no longer guaranteed to be converged at any point, warp-level primitives take an explicit mask naming the participating lanes, and `__syncwarp()` forces reconvergence.

### 5. What determines occupancy? How do you control it?

Occupancy is resident warps divided by the maximum warps per SM. It is set by whichever limit binds first: block size (warps per block against the warps-per-SM and blocks-per-SM limits), registers per thread (the 64K-register file is split among resident blocks), and shared memory per block. You control it through the launch configuration, `__launch_bounds__` or `-maxrregcount` to cap registers, and the amount of shared memory requested. Example: on an A100 with 256-thread blocks, 32 registers per thread allows 8 blocks (100%), while 64 registers per thread allows 4 blocks (50%).

**Follow-up: is higher occupancy always better? When can low occupancy be faster?**
No. Occupancy only matters as a means of hiding latency. Once there are enough eligible warps to keep the functional units busy, more warps add nothing. A kernel can hide latency with instruction-level parallelism instead: each thread does more independent work and keeps several loads in flight. Register-tiled GEMM deliberately uses many registers per thread, lowering occupancy but raising data reuse and ILP, and is faster for it. More resident warps can also thrash L1 and shared memory.

**Follow-up: achieved occupancy is far below theoretical. What could cause it?**
Theoretical occupancy is what the resources allow; achieved occupancy is what was actually resident on average. Common gaps: too few blocks to fill every SM, or a tail effect where the last wave leaves SMs idle; load imbalance, where some warps run much longer and their block holds its slot until the slowest warp finishes; and blocks that finish at different times, leaving gaps before the next block is dispatched.
