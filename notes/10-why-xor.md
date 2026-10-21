# Why XOR shows up everywhere in CUDA

XOR appears in two places that look unrelated: warp reductions (`__shfl_xor_sync`) and shared-memory swizzles (`tile[r][c ^ r]`). Same two properties are doing the work in both.

## Property 1: `x ↦ x ^ m` is an involution

`(x ^ m) ^ m = x`. So for a fixed mask `m`, lanes pair up: lane `i` talks to lane `i ^ m`, and that lane talks back to `i`. 32 lanes form 16 disjoint pairs, every lane has exactly one partner, and the exchange is symmetric.

That is exactly a butterfly network. With masks 16, 8, 4, 2, 1:

```cuda
for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
```

After round `m`, each lane holds the sum over a group of `32 / m` lanes, and after 5 rounds every lane holds the full sum. `__shfl_down_sync` is **not** a permutation (high lanes read from out of range and keep their own value), so only lane 0 ends up correct. That's why softmax and RMSNorm, where every thread needs the result, prefer the XOR form.

## Property 2: XOR is a bijection along each argument (a Latin square)

For a fixed row `r`, `c ↦ c ^ r` permutes `0..31`; for a fixed column `c`, `r ↦ c ^ r` also permutes `0..31`. Laid out as a table, `c ^ r` is a **Latin square**: every value appears once per row and once per column.

Transpose stores logical `(r, c)` at physical column `c ^ r`:

- **Writing a row** (fixed `r`, lanes vary `c`): physical columns `c ^ r` are a permutation → 32 distinct banks.
- **Reading a column** (fixed `c`, lanes vary `r`): physical columns `c ^ r` are also a permutation → 32 distinct banks.

Both directions are conflict-free with **no extra memory**. 4 × 4 example (row `r`, logical column `c`, entry = physical column):

```
        c=0 c=1 c=2 c=3
r=0      0   1   2   3
r=1      1   0   3   2
r=2      2   3   0   1
r=3      3   2   1   0
```

Every row and every column is a permutation of 0..3.

## Why XOR rather than padding or `(c + r) % 32`

- **Padding** (`[32][33]`) also fixes conflicts, but wastes a column (128 bytes per 32 × 32 float tile) and breaks power-of-two row alignment. That matters later: `float4` loads, `ldmatrix`, and TMA all want 16-byte (or larger) aligned rows.
- **`(c + r) % 32`** is also a Latin square, but addition carries across bits. XOR acts on each bit independently, so you can swizzle **only some bits** of an address (e.g. the bits that select a 16-byte chunk within a 128-byte row) and leave the low bits — the position inside an aligned vector — untouched. That's the form CUTLASS/CuTe swizzles and Hopper TMA swizzle modes take.
- Requirement: the tile width must be a power of two, so XOR stays inside the range.

Kernel: [04_transpose.cu](../kernels/cuda/04_transpose.cu) (padding and XOR variants side by side, with a 32 × 8 block where each thread moves four elements).
