# Coalescing vs. bank conflicts: two paths, two opposite-sounding rules

I kept mixing up three ideas — coalescing, bank conflicts, vectorized access — because they all sound like "make the addresses nice". They live on different hardware paths with different rules. Units below are bytes throughout.

```
global path:   HBM/DRAM  -> L2 -> L1 -> registers     unit: 32-byte sector
shared path:   shared memory -> registers             unit: 32 banks x 4 bytes
```

## Rule 1 (global path): a warp's addresses should cluster

Memory instructions are issued per **warp**. The memory system moves data in **32-byte sectors**. What matters is how many distinct sectors one warp instruction touches.

- 32 threads reading 32 consecutive floats = 128 bytes = **4 sectors**, all bytes used. Coalesced.
- 32 threads reading floats with a stride of 32 floats (a column of a row-major matrix) = **32 sectors**, 4 useful bytes out of each 32 → 1/8 of the bandwidth.

"Coalesced" is a property of the **warp's** address pattern, not of one thread. One thread reading 4 bytes is fine, if its 31 neighbours read the next 124.

## Rule 2 (shared path): a warp's addresses should spread across banks

Shared memory is split into **32 banks**, each 4 bytes wide; consecutive 4-byte words go to consecutive banks (`bank = (byte_address / 4) % 32`). Each bank serves one word per cycle. If several lanes of the **same warp, in the same instruction**, hit **different words in the same bank**, those accesses serialize: an n-way conflict.

Exceptions and details:

- Many lanes reading the **same word** = broadcast, no conflict.
- `s[tid]` (consecutive) hits 32 distinct banks. Consecutive access satisfies *both* rules at once; the rules only diverge for strided patterns.
- A stride of `s` words gives a `gcd(s, 32)`-way conflict. Column of a `float tile[32][32]` → stride 32 → **32-way**, the worst case. Padding to `[32][33]` makes the stride 33, which is odd, so `gcd = 1`.
- "Spread out" means across **banks** (address mod 128 bytes), not physically far apart.

## Vectorized access is orthogonal

`float4` loads change **how many bytes one thread moves per instruction** (16 instead of 4), not the warp-level pattern. Same total bytes, 4× fewer instructions, more bytes in flight per thread. It needs 16-byte alignment and a plan for the tail. It is not a substitute for coalescing: 32 lanes doing strided `float4` loads still waste sectors.

## Why transpose needs shared memory

A naive transpose reads rows and writes columns (or vice versa), so one side is always strided on the global path. The shared-memory tile exists to satisfy **Rule 1 on both sides**: read rows of the input coalesced, write rows of the output coalesced, and do the actual row↔column exchange inside shared memory — where it then breaks **Rule 2**, which padding or a swizzle fixes ([note 10](10-why-xor.md)). The smem tile isn't there to "make transposition natural"; it's there because global memory punishes strided access ~8× and shared memory punishes it in a way that is cheap to fix.
