# Working in gpu-kernel-lab

Purpose: GPU kernels and numerical reasoning.

CUDA examples live in kernels/cuda/, Triton in kernels/triton/, PyTorch demonstrations in demos/, and CPU helpers in tools/. Use a matching NVIDIA toolchain for GPU checks. The shared headers must remain alongside the CUDA examples.

## Commands

Run from the repository root unless a command says otherwise.

```bash
make -C kernels/cuda 02_reduce ARCH=sm_75
./kernels/cuda/02_reduce
```

## Editing rules

- Keep changes focused on the request. Do not add test files, test dependencies, or scaffolding unless requested or required by an existing build. Run relevant existing checks.
- Keep README commands and relative links aligned with the actual files.
- Report measured, estimated, source-reported, and unverified results accurately. Include environment and assumptions with performance claims.
- Preserve source copyright and license provenance; consult LICENSES.md before moving or adapting material.
- Never commit credentials, private paths, datasets, raw internal logs, or generated caches.
- Use docs/content-routing.md to keep the four companion repositories distinct.
