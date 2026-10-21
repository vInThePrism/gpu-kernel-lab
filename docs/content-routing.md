# Where new work belongs

| Work | Repository | Typical destination |
|---|---|---|
| CUDA, Triton, or CuTe operators; layouts; synchronization; kernel measurements | [gpu-kernel-lab](https://github.com/vInThePrism/gpu-kernel-lab) | kernels/, notes/, tools/ |
| Attention, MoE, parallelism, inference concepts, and paper derivations | [ai-infra-notebook](https://github.com/vInThePrism/ai-infra-notebook) | notes/ |
| Training bottlenecks, negative results, trace diagnosis, data loading, and checkpoint experiments | [training-perf-field-notes](https://github.com/vInThePrism/training-perf-field-notes) | notes/, tools/src/perfdoctor/ |
| Model FLOPs, LoRA/frozen-state accounting, packing contracts, and MFU collection | [logical-mfu](https://github.com/vInThePrism/logical-mfu) | logical_mfu/, docs/ |

An RMSNorm implementation belongs in the kernel lab. An investigation of why its gain shrinks in multi-GPU training belongs in field notes. A correction to the MFU formula belongs in logical-mfu. Link these pieces instead of copying the same explanation into every repository.

Update the existing note when the same concept improves. Use a new file for a distinct question. Keep raw chats, temporary logs, and incomplete thoughts locally until they become useful to another reader. Preserve public-safe evidence as small tables or reproducible inputs rather than committing large raw traces.

Use concrete commit descriptions, such as `Explain the WAR barrier between reduction calls` or `Document checkpoint final-drain timing`.
