# Benchmarks

Local inference on an RTX 4070 Super (12 GB VRAM) via Ollama. Generation speed = `eval rate` from `ollama run --verbose`; VRAM = `nvidia-smi` used minus idle baseline (~1.1 GB).

| Date | Model | Quant | Context | Processor | VRAM (model) | Prompt eval (tok/s) | Generation (tok/s) |
|---|---|---|---|---|---|---|---|
| 2026-10-27 | llama3.1:8b | Q4_K_M | 4096 | 100% GPU | ~5.4 GB | 822* | 64.58 |
| 2026-10-27 | llama3.1:8b-instruct-q8_0 | Q8_0 | 4096 | 100% GPU | ~8.9 GB | 629* | 42.53 |

\* Prompt eval on a warm second turn processes only ~20 new tokens (the rest come from the prompt cache), so it is not a reliable prefill measurement.

Method: fresh chat, prompt 1 "Explain how a car engine works in detail." (warm-up, not recorded), prompt 2 "Explain how a jet engine works in detail." (recorded).

## Q4 vs Q8

- Generation is memory-bandwidth bound: every token reads all weights once, so tok/s ≈ bandwidth ÷ model size. 4070 Super ≈ 504 GB/s.
- Predicted Q8 ≈ 38 tok/s (504 / 8.5 GB × 65% efficiency seen on Q4). Measured 42.5 — prediction holds.
- Q8 costs 34% speed and ~3.5 GB more VRAM for higher weight precision.
- **Answer quality (jet engine prompt, n=1 each):** both contained factual errors. Q4 was more ambitious (Brayton cycle, thrust equation) but more confidently wrong: it claimed jets run a "four-stroke cycle," mislabeled a Brayton stage and listed a fan as universal. Q8 was shallower with fewer outright errors. Slight edge to Q8, but not conclusive: one sample each at default temperature, and both answers were contaminated by the prior car-engine turn in the chat history ("four-stroke" leaked from it).
- Takeaway: an 8B model writes confident, well-formatted, partly wrong technical explanations at either quant. Quality comparisons need multiple samples, fresh context per prompt, and fact-checking.

## Context length (llama3.1:8b Q4_K_M)

| num_ctx | SIZE (`ollama ps`) | Processor | VRAM used (total) | Generation (tok/s) |
|---|---|---|---|---|
| 4,096 | 5.3 GB | 100% GPU | 6,874 MiB | ~65 |
| 16,384 | 7.0 GB | 100% GPU | 8,518 MiB | — |
| 65,536 | 14 GB | 26%/74% CPU/GPU | 11,585 MiB (full) | 25.19 |

- **KV cache grows linearly:** ~138 KB/token (4K→16K) and ~142 KB/token (16K→64K). Theory for Llama 3.1 8B: 2 (K+V) × 32 layers × 8 KV heads × 128 dim × 2 bytes (fp16) = 128 KiB/token → 64K context ≈ 8 GiB of KV cache on top of 4.9 GB of weights.
- **Memory doesn't degrade gracefully, it spills:** at 64K the model no longer fits in 12 GB, so Ollama put 26% of it in system RAM on the CPU. Output quality is unchanged; speed drops because system RAM bandwidth is a fraction of VRAM's.
- **Spill cost:** 26% of the model on CPU made generation 2.6× slower (64.6 → 25.2 tok/s). Every token passes through all layers in sequence, so the slow CPU part (~1.3 GB at ~60 GB/s RAM) takes longer than the large GPU part (~3.6 GB at ~504 GB/s). The slowest stage sets the speed.
- **Practical rule:** the KV cache is reserved for the full `num_ctx` up front. This run used 52 tokens of a 65,536 window and paid ~8 GB VRAM and 61% of its speed for it. Set `num_ctx` to what the task needs.
- My predictions before measuring (both wrong): "grows but not linearly" and "won't spill, will lose accuracy."

## Laptop CPU inference (Ryzen 5 7530U)

HP Envy x360 15: 8 GB LPDDR4X-4266 (~68 GB/s), no CUDA. Ollama 0.40.12, 100% CPU (Ollama detects the integrated Radeon via Vulkan but skips integrated GPUs by default). VS Code and Claude closed while measuring.

| Date | Model | Quant | File | Loaded (`ollama ps`) | Processor | Prompt eval (tok/s) | Generation (tok/s) |
|---|---|---|---|---|---|---|---|
| 2026-10-27 | llama3.2:1b | Q8_0 | 1.3 GB | — | 100% CPU | 49.7* | 20.6 (17.0 at ~850 ctx) |
| 2026-10-27 | llama3.2:3b | Q4_K_M | 1.96 GB | 2.6 GB | 100% CPU | 38.0 | 12.48 |

\* 34-token prompt: too short for a reliable prefill measurement.

### Prediction vs actual

- Same rule as the GPU: tok/s ≈ bandwidth ÷ model bytes × efficiency, where efficiency = measured ÷ ceiling.
- **1B:** ceiling 68 ÷ 1.3 = 52.5 tok/s. Predicted ~33 using the GPU's 63% efficiency. Measured 20.6 = **39%**. The prediction was too high because I assumed the CPU would be as efficient as the GPU.
- **3B:** ceiling 68 ÷ 1.96 = 34.7 tok/s. Predicted ~13.6 using the 1B's measured 39%. Measured 12.48 = **36%**, within 8% of the prediction. Once a machine's efficiency is measured, it predicts other models on that machine.
- **Why the CPU is less efficient (36–39% vs 63%):** using full bandwidth needs a huge number of memory reads in flight at once. A GPU's thousands of cores keep the bus busy; 6 CPU cores can't, and they also have to do all the dequantizing and math themselves.
- **Why 2.6 GB loaded for a 1.96 GB file:** the KV cache is reserved for the full context up front. Llama 3.2 3B: 2 (K+V) × 28 layers × 8 KV heads × 128 dim × 2 bytes = 112 KiB/token × 4,096 ≈ 0.47 GB, plus ~0.17 GB of compute buffers. File size is never memory size.
- **Why the 1B slowed at ~850 tokens (20.6 → 17.0):** every new token attends to all earlier tokens. The extra KV reads are small (32 KiB/token × 850 ≈ 27 MB, ~2% of the weights), so most of the 17% drop is attention compute that grows with context, which a weak CPU feels. n=1, so part of it may be noise.
- **Practical read:** a 3B Q4 model is usable on CPU-only hardware (12.5 tok/s beats reading speed), but prefill at ~38 tok/s means a 500-token prompt waits ~13 s before the first token. Speed alone doesn't make 3B the right model; quality still has to be tested.

## PyTorch: CPU vs GPU matrix multiply (desktop)

`gpu_matmul.py`: one 4096 × 4096 fp32 matmul (2 × 4096³ ≈ 137 billion operations), torch 2.15.0+cu134, n=1.

| Device | Predicted | Measured | Achieved |
|---|---|---|---|
| CPU | 137 ms (at ~1 trillion ops/s) | 233.9 ms | ~0.59 trillion ops/s |
| RTX 4070 Super | 4.6 ms (at ~30 trillion ops/s) | 7.3 ms | ~18.8 trillion ops/s |
| Speedup | ~30× | ~32× | |

- **GPU calls are asynchronous:** `A_gpu @ B_gpu` only queues the work, so the timer needs `torch.cuda.synchronize()` before it stops.
- **Warm-up needs a sync too:** without one after the warm-up multiply, the timed run waited behind it in the GPU queue and measured 11.0 ms instead of 7.3 ms, a 50% error with no warning.
- Matrix × matrix is the prefill shape (compute bound), which is where a GPU's thousands of cores pay off. Generation is matrix × vector (bandwidth bound), so the gap there should be smaller.

## Notes

- **First-request warm-up:** the very first prompt after loading showed prompt eval at 1.07 tok/s (18.7 s for 20 tokens) and generation at 61.06 tok/s. Treated as one-time CUDA initialization; the warm second run is recorded above.
- **Chat = growing prompt:** the model is stateless, so each turn resends the whole conversation. Turn 2 was 2,870 tokens (70% of the 4,096 context) after two questions.
 
Laptop specs Ryzen 5 7530U, ~68 GB/s, CPU only
Model	Quant	File	Loaded	Gen (tok/s)	Prefill (tok/s)
llama3.2:1b	Q8_0	1.3 GB	?	20.6 fresh / 17.0 at ~850 ctx	49.7 (noisy, 34-token prompt)
llama3.2:3b	Q4_K_M	1.96 GB	2.6 GB	12.48	38.0