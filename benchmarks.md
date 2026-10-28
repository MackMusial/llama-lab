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

## Notes

- **First-request warm-up:** the very first prompt after loading showed prompt eval at 1.07 tok/s (18.7 s for 20 tokens) and generation at 61.06 tok/s. Treated as one-time CUDA initialization; the warm second run is recorded above.
- **Chat = growing prompt:** the model is stateless, so each turn resends the whole conversation. Turn 2 was 2,870 tokens (70% of the 4,096 context) after two questions.
