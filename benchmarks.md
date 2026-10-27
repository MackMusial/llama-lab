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

## Notes

- **First-request warm-up:** the very first prompt after loading showed prompt eval at 1.07 tok/s (18.7 s for 20 tokens) and generation at 61.06 tok/s. Treated as one-time CUDA initialization; the warm second run is recorded above.
- **Chat = growing prompt:** the model is stateless, so each turn resends the whole conversation. Turn 2 was 2,870 tokens (70% of the 4,096 context) after two questions.
