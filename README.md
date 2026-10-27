# llama-lab

Learning to run, fine-tune and serve a Llama model locally — on my own GPU, no cloud required.

The goal is a custom model built by fine-tuning an existing Llama (1B–8B) with LoRA/QLoRA, then serving it through Ollama. Not training from scratch.

## Hardware

- NVIDIA RTX 4070 Super, 12 GB VRAM
- Enough for quantized 8B inference and QLoRA fine-tuning of 8B models

## Roadmap

- [ ] **Phase 1 — Run a model locally:** Ollama, an 8B instruct model, quantization (Q4/Q8), context length, tokens/sec
- [ ] **Phase 2 — Learn the toolkit:** Python venv, PyTorch + CUDA, Hugging Face `transformers` and `datasets`
- [ ] **Phase 3 — Fine-tune:** LoRA/QLoRA with Unsloth (or PEFT + TRL) on a small JSONL instruction dataset; compare before vs. after
- [ ] **Phase 4 — Bring it home:** merge the adapter or export to GGUF, load into Ollama with a Modelfile

## Repo layout

Only code, small datasets (JSONL), Modelfiles and notes live here. Model weights, checkpoints and the venv are kept outside the repo (see `.gitignore`).
