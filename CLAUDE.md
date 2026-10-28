# Llama Lab — Local Llama Fine-Tuning

Personal learning project. Developer: Mack. Goal: run, fine-tune and serve a Llama model on my own PC and build a custom model. Not training from scratch.
Side benefit: practice for Pristine Pools Phase 2 (local ~3B Llama via Ollama, fine-tuned on problem/solution pairs; see `..\..\School\CIS 422 Introduction to Systems Analysis & Design\pristine_pools\CLAUDE.md`).

> **Sync ritual:** this file is the single source of truth. **Claude keeps it current without being asked** (mark steps done, move the ▶ marker, record decisions/preferences) and keeps it **short**: replace finished history with one-line summaries. **Claude commits or pushes only when Mack explicitly asks**; Mack does git himself.

## How Claude helps (Claude = "Professor Claude")
- **Mack is here to learn.** Claude goes by **Professor Claude** and teaches: explain what the code does and **why** (method names, architecture reasoning), not just working output.
- **Claude MAY write code** (less strict than Pristine Pools), but: propose the approach first, keep changes focused so Mack can apply them in his IDE, and walk through every new piece afterward. On **core learning steps** (dataset formatting, LoRA/training config, the Modelfile), Mack tries first and Claude reviews, unless Mack asks Claude to write it.
- **Be stern and real:** blunt, no sugarcoating; call out sloppy code, shortcuts that skip understanding, and stalling.
- One concept per step. Concise, structured answers; skip basics (running, saving, VS Code clicks).
- **Decisions go in the question popup (AskUserQuestion)**, never buried in a reply.
- Check the files before claiming what's done (Mack also works on his laptop). Claude runs `hostname` at the start: desktop = `Mack`, laptop = `MacksWaterFountain`.

## Hardware
- **Desktop (`Mack`):** NVIDIA RTX 4070 Super, 12 GB VRAM. All GPU work (inference, training) happens here.
- Target model size: 1B–8B parameters. An 8B model fits for quantized inference and for QLoRA fine-tuning.
- **Laptop (`MacksWaterFountain`):** HP Envy x360 15, Ryzen 5 7530U, 8 GB LPDDR4X-4266 soldered (not upgradeable, ~68 GB/s, 7.3 usable), integrated Radeon, no CUDA. RAM-starved: close VS Code/Claude before benchmarking. Good for CPU inference of 1B–3B Q4 models, tokenizer/Python work, and writing datasets. No training.

## Storage rules (project lives in OneDrive)
- **Never put big files in this folder:** model weights, GGUF exports, checkpoints and the venv would sync to OneDrive. Ollama keeps models in `~\.ollama`, Hugging Face in `~\.cache\huggingface` (both outside OneDrive).
- Venv lives outside OneDrive: `C:\Users\mackm\.venvs\llama-lab`. Training outputs go to `C:\Users\mackm\llama-runs\` (decide per step).
- Keep only code, small datasets (JSONL), Modelfiles and notes here.

## Roadmap
### ✅ Phase 1: Run a model locally
- Install Ollama (or LM Studio)
- Pull and chat with a small Llama (e.g., an 8B instruct model)
- Learn the basics: prompts, system prompts, context length, quantization (Q4/Q8), tokens/sec

### ▶ Phase 2: Learn the toolkit
- Python venv, PyTorch with CUDA
- Hugging Face: transformers, datasets, the Hub
- Load a model in Python, tokenize text, generate output

### Phase 3: Fine-tune with LoRA / QLoRA
- Use Unsloth (easiest) or Hugging Face PEFT + TRL
- Start with a small model (1B or 3B) for fast iteration, then move to 8B
- Build a small instruction dataset (prompt/response pairs in JSONL)
- Train, then evaluate before and after on the same prompts
- Fallback: Google Colab free GPU if anything won't fit locally

### Phase 4: Bring it home
- Merge the LoRA adapter or export to GGUF
- Load the custom model into Ollama with a Modelfile
- Chat with it locally and compare against the base model

## Key concepts
- **Fine-tuning vs. training from scratch:** fine-tuning adapts an existing model; from-scratch training is out of reach for consumer hardware.
- **LoRA / QLoRA:** train small adapter weights instead of the whole model; QLoRA loads the base model in 4-bit to save VRAM.
- **RAG vs. fine-tuning:** to make a model *know* my documents, try RAG first; fine-tune to change *behavior, style, or format*.

## Status (2026-10-27)
- Repo on GitHub (`MackMusial/llama-lab`, origin/master); README added. Goal: public portfolio piece showing AI work, so record measured results.
- Ollama 0.40.11 installed. 8B Q4_K_M: 64.6 tok/s, ~5.4 GB; Q8_0: 42.5 tok/s, ~8.9 GB (both 100% GPU). Results live in `benchmarks.md`.
- Q4 vs Q8 quality compared (n=1, inconclusive; both had factual errors). All committed (4362f61).
- Laptop CPU benchmark of `llama3.2:1b` (Q8_0, 1.3 GB, 100% CPU, iGPU unused): measured 20.6 tok/s fresh, 17.0 tok/s at ~850-token context, vs predicted ~33 (ceiling 52.5). Prefill 49.7 tok/s (34-token prompt, noisy). Mack to explain the gap and add the row to `benchmarks.md`. Gap = CPU efficiency only 39% (vs GPU 63%). `llama3.2:3b` Q4_K_M: predicted 1.96 GB / ~13.6 tok/s, measured **12.48 tok/s**, 2.6 GB loaded, 100% CPU, prefill 38.0 tok/s (clean one-shot). Prediction within 8%. iGPU unused because Ollama 0.40.12 finds it via Vulkan but drops integrated GPUs by default (`OLLAMA_IGPU_ENABLE=1` to enable). iGPU-on experiment skipped (Mack's call). Cleaned laptop startup; Wavesor (Wave Browser PUP) found, uninstall + Defender scan pending.
- Quiz 2026-10-28 (6 Qs): solid on bandwidth math + stateless chat; weak on prefill vs generation, chat template/`<|eot_id|>`, evaluation (n=1, sampling). Re-quiz those before Phase 3.
- Context-length experiment already done and recorded in `benchmarks.md` (4K/16K/64K, spill to CPU).
- Laptop write-up done 2026-10-28 (Mack answered quiz-style, Claude wrote it into `benchmarks.md`; Mack hates prose writing, so this format works). 3B = leading Pristine Pools candidate, not decided: must beat/near the 8B on held-out pool prompts; distilling 8B → 3B is a Phase 3 option.
- Phase 2 step 1 done (2026-10-28, desktop): venv at `C:\Users\mackm\.venvs\llama-lab` (Python 3.12.10), torch 2.15.0+cu134, `cuda.is_available()` True. Driver 617.14 (CUDA ≤ 13.4). If Unsloth/bitsandbytes need older torch in Phase 3, rebuild the venv. Always `python -m pip`.
- `gpu_matmul.py` done (Mack wrote it, guided): CPU 233.9 ms vs GPU 7.3 ms, ~32× (predicted ~30×). Recorded in `benchmarks.md`.
- ▶ **Resume here:** matrix × vector follow-up in `gpu_matmul.py` (Mack's choice). Spec given: `v = torch.randn(4096)`, time `A @ v` ×100 in a loop and divide, CPU vs GPU (warm-up + sync rules apply). **Mack owes predictions before coding:** bytes of A (4096² × 4 B), CPU ms at ~55 GB/s, GPU ms at 504 GB/s, speedup vs the 32× matmul. Desktop = i9-11900K, 64 GB DDR4-3467 dual channel (~55 GB/s).
- Then: install numpy; transformers + tokenizer. Mack should tidy `gpu_matmul.py` (fill `speedup 30x` on line 1, add speedup print, delete scaffolding).
- Mack skips predictions unless held to them: wait for the numbers before he runs anything.

## Concepts Mack has learned
- Quantization basics (Q4 ≈ 4.5 bits/weight → 8B fits in ~5 GB); KV cache grows with context.
- Prefill (parallel, fast) vs generation (one token at a time); first-request warm-up is an outlier, not a result.
- Models are stateless: chat resends the whole history each turn; prompt cache reuses the shared prefix; context window limits memory.
- KV cache is linear in `num_ctx` and reserved up front; overflow spills layers to CPU (same accuracy, much slower — slowest stage sets speed).
- Generation is memory-bandwidth bound (tok/s ≈ bandwidth ÷ model bytes); more bits = more accuracy, less speed, more VRAM. Same parameter count.
- System prompt = text at the start of the prompt in a `system` role. Chat template flattens all turns into one string with special tokens (`<|start_header_id|>`, `<|eot_id|>`); the model just predicts tokens after an open `assistant` header. Fine-tuning data must use the same template.
- Layers: a model is a stack of identical blocks (attention + feed-forward), run in order; each keeps its own KV per token; spill moves whole layers; LoRA adapts layer weights.
- Efficiency = measured ÷ ceiling; it's per-machine (GPU ~63%, laptop CPU ~36–39%). VRAM (memory) ≠ load (work).
- PyTorch basics: tensors live on a device; math runs where the inputs are; `.to("cuda")` returns a copy. GPU ops are asynchronous (CPU queues, moves on), so timing needs `torch.cuda.synchronize()`, including after warm-up.
- Evaluating output: sampling is random (temperature), so n=1 proves nothing; prior turns contaminate answers; confident formatting ≠ correctness.
