# RD-NET: Inference-Time Drift Reduces Collapse in Frozen LLMs

RD-NET is a simple inference-time modification for frozen large language models that reduces repetition collapse using scheduled stochastic drift and fast-weight memory.

This repo contains a minimal reproducible experiment showing that adding a small drifted fast-weight memory to a **frozen** Llama-3.1-8B model reduces repetition collapse during long-form text generation.

No training.  
No LoRA.  
No fine-tuning.  
Weights stay frozen.

The only thing changing is a small Gaussian drift term applied to an untrained fast-weight memory at inference time.

---

## What the demo does

- Loads `meta-llama/Meta-Llama-3.1-8B` from Hugging Face in eval mode  
- Attaches a random fast-weight memory in hidden space (keys/values are fixed, untrained)  
- At each generation step:
  - Reads the last hidden state
  - Computes a memory read with small Gaussian drift on the keys
  - Projects this into vocab space and adds it to the logits
- Compares long-form sampling **with** and **without** this drift  
- Logs a simple char-level rep-4 score over the last 50k characters

It is a **sampling experiment**, not a training method.

---

## Files

- `rd_demo_final.py`  
  Minimal, self-contained script that:
  - runs vanilla vs RD-Net generation
  - writes `vanilla_log.txt` and `rdnet_log.txt`
  - prints:  
    `k tokens | rep-4 = <score> | drift = <value>`

- `rd_wrapper.py`  
  Lightweight wrapper to use the same drift mechanism as a plug-in around any Hugging Face causal LM.

---

## Running the experiment

You need Python, `torch`, and `transformers`.

```bash
pip install torch transformers
python rd_demo_final.pyThis will:
	•	Run a 20k-token vanilla generation (no drift)
	•	Run a 150k-token RD-Net generation (with drift)
	•	Produce:
	•	vanilla_log.txt
	•	rdnet_log.txt

Each log line looks like:10k tokens | rep-4 = 0.4321 | drift = 0.0720Typical behavior I’ve observed:
	•	Vanilla: rep-4 climbs quickly as the model falls into loops / copy-paste patterns
	•	RD-Net: rep-4 stays lower for much longer with the same base model, same prompt, same seed

If your results disagree, please open an issue.

⸻

What this is NOT
	•	Not a claim of improved reasoning or accuracy
	•	Not reinforcement learning
	•	Not fine-tuning or weight updates
	•	Not a benchmark win

This is a single, specific empirical observation:

Inference-time drift via a small fast-weight memory changes long-form collapse behavior in a frozen model.

---

## Why This Matters

Large language models often suffer **repetition collapse**, especially when generating tens of thousands of tokens from a single prompt.  
Once the model falls into a repeated n-gram attractor, it rarely escapes without external intervention.

Most attempts to solve this involve:

- Temperature tricks  
- Penalizing repetition  
- Reinforcement-based rewriting  
- Additional training, LoRA, or adapters  

**RD-NET takes a different approach:**  
It perturbs internal retrieval pathways during inference using scheduled stochastic drift applied to a fast-weight memory projection. The base model weights remain untouched.

---

## Method Summary

RD-NET introduces three components:

| Component | Description |
|----------|-------------|
| **Fast-weight memory** | A random (untrained) K/V matrix the model can consult alongside its activations |
| **Stochastic drift** | Small Gaussian noise added to memory keys each step |
| **Decay schedule** | Drift starts stronger and gradually reduces to a floor |

This nudges the model away from local minima while keeping it coherent.

---

## Key Properties

- No training
- No LoRA or adapters
- No fine-tuning
- Works on frozen models
- Works at inference time only
- Compatible with any HuggingFace LLM

---

## Results Snapshot (Early)

| Model | Max Tokens Before Collapse | Rep-4 Trend | Notes |
|-------|----------------------------|-------------|-------|
| Llama-3.1-8B (Vanilla) | ~10k–20k | rises quickly to ~0.80 | collapses to loops |
| Llama-3.1-8B + RD-NET | 100k–150k tested | stays <0.20–0.30 significantly longer | still degrades eventually, but more slowly |

Logs are included in the repo.

---

## Run It
⸻

In short: this isn’t a benchmark win — it’s a reproducible phenomenon that deserves investigation.

Status / TODO
	•	Minimal reproducible script on Llama-3.1-8B
	•	Test on other model families (Mistral, Qwen, etc.)
	•	Token-level metrics and entropy curves
	•	Ablations over drift schedule / memory size
	•	Proper write-up with plots

⸻

License

MIT. Use it, break it, improve it. Attribution appreciated.
