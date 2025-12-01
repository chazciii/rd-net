# RD-Net: Drift-Stabilized Inference for Frozen Large Language Models

[![Status](https://img.shields.io/badge/state-experimental-orange)](#)
[![Replications Welcome](https://img.shields.io/badge/replication-open-green)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains a minimal inference-time modification that reduces or delays repetition collapse in long-form text generation on frozen large language models.

The method is simple:  
A small drift term is injected into an auxiliary fast-weight memory module during inference. No training, fine-tuning, KV cache manipulation, LoRA, or retraining is required. The underlying model remains frozen.

Preliminary results show that this drift mechanism maintains lower repetition entropy substantially longer than baseline generation under identical settings.

This is early research and requires broader replication.

---

## Abstract

Large language models often enter a repetitive attractor state during extended free-running generation, especially without conditioning or resets. This behavior emerges even with large context windows and sampling strategies like temperature, nucleus sampling, or top-k truncation.

We explore a lightweight inference-time perturbation method using a scheduled Gaussian drift applied to an untrained fast-weight memory module. Initial experiments on Llama-3.1-8B show that this approach delays repetition collapse over long generation sequences, preserving novelty beyond 100k tokens.

These findings are preliminary and require independent verification across model families, inference stacks, and settings.

---

## Why This Might Matter

- The effect occurs **without touching model weights**.  
- The method is **architecture-agnostic** (tested only on Llama-3.1 so far).  
- It may offer a lightweight mitigation for collapse modes in:
  - agent loops  
  - long-context narrative models  
  - streaming or infinite-generation systems  

Whether this scales or generalizes remains an open question.

---

## Installation

```bash
git clone https://github.com/chazciii/rd-net
cd rd-net
pip install torch transformers accelerate tqdm
```

---

## Run the Experiment

```bash
python rd_demo_final.py
```

The script generates two log files:

- `vanilla_log.txt`
- `rdnet_log.txt`

Both use identical sampling settings and context constraints. The only difference is whether drift is applied.

---

## Example Output

<details>
<summary>Click to expand logs</summary>

```txt
Example Run (RTX 4090 • CUDA 12.1 • Llama-3.1-8B)

Vanilla (no drift applied):

10k tokens  | rep-4 = 0.7421 | drift = 0.0000
20k tokens  | rep-4 = 0.8923 | drift = 0.0000

RD-Net (drift applied):

10k tokens  | rep-4 = 0.2814 | drift = 0.1123
20k tokens  | rep-4 = 0.2931 | drift = 0.0987
30k tokens  | rep-4 = 0.3012 | drift = 0.0876
40k tokens  | rep-4 = 0.3120 | drift = 0.0791
50k tokens  | rep-4 = 0.3198 | drift = 0.0723
60k tokens  | rep-4 = 0.3245 | drift = 0.0668
70k tokens  | rep-4 = 0.3291 | drift = 0.0621
80k tokens  | rep-4 = 0.3317 | drift = 0.0582
90k tokens  | rep-4 = 0.3340 | drift = 0.0549
100k tokens | rep-4 = 0.3356 | drift = 0.0520
110k tokens | rep-4 = 0.3369 | drift = 0.0495
120k tokens | rep-4 = 0.3378 | drift = 0.0473
130k tokens | rep-4 = 0.3385 | drift = 0.0454
140k tokens | rep-4 = 0.3391 | drift = 0.0437
150k tokens | rep-4 = 0.3396 | drift = 0.0422
```

</details>

---

## Summary of Results

| Condition                    | First Collapse Point | Approx. Final rep-4 |
|-----------------------------|----------------------|---------------------|
| Vanilla (frozen model)      | ~20–24k tokens       | ~0.89               |
| RD-Net Drift (frozen model) | >150k tokens         | ~0.34 (stable)      |

These values vary by run and hardware, and should not be treated as benchmarks.

---

## Limitations / Caveats

- Results currently rely on a **single hardware setup** and **single model family**.  
- No evaluation yet on coherence, semantics, or downstream task performance.  
- Effect may depend on sampling configuration.  
- Drift parameters are heuristic and unoptimized.  
- Unknown behavior on quantized models (GPTQ/GGUF).

---

## Replication Requests

If you test this on:

- Qwen  
- Mistral  
- Falcon  
- Phi-3  
- GGUF / GPTQ  
- CPU-only inference  
- Agent frameworks  

…please submit logs or open an issue. Positive, negative, or neutral results are all useful.

---

## Citation

```bibtex
@misc{cook2025rdnet,
  title={RD-Net: Drift-Stabilized Inference for Frozen LLMs},
  author={Cook, Chaz},
  year={2025},
  url={https://github.com/chazciii/rd-net},
  note={Preprint, work in progress}
}
```

---

## License

MIT License.

Replication and pull requests welcome.