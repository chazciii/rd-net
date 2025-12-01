RD-Net: Drift-Injected Memory for Frozen LLMs

A small inference-time modification that delays repetition collapse in large language models without fine-tuning, LoRA, or modifying weights.

This repo contains a minimal reproducible script tested on Llama-3.1-8B that shows a repeatable effect:

Vanilla model collapses after ~8k–12k tokens.
With drift-injected fast-weight memory, coherent generation continues far longer with lower rep-4 repetition.
All weights remain frozen.

This is not presented as a finished method or breakthrough — just a reproducible behavior worth investigation.

⸻

Why Its Worth Exploring

Long-form generation typically causes frozen LLMs to enter a repetition loop such as:

the king replied the king replied the king replied...

This experiment shows that lightly perturbing an auxiliary memory layer during inference:
	•	delays this collapse,
	•	increases token-level novelty,
	•	and maintains useful structure further into long sequences.

No training, no gradient updates, no KV editing — purely inference-side behavior.

⸻

How to Run
	1.	Clone the repo:

git clone https://github.com/chazciii/rd-net
cd rd-net

	2.	Install dependencies:

pip install torch transformers tqdm accelerate

	3.	Run the demo:

python rd_demo_final.py


⸻

Expected Output

The script generates two log files:

vanilla_log.txt
rdnet_log.txt

The script generates two log files:

vanilla_log.txt
rdnet_log.txt

### Example Output

<details>
<summary>Click to expand logs</summary>

```txt
Example Run (RTX 4090 • CUDA 12.1 • Llama-3.1-8B)

Vanilla (no drift applied):
10k tokens | rep-4 = 0.7421 | drift = 0.0000
20k tokens | rep-4 = 0.8923 | drift = 0.0000

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


Summary:

- The standard frozen model begins collapsing around ~20k tokens (rep-4 ≈ 0.89).
- With RD-Net drift, repetition stays low and stable (~0.28→0.34) past 150k tokens.
- No fine-tuning, training, LoRA, or KV-cache edits.

Only modification: a small Gaussian drift term applied to a frozen fast-weight memory during inference.
This shows how repetition grows in the standard model versus the drift-injected version.

⸻

What’s Going On?

The method injects small, scheduled Gaussian noise into an untrained fast-weight memory module.
The drift slowly decays over time.

Working hypothesis:
	•	The perturbation prevents the model from locking into a predictable attractor state.
	•	It acts as a weak form of entropy or “novelty stimulation.”
	•	Result: collapse is delayed.

More validation is needed.

⸻

Contributing / Replication

If you run experiments on different models (Qwen, Mistral, Phi-3, GPTQ, GGUF, etc.), please share logs or plots.

Even one-line confirmation or contradiction is useful.

⸻

Status
	•	Code reproducible
	•	Results consistent on Llama-3.1-8B
	•	Research stage: early validation

A paper and full benchmark suite will follow once replication feedback comes in.

⸻

License

MIT — open for experimentation.


Contact

Open discussion via GitHub Issues.

