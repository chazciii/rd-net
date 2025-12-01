RD-Net: Drift-Injected Memory for Frozen LLMs

A small inference-time modification that delays repetition collapse in large language models without fine-tuning, LoRA, or modifying weights.

This repo contains a minimal reproducible script tested on Llama-3.1-8B that shows a repeatable effect:

Vanilla model collapses after ~8k–12k tokens.
With drift-injected fast-weight memory, coherent generation continues far longer with lower rep-4 repetition.
All weights remain frozen.

This is not presented as a finished method or breakthrough — just a reproducible behavior worth investigation.