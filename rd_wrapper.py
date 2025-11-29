# rd_wrapper.py — Lightweight wrapper for drifted inference
# Plug-and-play version for running RD-Net behavior on any Hugging Face model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math


class DriftScheduler:
    """Simple drift decay schedule. Controls how much noise is applied."""
    def __init__(self, initial=0.12, final=0.006, decay_steps=200_000):
        self.initial = initial
        self.final = final
        self.decay_steps = decay_steps
        self.step = 0

    def reset(self):
        self.step = 0

    def get(self):
        """Returns current drift level, decaying over time."""
        self.step += 1
        p = min(self.step / self.decay_steps, 1.0)
        return self.final + (self.initial - self.final) * max(0.0, 1.0 - math.log10(1 + 9 * p))


class FastWeightMemory(nn.Module):
    """Random fast-weight memory with drifted Gaussian noise applied to keys."""

    def __init__(self, dim, size=2048):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(size, dim) * 0.02, requires_grad=False)
        self.values = nn.Parameter(torch.randn(size, dim) * 0.02, requires_grad=False)

    def forward(self, query, drift=0.0):
        k = self.keys
        if drift > 0:
            k = k + torch.randn_like(k) * drift  # core drift operation

        k_norm = F.normalize(k, dim=-1)
        q_norm = F.normalize(query, dim=-1)

        sim = torch.matmul(k_norm, q_norm.t())
        attn = F.softmax(sim / 0.1, dim=0)

        return torch.matmul(attn.t(), self.values)


class RDWrapper:
    """Drop-in wrapper for inference with drifted fast-weight memory."""

    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B", drift_strength=0.12):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.model.config.output_hidden_states = True

        dim = self.model.config.hidden_size
        self.memory = FastWeightMemory(dim).to(self.model.device)
        self.scheduler = DriftScheduler(initial=drift_strength)
        self.proj = nn.Linear(dim, self.model.config.vocab_size, bias=False).to(self.model.device)
        self.proj.weight.data.normal_(mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=2048, temperature=0.8, top_p=0.95):
        """Runs generation with drift active. Intended for long-form or agent-style loops."""

        self.scheduler.reset()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = inputs.input_ids

        for _ in range(max_new_tokens):
            drift = self.scheduler.get()

            outputs = self.model(
                generated,
                use_cache=True,
                output_hidden_states=True,
            )

            last_hidden = outputs.hidden_states[-1][:, -1]
            drift_vec = self.memory(last_hidden, drift)
            drift_logits = self.proj(drift_vec)

            logits = outputs.logits[:, -1, :] + drift_logits
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


# Example use:
# rd = RDWrapper("meta-llama/Meta-Llama-3.1-8B")
# print(rd.generate("The future of AI is", max_new_tokens=500))
