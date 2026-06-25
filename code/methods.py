from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset


def get_base_model(
    model_name: str,
    quantize: bool = True,
) -> Tuple[nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> nn.Module:
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


class EWCRegularizer:
    def __init__(self) -> None:
        self._fishers: List[Dict[str, torch.Tensor]] = []
        self._anchors: List[Dict[str, torch.Tensor]] = []

    def compute_fisher(
        self,
        model: nn.Module,
        tokenized_batches: List[Dict[str, torch.Tensor]],
        device: str = "cuda",
        n_batches: int = 50,
    ) -> None:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        anchor: Dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        processed = 0
        for batch in tokenized_batches:
            if processed >= n_batches:
                break
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(**batch_gpu)
            outputs.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # F_i ≈ E[(∂ log p / ∂θ_i)^2]: squared gradient as diagonal Fisher estimate
                    fisher[n] += p.grad.detach().pow(2)
            processed += 1

        for key in fisher:
            fisher[key] /= max(1, processed)

        self._fishers.append(fisher)
        self._anchors.append(anchor)

    def penalty(self, model: nn.Module, lam: float = 1.0) -> torch.Tensor:
        if not self._fishers:
            return torch.tensor(0.0)
        device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cpu"),
        )
        loss = torch.tensor(0.0, device=device)
        for fisher, anchor in zip(self._fishers, self._anchors):
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    loss = loss + (fisher[n].to(device) * (p - anchor[n].to(device)).pow(2)).sum()
        return (lam / 2.0) * loss


class ReplayBuffer:
    def __init__(self) -> None:
        self._tasks: List[Dataset] = []

    def add_task(self, dataset: Dataset) -> None:
        self._tasks.append(dataset)

    def __len__(self) -> int:
        return sum(len(d) for d in self._tasks)

    def sample(self, n: int) -> List[dict]:
        if not self._tasks:
            return []
        pool: List[dict] = [ex for ds in self._tasks for ex in ds.to_list()]
        if len(pool) <= n:
            return pool
        return random.sample(pool, n)


class OLoRAHelper:
    def __init__(self) -> None:
        self._frozen_A: List[Dict[str, torch.Tensor]] = []

    def snapshot_lora_a(self, model: nn.Module) -> None:
        snapshot: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora_A"):
                for adapter_key, linear in module.lora_A.items():
                    snapshot[f"{name}.lora_A.{adapter_key}"] = linear.weight.detach().clone()
        self._frozen_A.append(snapshot)

    def orthogonality_loss(self, model: nn.Module, mu: float = 0.1) -> torch.Tensor:
        if not self._frozen_A:
            return torch.tensor(0.0)
        device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cpu"),
        )
        current_A: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora_A"):
                for adapter_key, linear in module.lora_A.items():
                    current_A[f"{name}.lora_A.{adapter_key}"] = linear.weight

        loss = torch.tensor(0.0, device=device)
        for frozen_snapshot in self._frozen_A:
            for param_name, A_curr in current_A.items():
                if param_name in frozen_snapshot:
                    A_prev = frozen_snapshot[param_name].to(device)
                    loss = loss + (A_curr.t() @ A_prev).norm(p="fro").pow(2)
        return mu * loss


class MMRTrainer:
    def __init__(self, replay_ratio: float = 0.06) -> None:
        self.replay_ratio = replay_ratio
        self.buffer = ReplayBuffer()

    def add_task_to_buffer(self, dataset: Dataset) -> None:
        self.buffer.add_task(dataset)

    def mix_batch(self, current_batch: List[dict]) -> List[dict]:
        if not self.buffer._tasks:
            return current_batch
        n_replay = max(1, int(self.replay_ratio * len(current_batch)))
        n_curr = len(current_batch) - n_replay
        return current_batch[:n_curr] + self.buffer.sample(n_replay)
