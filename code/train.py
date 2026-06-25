from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data_loader import load_task, build_replay_buffer, TASK_SEQUENCE, LABEL_MAPS
from methods import (
    attach_lora,
    EWCRegularizer,
    ReplayBuffer,
    OLoRAHelper,
    MMRTrainer,
)
from evaluate import task_metric


def _tokenize_batch(
    batch: List[dict],
    tokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    full_texts = [ex["prompt"] + " " + ex["target"] for ex in batch]
    enc = tokenizer(
        full_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _get_optimizer(model: nn.Module, lr: float = 2e-4):
    try:
        from bitsandbytes.optim import PagedAdamW8bit
        return PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    except ImportError:
        return AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )


def evaluate_task(
    model: nn.Module,
    tokenizer,
    task_name: str,
    n_eval: int = 500,
    max_length: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> float:
    inv_label_map = {v: k for k, v in LABEL_MAPS[task_name].items()}
    val_ds = load_task(task_name, split="validation", n_examples=n_eval)

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for i in range(0, len(val_ds), batch_size):
        batch = val_ds[i: i + batch_size]
        prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
        true_labels = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]

        inputs = tokenizer(
            prompts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for j, out_ids in enumerate(output_ids):
            gen_text = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True).strip().lower()
            pred = next(
                (idx for label_str, idx in inv_label_map.items() if label_str in gen_text),
                0,
            )
            all_preds.append(pred)
            all_labels.append(int(true_labels[j]))

    return task_metric(all_preds, all_labels, task_name)


def train_on_task(
    model: nn.Module,
    tokenizer,
    task_name: str,
    method: str,
    hparams: Dict[str, Any],
    prior_state: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = hparams.get("lr", 2e-4)
    batch_size = hparams.get("batch_size", 8)
    max_seq_len = hparams.get("max_seq_len", 256)
    epochs = hparams.get("epochs", 2)
    n_train = hparams.get("n_train", None)
    ewc_lambda = hparams.get("ewc_lambda", 500.0)
    replay_buffer_size = hparams.get("replay_buffer_size", 200)
    olora_mu = hparams.get("olora_mu", 0.1)
    mmr_replay_ratio = hparams.get("mmr_replay_ratio", 0.06)

    if prior_state is None:
        prior_state = {}

    train_ds = load_task(task_name, split="train", n_examples=n_train)
    train_list = train_ds.to_list()

    optimizer = _get_optimizer(model, lr=lr)
    total_steps = max(1, (len(train_list) // batch_size) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    ewc: Optional[EWCRegularizer] = prior_state.get("ewc")
    olora_helper: Optional[OLoRAHelper] = prior_state.get("olora_helper")
    mmr_trainer: Optional[MMRTrainer] = prior_state.get("mmr_trainer")
    replay_buf: Optional[ReplayBuffer] = prior_state.get("replay_buffer")

    if method == "lora_ewc" and ewc is None:
        ewc = EWCRegularizer()
    if method == "olora" and olora_helper is None:
        olora_helper = OLoRAHelper()
    if method == "mmr" and mmr_trainer is None:
        mmr_trainer = MMRTrainer(replay_ratio=mmr_replay_ratio)
    if method == "lora_replay" and replay_buf is None:
        replay_buf = ReplayBuffer()

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    model.train()
    total_loss = 0.0
    steps = 0

    for _epoch in range(epochs):
        random.shuffle(train_list)
        for i in range(0, len(train_list), batch_size):
            current_batch = train_list[i: i + batch_size]
            if not current_batch:
                continue

            if method == "lora_replay" and replay_buf is not None:
                n_replay = max(1, int(0.06 * len(current_batch)))
                replayed = replay_buf.sample(n_replay)
                if replayed:
                    current_batch = current_batch[: len(current_batch) - n_replay] + replayed
            elif method == "mmr" and mmr_trainer is not None:
                current_batch = mmr_trainer.mix_batch(current_batch)

            tensors = _tokenize_batch(current_batch, tokenizer, max_seq_len, device)
            optimizer.zero_grad()
            outputs = model(**tensors)
            loss = outputs.loss

            if method == "lora_ewc" and ewc is not None:
                loss = loss + ewc.penalty(model, lam=ewc_lambda)
            elif method == "olora" and olora_helper is not None:
                loss = loss + olora_helper.orthogonality_loss(model, mu=olora_mu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            steps += 1

    wall_time = time.time() - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    if method == "lora_ewc" and ewc is not None:
        fisher_batches = [
            _tokenize_batch([ex], tokenizer, max_seq_len, device="cpu")
            for ex in train_list[:200]
        ]
        ewc.compute_fisher(model, fisher_batches, device=device, n_batches=50)
        prior_state["ewc"] = ewc

    if method == "lora_replay" and replay_buf is not None:
        replay_buf.add_task(build_replay_buffer(task_name, n_per_task=replay_buffer_size))
        prior_state["replay_buffer"] = replay_buf

    if method == "olora" and olora_helper is not None:
        olora_helper.snapshot_lora_a(model)
        prior_state["olora_helper"] = olora_helper

    if method == "mmr" and mmr_trainer is not None:
        mmr_trainer.add_task_to_buffer(build_replay_buffer(task_name, n_per_task=replay_buffer_size))
        prior_state["mmr_trainer"] = mmr_trainer

    eval_score = evaluate_task(model, tokenizer, task_name, device=device)

    metrics = {
        "train_loss": total_loss / max(1, steps),
        "eval_metric": eval_score,
        "wall_time_s": wall_time,
        "peak_mem_gb": peak_mem_gb,
    }
    return model, metrics, prior_state
