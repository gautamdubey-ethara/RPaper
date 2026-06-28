from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data_loader import TASK_SEQUENCE
from methods import get_base_model, attach_lora
from train import train_on_task, evaluate_task
from evaluate import average_accuracy, forgetting, backward_transfer

DEFAULT_CONFIG: Dict = {
    "lr": 2e-4,
    "batch_size": 8,
    "max_seq_len": 256,
    "epochs": 2,
    "n_train": 2000,
    "ewc_lambda": 500.0,
    "replay_buffer_size": 200,
    "olora_mu": 0.1,
    "mmr_replay_ratio": 0.06,
    "quantize": True,
}

MODEL_IDS: Dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}

VALID_METHODS = {"full_ft", "lora", "lora_ewc", "lora_replay", "olora", "mmr"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    model_key: str,
    method: str,
    seed: int,
    hparams: Dict,
    output_dir: Path,
) -> None:
    print(f"\n{'='*60}\n  Model={model_key}  Method={method}  Seed={seed}\n{'='*60}")
    _set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = hparams["quantize"] and method != "full_ft"
    model, tokenizer = get_base_model(MODEL_IDS[model_key], quantize=quantize)
    if method != "full_ft":
        model = attach_lora(model)
    else:
        model.gradient_checkpointing_enable()
        hparams = dict(hparams)
        hparams["batch_size"] = 2

    K = len(TASK_SEQUENCE)
    R = np.zeros((K, K))
    timing: Dict[str, float] = {}
    memory: Dict[str, float] = {}
    prior_state: Dict = {}

    for i, task_name in enumerate(TASK_SEQUENCE):
        print(f"  [{i+1}/{K}] Training on {task_name} ...")
        model, metrics, prior_state = train_on_task(
            model, tokenizer, task_name, method, hparams, prior_state
        )
        timing[task_name] = metrics["wall_time_s"]
        memory[task_name] = metrics["peak_mem_gb"]
        print(
            f"    loss={metrics['train_loss']:.4f}  "
            f"eval={metrics['eval_metric']:.4f}  "
            f"time={metrics['wall_time_s']:.1f}s  "
            f"mem={metrics['peak_mem_gb']:.2f}GB"
        )

        for j in range(i + 1):
            score = evaluate_task(model, tokenizer, TASK_SEQUENCE[j], device=device)
            R[i, j] = score
            print(f"    R[{i},{j}] ({TASK_SEQUENCE[j]}) = {score:.4f}")

    aa = average_accuracy(R)
    f_score = forgetting(R)
    bwt = backward_transfer(R)
    total_time = sum(timing.values())
    peak_mem = max(memory.values())

    print(f"\n  AA={aa:.4f}  F={f_score:.4f}  BWT={bwt:.4f}  "
          f"total_time={total_time:.1f}s  peak_mem={peak_mem:.2f}GB")

    result = {
        "model": model_key,
        "method": method,
        "seed": seed,
        "R_matrix": R.tolist(),
        "average_accuracy": aa,
        "forgetting": f_score,
        "backward_transfer": bwt,
        "timing_per_task": timing,
        "memory_per_task": memory,
        "total_time_s": total_time,
        "peak_mem_gb": peak_mem,
        "hparams": hparams,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_{method}_{seed}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved → {out_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run continual fine-tuning experiments.")
    parser.add_argument("--model", default="qwen", choices=list(MODEL_IDS))
    parser.add_argument("--methods", default=",".join(sorted(VALID_METHODS)))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n_train", type=int, default=DEFAULT_CONFIG["n_train"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)

    unknown = set(methods) - VALID_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Choose from {VALID_METHODS}.")

    hparams = dict(DEFAULT_CONFIG)
    hparams["n_train"] = args.n_train
    hparams["epochs"] = args.epochs

    for method in methods:
        for seed in seeds:
            out_path = output_dir / f"{args.model}_{method}_{seed}.json"
            if out_path.exists():
                print(f"  Skipping {out_path} (already exists)")
                continue
            run_single_experiment(args.model, method, seed, hparams, output_dir)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


if __name__ == "__main__":
    main()
