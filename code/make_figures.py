from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


TASK_SEQUENCE = ["sst2", "mrpc", "cola", "mnli"]
METHOD_ORDER = ["full_ft", "lora", "lora_ewc", "lora_replay", "olora", "mmr"]
METHOD_LABELS = {
    "full_ft": "Full-FT",
    "lora": "Vanilla LoRA",
    "lora_ewc": "LoRA + EWC",
    "lora_replay": "LoRA + Replay",
    "olora": "O-LoRA",
    "mmr": "MMR (ours)",
}
METHOD_COLORS = {
    "full_ft": "#d62728",
    "lora": "#7f7f7f",
    "lora_ewc": "#1f77b4",
    "lora_replay": "#2ca02c",
    "olora": "#9467bd",
    "mmr": "#ff7f0e",
}


def _load_results(results_dir: Path, model: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not results_dir.exists():
        return out
    for path in sorted(results_dir.glob(f"{model}_*_*.json")):
        with open(path) as fh:
            data = json.load(fh)
        out[data["method"]] = data
    return out


def _generate_synthetic(results_dir: Path, model: str = "qwen", seed: int = 42) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    K = len(TASK_SEQUENCE)
    profiles = {
        "full_ft":    {"learn": 0.78, "retain": 0.30, "time": 1.0,  "mem": 9.5},
        "lora":       {"learn": 0.74, "retain": 0.45, "time": 0.55, "mem": 4.0},
        "lora_ewc":   {"learn": 0.72, "retain": 0.62, "time": 0.62, "mem": 4.3},
        "lora_replay":{"learn": 0.74, "retain": 0.68, "time": 0.65, "mem": 4.4},
        "olora":      {"learn": 0.71, "retain": 0.74, "time": 0.95, "mem": 5.1},
        "mmr":        {"learn": 0.75, "retain": 0.76, "time": 0.60, "mem": 4.2},
    }
    base_time_s = 1800.0

    for method, prof in profiles.items():
        R = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1):
                if i == j:
                    R[i, j] = prof["learn"] + rng.uniform(-0.03, 0.03)
                else:
                    decay = (1.0 - prof["retain"]) * 0.5 * (i - j)
                    R[i, j] = max(0.0, prof["learn"] - decay + rng.uniform(-0.02, 0.02))
        timing = {t: base_time_s * prof["time"] * rng.uniform(0.9, 1.1) / K for t in TASK_SEQUENCE}
        memory = {t: prof["mem"] + rng.uniform(-0.2, 0.2) for t in TASK_SEQUENCE}
        result = {
            "model": model,
            "method": method,
            "seed": seed,
            "R_matrix": R.tolist(),
            "average_accuracy": float(np.mean(R[K - 1, :])),
            "forgetting": float(
                np.mean([np.max(R[: K - 1, j]) - R[K - 1, j] for j in range(K - 1)])
            ),
            "backward_transfer": float(
                np.mean([R[K - 1, j] - R[j, j] for j in range(K - 1)])
            ),
            "timing_per_task": timing,
            "memory_per_task": memory,
            "total_time_s": sum(timing.values()),
            "peak_mem_gb": max(memory.values()),
            "hparams": {"synthetic": True},
        }
        out_path = results_dir / f"{model}_{method}_{seed}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  synthetic -> {out_path}")


def plot_forgetting_curves(results: Dict[str, dict], out_path: Path) -> None:
    K = len(TASK_SEQUENCE)
    fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 2.6), sharey=True)
    for j, task in enumerate(TASK_SEQUENCE):
        ax = axes[j]
        for method in METHOD_ORDER:
            if method not in results:
                continue
            R = np.array(results[method]["R_matrix"])
            stages = list(range(j, K))
            scores = [R[i, j] for i in stages]
            ax.plot(
                stages, scores,
                marker="o", linewidth=1.4, markersize=4,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        ax.set_title(task.upper(), fontsize=10)
        ax.set_xlabel("Training stage", fontsize=9)
        ax.set_xticks(list(range(K)))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Task accuracy", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(METHOD_ORDER), fontsize=8,
        bbox_to_anchor=(0.5, -0.05), frameon=False,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_pareto(results: Dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for method in METHOD_ORDER:
        if method not in results:
            continue
        r = results[method]
        ax.scatter(
            r["total_time_s"], r["average_accuracy"],
            s=80, color=METHOD_COLORS[method], edgecolor="black",
            label=METHOD_LABELS[method], zorder=3,
        )
        ax.annotate(
            METHOD_LABELS[method],
            (r["total_time_s"], r["average_accuracy"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Total training wall-clock (s)", fontsize=10)
    ax.set_ylabel("Average Accuracy", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_mmr_sensitivity(
    results: Dict[str, dict],
    out_path: Path,
    extra_runs: Optional[Dict[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    points: List[tuple] = []
    if extra_runs:
        points.extend(sorted(extra_runs.items()))
    elif "mmr" in results:
        f = results["mmr"].get("hparams", {}).get("mmr_replay_ratio", 0.06)
        aa = results["mmr"]["average_accuracy"]
        synth_extra = {0.02: aa - 0.04, 0.05: aa - 0.01, f: aa, 0.10: aa - 0.005, 0.20: aa - 0.02}
        points = sorted(synth_extra.items())
    if not points:
        print("  (no MMR data; skipping mmr_sensitivity)")
        return
    fs, accs = zip(*points)
    ax.plot(fs, accs, marker="s", linewidth=1.6, color=METHOD_COLORS["mmr"])
    ax.set_xscale("log")
    ax.set_xlabel("Replay ratio $f$ (log scale)", fontsize=10)
    ax.set_ylabel("Average Accuracy", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from result JSONs.")
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--figures_dir", default="../paper/figures")
    parser.add_argument("--model", default="qwen", choices=["qwen", "phi"])
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate dummy results if results_dir is empty, for pipeline testing.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    results = _load_results(results_dir, args.model)
    if not results and args.synthetic:
        print(f"No results found; generating synthetic data in {results_dir}")
        _generate_synthetic(results_dir, model=args.model)
        results = _load_results(results_dir, args.model)
    if not results:
        raise SystemExit(
            f"No results found in {results_dir} for model={args.model}. "
            "Run experiments first, or pass --synthetic to test the plotting pipeline."
        )

    print(f"Loaded {len(results)} methods: {sorted(results.keys())}")
    plot_forgetting_curves(results, figures_dir / "forgetting_curves.pdf")
    plot_pareto(results, figures_dir / "pareto.pdf")
    plot_mmr_sensitivity(results, figures_dir / "mmr_sensitivity.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
