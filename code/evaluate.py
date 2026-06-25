from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def average_accuracy(R: np.ndarray) -> float:
    K = R.shape[0]
    return float(np.mean(R[K - 1, :]))


def forgetting(R: np.ndarray) -> float:
    K = R.shape[0]
    if K < 2:
        return 0.0
    total = sum(
        float(np.max(R[: K - 1, j])) - R[K - 1, j]
        for j in range(K - 1)
    )
    return total / (K - 1)


def backward_transfer(R: np.ndarray) -> float:
    K = R.shape[0]
    if K < 2:
        return 0.0
    total = sum(R[K - 1, j] - R[j, j] for j in range(K - 1))
    return total / (K - 1)


def task_metric(predictions: List[int], labels: List[int], task: str) -> float:
    if task in ("sst2", "mnli"):
        return float(accuracy_score(labels, predictions))
    if task == "mrpc":
        return float(f1_score(labels, predictions, average="binary"))
    if task == "cola":
        return float(matthews_corrcoef(labels, predictions))
    raise ValueError(f"Unknown task {task!r}")
