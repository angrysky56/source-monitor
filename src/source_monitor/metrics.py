"""Small dependency-free metrics: AUROC (rank-based, tie-aware) + selective accuracy."""

from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray | list[float], labels: np.ndarray | list[int]) -> float:
    """
    Area under the ROC curve via the rank-sum (Mann-Whitney) identity, with
    average ranks for ties. labels: 1 = positive class. Returns nan if only
    one class is present.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selective_acc(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the `coverage` fraction of examples with highest confidence."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = max(1, int(round(coverage * len(conf))))
    keep = np.argsort(-conf, kind="mergesort")[:n]
    return float(correct[keep].mean())
