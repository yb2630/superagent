"""Frozen-encoder multi-label head: one linear layer trained with BCE plus L2.

The head maps a 1024-dimensional frozen Qwen3 embedding to five category logits.  Each
label has an independent sigmoid and the loss is binary cross-entropy plus an L2 penalty on
the weights (bias terms are not penalized).  Training is full-batch AdamW on CPU with one
thread and a fixed seed, so a fit is reproducible for the same features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import HEAD_EPOCHS, HEAD_LEARNING_RATE, LABELS


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))


def _logit(probability: float, eps: float = 1e-6) -> float:
    probability = min(max(float(probability), eps), 1.0 - eps)
    return math.log(probability / (1.0 - probability))


@dataclass
class LinearHead:
    labels: tuple[str, ...]
    weights: np.ndarray  # (n_labels, dim)
    bias: np.ndarray  # (n_labels,)
    mean: np.ndarray  # (dim,)
    scale: np.ndarray  # (dim,)
    hyperparameters: dict

    def logits(self, X: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.weights.T + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(X))

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "hyperparameters": self.hyperparameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LinearHead:
        return cls(
            labels=tuple(data["labels"]),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            hyperparameters=dict(data.get("hyperparameters") or {}),
        )


def fit_linear_head(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    labels: tuple[str, ...] = LABELS,
    seed: int = 20261121,
    epochs: int = HEAD_EPOCHS,
    learning_rate: float = HEAD_LEARNING_RATE,
    l2: float = 0.1,
) -> LinearHead:
    import torch

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"bad shapes X={X.shape} Y={Y.shape}")
    if X.shape[1] == 0 or X.shape[0] == 0:
        raise ValueError("no features or examples")
    if Y.shape[1] != len(labels):
        raise ValueError(f"Y has {Y.shape[1]} columns for {len(labels)} labels")
    n, dimension = X.shape
    n_labels = Y.shape[1]

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)

    # One CPU thread keeps reductions and the fitted weights reproducible.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = torch.nn.Linear(dimension, n_labels)
    with torch.no_grad():
        base = np.clip(Y.mean(axis=0), 1e-3, 1.0 - 1e-3)
        model.bias.copy_(torch.tensor(np.log(base / (1.0 - base)), dtype=torch.float32))
        model.weight.zero_()

    constant = [bool(Y[:, index].min() == Y[:, index].max()) for index in range(n_labels)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    features = torch.tensor((X - mean) / scale, dtype=torch.float32)
    targets = torch.tensor(Y, dtype=torch.float32)
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        if l2:
            loss = loss + 0.5 * float(l2) * (model.weight * model.weight).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights = model.weight.detach().cpu().numpy().astype(np.float64).copy()
        bias = model.bias.detach().cpu().numpy().astype(np.float64).copy()

    for index, is_constant in enumerate(constant):
        if is_constant:
            weights[index] = 0.0
            positives = int(Y[:, index].sum())
            bias[index] = _logit((positives + 1.0) / (n + 2.0))

    return LinearHead(
        labels=tuple(labels),
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
        hyperparameters={
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "loss": "binary_cross_entropy_with_logits + 0.5*l2*||W||^2",
            "optimizer": "AdamW(weight_decay=0)",
            "constant_labels": [labels[index] for index in range(n_labels) if constant[index]],
        },
    )


def head_predict(head: LinearHead, X: np.ndarray) -> np.ndarray:
    return head.predict_proba(np.asarray(X, dtype=np.float64))


__all__ = ["LinearHead", "fit_linear_head", "head_predict", "sigmoid"]
