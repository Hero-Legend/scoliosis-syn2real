"""Functions extracted without body changes from the existing training implementation."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from collections import Counter


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def amp_optimizer_step_succeeded(scale_before: float, scale_after: float) -> bool:
    """Whether GradScaler executed rather than skipped an optimizer update."""

    if not math.isfinite(scale_before) or not math.isfinite(scale_after):
        raise ValueError("AMP scales must be finite")
    if scale_before <= 0 or scale_after <= 0:
        raise ValueError("AMP scales must be positive")
    return scale_after >= scale_before


def ordinal_targets(grades: np.ndarray) -> np.ndarray:
    grades = np.asarray(grades, dtype=np.int64)
    if grades.ndim != 1 or np.any((grades < 0) | (grades > 2)):
        raise ValueError("grades must be a one-dimensional array in {0,1,2}")
    return np.stack([grades > 0, grades > 1], axis=1).astype(np.float32)


def decode_ordinal_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("ordinal probabilities must have shape [N,2]")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("ordinal probabilities must be finite and lie in [0,1]")
    if np.any(probabilities[:, 0] + 1e-12 < probabilities[:, 1]):
        raise ValueError("ordinal probabilities violate cumulative monotonicity")
    return (probabilities > 0.5).sum(axis=1).astype(np.int64)


def confusion_matrix_3(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth and prediction must be aligned vectors")
    if np.any((truth < 0) | (truth > 2) | (prediction < 0) | (prediction > 2)):
        raise ValueError("grades must lie in {0,1,2}")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def quadratic_weighted_kappa(truth: np.ndarray, prediction: np.ndarray) -> float:
    matrix = confusion_matrix_3(truth, prediction).astype(np.float64)
    n = matrix.sum()
    if n <= 0:
        raise ValueError("empty evaluation")
    truth_hist = matrix.sum(axis=1)
    pred_hist = matrix.sum(axis=0)
    expected = np.outer(truth_hist, pred_hist) / n
    weights = np.fromfunction(lambda i, j: ((i - j) / 2.0) ** 2, (3, 3), dtype=float)
    denominator = float((weights * expected).sum())
    numerator = float((weights * matrix).sum())
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    return float(1.0 - numerator / denominator)


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    matrix = confusion_matrix_3(truth, prediction)
    recalls = []
    f1s = []
    for grade in range(3):
        tp = int(matrix[grade, grade])
        fn = int(matrix[grade, :].sum() - tp)
        fp = int(matrix[:, grade].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    errors = np.abs(prediction - truth)
    return {
        "qwk": quadratic_weighted_kappa(truth, prediction),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "grade_mae": float(np.mean(errors)),
        "severe_error_rate": float(np.mean(errors >= 2)),
        "per_grade_recall": recalls,
        "confusion_matrix": matrix.tolist(),
        "case_count": int(len(truth)),
    }


def expected_microsteps(job: dict, epochs: int, micro_batch_size: int) -> int:
    if job["job_type"] == "SOURCE_PRETRAIN":
        per_epoch = math.ceil(int(job["source_cases"]) / micro_batch_size)
    elif int(job["source_cases"]) > 0:
        if micro_batch_size % 2:
            raise ValueError("joint micro-batch must be even")
        per_epoch = math.ceil(int(job["source_cases"]) / (micro_batch_size // 2))
    else:
        per_epoch = math.ceil(int(job["target_visible_cases"]) / micro_batch_size)
    return per_epoch * epochs


def validate_budget_rows(rows: list[dict], expected_case_count: int) -> None:
    if len(rows) != expected_case_count:
        raise ValueError("budget authority does not cover TrainDev exactly")
    counts = Counter(row["case_id"] for row in rows)
    if any(value != 1 for value in counts.values()):
        raise ValueError("budget authority duplicates a TrainDev case")
