"""Functions extracted without body changes from the existing training implementation."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from engine.utils import quadratic_weighted_kappa, classification_metrics
R1 = "C1_R1_REAL_ONLY_CE"
R2 = "C1_R2_SYN_PRETRAIN_CE"


def group_resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups == group) for group in sampled]
    return np.concatenate(pieces)


def hierarchical_bootstrap(indexed: dict, budget: int, candidate: str, reference: str, seeds: list[int], repetitions: int, rng_seed: int) -> dict:
    rng = np.random.default_rng(rng_seed)
    template = indexed[(budget, candidate, seeds[0])]
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = group_resample_indices(template["_groups"], rng)
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        differences = []
        for seed in sampled_seeds:
            a = indexed[(budget, candidate, int(seed))]
            b = indexed[(budget, reference, int(seed))]
            truth = a["_truth"][indices]
            differences.append(
                quadratic_weighted_kappa(truth, a["_predicted"][indices])
                - quadratic_weighted_kappa(truth, b["_predicted"][indices])
            )
        draws[repetition] = float(np.mean(differences))
    return {
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": rng_seed,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "one_sided_p": float((1 + np.sum(draws <= 0)) / (repetitions + 1)),
        "bootstrap_probability_gain_gt_zero": float(np.mean(draws > 0)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def risk_coverage(rows: list[dict], coverage_grid: list[float]) -> list[dict]:
    output = []
    for row in rows:
        order = np.argsort(-row["_confidence"], kind="stable")
        n = len(order)
        for coverage in coverage_grid:
            count = max(1, int(math.ceil(coverage * n)))
            selected = order[:count]
            truth = row["_truth"][selected]
            predicted = row["_predicted"][selected]
            metrics = classification_metrics(truth, predicted)
            output.append({
                "method_id": row["method_id"],
                "budget_percent": row["budget_percent"],
                "label_budget_seed": row["label_budget_seed"],
                "target_coverage": coverage,
                "retained_cases": count,
                "empirical_coverage": count / n,
                "qwk": metrics["qwk"],
                "grade_mae": metrics["grade_mae"],
                "severe_error_rate": metrics["severe_error_rate"],
            })
    return output
