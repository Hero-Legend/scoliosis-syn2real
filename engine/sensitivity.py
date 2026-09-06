"""Functions extracted without body changes from the existing training implementation."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from engine import statistics as frozen
from engine.utils import quadratic_weighted_kappa


def draw_summary(draws: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(draws)
    if not np.all(finite):
        raise RuntimeError(f"Non-finite bootstrap draws: {int(np.sum(~finite))}")
    return {
        "repetitions": int(draws.size),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "median": float(np.median(draws)),
        "probability_gt_zero": float(np.mean(draws > 0)),
        "one_sided_monte_carlo_p": float((1 + np.sum(draws <= 0)) / (draws.size + 1)),
    }


def qwk_gain(indexed: dict, budget: int, seed: int, indices: np.ndarray) -> float:
    candidate = indexed[(budget, frozen.R2, seed)]
    reference = indexed[(budget, frozen.R1, seed)]
    truth = candidate["_truth"][indices]
    return float(
        quadratic_weighted_kappa(truth, candidate["_predicted"][indices])
        - quadratic_weighted_kappa(truth, reference["_predicted"][indices])
    )


def group_only_gain_draws(
    indexed: dict, budget: int, seeds: list[int], repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    template = indexed[(budget, frozen.R2, seeds[0])]
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = frozen.group_resample_indices(template["_groups"], rng)
        draws[repetition] = np.mean([qwk_gain(indexed, budget, seed, indices) for seed in seeds])
    return draws


def seed_only_gain_draws(
    indexed: dict, budget: int, seeds: list[int], repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.arange(len(indexed[(budget, frozen.R2, seeds[0])]["_truth"]))
    gains = np.asarray([qwk_gain(indexed, budget, seed, indices) for seed in seeds])
    sampled = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
    return gains[sampled].mean(axis=1)


def hierarchical_gain_draws(
    indexed: dict, budget: int, seeds: list[int], repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    template = indexed[(budget, frozen.R2, seeds[0])]
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = frozen.group_resample_indices(template["_groups"], rng)
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        draws[repetition] = np.mean(
            [qwk_gain(indexed, budget, int(seed), indices) for seed in sampled_seeds]
        )
    return draws


def threshold_margin(
    indexed: dict, budget: int, seed: int, indices: np.ndarray, fraction: float
) -> float:
    candidate = indexed[(budget, frozen.R2, seed)]
    reference = indexed[(100, frozen.R1, seed)]
    truth = candidate["_truth"][indices]
    candidate_qwk = quadratic_weighted_kappa(truth, candidate["_predicted"][indices])
    reference_qwk = quadratic_weighted_kappa(truth, reference["_predicted"][indices])
    return float(candidate_qwk - fraction * reference_qwk)


def threshold_draws(
    indexed: dict,
    budget: int,
    seeds: list[int],
    repetitions: int,
    rng: np.random.Generator,
    fraction: float,
    resample_groups: bool,
    resample_seeds: bool,
) -> np.ndarray:
    template = indexed[(budget, frozen.R2, seeds[0])]
    all_indices = np.arange(len(template["_truth"]))
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = (
            frozen.group_resample_indices(template["_groups"], rng)
            if resample_groups
            else all_indices
        )
        sampled_seeds = (
            rng.choice(seeds, size=len(seeds), replace=True) if resample_seeds else seeds
        )
        draws[repetition] = np.mean(
            [threshold_margin(indexed, budget, int(seed), indices, fraction) for seed in sampled_seeds]
        )
    return draws
