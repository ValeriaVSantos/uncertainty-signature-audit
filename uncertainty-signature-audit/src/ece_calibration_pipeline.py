"""Recompute the ECE-style divergence reported in the paper.

The reference values are discourse-pragmatic proxy scores, not binary factual
accuracy. The metric is therefore described as an ECE-style divergence measure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_ece(
    confidences: np.ndarray, reference_scores: np.ndarray, n_bins: int = 10
) -> tuple[float, list[float], list[float]]:
    """Calculate weighted absolute bin divergence on a 0-100 scale."""
    confidences = np.asarray(confidences, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    if confidences.shape != reference_scores.shape:
        raise ValueError("confidences and reference_scores must have the same shape")
    if confidences.size == 0:
        raise ValueError("input arrays must not be empty")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if np.any(~np.isfinite(confidences)) or np.any(~np.isfinite(reference_scores)):
        raise ValueError("inputs must contain only finite values")
    if np.any((confidences < 0) | (confidences > 100)):
        raise ValueError("confidence values must be between 0 and 100")
    if np.any((reference_scores < 0) | (reference_scores > 100)):
        raise ValueError("reference scores must be between 0 and 100")

    boundaries = np.linspace(0, 100, n_bins + 1)
    divergence = 0.0
    bin_confidences: list[float] = []
    bin_reference_scores: list[float] = []
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = ((confidences >= lower) & (confidences <= upper)) if index == n_bins - 1 else ((confidences >= lower) & (confidences < upper))
        count = int(mask.sum())
        if count == 0:
            bin_confidences.append(float("nan"))
            bin_reference_scores.append(float("nan"))
            continue
        mean_confidence = float(confidences[mask].mean())
        mean_reference = float(reference_scores[mask].mean())
        divergence += (count / confidences.size) * abs(mean_reference - mean_confidence)
        bin_confidences.append(mean_confidence)
        bin_reference_scores.append(mean_reference)
    return divergence, bin_confidences, bin_reference_scores


def plot_reliability_diagram(faithful, sanitized, faithful_ece, sanitized_ece, output_path: Path) -> None:
    """Save the comparison diagram without opening an interactive window."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot([0, 100], [0, 100], "--", color="gray", label="Reference diagonal")
    for (confidences, references), label, color, score in (
        (faithful, "Faithful transcript", "#5b2c6f", faithful_ece),
        (sanitized, "Sanitized transcript", "#d35400", sanitized_ece),
    ):
        x_values = np.asarray(confidences, dtype=float)
        y_values = np.asarray(references, dtype=float)
        valid = ~(np.isnan(x_values) | np.isnan(y_values))
        axis.plot(x_values[valid], y_values[valid], "o-", color=color, linewidth=2, label=f"{label} (divergence: {score:.2f})")
    axis.set(title="Model confidence and pragmatic reference score")
    axis.set_xlabel("Mean model confidence")
    axis.set_ylabel("Mean pragmatic reference score")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.grid(linestyle=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"CONFIANCA_IA_FIEL", "CONFIANCA_IA_HIGIENIZADA", "NOTA_HUMANA"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    reference = frame["NOTA_HUMANA"].to_numpy()
    faithful_ece, faithful_conf, faithful_ref = calculate_ece(frame["CONFIANCA_IA_FIEL"].to_numpy(), reference, args.bins)
    sanitized_ece, sanitized_conf, sanitized_ref = calculate_ece(frame["CONFIANCA_IA_HIGIENIZADA"].to_numpy(), reference, args.bins)
    print(f"N: {len(frame)}")
    print(f"Faithful-layer divergence: {faithful_ece:.4f}")
    print(f"Sanitized-layer divergence: {sanitized_ece:.4f}")
    print(f"Absolute difference: {abs(faithful_ece - sanitized_ece):.4f}")
    plot_reliability_diagram((faithful_conf, faithful_ref), (sanitized_conf, sanitized_ref), faithful_ece, sanitized_ece, args.output)


if __name__ == "__main__":
    main()
