"""Evaluate reciprocity grace/expiry windows and plot the noCheat results."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MBD_ROOT = REPO_ROOT / "Generator" / "MBD_systems"
sys.path.insert(0, str(MBD_ROOT))

from cpm_enhanced_detector import process_cpm_enhanced_folder  # noqa: E402


DATA_ROOT = REPO_ROOT / "Simulation-Test" / "noCheat-test"
OUTPUT_ROOT = Path(__file__).resolve().parent / "created" / "grace_expiry_noCheat"
ATTACKS = {
    "randomPositionOffset": "Random position offset",
    "constantPositionOffset": "Constant position offset",
}
DETECTORS = {1: "1-edge", 2: "2-edge"}
CONFIGS = [
    (3, 6, "Current: 3 s grace / 6 s expiry"),
    (2, 4, "2 s grace / 4 s expiry"),
    (2, 6, "Proposed: 2 s grace / 6 s expiry"),
    (2, 8, "2 s grace / 8 s expiry"),
]


def rates(metrics):
    tp, tn, fp, fn = (int(metrics[key]) for key in ("tp", "tn", "fp", "fn"))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def run_study():
    rows = []
    for grace_s, expiry_s, label in CONFIGS:
        if expiry_s <= grace_s:
            raise ValueError("Expiry must be greater than grace")
        for attack, attack_label in ATTACKS.items():
            input_folder = DATA_ROOT / f"json_{attack}"
            for required_edges, detector_label in DETECTORS.items():
                print(
                    f"Running {detector_label}, {attack_label}, "
                    f"grace={grace_s}s, expiry={expiry_s}s...",
                    flush=True,
                )
                metrics, _ = process_cpm_enhanced_folder(
                    input_folder,
                    required_unreciprocated_edges=required_edges,
                    edge_grace_ns=grace_s * 1_000_000_000,
                    edge_ttl_ns=expiry_s * 1_000_000_000,
                )
                rows.append(
                    {
                        "configuration": label,
                        "grace_s": grace_s,
                        "expiry_s": expiry_s,
                        "attack": attack,
                        "attack_label": attack_label,
                        "required_edges": required_edges,
                        "detector": detector_label,
                        **rates(metrics),
                    }
                )
    return rows


def lookup(rows, grace_s, expiry_s, attack, required_edges):
    return next(
        row
        for row in rows
        if row["grace_s"] == grace_s
        and row["expiry_s"] == expiry_s
        and row["attack"] == attack
        and row["required_edges"] == required_edges
    )


def plot_current_vs_proposed(rows, output_path):
    metrics = [("fpr", "False positive rate"), ("tpr", "True positive rate"), ("f1", "F1 score")]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharey="col")
    x = np.arange(len(DETECTORS))
    width = 0.34
    colors = ["#777777", "#2F6B9A"]

    for row_index, (attack, attack_label) in enumerate(ATTACKS.items()):
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for config_index, (grace_s, expiry_s, config_label) in enumerate(
                [CONFIGS[0], CONFIGS[2]]
            ):
                values = [
                    100 * lookup(rows, grace_s, expiry_s, attack, edge_count)[metric]
                    for edge_count in DETECTORS
                ]
                bars = axis.bar(
                    x + (config_index - 0.5) * width,
                    values,
                    width,
                    color=colors[config_index],
                    label=config_label,
                )
                axis.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
            axis.set_title(title)
            axis.set_xticks(x, DETECTORS.values())
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                axis.set_ylabel(f"{attack_label}\nRate (%)")
            axis.set_ylim(0, 105 if metric != "fpr" else None)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    figure.suptitle("noCheat: 2-second grace period compared with current settings", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_expiry_sensitivity(rows, output_path):
    metrics = [("fpr", "False positive rate"), ("tpr", "True positive rate"), ("f1", "F1 score")]
    expiry_values = [4, 6, 8]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey="col")
    colors = {1: "#E07B39", 2: "#3978A8"}

    for row_index, (attack, attack_label) in enumerate(ATTACKS.items()):
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            all_values = []
            for edge_count, detector_label in DETECTORS.items():
                values = [
                    100 * lookup(rows, 2, expiry_s, attack, edge_count)[metric]
                    for expiry_s in expiry_values
                ]
                all_values.extend(values)
                axis.plot(
                    expiry_values,
                    values,
                    marker="o",
                    linewidth=2,
                    color=colors[edge_count],
                    label=detector_label,
                )
                for expiry_s, value in zip(expiry_values, values):
                    other_edge_count = 2 if edge_count == 1 else 1
                    other_value = 100 * lookup(
                        rows, 2, expiry_s, attack, other_edge_count
                    )[metric]
                    label_offset = 8 if value >= other_value else -15
                    axis.annotate(
                        f"{value:.1f}%",
                        (expiry_s, value),
                        xytext=(0, label_offset),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            axis.set_title(title)
            axis.set_xticks(expiry_values)
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                axis.set_ylabel(f"{attack_label}\nRate (%)")
            if row_index == 1:
                axis.set_xlabel("Edge expiry (seconds), with grace fixed at 2 seconds")
            axis.set_ylim(0, 105 if metric != "fpr" else max(all_values) + 5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    figure.suptitle("noCheat: expiry sensitivity with a 2-second grace period", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_results(rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    with (OUTPUT_ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_current_vs_proposed(rows, OUTPUT_ROOT / "current_vs_2s_grace.png")
    plot_expiry_sensitivity(rows, OUTPUT_ROOT / "2s_grace_expiry_sensitivity.png")


if __name__ == "__main__":
    results = run_study()
    write_results(results)
    print(f"Saved study outputs in {OUTPUT_ROOT}")
