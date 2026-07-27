import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Fixed detector metrics, stored as rates in the range [0, 1].
RESULTS = {
    (2, "randomPositionOffset"): {
        "false_positive_rate": 0.1418,
        "true_positive_rate": 0.9598,
        "f1_score": 0.7680,
    },
    (2, "constantPositionOffset"): {
        "false_positive_rate": 0.1264,
        "true_positive_rate": 0.1749,
        "f1_score": 0.2012,
    },
    (3, "randomPositionOffset"): {
        "false_positive_rate": 0.0719,
        "true_positive_rate": 0.9682,
        "f1_score": 0.8629,
    },
    (3, "constantPositionOffset"): {
        "false_positive_rate": 0.0570,
        "true_positive_rate": 0.1965,
        "f1_score": 0.2707,
    },
    (4, "randomPositionOffset"): {
        "false_positive_rate": 0.3011,
        "true_positive_rate": 0.9694,
        "f1_score": 0.6207,
    },
    (4, "constantPositionOffset"): {
        "false_positive_rate": 0.2602,
        "true_positive_rate": 0.6562,
        "f1_score": 0.4654,
    },
    (5, "randomPositionOffset"): {
        "false_positive_rate": 0.1206,
        "true_positive_rate": 0.9686,
        "f1_score": 0.7969,
    },
    (5, "constantPositionOffset"): {
        "false_positive_rate": 0.0939,
        "true_positive_rate": 0.5630,
        "f1_score": 0.5678,
    },
}

DETECTORS = {
    2: ("CAM Only", "kalman_cam_only"),
    3: ("Tsukada", "kalman_cam_cpm"),
    4: ("1-Edge Recip.", "kalman_cam_cpm_enhanced"),
    5: ("2-Edge Recip.", "kalman_cam_cpm_enhanced_two_edges"),
}

ATTACK_LABELS = {
    "randomPositionOffset": "Random Position Offset",
    "constantPositionOffset": "Constant Position Offset",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fixed Kalman detector comparisons.")
    parser.add_argument(
        "filename",
        help="Output filename suffix; the setting is added as a prefix",
    )
    parser.add_argument("--setting", default="urban", help="Label used in the title and output filename")
    parser.add_argument("--types", nargs="+", type=int, default=[2, 3, 4, 5], help="Detector types (2, 3, 4, 5)")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["randomPositionOffset", "constantPositionOffset"],
        help="Attack names",
    )
    parser.add_argument(
        "--include-f1",
        action="store_true",
        help="Add an F1 Score subplot to the generated figure",
    )
    return parser.parse_args()


def validate_args(args):
    unsupported = [detector_type for detector_type in args.types if detector_type not in DETECTORS]
    if unsupported:
        raise ValueError(f"Unsupported detector types: {unsupported}; choose from 2, 3, 4, 5")
    if len(set(args.types)) != len(args.types):
        raise ValueError("Detector types must not contain duplicates")
    if len(set(args.attacks)) != len(args.attacks):
        raise ValueError("Attacks must not contain duplicates")

    missing = [
        (detector_type, attack)
        for detector_type in args.types
        for attack in args.attacks
        if (detector_type, attack) not in RESULTS
    ]
    if missing:
        raise ValueError(f"No fixed metrics are defined for: {missing}")


def attack_label(attack):
    return ATTACK_LABELS.get(attack, attack.replace("_", " ").title())


def add_value_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_results(args, results, output_path):
    detector_labels = [DETECTORS[detector_type][0] for detector_type in args.types]
    x_positions = np.arange(len(args.types))
    bar_width = 0.78 / len(args.attacks)
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]

    plot_specs = [
        ("false_positive_rate", "False Positive Rate"),
        ("true_positive_rate", "True Positive Rate"),
    ]
    if args.include_f1:
        plot_specs.append(("f1_score", "F1 Score"))

    figure, axes = plt.subplots(1, len(plot_specs), figsize=(6.5 * len(plot_specs), 5.5))
    axes = np.atleast_1d(axes)
    plot_specs = [
        (metric_name, title, axes[index])
        for index, (metric_name, title) in enumerate(plot_specs)
    ]

    for metric_name, title, axis in plot_specs:
        all_values = []
        for attack_index, attack in enumerate(args.attacks):
            values = [results[(detector_type, attack)][metric_name] * 100 for detector_type in args.types]
            all_values.extend(values)
            offset = (attack_index - (len(args.attacks) - 1) / 2) * bar_width
            bars = axis.bar(
                x_positions + offset,
                values,
                bar_width,
                label=attack_label(attack),
                color=colors[attack_index % len(colors)],
            )
            add_value_labels(axis, bars)

        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_ylabel("Rate (%)")
        axis.set_xticks(x_positions, detector_labels)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if metric_name == "false_positive_rate":
            upper_limit = min(100, max(10, math.ceil((max(all_values, default=0) + 8) / 10) * 10))
            axis.set_ylim(0, upper_limit)
        else:
            axis.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(f"{args.setting.title()} Detection Performance", fontsize=15, weight="bold", y=0.98)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_metrics(args, results):
    print("\nDetector metrics")
    f1_header = f" {'F1':>8}" if args.include_f1 else ""
    print(f"{'Detector':<31} {'Attack':<26} {'FPR':>8} {'TPR':>8}{f1_header}")
    print("-" * (86 if args.include_f1 else 77))
    for detector_type in args.types:
        for attack in args.attacks:
            metrics = results[(detector_type, attack)]
            f1_value = f" {metrics['f1_score'] * 100:>7.2f}%" if args.include_f1 else ""
            print(
                f"{DETECTORS[detector_type][0]:<31} "
                f"{attack_label(attack):<26} "
                f"{metrics['false_positive_rate'] * 100:>7.2f}% "
                f"{metrics['true_positive_rate'] * 100:>7.2f}%"
                f"{f1_value}"
            )


def main():
    args = parse_args()
    try:
        validate_args(args)
        output_path = (
            Path(__file__).resolve().parent
            / "created"
            / f"{args.setting}_{args.filename}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(args, RESULTS, output_path)
        print_metrics(args, RESULTS)
        print(f"\nSaved figure in {output_path}")
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
