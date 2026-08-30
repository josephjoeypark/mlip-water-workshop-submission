#!/usr/bin/env python3
"""Plot sensitivity of transport estimates to the correlation time grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = (
    ("diffusion_vacf_pbc_nm2_ns", "VACF diffusion", r"$D_L$ (nm$^2$ ns$^{-1}$)"),
    ("viscosity_mPa_s", "Shear viscosity", r"$\eta$ (mPa s)"),
    (
        "diffusion_vacf_corrected_nm2_ns",
        "Corrected diffusion",
        r"$D_\infty$ (nm$^2$ ns$^{-1}$)",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        action="append",
        required=True,
        metavar="DT_FS=CSV",
        help="State table for one correlation grid; repeat for all grids.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_states(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        value, filename = spec.split("=", 1)
        frame = pd.read_csv(filename)
        frame["correlation_dt_fs"] = float(value)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    expected = {0.5, 1.0, 2.0, 5.0}
    observed = set(data["correlation_dt_fs"])
    if observed != expected:
        raise ValueError(f"Expected grids {expected}; received {observed}")
    return data


def write_long_table(data: pd.DataFrame, output: Path) -> None:
    columns = [
        "model_label",
        "backbone",
        "condition",
        "temperature_K",
        "replicas",
        "correlation_dt_fs",
    ]
    for metric, _, _ in METRICS:
        columns.extend([metric, f"{metric}_sem"])
    data[columns].sort_values(
        ["correlation_dt_fs", "model_label", "temperature_K"]
    ).to_csv(output, index=False)


def add_relative_differences(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_label", "temperature_K"]
    primary = data[data["correlation_dt_fs"] == 0.5].set_index(keys)
    rows = []
    for _, row in data.iterrows():
        if row["correlation_dt_fs"] == 0.5:
            continue
        reference = primary.loc[(row["model_label"], row["temperature_K"])]
        record = {key: row[key] for key in keys}
        record["correlation_dt_fs"] = row["correlation_dt_fs"]
        for metric, _, _ in METRICS:
            delta = row[metric] - reference[metric]
            record[f"{metric}_absolute_difference"] = delta
            record[f"{metric}_relative_difference_percent"] = 100 * delta / reference[metric]
        rows.append(record)
    return pd.DataFrame(rows)


def plot_absolute(data: pd.DataFrame, output: Path) -> None:
    models = list(dict.fromkeys(data["model_label"]))
    colors = dict(zip(models, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        for model in models:
            subset = data[data["model_label"] == model]
            for temperature, group in subset.groupby("temperature_K"):
                group = group.sort_values("correlation_dt_fs")
                ax.plot(
                    group["correlation_dt_fs"],
                    group[metric],
                    marker="o",
                    ms=3.5,
                    lw=1.0,
                    alpha=0.78,
                    color=colors[model],
                    label=model if temperature == min(subset["temperature_K"]) else None,
                )
        ax.set_title(title)
        ax.set_xlabel("Correlation grid (fs)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.5, 1, 2, 5])
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Transport estimates are stable to correlation-grid coarsening")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_relative(delta: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    for ax, (metric, title, _) in zip(axes, METRICS):
        column = f"{metric}_relative_difference_percent"
        groups = [
            delta.loc[delta["correlation_dt_fs"] == grid, column].abs().to_numpy()
            for grid in (1.0, 2.0, 5.0)
        ]
        ax.boxplot(groups, tick_labels=["1", "2", "5"], showfliers=True)
        ax.set_title(title)
        ax.set_xlabel("Correlation grid (fs)")
        ax.set_ylabel("Absolute change from 0.5 fs (%)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Sensitivity relative to the 0.5 fs primary analysis (24 states)")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt.rcParams["svg.hashsalt"] = "water-workshop-submission"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_states(args.states)
    delta = add_relative_differences(data)
    write_long_table(data, args.output_dir / "transport_spacing_states.csv")
    delta.to_csv(args.output_dir / "transport_spacing_differences.csv", index=False)
    plot_absolute(data, args.output_dir / "transport_spacing_absolute")
    plot_relative(delta, args.output_dir / "transport_spacing_relative")


if __name__ == "__main__":
    main()
