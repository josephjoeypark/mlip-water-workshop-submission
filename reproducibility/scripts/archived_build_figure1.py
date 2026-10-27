#!/usr/bin/env python3
"""Build Figure 1 from the frozen 123,200-update per-seed RMSE data."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model_training" / "results" / "cluster_rmse_all_seeds.csv"
OUTPUT = Path(__file__).resolve().parent / "figures"

FIGSIZE = (7.0, 5.0)
LAYOUT = {
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.10,
    "top": 0.855,
    "wspace": 0.12,
    "hspace": 0.31,
}

MODELS = ("MACE-OFF", "MACELES-OFF")
CONDITIONS = ("G1-G2", "G1-G6")
SEEDS = (17, 29, 43)
GROUPS = tuple(f"G{number}" for number in range(1, 7))
OPTIMIZER_UPDATES = 123_200

COLORS = {"G1-G2": "#E8A23B", "G1-G6": "#95CA42"}
LABELS = {"G1-G2": "G1–G2", "G1-G6": "G1–G6"}
BACKBONE_LABELS = {
    "MACE-OFF": "MACE-OFF23(M)",
    "MACELES-OFF": "MACELES-OFF",
}
ERRORBAR_STYLE = {
    "ecolor": "black",
    "elinewidth": 0.6,
    "capsize": 2.0,
    "capthick": 0.6,
}
GRID_STYLE = {"color": "0.80", "alpha": 0.45, "linewidth": 0.55}

plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    }
)


def load_and_validate() -> pd.DataFrame:
    """Load the two adapted conditions and enforce the frozen study design."""
    required_columns = {
        "model",
        "condition",
        "training_seed",
        "optimizer_updates",
        "cluster_group",
        "energy_rmse_meV_per_atom",
        "force_component_rmse_meV_per_A",
    }
    data = pd.read_csv(SOURCE)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data[
        data["model"].isin(MODELS) & data["condition"].isin(CONDITIONS)
    ].copy()
    if data.empty:
        raise ValueError("No G1-G2 or G1-G6 evaluation rows were found.")

    numeric_columns = (
        "training_seed",
        "optimizer_updates",
        "energy_rmse_meV_per_atom",
        "force_component_rmse_meV_per_A",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[list(numeric_columns)].isna().any().any():
        raise ValueError("Adapted-condition rows contain missing numerical values.")

    data["training_seed"] = data["training_seed"].astype(int)
    data["optimizer_updates"] = data["optimizer_updates"].astype(int)
    data["group_number"] = data["cluster_group"].str.extract(r"^G([1-6])$")[0]
    if data["group_number"].isna().any():
        invalid = sorted(
            data.loc[data["group_number"].isna(), "cluster_group"].unique()
        )
        raise ValueError(f"Invalid cluster-group labels: {invalid}")
    data["group_number"] = data["group_number"].astype(int)

    observed_updates = set(data["optimizer_updates"])
    if observed_updates != {OPTIMIZER_UPDATES}:
        raise ValueError(
            "Figure 1 requires only 123,200-update checkpoints; "
            f"found {sorted(observed_updates)}."
        )

    key = ["model", "condition", "training_seed", "cluster_group"]
    duplicates = data.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate per-seed evaluation rows:\n"
            + data.loc[duplicates, key].to_string(index=False)
        )

    expected = pd.MultiIndex.from_product(
        [MODELS, CONDITIONS, SEEDS, GROUPS], names=key
    )
    observed = pd.MultiIndex.from_frame(data[key])
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            "Per-seed Figure 1 grid is incomplete or unexpected. "
            f"Missing={list(missing)}, unexpected={list(unexpected)}"
        )

    return data


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.015,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )


def backbone_heading(axis: plt.Axes, model: str) -> None:
    axis.text(
        0.5,
        1.045,
        BACKBONE_LABELS[model],
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", **GRID_STYLE)
    axis.set_axisbelow(True)


def main() -> None:
    data = load_and_validate()
    summary = (
        data.groupby(["model", "condition", "group_number"], sort=False)
        .agg(
            energy_mean=("energy_rmse_meV_per_atom", "mean"),
            energy_sd=("energy_rmse_meV_per_atom", "std"),
            force_mean=("force_component_rmse_meV_per_A", "mean"),
            force_sd=("force_component_rmse_meV_per_A", "std"),
            seed_count=("training_seed", "nunique"),
        )
        .reset_index()
    )
    if not (summary["seed_count"] == len(SEEDS)).all():
        raise ValueError("Every plotted mean must contain exactly three training seeds.")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    figure.subplots_adjust(**LAYOUT)

    centers = np.arange(6)
    bar_width = 0.34

    for column, model in enumerate(MODELS):
        energy_axis = axes[0, column]
        force_axis = axes[1, column]

        for axis in (energy_axis, force_axis):
            axis.set_xlim(-0.55, 5.55)
            axis.set_xticks(centers, GROUPS)
            style_axis(axis)

        for offset, condition in (
            (-bar_width / 2, "G1-G2"),
            (bar_width / 2, "G1-G6"),
        ):
            rows = summary[
                (summary["model"] == model)
                & (summary["condition"] == condition)
            ].sort_values("group_number")
            if rows["group_number"].tolist() != list(range(1, 7)):
                raise ValueError(f"Incomplete group sequence for {model} / {condition}.")

            energy_axis.bar(
                centers + offset,
                rows["energy_mean"],
                width=bar_width,
                yerr=rows["energy_sd"],
                color=COLORS[condition],
                edgecolor="none",
                linewidth=0,
                error_kw=ERRORBAR_STYLE,
                zorder=3,
            )
            force_axis.bar(
                centers + offset,
                rows["force_mean"],
                width=bar_width,
                yerr=rows["force_sd"],
                color=COLORS[condition],
                edgecolor="none",
                linewidth=0,
                error_kw=ERRORBAR_STYLE,
                zorder=3,
            )

        backbone_heading(energy_axis, model)
        force_axis.set_xlabel("Water-cluster size")

    for axis in axes[0, :]:
        axis.set_ylim(0.0, 1.78)
        axis.set_yticks([0.0, 0.4, 0.8, 1.2, 1.6])
    for axis in axes[1, :]:
        axis.set_ylim(0.0, 25.0)
        axis.set_yticks([0, 5, 10, 15, 20, 25])

    axes[0, 0].set_ylabel(r"Energy RMSE (meV atom$^{-1}$)")
    axes[1, 0].set_ylabel(r"Force-component RMSE (meV $\mathrm{\AA}^{-1}$)")

    for axis, label in zip(axes.flat, ("a", "b", "c", "d")):
        panel_label(axis, label)

    handles = [
        Patch(facecolor=COLORS[condition], edgecolor="none", label=LABELS[condition])
        for condition in CONDITIONS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.55, 0.985),
        handlelength=1.6,
        columnspacing=1.25,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT / "figure1.png"
    pdf_path = OUTPUT / "figure1.pdf"
    figure.savefig(png_path, dpi=600, facecolor="white")
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    print(f"Validated {len(data)} rows at {OPTIMIZER_UPDATES:,} updates.")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
