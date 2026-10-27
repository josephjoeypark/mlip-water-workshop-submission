#!/usr/bin/env python3
"""Build Figure 3 from per-replica diffusion and viscosity CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "md_simulations" / "results"
OUTPUT = Path(__file__).resolve().parent / "figures"

TEMPERATURES = np.array([275, 295, 315, 336], dtype=float)
MODELS = ("MACE-OFF", "MACELES-OFF")
CONDITIONS = ("Baseline", "G1-G2", "G1-G6")
REPLICAS = (1, 2, 3)

FIGSIZE = (7.0, 5.0)
LAYOUT = {
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.10,
    "top": 0.855,
    "wspace": 0.12,
    "hspace": 0.31,
}

COLORS = {
    "Baseline": "#E14F62",
    "G1-G2": "#E8A23B",
    "G1-G6": "#95CA42",
}
LABELS = {"Baseline": "Baseline", "G1-G2": "G1–G2", "G1-G6": "G1–G6"}
MARKERS = {"Baseline": "o", "G1-G2": "s", "G1-G6": "^"}
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
EXPERIMENT_STYLE = {
    "color": "black",
    "linestyle": ":",
    "linewidth": 1.7,
    "marker": "o",
    "markersize": 3.2,
    "markerfacecolor": "black",
    "markeredgewidth": 0,
    "zorder": 4,
}

plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    }
)


def panel_label(axis, label: str) -> None:
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


def backbone_heading(axis, model: str) -> None:
    axis.text(
        0.5,
        1.045,
        BACKBONE_LABELS[model],
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
    )


def style_axis(axis) -> None:
    axis.grid(axis="y", **GRID_STYLE)
    axis.set_axisbelow(True)


def _expected_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [MODELS, CONDITIONS, TEMPERATURES.astype(int), REPLICAS],
        names=["model", "condition", "temperature_K", "replica"],
    )


def validate_replica_table(
    frame: pd.DataFrame, value_column: str, name: str
) -> pd.DataFrame:
    required = {
        "model",
        "condition",
        "temperature_K",
        "replica",
        "md_seed",
        "analyzed_ps",
        "correlation_timestep_fs",
        value_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    keys = ["model", "condition", "temperature_K", "replica"]
    if frame.duplicated(keys).any():
        duplicate_rows = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise ValueError(f"{name} contains duplicate state/replica rows:\n{duplicate_rows}")

    indexed = frame.set_index(keys).sort_index()
    expected = _expected_index().sort_values()
    missing_keys = expected.difference(indexed.index)
    extra_keys = indexed.index.difference(expected)
    if len(indexed) != 72 or len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"{name} must contain exactly the 72 expected rows "
            f"(6 model conditions x 4 temperatures x 3 replicas); "
            f"found {len(indexed)}, missing={list(missing_keys)}, "
            f"extra={list(extra_keys)}"
        )

    if not np.allclose(indexed["analyzed_ps"].to_numpy(dtype=float), 700.0):
        values = sorted(indexed["analyzed_ps"].unique().tolist())
        raise ValueError(f"{name} must use 700 ps for every row; found {values}")
    if not np.allclose(
        indexed["correlation_timestep_fs"].to_numpy(dtype=float), 0.5
    ):
        values = sorted(indexed["correlation_timestep_fs"].unique().tolist())
        raise ValueError(f"{name} must use 0.5 fs sampling for every row; found {values}")

    flat = indexed.reset_index()
    expected_seeds = (
        flat["temperature_K"] * 10_000 + flat["replica"] * 1_000 + 729
    )
    if not (flat["md_seed"] == expected_seeds).all():
        raise ValueError(f"{name} contains an invalid MD seed")

    values = indexed[value_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{name}.{value_column} must contain positive finite values")

    return flat


def validate_paired_coverage(
    diffusion: pd.DataFrame, viscosity: pd.DataFrame
) -> None:
    keys = ["model", "condition", "temperature_K", "replica"]
    paired_fields = ["md_seed", "analyzed_ps", "correlation_timestep_fs"]
    left = diffusion.sort_values(keys).reset_index(drop=True)
    right = viscosity.sort_values(keys).reset_index(drop=True)

    if not left[keys].equals(right[keys]):
        raise ValueError("Diffusion and viscosity do not have identical state/replica coverage")
    for field in paired_fields:
        if not np.allclose(
            left[field].to_numpy(dtype=float), right[field].to_numpy(dtype=float)
        ):
            raise ValueError(f"Diffusion and viscosity disagree in paired field {field!r}")


def load_experiment() -> pd.DataFrame:
    reference_path = RESULTS / "thermophysical_reference.csv"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Required experimental-reference table is missing: {reference_path}"
        )

    reference = pd.read_csv(reference_path)
    required = {
        "temperature_K",
        "oxygen_diffusion_1e-9_m2_s",
        "viscosity_mPa_s",
    }
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"{reference_path.name} is missing required columns: {missing}")
    reference = reference.loc[
        :,
        ["temperature_K", "oxygen_diffusion_1e-9_m2_s", "viscosity_mPa_s"],
    ].sort_values("temperature_K")
    if not np.array_equal(
        reference["temperature_K"].to_numpy(dtype=float), TEMPERATURES
    ):
        raise ValueError(
            f"{reference_path.name} must contain exactly 275, 295, 315, and 336 K"
        )
    return reference


def mean_and_sample_sd(
    rows: pd.DataFrame, field: str
) -> tuple[np.ndarray, np.ndarray]:
    grouped = rows.groupby("temperature_K", sort=True)[field]
    counts = grouped.count()
    if not (counts == 3).all():
        raise ValueError(
            f"Expected three replicas per temperature for {field}; "
            f"got {counts.to_dict()}"
        )
    return (
        grouped.mean().reindex(TEMPERATURES.astype(int)).to_numpy(dtype=float),
        grouped.std(ddof=1).reindex(TEMPERATURES.astype(int)).to_numpy(dtype=float),
    )


def model_errorbar(
    axis, mean: np.ndarray, sample_sd: np.ndarray, condition: str
) -> None:
    axis.errorbar(
        TEMPERATURES,
        mean,
        yerr=sample_sd,
        color=COLORS[condition],
        marker=MARKERS[condition],
        barsabove=False,
        label=LABELS[condition],
        zorder=3,
        **ERRORBAR_STYLE,
    )


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[condition],
            marker=MARKERS[condition],
            label=LABELS[condition],
        )
        for condition in CONDITIONS
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            linestyle=":",
            linewidth=1.7,
            marker="o",
            markersize=3.2,
            markerfacecolor="black",
            markeredgewidth=0,
            label="EXP (bulk)",
        )
    )
    return handles


def main() -> None:
    diffusion = validate_replica_table(
        pd.read_csv(RESULTS / "diffusion_per_replica.csv"),
        "oxygen_vacf_diffusion_nm2_ns",
        "diffusion_per_replica.csv",
    )
    viscosity = validate_replica_table(
        pd.read_csv(RESULTS / "viscosity_per_replica.csv"),
        "green_kubo_viscosity_mPa_s",
        "viscosity_per_replica.csv",
    )
    validate_paired_coverage(diffusion, viscosity)
    experiment = load_experiment()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    fig.subplots_adjust(**LAYOUT)

    for column, model in enumerate(MODELS):
        diffusion_axis = axes[0, column]
        viscosity_axis = axes[1, column]

        for condition in CONDITIONS:
            diffusion_rows = diffusion[
                (diffusion["model"] == model)
                & (diffusion["condition"] == condition)
            ]
            viscosity_rows = viscosity[
                (viscosity["model"] == model)
                & (viscosity["condition"] == condition)
            ]
            diffusion_mean, diffusion_sd = mean_and_sample_sd(
                diffusion_rows, "oxygen_vacf_diffusion_nm2_ns"
            )
            viscosity_mean, viscosity_sd = mean_and_sample_sd(
                viscosity_rows, "green_kubo_viscosity_mPa_s"
            )
            model_errorbar(diffusion_axis, diffusion_mean, diffusion_sd, condition)
            model_errorbar(viscosity_axis, viscosity_mean, viscosity_sd, condition)

        diffusion_axis.plot(
            experiment["temperature_K"],
            experiment["oxygen_diffusion_1e-9_m2_s"],
            **EXPERIMENT_STYLE,
        )
        viscosity_axis.plot(
            experiment["temperature_K"],
            experiment["viscosity_mPa_s"],
            **EXPERIMENT_STYLE,
        )

        backbone_heading(diffusion_axis, model)
        for axis in (diffusion_axis, viscosity_axis):
            axis.set_xlim(272, 339)
            axis.set_xticks(TEMPERATURES)
            style_axis(axis)

        diffusion_axis.set_ylim(0.0, 5.45)
        diffusion_axis.set_yticks([0, 1, 2, 3, 4, 5])
        viscosity_axis.set_ylim(0.0, 4.35)
        viscosity_axis.set_yticks([0, 1, 2, 3, 4])
        viscosity_axis.set_xlabel("Temperature (K)")

    axes[0, 0].set_ylabel(
        r"$D_{\mathrm{O}}^{(L)}$ ($10^{-9}$ m$^2$ s$^{-1}$)"
    )
    axes[1, 0].set_ylabel(r"$\eta$ (mPa s)")

    for axis, label in zip(axes.flat, ("a", "b", "c", "d")):
        panel_label(axis, label)

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.55, 0.985),
        handlelength=1.6,
        columnspacing=1.25,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT / "figure3.png"
    pdf_path = OUTPUT / "figure3.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
