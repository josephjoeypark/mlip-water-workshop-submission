#!/usr/bin/env python3
"""Build Figure 2 from the committed per-replica density and O--O RDF data."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "md_simulations" / "results"
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
CONDITIONS = ("Baseline", "G1-G2", "G1-G6")
TEMPERATURES = (275, 295, 315, 336)
REPLICAS = (1, 2, 3)

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


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    """Fail with a useful message when an input table has the wrong schema."""
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")


def validate_state_coverage(frame: pd.DataFrame, filename: str) -> None:
    """Require the complete 2 x 3 x 4 x 3 model-state-replica design."""
    expected = {
        "model": set(MODELS),
        "condition": set(CONDITIONS),
        "temperature_K": set(TEMPERATURES),
        "replica": set(REPLICAS),
    }
    for field, expected_values in expected.items():
        observed_values = set(frame[field])
        if observed_values != expected_values:
            raise ValueError(
                f"{filename}: {field} coverage is {sorted(observed_values)}; "
                f"expected {sorted(expected_values)}"
            )

    states = frame[["model", "condition", "temperature_K", "replica"]].drop_duplicates()
    expected_count = len(MODELS) * len(CONDITIONS) * len(TEMPERATURES) * len(REPLICAS)
    if len(states) != expected_count:
        raise ValueError(
            f"{filename} contains {len(states)} unique states; expected {expected_count}"
        )


def validate_density(density: pd.DataFrame) -> None:
    require_columns(
        density,
        {
            "model",
            "condition",
            "temperature_K",
            "replica",
            "md_seed",
            "analysis_start_ps",
            "analysis_end_ps",
            "analysis_frames",
            "density_g_cm3",
        },
        "density_per_replica.csv",
    )
    validate_state_coverage(density, "density_per_replica.csv")
    if density.duplicated(
        ["model", "condition", "temperature_K", "replica"]
    ).any():
        raise ValueError("density_per_replica.csv has duplicate state/replica rows")
    if not np.allclose(density["analysis_start_ps"], 500.0):
        raise ValueError("Every density row must use analysis_start_ps = 500")
    if not np.allclose(density["analysis_end_ps"], 1000.0):
        raise ValueError("Every density row must use analysis_end_ps = 1000")
    if not (density["analysis_frames"] == 5001).all():
        raise ValueError("Every density row must contain 5,001 analysis frames")
    expected_seeds = (
        density["temperature_K"] * 10_000 + density["replica"] * 1_000 + 729
    )
    if not (density["md_seed"] == expected_seeds).all():
        raise ValueError("density_per_replica.csv contains an invalid MD seed")
    if not np.isfinite(density["density_g_cm3"]).all():
        raise ValueError("density_per_replica.csv contains non-finite densities")


def validate_rdf(rdf: pd.DataFrame) -> None:
    require_columns(
        rdf,
        {"model", "condition", "temperature_K", "replica", "r_A", "g_OO"},
        "oo_rdf_per_replica.csv",
    )
    validate_state_coverage(rdf, "oo_rdf_per_replica.csv")
    key = ["model", "condition", "temperature_K", "replica", "r_A"]
    if rdf.duplicated(key).any():
        raise ValueError("oo_rdf_per_replica.csv has duplicate state/replica/radius rows")
    if not np.isfinite(rdf[["r_A", "g_OO"]].to_numpy()).all():
        raise ValueError("oo_rdf_per_replica.csv contains non-finite values")

    grids = []
    for _, rows in rdf.groupby(
        ["model", "condition", "temperature_K", "replica"], sort=False
    ):
        grid = rows.sort_values("r_A")["r_A"].to_numpy()
        if len(grid) == 0 or np.any(np.diff(grid) <= 0):
            raise ValueError("Every RDF state must have a non-empty, increasing r_A grid")
        grids.append(grid)
    reference_grid = grids[0]
    expected_grid = np.arange(0.01, 6.0, 0.02)
    if len(reference_grid) != 300 or not np.allclose(
        reference_grid, expected_grid, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("RDF grid must contain 300 bins from 0.01 to 5.99 A")
    if any(
        len(grid) != len(reference_grid)
        or not np.allclose(grid, reference_grid, rtol=0.0, atol=1.0e-12)
        for grid in grids[1:]
    ):
        raise ValueError("All RDF states must share the same radial grid")


def load_density_reference() -> pd.Series:
    """Read the committed IAPWS-95 density reference."""
    path = RESULTS / "thermophysical_reference.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required experimental-reference table is missing: {path}")

    reference = pd.read_csv(path)
    require_columns(reference, {"temperature_K"}, path.name)
    candidates = (
        "iapws95_density_g_cm3",
        "density_g_cm3",
        "experimental_density_g_cm3",
    )
    density_column = next((name for name in candidates if name in reference.columns), None)
    if density_column is None:
        raise ValueError(
            f"{path.name} must contain one of these density columns: {candidates}"
        )
    selected = reference[reference["temperature_K"].isin(TEMPERATURES)][
        ["temperature_K", density_column]
    ].drop_duplicates()
    if selected["temperature_K"].duplicated().any() or set(
        selected["temperature_K"]
    ) != set(TEMPERATURES):
        raise ValueError(f"{path.name} must contain one density for every plotted temperature")
    result = selected.set_index("temperature_K")[density_column].sort_index()
    if not np.isfinite(result).all():
        raise ValueError(f"{path.name} contains non-finite densities")
    return result


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
    density = pd.read_csv(RESULTS / "density_per_replica.csv")
    rdf = pd.read_csv(RESULTS / "oo_rdf_per_replica.csv")
    rdf_reference = pd.read_csv(RESULTS / "oo_rdf_reference.csv")
    require_columns(
        rdf_reference,
        {"r_A", "g_OO", "g_OO_uncertainty"},
        "oo_rdf_reference.csv",
    )
    validate_density(density)
    validate_rdf(rdf)
    if not np.isfinite(
        rdf_reference[["r_A", "g_OO", "g_OO_uncertainty"]].to_numpy()
    ).all():
        raise ValueError("oo_rdf_reference.csv contains non-finite values")
    density_reference = load_density_reference()

    rdf_295 = rdf[rdf["temperature_K"] == 295]
    rdf_reference = rdf_reference.sort_values("r_A")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        sharey="row",
        constrained_layout=False,
    )
    fig.subplots_adjust(**LAYOUT)

    for column, model in enumerate(MODELS):
        rdf_axis = axes[0, column]
        density_axis = axes[1, column]

        for condition in CONDITIONS:
            rdf_rows = rdf_295[
                (rdf_295["model"] == model)
                & (rdf_295["condition"] == condition)
            ]
            rdf_group = rdf_rows.groupby("r_A", sort=True)["g_OO"]
            rdf_mean = rdf_group.mean()
            rdf_sd = rdf_group.std(ddof=1)
            if not (rdf_group.count() == len(REPLICAS)).all():
                raise ValueError(f"Incomplete 295 K RDF replicas for {model}/{condition}")
            rdf_axis.plot(
                rdf_mean.index,
                rdf_mean.to_numpy(),
                color=COLORS[condition],
                marker=MARKERS[condition],
                markersize=2.2,
                markevery=8,
                markeredgewidth=0,
                label=LABELS[condition],
                zorder=3,
            )
            rdf_axis.fill_between(
                rdf_mean.index,
                rdf_mean.to_numpy() - rdf_sd.to_numpy(),
                rdf_mean.to_numpy() + rdf_sd.to_numpy(),
                color=COLORS[condition],
                alpha=0.24,
                edgecolor=COLORS[condition],
                linewidth=0.4,
                zorder=2,
            )

            density_rows = density[
                (density["model"] == model)
                & (density["condition"] == condition)
            ]
            density_group = density_rows.groupby("temperature_K", sort=True)[
                "density_g_cm3"
            ]
            density_mean = density_group.mean()
            density_sd = density_group.std(ddof=1)
            if not (density_group.count() == len(REPLICAS)).all():
                raise ValueError(f"Incomplete density replicas for {model}/{condition}")
            density_axis.errorbar(
                density_mean.index,
                density_mean.to_numpy(),
                yerr=density_sd.to_numpy(),
                color=COLORS[condition],
                marker=MARKERS[condition],
                barsabove=False,
                label=LABELS[condition],
                zorder=3,
                **ERRORBAR_STYLE,
            )

        rdf_axis.plot(
            rdf_reference["r_A"],
            rdf_reference["g_OO"],
            markevery=8,
            **EXPERIMENT_STYLE,
        )
        rdf_axis.fill_between(
            rdf_reference["r_A"],
            rdf_reference["g_OO"] - rdf_reference["g_OO_uncertainty"],
            rdf_reference["g_OO"] + rdf_reference["g_OO_uncertainty"],
            color="black",
            alpha=0.16,
            edgecolor="black",
            linewidth=0.4,
            zorder=1,
        )
        density_axis.plot(
            density_reference.index,
            density_reference.to_numpy(),
            **EXPERIMENT_STYLE,
        )

        backbone_heading(rdf_axis, model)

        rdf_axis.set_xlim(2.2, 5.5)
        rdf_axis.set_ylim(-0.05, 3.25)
        rdf_axis.set_yticks([0.0, 0.6, 1.2, 1.8, 2.4, 3.0])
        rdf_axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")

        density_axis.set_xlim(272, 339)
        density_axis.set_xticks(TEMPERATURES)
        density_axis.set_ylim(0.89, 1.23)
        density_axis.set_yticks([0.90, 0.96, 1.02, 1.08, 1.14, 1.20])
        density_axis.set_xlabel("Temperature (K)")

        style_axis(rdf_axis)
        style_axis(density_axis)

    axes[0, 0].set_ylabel(r"$g_{\mathrm{OO}}(r)$")
    axes[1, 0].set_ylabel(r"$\rho$ (g cm$^{-3}$)")

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
    fig.savefig(OUTPUT / "figure2.png", dpi=600, facecolor="white")
    fig.savefig(
        OUTPUT / "figure2.pdf",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"Saved PNG: {OUTPUT / 'figure2.png'}")
    print(f"Saved PDF: {OUTPUT / 'figure2.pdf'}")


if __name__ == "__main__":
    main()
