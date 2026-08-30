#!/usr/bin/env python3
"""Plot density times viscosity from the compact Figure 2 and Figure 3 data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper_figures"
OUTPUT = ROOT / "figures"

COLORS = {"Baseline": "#4C78A8", "G1-G2": "#F58518", "G1-G6": "#54A24B"}
MARKERS = {"Baseline": "o", "G1-G2": "s", "G1-G6": "^"}
FOUNDATIONS = ("MACEOFF23(M)", "MACELES-OFF")
CONDITIONS = ("Baseline", "G1-G2", "G1-G6")
MODEL_ORDER = tuple(f"{foundation} / {condition}" for foundation in FOUNDATIONS for condition in CONDITIONS)


def build_dataset() -> pd.DataFrame:
    density = pd.read_csv(DATA / "figure2_density_states.csv")
    viscosity = pd.read_csv(DATA / "figure3_transport_states_0p5fs.csv")

    density = density.rename(columns={"model": "model_label"})

    density_columns = [
        "model_label",
        "electrostatics",
        "condition",
        "temperature_K",
        "replicas",
        "density_mean_g_cm3",
        "density_replica_sd_g_cm3",
        "density_replica_sem_g_cm3",
        "iapws95_density_g_cm3",
    ]
    viscosity_columns = [
        "model_label",
        "temperature_K",
        "viscosity_mPa_s",
        "viscosity_mPa_s_sem",
        "viscosity_mPa_s_n",
        "experimental_viscosity_mPa_s",
    ]
    merged = density[density_columns].merge(
        viscosity[viscosity_columns],
        on=["model_label", "temperature_K"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 24 or set(merged.model_label) != set(MODEL_ORDER):
        raise ValueError("expected six matched models at four temperatures")

    merged = merged.rename(columns={"electrostatics": "backbone"})
    merged["density_times_viscosity_g_cm3_mPa_s"] = (
        merged.density_mean_g_cm3 * merged.viscosity_mPa_s
    )
    # Density and viscosity are separate compact state summaries. With no
    # cross-covariance available, propagate their reported SEMs independently.
    merged["density_times_viscosity_sem_g_cm3_mPa_s"] = np.sqrt(
        (merged.viscosity_mPa_s * merged.density_replica_sem_g_cm3) ** 2
        + (merged.density_mean_g_cm3 * merged.viscosity_mPa_s_sem) ** 2
    )
    merged["reference_density_times_viscosity_g_cm3_mPa_s"] = (
        merged.iapws95_density_g_cm3 * merged.experimental_viscosity_mPa_s
    )
    merged["ratio_to_reference"] = (
        merged.density_times_viscosity_g_cm3_mPa_s
        / merged.reference_density_times_viscosity_g_cm3_mPa_s
    )

    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    merged["_model_order"] = merged.model_label.map(order)
    merged = merged.sort_values(["_model_order", "temperature_K"]).drop(columns="_model_order")
    output_columns = [
        "model_label",
        "backbone",
        "condition",
        "temperature_K",
        "replicas",
        "density_mean_g_cm3",
        "density_replica_sd_g_cm3",
        "density_replica_sem_g_cm3",
        "viscosity_mPa_s",
        "viscosity_mPa_s_sem",
        "viscosity_mPa_s_n",
        "density_times_viscosity_g_cm3_mPa_s",
        "density_times_viscosity_sem_g_cm3_mPa_s",
        "iapws95_density_g_cm3",
        "experimental_viscosity_mPa_s",
        "reference_density_times_viscosity_g_cm3_mPa_s",
        "ratio_to_reference",
    ]
    merged = merged[output_columns]
    merged.to_csv(DATA / "density_times_viscosity_states.csv", index=False)
    return merged


def plot(states: pd.DataFrame) -> None:
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
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.78, wspace=0.12)

    reference = (
        states[
            [
                "temperature_K",
                "reference_density_times_viscosity_g_cm3_mPa_s",
            ]
        ]
        .drop_duplicates()
        .sort_values("temperature_K")
    )
    if len(reference) != 4:
        raise ValueError("reference product must be unique at each temperature")

    for column, backbone in enumerate(FOUNDATIONS):
        ax = axes[column]
        for condition in CONDITIONS:
            group = states[
                (states.backbone == backbone) & (states.condition == condition)
            ].sort_values("temperature_K")
            ax.errorbar(
                group.temperature_K,
                group.density_times_viscosity_g_cm3_mPa_s,
                yerr=group.density_times_viscosity_sem_g_cm3_mPa_s,
                color=COLORS[condition],
                marker=MARKERS[condition],
                capsize=2,
                label=condition,
            )
        ax.plot(
            reference.temperature_K,
            reference.reference_density_times_viscosity_g_cm3_mPa_s,
            color="black",
            marker="o",
            markersize=3.5,
            lw=1.7,
            label="IAPWS density × experimental viscosity",
        )
        ax.set_title(backbone)
        ax.set_xlabel("Temperature (K)")
        ax.set_xticks([275, 295, 315, 336])
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", alpha=0.25)
        ax.grid(axis="y", which="minor", alpha=0.10)
        ax.text(-0.12, 1.04, "ab"[column], transform=ax.transAxes, fontweight="bold")

    axes[0].set_ylabel(r"Density × viscosity, $\rho\eta$ (g cm$^{-3}$ mPa s)")
    handles = [
        Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=c)
        for c in CONDITIONS
    ] + [
        Line2D([], [], color="black", marker="o", markersize=3.5, lw=1.7, label="reference")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.91))
    fig.suptitle("Density–viscosity product across temperature", y=0.995, fontsize=10)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "density_times_viscosity.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / "density_times_viscosity.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    states = build_dataset()
    plot(states)
    print(DATA / "density_times_viscosity_states.csv")
    print(OUTPUT / "density_times_viscosity.png")
    print(OUTPUT / "density_times_viscosity.pdf")


if __name__ == "__main__":
    main()
