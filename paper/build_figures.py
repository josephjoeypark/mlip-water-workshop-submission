#!/usr/bin/env python3
"""Build the three manuscript composite figures from committed compact data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = ROOT / "model_training" / "data"
MD_DATA = ROOT / "md_simulations" / "data" / "published_results"
OUTPUT = ROOT / "paper" / "figures"

COLORS = {"Baseline": "#4C78A8", "G1-G2": "#F58518", "G1-G6": "#54A24B"}
FOUNDATION_STYLES = {"MACEOFF23(M)": "-", "MACELES-OFF": "--"}
MARKERS = {"Baseline": "o", "G1-G2": "s", "G1-G6": "^"}
FOUNDATIONS = ("MACEOFF23(M)", "MACELES-OFF")
CONDITIONS = ("Baseline", "G1-G2", "G1-G6")
MODEL_ORDER = tuple(f"{foundation} / {condition}" for foundation in FOUNDATIONS for condition in CONDITIONS)
# The array keys preserve the archived campaign's internal labels.  They are a
# storage detail, not public model names.
CURVE_KEYS = dict(
    zip(
        MODEL_ORDER,
        ("off_zs", "off_t12", "off_d00", "les_zs", "les_t12", "les_d00"),
    )
)


def setup_style() -> None:
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


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def shade_aggregation_regions(ax: plt.Axes, annotate: bool = False) -> None:
    ax.axvspan(0.65, 2.5, color="#DCE9F4", alpha=0.55, lw=0)
    ax.axvspan(2.5, 6.35, color="#F6E4CF", alpha=0.45, lw=0)
    if annotate:
        ax.text(
            1.55,
            0.98,
            "shared coverage",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.7,
            color="0.30",
        )
        ax.text(
            4.45,
            0.98,
            "G1-G6-only coverage",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.7,
            color="0.30",
        )


def build_figure1() -> None:
    summary = pd.read_csv(MODEL_DATA / "force_energy_results.csv")
    seed17 = pd.read_csv(MODEL_DATA / "force_energy_seed17.csv")
    summary["group"] = summary["cluster_group"].str.extract(r"(\d+)").astype(int)
    seed17["group"] = seed17["cluster_group"].str.extract(r"(\d+)").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), sharex="col", constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.88, wspace=0.18, hspace=0.12)

    for column, backbone in enumerate(FOUNDATIONS):
        energy_ax = axes[0, column]
        force_ax = axes[1, column]
        for ax in (energy_ax, force_ax):
            shade_aggregation_regions(ax, annotate=ax is energy_ax)
            ax.set_yscale("log")
            ax.set_xlim(0.75, 6.25)
            ax.set_xticks(range(1, 7), [f"G{i}" for i in range(1, 7)])
            ax.grid(axis="y", which="major", alpha=0.25)
            ax.grid(axis="y", which="minor", alpha=0.10)

        for condition in ("G1-G2", "G1-G6"):
            group = summary[(summary.backbone == backbone) & (summary.condition == condition)].sort_values("group")
            color = COLORS[condition]
            energy_ax.errorbar(
                group.group,
                group.total_energy_rmse_mean_meV_per_atom,
                yerr=group.total_energy_rmse_sample_sd_meV_per_atom,
                color=color,
                marker=MARKERS[condition],
                capsize=2,
                label=condition,
                zorder=3,
            )
            force_ax.errorbar(
                group.group,
                group.total_force_component_rmse_mean_meV_per_A,
                yerr=group.total_force_component_rmse_sample_sd_meV_per_A,
                color=color,
                marker=MARKERS[condition],
                capsize=2,
                label=condition,
                zorder=3,
            )
            deployed = seed17[(seed17.backbone == backbone) & (seed17.condition == condition)].sort_values("group")
            energy_ax.scatter(
                deployed.group,
                deployed.total_energy_rmse_meV_per_atom,
                marker="D",
                s=17,
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                zorder=5,
            )
            force_ax.scatter(
                deployed.group,
                deployed.total_force_component_rmse_meV_per_A,
                marker="D",
                s=17,
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                zorder=5,
            )

        zs = summary[(summary.backbone == backbone) & (summary.condition == "Baseline")].sort_values("group")
        force_ax.plot(
            zs.group,
            zs.total_force_component_rmse_mean_meV_per_A,
            color=COLORS["Baseline"],
            marker=MARKERS["Baseline"],
            label="Baseline",
            zorder=3,
        )
        energy_ax.set_title(backbone)
        force_ax.set_xlabel("Aggregation group")

    axes[0, 0].set_ylabel("Energy RMSE (meV atom$^{-1}$)")
    axes[1, 0].set_ylabel(r"Force-component RMSE (meV $\mathrm{\AA}^{-1}$)")
    panel_label(axes[0, 0], "a")
    panel_label(axes[0, 1], "b")
    panel_label(axes[1, 0], "c")
    panel_label(axes[1, 1], "d")

    legend = [
        Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=c) for c in CONDITIONS
    ]
    legend.append(
        Line2D([], [], color="0.25", marker="D", markerfacecolor="white", linestyle="none", label="deployed seed 17")
    )
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.985))
    fig.suptitle("Molecular transfer across aggregation order", y=0.995, fontsize=10)
    save(fig, "figure1_molecular_transfer")


def model_parts(model: str, condition: str, electrostatics: str) -> tuple[str, str]:
    del model
    return electrostatics, condition


def build_figure2() -> None:
    density = pd.read_csv(MD_DATA / "density.csv")
    rdf = pd.read_csv(MD_DATA / "rdf_295K.csv")
    experiment = pd.read_csv(MD_DATA / "soper_298K.csv")
    density[["backbone", "plot_condition"]] = density.apply(
        lambda row: model_parts(row.model, row.condition, row.electrostatics), axis=1, result_type="expand"
    )
    rdf[["backbone", "plot_condition"]] = rdf.apply(
        lambda row: model_parts(row.model, row.condition, row.electrostatics), axis=1, result_type="expand"
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.0), sharey="row", constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.91, wspace=0.12, hspace=0.38)
    for column, backbone in enumerate(FOUNDATIONS):
        axes[0, column].set_title(backbone)
        subset = density[density.backbone == backbone]
        for condition in CONDITIONS:
            group = subset[subset.plot_condition == condition].sort_values("temperature_K")
            axes[0, column].errorbar(
                group.temperature_K,
                group.density_mean_g_cm3,
                yerr=group.density_replica_sd_g_cm3,
                color=COLORS[condition],
                marker=MARKERS[condition],
                capsize=2,
                label=condition,
            )
        reference = subset[["temperature_K", "iapws95_density_g_cm3"]].drop_duplicates().sort_values("temperature_K")
        axes[0, column].plot(reference.temperature_K, reference.iapws95_density_g_cm3, color="black", lw=1.7, label="IAPWS-95")
        axes[0, column].set_xticks([275, 295, 315, 336])
        axes[0, column].set_xlabel("Temperature (K)")
        axes[0, column].grid(alpha=0.22)

        rsub = rdf[rdf.backbone == backbone]
        for condition in CONDITIONS:
            group = rsub[rsub.plot_condition == condition].sort_values("r_A")
            axes[1, column].plot(group.r_A, group.g_OO_mean, color=COLORS[condition], label=condition)
            axes[1, column].fill_between(
                group.r_A,
                group.g_OO_mean - group.g_OO_replica_sd,
                group.g_OO_mean + group.g_OO_replica_sd,
                color=COLORS[condition],
                alpha=0.13,
                lw=0,
            )
            axes[2, column].plot(group.r_A, group.g_OH_intermolecular_mean, color=COLORS[condition], label=condition)
            axes[2, column].fill_between(
                group.r_A,
                group.g_OH_intermolecular_mean - group.g_OH_intermolecular_replica_sd,
                group.g_OH_intermolecular_mean + group.g_OH_intermolecular_replica_sd,
                color=COLORS[condition],
                alpha=0.13,
                lw=0,
            )

        oo = experiment[experiment.pair == "OO"].sort_values("r_A")
        oh = experiment[experiment.pair == "OH_intermolecular"].sort_values("r_A")
        axes[1, column].plot(oo.r_A, oo.g_r, color="black", lw=1.6, label="Soper (298 K)")
        axes[1, column].fill_between(
            oo.r_A, oo.g_r - oo.g_r_uncertainty, oo.g_r + oo.g_r_uncertainty, color="black", alpha=0.10, lw=0
        )
        axes[2, column].plot(oh.r_A, oh.g_r, color="black", lw=1.6, label="Soper (298 K)")
        axes[2, column].fill_between(
            oh.r_A, oh.g_r - oh.g_r_uncertainty, oh.g_r + oh.g_r_uncertainty, color="black", alpha=0.10, lw=0
        )
        axes[1, column].set_xlim(2.2, 5.5)
        axes[2, column].set_xlim(1.3, 4.5)
        axes[1, column].set_ylim(-0.05, 3.25)
        axes[2, column].set_ylim(-0.03, 1.80)
        axes[1, column].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        axes[2, column].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        for row in (1, 2):
            axes[row, column].grid(alpha=0.20)

    axes[0, 0].set_ylabel(r"Density (g cm$^{-3}$)")
    axes[1, 0].set_ylabel(r"$g_{\mathrm{OO}}(r)$")
    axes[2, 0].set_ylabel(r"Intermolecular $g_{\mathrm{OH}}(r)$")
    for ax, label in zip(axes.ravel(), "abcdef"):
        panel_label(ax, label)

    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=c) for c in CONDITIONS]
    handles += [Line2D([], [], color="black", lw=1.7, label="experiment")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.965))
    fig.suptitle("Transfer to equilibrium density and local liquid structure", y=0.997, fontsize=10)
    save(fig, "figure2_equilibrium_structure")


def experimental_diffusion(temperature: np.ndarray) -> np.ndarray:
    return 16.35 * ((temperature / 215.05) - 1.0) ** 2.063


def experimental_viscosity(temperature: np.ndarray) -> np.ndarray:
    reduced = temperature / 300.0
    coefficients = ((280.68, -1.9), (511.45, -7.7), (61.131, -19.6), (0.45903, -40.0))
    return sum(a * reduced**b for a, b in coefficients) / 1000.0


def build_figure3() -> None:
    states = pd.read_csv(MD_DATA / "transport_0p5fs.csv")
    curves = np.load(MD_DATA / "transport_curves_295K_0p5fs.npz")
    decomposition = pd.read_csv(MD_DATA / "stress_decomposition_295K.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.35), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.88, wspace=0.23, hspace=0.27)
    ax_d, ax_eta, ax_running, ax_stress = axes.ravel()

    temp_grid = np.linspace(273.0, 340.0, 250)
    ax_d.plot(temp_grid, experimental_diffusion(temp_grid), color="black", lw=1.8, label="experiment")
    ax_eta.plot(temp_grid, experimental_viscosity(temp_grid), color="black", lw=1.8, label="experiment")

    for model in MODEL_ORDER:
        group = states[states.model_label == model].sort_values("temperature_K")
        backbone, condition = model.split(" / ")
        style = FOUNDATION_STYLES[backbone]
        color = COLORS[condition]
        ax_d.errorbar(
            group.temperature_K,
            group.diffusion_vacf_corrected_nm2_ns,
            yerr=group.diffusion_vacf_corrected_nm2_ns_sem,
            color=color,
            ls=style,
            marker=MARKERS[condition],
            capsize=1.8,
            label=model,
        )
        ax_eta.errorbar(
            group.temperature_K,
            group.viscosity_mPa_s,
            yerr=group.viscosity_mPa_s_sem,
            color=color,
            ls=style,
            marker=MARKERS[condition],
            capsize=1.8,
            label=model,
        )
        key = CURVE_KEYS[model]
        time = curves["diffusion_time_ps"]
        mask = time <= 10.0
        mean = curves[f"{key}_running_D_mean"]
        sem = curves[f"{key}_running_D_sem"]
        ax_running.plot(time[mask], mean[mask], color=color, ls=style, label=model)
        ax_running.fill_between(time[mask], mean[mask] - sem[mask], mean[mask] + sem[mask], color=color, alpha=0.10, lw=0)

    ax_d.set_ylabel(r"Corrected $D_\infty$ (nm$^2$ ns$^{-1}$)")
    ax_eta.set_ylabel(r"Viscosity $\eta$ (mPa s)")
    ax_d.set_xlabel("Temperature (K)")
    ax_eta.set_xlabel("Temperature (K)")
    ax_eta.set_yscale("log")
    ax_d.set_xticks([275, 295, 315, 336])
    ax_eta.set_xticks([275, 295, 315, 336])
    ax_running.axvspan(5.0, 10.0, color="0.7", alpha=0.22, lw=0)
    ax_running.text(7.5, 0.97, "5–10 ps", transform=ax_running.get_xaxis_transform(), ha="center", va="top", color="0.35")
    ax_running.set(xlabel="VACF integration limit (ps)", ylabel=r"Raw $D_L(t)$ (nm$^2$ ns$^{-1}$)", xlim=(0, 10))

    stress_time = curves["stress_time_ps"]
    stress_mask = stress_time <= 2.0
    stress_models = ("MACEOFF23(M) / G1-G6", "MACELES-OFF / G1-G6")
    for model in stress_models:
        key = CURVE_KEYS[model]
        style = FOUNDATION_STYLES[model.split(" / ")[0]]
        mean = curves[f"{key}_stress_normalized_mean"]
        sem = curves[f"{key}_stress_normalized_sem"]
        ax_stress.plot(
            stress_time[stress_mask], mean[stress_mask], color=COLORS["G1-G6"],
            ls=style, label=model.split(" / ")[0],
        )
        ax_stress.fill_between(
            stress_time[stress_mask], mean[stress_mask] - sem[stress_mask], mean[stress_mask] + sem[stress_mask],
            color=COLORS["G1-G6"], alpha=0.12, lw=0,
        )
    ax_stress.axhline(0.0, color="0.45", lw=0.7)
    ax_stress.set(
        xlabel="Correlation time (ps)",
        ylabel=r"$C_\sigma(t)/C_\sigma(0)$",
        xlim=(0, 2.0),
    )
    ax_stress.legend(frameon=False, loc="upper right")

    inset = inset_axes(ax_stress, width="46%", height="42%", loc="center right", borderpad=1.0)
    eta_mask = stress_time <= 10.0
    for model in stress_models:
        key = CURVE_KEYS[model]
        style = FOUNDATION_STYLES[model.split(" / ")[0]]
        mean = curves[f"{key}_running_eta_mean"]
        sem = curves[f"{key}_running_eta_sem"]
        inset.plot(stress_time[eta_mask], mean[eta_mask], color=COLORS["G1-G6"], ls=style)
        inset.fill_between(
            stress_time[eta_mask], mean[eta_mask] - sem[eta_mask], mean[eta_mask] + sem[eta_mask],
            color=COLORS["G1-G6"], alpha=0.10, lw=0,
        )
    inset.axvspan(5.0, 10.0, color="0.7", alpha=0.20, lw=0)
    inset.set(xlim=(0, 10), xlabel="$t$ (ps)", ylabel=r"$\eta(t)$")
    inset.tick_params(labelsize=5.7)
    inset.xaxis.label.set_size(6.0)
    inset.yaxis.label.set_size(6.0)

    means = decomposition.groupby("model_label").mean(numeric_only=True)
    off, les = means.loc[stress_models[0]], means.loc[stress_models[1]]
    eta_ratio = les.eta_5_10ps_mPa_s / off.eta_5_10ps_mPa_s
    g_ratio = les.G_infinity_Pa / off.G_infinity_Pa
    tau_ratio = les.tau_bar_5_10ps_ps / off.tau_bar_5_10ps_ps
    ax_stress.text(
        0.98,
        0.05,
        rf"MACELES/MACEOFF: $\eta$ {eta_ratio:.2f}$\times$, $G_\infty$ {g_ratio:.2f}$\times$, $\bar{{\tau}}$ {tau_ratio:.2f}$\times$",
        transform=ax_stress.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
    )

    for ax in axes.ravel():
        ax.grid(alpha=0.20)
    for ax, label in zip(axes.ravel(), "abcd"):
        panel_label(ax, label)

    encoding = [
        Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=c) for c in CONDITIONS
    ] + [
        Line2D([], [], color="0.25", ls="-", label="MACEOFF23(M)"),
        Line2D([], [], color="0.25", ls="--", label="MACELES-OFF"),
        Line2D([], [], color="black", lw=1.7, label="experiment"),
    ]
    fig.legend(handles=encoding, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.55, 0.975))
    fig.suptitle("Transport and collective relaxation from native 0.5 fs records", y=0.997, fontsize=10)
    save(fig, "figure3_transport_relaxation")


def main() -> None:
    setup_style()
    build_figure1()
    build_figure2()
    build_figure3()


if __name__ == "__main__":
    main()
