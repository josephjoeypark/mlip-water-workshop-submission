#!/usr/bin/env python3
"""Package trajectory-level transport output into compact Figure 3 inputs."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


KB = 1.380649e-23
KEYS = {
    "MACEOFF23(M) / Baseline": "off_zs", "MACEOFF23(M) / G1-G2": "off_t12",
    "MACEOFF23(M) / G1-G6": "off_d00", "MACELES-OFF / Baseline": "les_zs",
    "MACELES-OFF / G1-G2": "les_t12", "MACELES-OFF / G1-G6": "les_d00",
}


def mean_sem(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    length = min(map(len, values)); stack = np.stack([value[:length] for value in values])
    return stack.mean(axis=0), stack.std(axis=0, ddof=1) / math.sqrt(len(stack))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lanes = pd.read_csv(args.analysis / "lanes.csv")
    states = pd.read_csv(args.analysis / "states.csv")
    arrays, decomposition = {}, []
    for model, key in KEYS.items():
        group = lanes[(lanes.model_label == model) & (lanes.temperature_K == 295)]
        curves = [np.load(args.analysis / row.curve_file) for row in group.itertuples()]
        diffusion = [curve["running_diffusion_pbc_nm2_ns"] for curve in curves]
        d_mean, d_sem = mean_sem(diffusion)
        if "diffusion_time_ps" not in arrays:
            arrays["diffusion_time_ps"] = curves[0]["vacf_time_ps"][:len(d_mean)]
        arrays[f"{key}_running_D_mean"], arrays[f"{key}_running_D_sem"] = d_mean, d_sem
        stress = [curve["stress_diagonal_Pa2"] / curve["stress_diagonal_Pa2"][0] for curve in curves]
        s_mean, s_sem = mean_sem(stress)
        eta = [curve["running_viscosity_mPa_s"] for curve in curves]
        e_mean, e_sem = mean_sem(eta)
        if "stress_time_ps" not in arrays:
            arrays["stress_time_ps"] = curves[0]["stress_time_ps"][:len(s_mean)]
        arrays[f"{key}_stress_normalized_mean"], arrays[f"{key}_stress_normalized_sem"] = s_mean, s_sem
        arrays[f"{key}_running_eta_mean"], arrays[f"{key}_running_eta_sem"] = e_mean, e_sem
        if model.endswith("G1-G6"):
            for row, curve in zip(group.itertuples(), curves):
                time = curve["stress_time_ps"]; mask = (time >= 5) & (time <= 10)
                c0 = float(curve["stress_diagonal_Pa2"][0]); eta_window = float(curve["running_viscosity_mPa_s"][mask].mean())
                modulus = row.volume_A3 * 1e-30 * c0 / (KB * row.mean_temperature_K)
                decomposition.append({
                    "model_label": model, "backbone": row.backbone, "temperature_K": 295,
                    "md_replica": row.md_replica, "volume_A3": row.volume_A3,
                    "mean_temperature_K": row.mean_temperature_K, "C_sigma_0_Pa2": c0,
                    "G_infinity_Pa": modulus, "eta_5_10ps_mPa_s": eta_window,
                    "tau_bar_5_10ps_ps": eta_window * 1e9 / modulus,
                })
        for curve in curves:
            curve.close()
    args.output.mkdir(parents=True, exist_ok=True)
    states.to_csv(args.output / "transport_0p5fs.csv", index=False)
    np.savez_compressed(args.output / "transport_curves_295K_0p5fs.npz", **arrays)
    pd.DataFrame(decomposition).to_csv(args.output / "stress_decomposition_295K.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
