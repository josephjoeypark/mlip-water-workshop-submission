#!/usr/bin/env python3
"""Aggregate per-seed cluster evaluations into the two Figure 1 CSV files."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    values[["backbone", "condition"]] = values.model_label.str.split(" / ", n=1, expand=True)
    rows = []
    for (backbone, condition, cluster), group in values.groupby(["backbone", "condition", "cluster_group"], sort=False):
        seeds = group.training_seed.dropna().astype(int).tolist()
        rows.append({
            "backbone": backbone, "condition": condition, "cluster_group": cluster,
            "training_seeds": ",".join(map(str, seeds)) if seeds else "released",
            "total_energy_rmse_mean_meV_per_atom": group.total_energy_rmse_meV_per_atom.mean(),
            "total_energy_rmse_sample_sd_meV_per_atom": group.total_energy_rmse_meV_per_atom.std(ddof=1),
            "total_force_component_rmse_mean_meV_per_A": group.total_force_component_rmse_meV_per_A.mean(),
            "total_force_component_rmse_sample_sd_meV_per_A": group.total_force_component_rmse_meV_per_A.std(ddof=1),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "figure1_cluster_rmse_by_group.csv", index=False)
    seed17 = values[(values.training_seed == 17) | values.training_seed.isna()].copy()
    seed17["optimizer_updates"] = seed17.optimizer_updates.where(seed17.condition != "Baseline")
    columns = ["model_label", "backbone", "condition", "training_seed", "optimizer_updates",
               "cluster_group", "configurations", "total_energy_rmse_meV_per_atom",
               "total_force_component_rmse_meV_per_A"]
    seed17[columns].to_csv(args.output / "figure1_seed17.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
