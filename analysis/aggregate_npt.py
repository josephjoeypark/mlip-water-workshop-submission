#!/usr/bin/env python3
"""Aggregate completed NPT lanes into Figure 2 density and RDF tables."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parents[1] / "runs/npt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "runs/npt_aggregate")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/study.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    reference = pd.read_csv(Path(__file__).parents[1] / "data/paper_figures/figure2_density_states.csv")
    iapws = reference.groupby("temperature_K").iapws95_density_g_cm3.first().to_dict()
    density_rows, rdf_rows = [], []
    for model_id, model in config["models"].items():
        label = f"{model['foundation']} / {model['condition']}"
        for temperature in config["md"]["temperatures_K"]:
            analyses, rdfs = [], []
            for replica in range(1, config["md"]["replicas"] + 1):
                lane = args.runs / model_id / f"T{temperature}" / f"replica{replica}"
                analyses.append(json.loads((lane / "npt_analysis.json").read_text()))
                if temperature == 295:
                    rdfs.append(pd.read_csv(lane / "rdf.csv"))
            values = pd.Series([item["density_mean_g_per_cm3"] for item in analyses])
            mean, sd = values.mean(), values.std(ddof=1)
            density_rows.append({
                "model": label, "condition": model["condition"], "electrostatics": model["foundation"],
                "temperature_K": temperature, "replicas": len(values), "density_mean_g_cm3": mean,
                "density_replica_sd_g_cm3": sd, "density_replica_sem_g_cm3": sd / math.sqrt(len(values)),
                "iapws95_density_g_cm3": iapws[temperature], "error_g_cm3": mean - iapws[temperature],
                "relative_error_percent": 100 * (mean / iapws[temperature] - 1),
            })
            if rdfs:
                joined = pd.concat([frame.assign(replica=i + 1) for i, frame in enumerate(rdfs)])
                for radius, group in joined.groupby("r_A", sort=True):
                    rdf_rows.append({
                        "model": label, "condition": model["condition"], "electrostatics": model["foundation"],
                        "temperature_K": temperature, "replicas": len(rdfs), "r_A": radius,
                        "g_OO_mean": group.g_OO.mean(), "g_OO_replica_sd": group.g_OO.std(ddof=1),
                        "g_OH_intermolecular_mean": group.g_OH_intermolecular.mean(),
                        "g_OH_intermolecular_replica_sd": group.g_OH_intermolecular.std(ddof=1),
                    })
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(density_rows).to_csv(args.output / "figure2_density_states.csv", index=False)
    pd.DataFrame(rdf_rows).to_csv(args.output / "figure2_rdf_states_295K.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
