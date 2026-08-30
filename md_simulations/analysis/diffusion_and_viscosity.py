#!/usr/bin/env python3
"""Analyze the six-model NVT campaign and write diffusion/viscosity tables."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parents[2] / "runs")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[2] / "runs/transport")
    parser.add_argument("--prefix-ps", type=float, default=700.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "configs/study.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    temperatures = config["md"]["temperatures_K"]
    rows = []
    for model_id, model in config["models"].items():
        for temperature in temperatures:
            for replica in range(1, config["md"]["replicas"] + 1):
                rows.append({
                    "lane_id": len(rows), "pack_id": 0,
                    "model_label": f"{model['foundation']} / {model['condition']}",
                    "backbone": model["foundation"], "condition": model["condition"],
                    "training_seed": "" if model["condition"] == "Baseline" else config["training"]["seed"],
                    "temperature_K": temperature, "md_replica": replica,
                    "md_seed": temperature * 10000 + replica * 1000 + 729,
                    "stored_precision": model["stored_precision"],
                    "model_path": str((args.config.parent / model["path"]).resolve()),
                    "model_sha256": model["sha256"],
                    "output_dir": str((args.runs / "nvt" / model_id / f"T{temperature}" / f"replica{replica}").resolve()),
                })
    manifest = args.output.parent / "transport_lane_manifest.tsv"
    command = [sys.executable, Path(__file__).parent / "internal/transport_estimators.py",
               "--manifest", manifest, "--output-dir", args.output, "--workers", args.workers,
               "--target-prefix-ps", args.prefix_ps, "--complete-segments-only"]
    print("$ " + " ".join(map(str, command)))
    if args.dry_run:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    subprocess.run([str(value) for value in command], check=True)


if __name__ == "__main__":
    main()
