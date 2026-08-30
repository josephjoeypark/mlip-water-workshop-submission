#!/usr/bin/env python3.11
"""Build the two manuscript tables from frozen parameter and BEC values."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "paper_tables"

PARAMETERS = (
    {
        "model": "MACEOFF23(M)",
        "canonical_base_or_g1_g6_fused_parameters": 1_428_368,
        "g1_g2_fused_export_parameters": 1_428_376,
        "mace_lora_parameters": 235_392,
        "les_lora_parameters": 0,
        "total_lora_parameters": 235_392,
        "total_during_lora_training": 1_663_760,
        "trainable_fraction_percent": 14.15,
    },
    {
        "model": "MACELES-OFF",
        "canonical_base_or_g1_g6_fused_parameters": 1_891_360,
        "g1_g2_fused_export_parameters": 1_891_368,
        "mace_lora_parameters": 235_392,
        "les_lora_parameters": 117_824,
        "total_lora_parameters": 353_216,
        "total_during_lora_training": 2_244_576,
        "trainable_fraction_percent": 15.74,
    },
)

MODEL_ORDER = (
    "MACELES-OFF / Baseline",
    "MACELES-OFF / G1-G2",
    "MACELES-OFF / G1-G6",
)
REFERENCE = {"H": (0.430, 0.080), "O": (-0.860, 0.092)}


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_bec_rows() -> list[dict]:
    source = ROOT / "data" / "response" / "les_bec_64water_frame0.csv"
    grouped: dict[tuple[str, str], list[float]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["element"])
            grouped.setdefault(key, []).append(float(row["BEC_trace_over_3"]))

    rows = [
        {
            "source_or_model": "RPBE-D3 reference",
            "frame": "liquid-water ensemble",
            "H_mean_e": REFERENCE["H"][0],
            "H_sd_e": REFERENCE["H"][1],
            "O_mean_e": REFERENCE["O"][0],
            "O_sd_e": REFERENCE["O"][1],
            "H_raw_mean_e": "",
            "O_raw_mean_e": "",
            "sign_treatment": "conventional",
            "two_H_plus_O_e": 0.0,
        }
    ]
    for model in MODEL_ORDER:
        hydrogen = grouped[(model, "H")]
        oxygen = grouped[(model, "O")]
        # The latent polarization has a global sign ambiguity.  Apply one
        # global sign flip for comparison with the conventional water parity.
        h_mean = -statistics.fmean(hydrogen)
        o_mean = -statistics.fmean(oxygen)
        rows.append(
            {
                "source_or_model": model,
                "frame": 0,
                "H_mean_e": h_mean,
                "H_sd_e": statistics.pstdev(hydrogen),
                "O_mean_e": o_mean,
                "O_sd_e": statistics.pstdev(oxygen),
                "H_raw_mean_e": -h_mean,
                "O_raw_mean_e": -o_mean,
                "sign_treatment": "raw LES multiplied by -1",
                "two_H_plus_O_e": 2.0 * h_mean + o_mean,
            }
        )
    return rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "table1_model_parameters.csv", list(PARAMETERS))
    write_rows(OUTPUT / "table2_bec_comparison.csv", build_bec_rows())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
