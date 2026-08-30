#!/usr/bin/env python3
"""Evaluate one fused model on the G1--G6 molecular-cluster split."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from ase.io import iread
from mace.calculators import MACECalculator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-updates", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    calculator = MACECalculator(models=model, device=args.device, default_dtype="float64", model_type="MACE")
    totals = {group: {"n": 0, "energy_sq": 0.0, "force_sq": 0.0, "force_n": 0} for group in range(1, 7)}
    for atoms in iread(args.data, index=":"):
        group = len(atoms) // 3
        if group not in totals or 3 * group != len(atoms):
            continue
        reference_energy = float(atoms.info["REF_energy"])
        reference_forces = np.asarray(atoms.arrays["REF_forces"], dtype=float)
        atoms.calc = calculator
        energy_error_per_atom = (atoms.get_potential_energy() - reference_energy) / len(atoms)
        force_error = atoms.get_forces() - reference_forces
        totals[group]["n"] += 1
        totals[group]["energy_sq"] += energy_error_per_atom**2
        totals[group]["force_sq"] += float(np.sum(force_error**2))
        totals[group]["force_n"] += force_error.size

    rows = []
    for group, values in totals.items():
        if not values["n"]:
            raise ValueError(f"no G{group} configurations in {args.data}")
        rows.append({
            "model_label": args.model_label,
            "training_seed": "" if args.seed is None else args.seed,
            "optimizer_updates": "" if args.optimizer_updates is None else args.optimizer_updates,
            "cluster_group": f"G{group}",
            "configurations": values["n"],
            "total_energy_rmse_meV_per_atom": 1000 * math.sqrt(values["energy_sq"] / values["n"]),
            "total_force_component_rmse_meV_per_A": 1000 * math.sqrt(values["force_sq"] / values["force_n"]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
