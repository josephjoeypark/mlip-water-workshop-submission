#!/usr/bin/env python3
"""Train-only residual scales and joint auxiliary-gradient calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    PilotCollator,
    PilotDataset,
    batch_to_device,
    calibration_records,
    sha256,
)
from loss import calibrated_lambdas, objective_terms
from model import load_model, trainable_state


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if "test" in args.train.name.lower():
        raise ValueError("locked-test inputs are forbidden")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_manifest = json.loads(args.source_manifest.read_text())
    if not source_manifest.get("passed"):
        raise ValueError("source data manifest did not pass")
    if sha256(args.train) != source_manifest["products"]["train"]["sha256"]:
        raise ValueError("training JSONL differs from gated source")

    dataset = PilotDataset(args.train)
    model = load_model(args.foundation, seed=args.seed, device=device)
    initial_hash = state_sha256(trainable_state(model))
    collator = PilotCollator(
        atomic_numbers=model.trunk.atomic_numbers.detach().cpu().tolist(),
        cutoff=float(model.trunk.r_max),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    sums = {
        "D11_energy": 0.0,
        "D11_force": 0.0,
        "D11N_energy": 0.0,
        "D11N_force": 0.0,
    }
    energy_count = 0
    force_count = 0
    model.eval()
    for host_batch in loader:
        batch = batch_to_device(host_batch, device)
        current = model(
            batch["clusters"],
            batch["monomers"],
            batch["monomer_owner"],
            training=False,
            compute_counterfactual=True,
        )
        atoms = batch["reference"]["num_atoms"]
        for condition, prefix in (("D11", ""), ("D11N", "null_")):
            energy_residual = (
                current["interaction_energy"]
                - batch["reference"][f"{prefix}interaction_energy"]
            ) / atoms
            force_residual = (
                current["interaction_forces"]
                - batch["reference"][f"{prefix}interaction_forces"]
            )
            sums[f"{condition}_energy"] += float(energy_residual.square().sum().cpu())
            sums[f"{condition}_force"] += float(force_residual.square().sum().cpu())
        energy_count += len(host_batch["records"])
        force_count += int(batch["reference"]["total_forces"].numel())
    scales = {
        f"{condition}_energy_eV_per_atom": (sums[f"{condition}_energy"] / energy_count) ** 0.5
        for condition in ("D11", "D11N")
    }
    scales.update(
        {
            f"{condition}_force_eV_per_A": (sums[f"{condition}_force"] / force_count) ** 0.5
            for condition in ("D11", "D11N")
        }
    )
    if not all(np.isfinite(value) and value > 0.0 for value in scales.values()):
        raise FloatingPointError(f"invalid train-only scales: {scales}")

    host_calibration = collator(calibration_records(dataset))
    batch = batch_to_device(host_calibration, device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    calibrations: dict[str, Any] = {}
    record_keys = [record["record_key"] for record in host_calibration["records"]]
    for condition in ("D11", "D11N"):
        model.zero_grad(set_to_none=True)
        current = model(
            batch["clusters"],
            batch["monomers"],
            batch["monomer_owner"],
            training=True,
            compute_counterfactual=True,
        )
        terms = objective_terms(
            current, batch["reference"], condition=condition, scales=scales
        )
        calibration = calibrated_lambdas(terms, parameters, target_fraction=0.075)
        calibration["initial_terms"] = {
            name: float(value.detach().cpu()) for name, value in terms.items()
        }
        calibration["record_keys"] = record_keys
        calibrations[condition] = calibration

    result = {
        "schema_version": "single-mace-counterfactual-calibration-v1",
        "passed": True,
        "foundation": str(args.foundation.resolve()),
        "foundation_sha256": sha256(args.foundation),
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "source_manifest_sha256": sha256(args.source_manifest),
        "seed": args.seed,
        "dtype": "float64",
        "initial_trainable_sha256": initial_hash,
        "data_audit": dataset.audit,
        "scales": scales,
        "calibrations": calibrations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
