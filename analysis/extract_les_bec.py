#!/usr/bin/env python3
"""Extract model-native per-atom Born effective charge tensors from MACELES."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


MODELS = {
    "MACELES-OFF / Baseline": (
        "maceles-off-baseline.model",
        "44c0f363dd577eafc0912341f44e658fcf5d58d02b88da8682562d654889cf80",
    ),
    "MACELES-OFF / G1-G2": (
        "maceles-off-g1-g2.model",
        "4eced609dada4a0a6a266140477899c40b95a8f4ee3819044669f66b788fdc4d",
    ),
    "MACELES-OFF / G1-G6": (
        "maceles-off-g1-g6.model",
        "d0b58f57440dfbb7c946e7b18252eb44dd349cf016237eddd1885ff8ab9e1bb2",
    ),
}
COMPONENTS = tuple(a + b for a in "xyz" for b in "xyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--frame", default=0, type=int)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_water_box(atoms) -> None:
    symbols = np.asarray(atoms.get_chemical_symbols())
    if len(atoms) != 192:
        raise ValueError(f"expected 192 atoms (64 waters), found {len(atoms)}")
    if np.count_nonzero(symbols == "O") != 64 or np.count_nonzero(symbols == "H") != 128:
        raise ValueError("expected exactly 64 oxygen and 128 hydrogen atoms")
    if not bool(np.all(atoms.pbc)):
        raise ValueError("the 64-water frame must retain periodic boundary conditions")
    for start in range(0, len(atoms), 3):
        if sorted(symbols[start : start + 3].tolist()) != ["H", "H", "O"]:
            raise ValueError("expected each contiguous three-atom record to be one H2O")


def evaluate_bec(model_path: Path, atoms, device: str) -> tuple[np.ndarray, dict]:
    model = torch.load(model_path, map_location="cpu", weights_only=False).double().eval()
    calculator = MACECalculator(
        models=model,
        device=device,
        default_dtype="float64",
        model_type="MACE",
        compute_bec=True,
    )
    batch = calculator._atoms_to_batch(atoms)  # Matches production graph construction.
    data = batch.to_dict()
    model_dtype = next(model.parameters()).dtype
    for key in batch.keys:
        value = data[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.to(device=calculator.device, dtype=model_dtype)
        elif torch.is_tensor(value):
            data[key] = value.to(device=calculator.device)

    # BEC is a derivative of LES polarization with respect to positions.
    with torch.enable_grad():
        output = model(
            data,
            training=False,
            compute_force=False,
            compute_stress=False,
            compute_bec=True,
        )
    raw = np.asarray(output["BEC"].detach().cpu(), dtype=np.float64)
    raw_shape = list(raw.shape)
    if raw.ndim == 4:
        bec = np.sum(raw, axis=1)
    else:
        bec = raw
    if bec.shape != (len(atoms), 3, 3) or not np.all(np.isfinite(bec)):
        raise ValueError(f"invalid BEC output shape {bec.shape} from {model_path}")
    audit = {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "ewald_normalization_factor": float(model.les.ewald.norm_factor),
        "bec_epsilon_factor": float(model.les.bec.epsilon_factor),
        "bec_normalization_factor": float(model.les.bec.normalization_factor),
        "native_output_shape": raw_shape,
    }
    del output, data, batch, calculator, model
    gc.collect()
    return bec, audit


def main() -> int:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    atoms = read(args.input, index=args.frame)
    validate_water_box(atoms)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    symbols = atoms.get_chemical_symbols()

    values: dict[str, np.ndarray] = {}
    model_metadata: dict[str, dict] = {}
    for label, (filename, expected_hash) in MODELS.items():
        path = args.models_dir / filename
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch for {path}: {actual_hash}")
        bec, audit = evaluate_bec(path, atoms, args.device)
        values[label] = bec
        tensor_sum = np.sum(bec, axis=0)
        model_metadata[label] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "acoustic_sum_tensor": tensor_sum.tolist(),
            "acoustic_sum_max_abs": float(np.max(np.abs(tensor_sum))),
            "tensor_min": float(np.min(bec)),
            "tensor_max": float(np.max(bec)),
            **audit,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "model",
                "atom_index_0based",
                "atom_id_1based",
                "molecule_id_1based",
                "element",
                "x_A",
                "y_A",
                "z_A",
                *("BEC_" + component for component in COMPONENTS),
                "BEC_trace_over_3",
            ]
        )
        for label in MODELS:
            for index, (symbol, position) in enumerate(zip(symbols, positions)):
                tensor = values[label][index]
                writer.writerow(
                    [
                        args.frame,
                        label,
                        index,
                        index + 1,
                        index // 3 + 1,
                        symbol,
                        *(f"{coordinate:.10f}" for coordinate in position),
                        *(f"{value:.12g}" for value in tensor.reshape(-1)),
                        f"{np.trace(tensor) / 3.0:.12g}",
                    ]
                )

    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata = {
        "schema_version": "three-maceles-born-effective-charge-v1",
        "quantity": "model-native per-atom Born effective charge tensor",
        "tensor_convention": "BEC_ab = dP_a / dr_b",
        "postprocessing": "none; raw model sign and acoustic-sum residual retained",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256(args.input),
            "frame": args.frame,
            "atoms": len(atoms),
            "water_molecules": 64,
            "cell_A": np.asarray(atoms.cell).tolist(),
            "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        },
        "models": model_metadata,
        "output": str(args.output.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
