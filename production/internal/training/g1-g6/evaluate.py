#!/usr/bin/env python3
"""Validation-only mechanistic evaluation, ordinary export, and fusion parity."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mace.modules.lora import merge_lora_weights
from torch.utils.data import DataLoader

from data import PilotCollator, PilotDataset, batch_to_device, sha256
from model import load_model, load_trainable_state


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for surface in ("total", "one_body", "interaction"):
        energy = np.asarray([row[f"{surface}_energy_error_eV_per_atom"] for row in rows])
        components = np.concatenate([row[f"{surface}_force_error_eV_per_A"].ravel() for row in rows])
        vectors = np.concatenate(
            [np.linalg.norm(row[f"{surface}_force_error_eV_per_A"], axis=1) for row in rows]
        )
        result[f"{surface}_energy_rmse_meV_per_atom"] = 1000.0 * math.sqrt(float(np.mean(energy**2)))
        result[f"{surface}_energy_signed_bias_meV_per_atom"] = 1000.0 * float(np.mean(energy))
        result[f"{surface}_force_component_rmse_meV_per_A"] = 1000.0 * math.sqrt(float(np.mean(components**2)))
        result[f"{surface}_force_component_p95_meV_per_A"] = 1000.0 * float(np.quantile(np.abs(components), 0.95))
        result[f"{surface}_force_vector_p95_meV_per_A"] = 1000.0 * float(np.quantile(vectors, 0.95))
    return result


def direct(model: torch.nn.Module, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    output = model(
        batch["clusters"],
        training=False,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    return output["energy"].detach().cpu().numpy(), output["forces"].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--backbone", choices=("OFF", "LES"), default="OFF")
    parser.add_argument("--condition", choices=("G1-G6", "D00", "D11", "D11N"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=39600)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if "test" in args.valid.name.lower():
        raise ValueError("locked-test inputs are forbidden")
    torch.set_default_dtype(torch.float64)
    source = json.loads(args.source_manifest.read_text())
    if sha256(args.valid) != source["products"]["valid"]["sha256"]:
        raise ValueError("validation data hash differs from gated source")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "fixed_update_final_ema":
        raise ValueError("evaluation requires the fixed-update final EMA")
    if int(checkpoint["global_step"]) != args.expected_step:
        raise ValueError(
            f"evaluation checkpoint step {checkpoint['global_step']} differs from "
            f"expected fixed step {args.expected_step}"
        )
    if checkpoint["condition"] != args.condition or int(checkpoint["seed"]) != args.seed:
        raise ValueError("checkpoint identity differs")

    device = torch.device(args.device)
    model = load_model(args.foundation, seed=args.seed, device=device)
    load_trainable_state(model, checkpoint["trainable_state"])
    model.eval()
    dataset = PilotDataset(args.valid)
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
    rows: list[dict[str, Any]] = []
    parity_host = None
    for host_batch in loader:
        if parity_host is None:
            parity_host = host_batch
        batch = batch_to_device(host_batch, device)
        output = model(
            batch["clusters"],
            batch["monomers"],
            batch["monomer_owner"],
            training=False,
            compute_counterfactual=True,
        )
        refs = batch["reference"]
        graph_count = len(host_batch["records"])
        atom_offsets = np.cumsum([0] + [len(record["symbols"]) for record in host_batch["records"]])
        for index in range(graph_count):
            start, stop = int(atom_offsets[index]), int(atom_offsets[index + 1])
            row: dict[str, Any] = {
                "record_key": host_batch["records"][index]["record_key"],
                "parent_id": int(host_batch["records"][index]["parent_id"]),
                "cluster_size": int(host_batch["records"][index]["cluster_size"]),
            }
            for surface, prediction_energy, prediction_force in (
                ("total", output["energy"], output["forces"]),
                ("one_body", output["one_body_energy"], output["one_body_forces"]),
                ("interaction", output["interaction_energy"], output["interaction_forces"]),
            ):
                row[f"{surface}_energy_error_eV_per_atom"] = float(
                    (
                        prediction_energy[index]
                        - refs[f"{surface}_energy"][index]
                    ).detach().cpu()
                    / refs["num_atoms"][index].detach().cpu()
                )
                row[f"{surface}_force_error_eV_per_A"] = (
                    prediction_force[start:stop] - refs[f"{surface}_forces"][start:stop]
                ).detach().cpu().numpy()
            rows.append(row)
    if len(rows) != len(dataset):
        raise RuntimeError("validation evaluation is incomplete")
    per_size = {
        f"G{size}": group_metrics([row for row in rows if row["cluster_size"] == size])
        for size in range(1, 7)
    }
    all_metrics = group_metrics(rows)
    macro_g2_g6 = {
        name: float(np.mean([per_size[f"G{size}"][name] for size in range(2, 7)]))
        for name in per_size["G2"]
        if "rmse" in name
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    foundation = "MACEOFF23(M)" if args.backbone == "OFF" else "MACELES-OFF"
    model_label = f"{foundation} / G1-G6"
    file_slug = "maceoff23m" if args.backbone == "OFF" else "maceles-off"
    native_path = args.output_dir / f"{file_slug}-g1-g6-seed{args.seed}-native.model"
    fused_path = args.output_dir / f"{file_slug}-g1-g6-seed{args.seed}-fused.model"
    native_trunk = copy.deepcopy(model.trunk).cpu()
    fused_trunk = merge_lora_weights(copy.deepcopy(native_trunk), inplace=True).cpu()
    if any("lora_A" in name or "lora_B" in name for name, _ in fused_trunk.named_parameters()):
        raise RuntimeError("fused export retained LoRA parameters")
    torch.save(native_trunk, native_path)
    torch.save(fused_trunk, fused_path)

    if parity_host is None:
        raise RuntimeError("no parity batch")
    parity_batch = batch_to_device(parity_host, device)
    native_energy, native_force = direct(native_trunk.to(device), parity_batch)
    fused_energy, fused_force = direct(fused_trunk.to(device), parity_batch)
    parity = {
        "maximum_native_fused_energy_discrepancy_eV": float(np.max(np.abs(native_energy - fused_energy))),
        "maximum_native_fused_force_discrepancy_eV_per_A": float(np.max(np.abs(native_force - fused_force))),
    }
    parity["passed"] = bool(
        parity["maximum_native_fused_energy_discrepancy_eV"] <= 1.0e-8
        and parity["maximum_native_fused_force_discrepancy_eV_per_A"] <= 1.0e-8
    )
    if not parity["passed"]:
        raise RuntimeError(f"native/fused parity failed: {parity}")

    record_error_path = args.output_dir / "per_record_errors.npz"
    record_error_payload: dict[str, np.ndarray] = {
        "record_key": np.asarray([row["record_key"] for row in rows]),
        "parent_id": np.asarray([row["parent_id"] for row in rows], dtype=np.int64),
        "cluster_size": np.asarray([row["cluster_size"] for row in rows], dtype=np.int8),
    }
    for surface in ("total", "one_body", "interaction"):
        force_errors = [row[f"{surface}_force_error_eV_per_A"] for row in rows]
        record_error_payload[f"{surface}_energy_error_eV_per_atom"] = np.asarray(
            [row[f"{surface}_energy_error_eV_per_atom"] for row in rows], dtype=np.float64
        )
        record_error_payload[f"{surface}_force_sse_eV2_per_A2"] = np.asarray(
            [float(np.sum(np.square(value))) for value in force_errors], dtype=np.float64
        )
        record_error_payload[f"{surface}_force_components"] = np.asarray(
            [value.size for value in force_errors], dtype=np.int64
        )
    np.savez_compressed(record_error_path, **record_error_payload)

    result = {
        "schema_version": "single-mace-counterfactual-validation-v1",
        "scope": "common_parent_held_out_G1_G6_validation_only",
        "model_label": model_label,
        "backbone": args.backbone,
        "condition": args.condition,
        "seed": args.seed,
        "global_step": int(checkpoint["global_step"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "foundation_sha256": sha256(args.foundation),
        "validation": str(args.valid.resolve()),
        "validation_sha256": sha256(args.valid),
        "records": len(rows),
        "data_audit": dataset.audit,
        "per_size": per_size,
        "pooled_G1_G6": all_metrics,
        "equal_size_macro_G2_G6": macro_g2_g6,
        "native_model": {"path": str(native_path.resolve()), "sha256": sha256(native_path)},
        "fused_model": {"path": str(fused_path.resolve()), "sha256": sha256(fused_path)},
        "per_record_errors": {
            "path": str(record_error_path.resolve()),
            "sha256": sha256(record_error_path),
        },
        "native_fused_parity": parity,
    }
    output = args.output_dir / "validation_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
