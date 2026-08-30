#!/usr/bin/env python3
"""Freeze a train/validation-only label and provenance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import PilotDataset, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any("test" in path.name.lower() for path in (args.train, args.valid)):
        raise ValueError("locked-test inputs are forbidden")
    source = json.loads(args.source_manifest.read_text())
    if not source.get("passed"):
        raise ValueError("source data manifest failed")
    train = PilotDataset(args.train)
    valid = PilotDataset(
        args.valid,
        null_size_means=train.audit["null_size_energy_means_eV"],
    )
    checks = {
        "train_hash": sha256(args.train) == source["products"]["train"]["sha256"],
        "valid_hash": sha256(args.valid) == source["products"]["valid"]["sha256"],
        "train_count": len(train) == 8800,
        "valid_count": len(valid) == 1100,
        "train_parents": train.audit["parents"] == 800,
        "valid_parents": valid.audit["parents"] == 100,
        "g6_monomer_internal_coordinates": max(
            train.audit["g6_monomer_internal_coordinate_max_A"],
            valid.audit["g6_monomer_internal_coordinate_max_A"],
        ) <= 1.0e-8,
        "exact_energy_closure": max(
            train.audit["exact_energy_closure_max_eV"],
            valid.audit["exact_energy_closure_max_eV"],
        ) <= 1.0e-12,
        "exact_force_closure": max(
            train.audit["exact_force_closure_max_eV_per_A"],
            valid.audit["exact_force_closure_max_eV_per_A"],
        ) <= 1.0e-12,
    }
    result = {
        "schema_version": "single-mace-counterfactual-data-audit-v1",
        "scope": "train_and_parent_held_out_validation_only",
        "passed": all(checks.values()),
        "checks": checks,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "train": train.audit,
        "valid": valid.audit,
        "label_definition": {
            "one_body": "G6-only shifted fragment energy and GD force joined by internal water geometry",
            "interaction": "authoritative total minus G6-only one-body",
            "null_energy": "0.1 * (authoritative total - train-only size mean)",
            "null_force": "0.1 * authoritative total force",
            "raw_eda_int_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
