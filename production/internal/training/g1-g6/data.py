"""G6-only one-body labels and paired cluster/monomer MACE batches."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = "conservative-1b-int-data-v1"
NULL_ALPHA = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PilotDataset(Dataset):
    """Load train/valid JSONL and replace branch labels by exact G6 subtraction."""

    def __init__(
        self,
        path: Path | str,
        *,
        null_size_means: dict[int, float] | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if "test" in self.path.name.lower():
            raise ValueError("locked-test inputs are forbidden in this pilot")
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(f"{self.path}:{line_number}: unexpected schema")
                records.append(record)
        if not records:
            raise ValueError(f"no records in {self.path}")

        g1_by_parent: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            if int(record["cluster_size"]) != 1:
                continue
            parent = int(record["parent_id"])
            g1_by_parent.setdefault(parent, []).append(record)
        for parent, monomers in g1_by_parent.items():
            monomers.sort(key=lambda value: int(value["source"]["fragment_index"]))
            indices = [int(value["source"]["fragment_index"]) for value in monomers]
            if indices != list(range(1, 7)):
                raise ValueError(f"parent {parent}: G6 monomer indices changed")

        internal_coordinate_max = 0.0
        absolute_translation_max = 0.0
        force_closure_max = 0.0
        energy_closure_max = 0.0
        by_size_totals: dict[int, list[float]] = {size: [] for size in range(2, 7)}
        for record in records:
            size = int(record["cluster_size"])
            if size >= 2:
                by_size_totals[size].append(float(record["total_energy_eV"]))
        if null_size_means is None:
            size_means = {
                size: float(np.mean(values)) for size, values in by_size_totals.items()
            }
        else:
            size_means = {int(size): float(value) for size, value in null_size_means.items()}
            if set(size_means) != set(range(2, 7)):
                raise ValueError("null size means must cover G2--G6")

        transformed: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            parent = int(record["parent_id"])
            size = int(record["cluster_size"])
            if parent not in g1_by_parent:
                raise ValueError(f"parent {parent}: no G6 monomer labels")
            positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
            if positions.shape != (3 * size, 3):
                raise ValueError(f"{record['record_key']}: cluster shape changed")
            cluster_waters = [positions[3 * index : 3 * (index + 1)] for index in range(size)]
            g6_waters = [
                np.asarray(value["positions_angstrom"], dtype=np.float64)
                for value in g1_by_parent[parent]
            ]
            costs = np.asarray(
                [
                    [
                        float(
                            np.max(
                                np.abs(
                                    (cluster_water - cluster_water[0])
                                    - (g6_water - g6_water[0])
                                )
                            )
                        )
                        for g6_water in g6_waters
                    ]
                    for cluster_water in cluster_waters
                ],
                dtype=np.float64,
            )
            assignment = min(
                itertools.permutations(range(6), size),
                key=lambda indices: max(costs[index, value] for index, value in enumerate(indices)),
            )
            match_error = max(costs[index, value] for index, value in enumerate(assignment))
            internal_coordinate_max = max(internal_coordinate_max, float(match_error))
            monomers = [g1_by_parent[parent][index] for index in assignment]
            for cluster_water, g6_index in zip(cluster_waters, assignment, strict=True):
                translation = cluster_water[0] - g6_waters[g6_index][0]
                absolute_translation_max = max(
                    absolute_translation_max, float(np.max(np.abs(translation)))
                )
            one_energy = float(sum(float(value["total_energy_eV"]) for value in monomers))
            one_forces = np.concatenate(
                [np.asarray(value["total_forces_eV_per_angstrom"], dtype=np.float64) for value in monomers],
                axis=0,
            )
            total_energy = float(record["total_energy_eV"])
            total_forces = np.asarray(
                record["total_forces_eV_per_angstrom"], dtype=np.float64
            )
            interaction_energy = total_energy - one_energy
            interaction_forces = total_forces - one_forces
            if size == 1:
                null_energy = 0.0
                null_forces = np.zeros_like(total_forces)
            else:
                null_energy = NULL_ALPHA * (total_energy - size_means[size])
                null_forces = NULL_ALPHA * total_forces
            record.update(
                {
                    "one_body_energy_eV": one_energy,
                    "one_body_forces_eV_per_angstrom": one_forces.tolist(),
                    "interaction_energy_eV": interaction_energy,
                    "interaction_forces_eV_per_angstrom": interaction_forces.tolist(),
                    "null_interaction_energy_eV": null_energy,
                    "null_interaction_forces_eV_per_angstrom": null_forces.tolist(),
                }
            )
            energy_closure_max = max(
                energy_closure_max,
                abs(total_energy - one_energy - interaction_energy),
            )
            force_closure_max = max(
                force_closure_max,
                float(np.max(np.abs(total_forces - one_forces - interaction_forces))),
            )
            transformed.append(record)
        if internal_coordinate_max > 1.0e-8:
            raise ValueError(
                "G6-only monomer internal-coordinate mismatch: "
                f"{internal_coordinate_max} A"
            )
        self.records = transformed
        self.audit = {
            "records": len(self.records),
            "parents": len(g1_by_parent),
            "sha256": sha256(self.path),
            "g6_monomer_internal_coordinate_max_A": internal_coordinate_max,
            "g6_periodic_image_translation_max_A": absolute_translation_max,
            "exact_energy_closure_max_eV": energy_closure_max,
            "exact_force_closure_max_eV_per_A": force_closure_max,
            "null_alpha": NULL_ALPHA,
            "null_size_energy_means_eV": size_means,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _configuration(record: dict[str, Any], atom_slice: slice | None = None):
    from ase import Atoms
    from mace.data import config_from_atoms

    symbols = np.asarray(record["symbols"])
    positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
    if atom_slice is not None:
        symbols = symbols[atom_slice]
        positions = positions[atom_slice]
    return config_from_atoms(Atoms(symbols=symbols.tolist(), positions=positions, pbc=False))


class PilotCollator:
    def __init__(self, *, atomic_numbers: Sequence[int], cutoff: float) -> None:
        from mace.tools import AtomicNumberTable

        self.z_table = AtomicNumberTable([int(value) for value in atomic_numbers])
        self.cutoff = float(cutoff)

    def _graph(self, record: dict[str, Any], atom_slice: slice | None = None):
        from mace.data import AtomicData

        return AtomicData.from_config(
            _configuration(record, atom_slice),
            z_table=self.z_table,
            cutoff=self.cutoff,
        )

    def __call__(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from mace.tools.torch_geometric import Batch

        if not records:
            raise ValueError("cannot collate an empty batch")
        clusters = [self._graph(record) for record in records]
        monomers = []
        owners: list[int] = []
        for owner, record in enumerate(records):
            for index in range(int(record["cluster_size"])):
                monomers.append(self._graph(record, slice(3 * index, 3 * (index + 1))))
                owners.append(owner)
        cluster_batch = Batch.from_data_list(clusters)
        monomer_batch = Batch.from_data_list(monomers)
        if not torch.equal(cluster_batch.positions, monomer_batch.positions):
            raise RuntimeError("cluster and monomer flattened coordinates differ")

        def energy(name: str) -> torch.Tensor:
            return torch.tensor(
                [float(record[name]) for record in records],
                dtype=torch.get_default_dtype(),
            )

        def forces(name: str) -> torch.Tensor:
            return torch.tensor(
                np.concatenate(
                    [np.asarray(record[name], dtype=np.float64) for record in records],
                    axis=0,
                ),
                dtype=torch.get_default_dtype(),
            )

        reference = {
            "total_energy": energy("total_energy_eV"),
            "total_forces": forces("total_forces_eV_per_angstrom"),
            "one_body_energy": energy("one_body_energy_eV"),
            "one_body_forces": forces("one_body_forces_eV_per_angstrom"),
            "interaction_energy": energy("interaction_energy_eV"),
            "interaction_forces": forces("interaction_forces_eV_per_angstrom"),
            "null_interaction_energy": energy("null_interaction_energy_eV"),
            "null_interaction_forces": forces("null_interaction_forces_eV_per_angstrom"),
            "num_atoms": torch.tensor(
                [len(record["symbols"]) for record in records],
                dtype=torch.get_default_dtype(),
            ),
            "cluster_size": torch.tensor(
                [int(record["cluster_size"]) for record in records], dtype=torch.long
            ),
        }
        return {
            "records": list(records),
            "clusters": cluster_batch,
            "monomers": monomer_batch,
            "monomer_owner": torch.tensor(owners, dtype=torch.long),
            "reference": reference,
        }


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        "records": batch["records"],
        "clusters": batch["clusters"].to(device),
        "monomers": batch["monomers"].to(device),
        "monomer_owner": batch["monomer_owner"].to(device),
        "reference": {name: value.to(device) for name, value in batch["reference"].items()},
    }


def calibration_records(dataset: PilotDataset) -> list[dict[str, Any]]:
    return [
        next(record for record in dataset.records if int(record["cluster_size"]) == size)
        for size in range(1, 7)
    ]
