#!/usr/bin/env python3
"""Analyze one 1 ns NpT lane and select its representative-volume frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase.io.trajectory import Trajectory


AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write analysis here instead of into --run-dir",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="analyze a saved trajectory that ended before exactly 1000 ps",
    )
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--temperature-K", required=True, type=float)
    parser.add_argument("--analysis-start-ps", default=500.0, type=float)
    parser.add_argument("--timestep-fs", default=1.0, type=float)
    parser.add_argument("--trajectory-every", default=100, type=int)
    parser.add_argument("--rdf-stride", default=10, type=int)
    parser.add_argument("--rdf-bin-width-A", default=0.02, type=float)
    parser.add_argument("--rdf-maximum-A", default=6.0, type=float)
    parser.add_argument("--blocks", default=10, type=int)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def water_indices(atoms) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    symbols = atoms.get_chemical_symbols()
    oxygen: list[int] = []
    hydrogen: list[int] = []
    bonded: set[tuple[int, int]] = set()
    if len(atoms) != 192:
        raise ValueError("expected 64 waters / 192 atoms")
    for start in range(0, len(atoms), 3):
        group = list(range(start, start + 3))
        o = [index for index in group if symbols[index] == "O"]
        h = [index for index in group if symbols[index] == "H"]
        if len(o) != 1 or len(h) != 2:
            raise ValueError("expected contiguous O/H/H water records")
        oxygen.extend(o)
        hydrogen.extend(h)
        bonded.update((o[0], item) for item in h)
    return np.asarray(oxygen), np.asarray(hydrogen), bonded


def mic_distances(first: np.ndarray, second: np.ndarray, cell: np.ndarray) -> np.ndarray:
    displacement = first[:, None, :] - second[None, :, :]
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    displacement = fractional @ cell
    return np.sqrt(np.einsum("ijk,ijk->ij", displacement, displacement))


def rdf_blocks(
    trajectory: Trajectory,
    frame_indices: np.ndarray,
    oxygen: np.ndarray,
    hydrogen: np.ndarray,
    bonded: set[tuple[int, int]],
    edges: np.ndarray,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunks = np.array_split(frame_indices, blocks)
    shell = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    oo_pair_count = len(oxygen) * (len(oxygen) - 1) / 2
    oh_mask = np.ones((len(oxygen), len(hydrogen)), dtype=bool)
    oxygen_lookup = {value: index for index, value in enumerate(oxygen.tolist())}
    hydrogen_lookup = {value: index for index, value in enumerate(hydrogen.tolist())}
    for o, h in bonded:
        oh_mask[oxygen_lookup[o], hydrogen_lookup[h]] = False
    oh_pair_count = int(np.count_nonzero(oh_mask))
    oo_values: list[np.ndarray] = []
    oh_values: list[np.ndarray] = []
    for chunk in chunks:
        if not len(chunk):
            raise ValueError("empty RDF block")
        oo_hist = np.zeros(len(edges) - 1, dtype=np.float64)
        oh_hist = np.zeros(len(edges) - 1, dtype=np.float64)
        oo_norm = np.zeros(len(edges) - 1, dtype=np.float64)
        oh_norm = np.zeros(len(edges) - 1, dtype=np.float64)
        for frame_index in chunk:
            atoms = trajectory[int(frame_index)]
            positions = np.asarray(atoms.positions, dtype=np.float64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            volume = float(abs(np.linalg.det(cell)))
            oo = mic_distances(positions[oxygen], positions[oxygen], cell)
            oo_hist += np.histogram(oo[np.triu_indices(len(oxygen), 1)], bins=edges)[0]
            oh = mic_distances(positions[oxygen], positions[hydrogen], cell)
            oh_hist += np.histogram(oh[oh_mask], bins=edges)[0]
            oo_norm += oo_pair_count * shell / volume
            oh_norm += oh_pair_count * shell / volume
        oo_values.append(oo_hist / oo_norm)
        oh_values.append(oh_hist / oh_norm)
    oo_array = np.asarray(oo_values)
    oh_array = np.asarray(oh_values)
    return (
        oo_array.mean(axis=0),
        oo_array.std(axis=0, ddof=1) / math.sqrt(blocks),
        oh_array.mean(axis=0),
        oh_array.std(axis=0, ddof=1) / math.sqrt(blocks),
        0.5 * (edges[:-1] + edges[1:]),
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.analysis_start_ps < 1000.0:
        raise ValueError("analysis-start-ps must lie in [0, 1000)")
    if args.blocks < 2:
        raise ValueError("at least two time blocks are required")
    if args.rdf_stride < 1:
        raise ValueError("rdf-stride must be positive")
    summary_path = args.run_dir / "summary.json"
    trajectory_path = args.run_dir / "trajectory.traj"
    if not trajectory_path.is_file():
        raise FileNotFoundError("NpT trajectory is missing")
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    if summary is None and not args.allow_partial:
        raise FileNotFoundError("NpT summary is missing")
    if summary is not None:
        if not summary.get("complete") and not args.allow_partial:
            raise ValueError("NpT trajectory is incomplete")
        if summary.get("model_sha256") != args.model_sha256:
            raise ValueError("NpT model hash mismatch")
        if not math.isclose(float(summary["temperature_K"]), args.temperature_K):
            raise ValueError("NpT temperature mismatch")
    expected_steps = int(round(1000.0 * 1000.0 / args.timestep_fs))
    if not args.allow_partial and summary.get("completed_production_steps") != expected_steps:
        raise ValueError("NpT production is not exactly 1 ns")

    trajectory = Trajectory(trajectory_path, "r")
    if len(trajectory) < 2:
        raise ValueError("empty NpT trajectory")
    interval_ps = args.trajectory_every * args.timestep_fs / 1000.0
    times = np.arange(len(trajectory), dtype=np.float64) * interval_ps
    selected_indices = np.flatnonzero(times >= args.analysis_start_ps - 1.0e-9)
    if not len(selected_indices):
        raise ValueError("NpT trajectory does not reach the analysis window")
    if not args.allow_partial and times[selected_indices[-1]] < 999.9:
        raise ValueError("NpT trajectory does not cover the requested analysis window")
    first_atoms = trajectory[0]
    oxygen, hydrogen, bonded = water_indices(first_atoms)
    mass = float(np.sum(first_atoms.get_masses()))
    volumes = np.asarray([trajectory[int(index)].get_volume() for index in selected_indices])
    densities = mass * AMU_PER_ANGSTROM3_TO_G_PER_CM3 / volumes
    density_chunks = np.array_split(np.arange(len(selected_indices)), args.blocks)
    density_blocks = []
    for block_index, chunk in enumerate(density_chunks):
        if not len(chunk):
            raise ValueError("empty density block")
        block_times = times[selected_indices[chunk]]
        block_densities = densities[chunk]
        density_blocks.append(
            {
                "block": block_index,
                "start_ps": float(block_times[0]),
                "end_ps": float(block_times[-1]),
                "frames": int(len(chunk)),
                "density_mean_g_per_cm3": float(np.mean(block_densities)),
            }
        )
    block_means = np.asarray(
        [item["density_mean_g_per_cm3"] for item in density_blocks],
        dtype=np.float64,
    )
    density_block_sem = float(np.std(block_means, ddof=1) / math.sqrt(args.blocks))
    midpoint = len(densities) // 2
    early_density = float(np.mean(densities[:midpoint]))
    late_density = float(np.mean(densities[midpoint:]))
    mean_volume = float(np.mean(volumes))
    representative_position = int(np.argmin(np.abs(volumes - mean_volume)))
    representative_index = int(selected_indices[representative_position])
    representative = trajectory[representative_index]
    representative.write(output_dir / "representative.traj")

    centered_time = times[selected_indices] - np.mean(times[selected_indices])
    density_slope = float(np.dot(centered_time, densities - np.mean(densities)) / np.dot(centered_time, centered_time))
    rdf_indices = selected_indices[:: args.rdf_stride]
    edges = np.arange(0.0, args.rdf_maximum_A + 0.5 * args.rdf_bin_width_A, args.rdf_bin_width_A)
    oo_mean, oo_sem, oh_mean, oh_sem, radius = rdf_blocks(
        trajectory,
        rdf_indices,
        oxygen,
        hydrogen,
        bonded,
        edges,
        args.blocks,
    )
    with (output_dir / "rdf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("r_A", "g_OO", "g_OO_block_sem", "g_OH_intermolecular", "g_OH_block_sem"))
        writer.writerows(zip(radius, oo_mean, oo_sem, oh_mean, oh_sem))

    with (output_dir / "density_blocks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(density_blocks[0]))
        writer.writeheader()
        writer.writerows(density_blocks)

    exact_1ns = bool(
        summary is not None
        and summary.get("complete")
        and summary.get("completed_production_steps") == expected_steps
        and times[selected_indices[-1]] >= 999.9
    )
    payload = {
        "schema_version": "eight-model-b123200-npt-analysis-v2",
        "complete": exact_1ns,
        "preliminary": not exact_1ns,
        "source_trajectory": str(trajectory_path),
        "model_sha256": args.model_sha256,
        "temperature_K": args.temperature_K,
        "analysis_start_ps": args.analysis_start_ps,
        "analysis_end_ps": float(times[selected_indices[-1]]),
        "analysis_frames": int(len(selected_indices)),
        "analysis_duration_ps": float(
            times[selected_indices[-1]] - times[selected_indices[0]]
        ),
        "rdf_frames": int(len(rdf_indices)),
        "rdf_blocks": args.blocks,
        "rdf_stride_frames": args.rdf_stride,
        "rdf_sampling_interval_ps": float(interval_ps * args.rdf_stride),
        "rdf_definition": {
            "OO": "distinct oxygen pairs",
            "OH": "intermolecular oxygen-hydrogen pairs; two covalent O-H pairs per water excluded",
            "normalization": "instantaneous-cell ideal-pair shell population",
        },
        "density_mean_g_per_cm3": float(np.mean(densities)),
        "density_sample_sd_g_per_cm3": float(np.std(densities, ddof=1)),
        "density_block_sem_g_per_cm3": density_block_sem,
        "density_blocks": density_blocks,
        "density_first_half_g_per_cm3": early_density,
        "density_second_half_g_per_cm3": late_density,
        "density_second_minus_first_half_g_per_cm3": late_density - early_density,
        "density_linear_slope_g_per_cm3_per_ps": density_slope,
        "mean_volume_A3": mean_volume,
        "representative_frame_index": representative_index,
        "representative_time_ps": float(times[representative_index]),
        "representative_volume_A3": float(representative.get_volume()),
        "outputs": {
            "representative": "representative.traj",
            "rdf": "rdf.csv",
            "density_blocks": "density_blocks.csv",
        },
    }
    atomic_json(output_dir / "npt_analysis.json", payload)
    if exact_1ns:
        (output_dir / "NPT_COMPLETE").touch()
    else:
        (output_dir / "PRELIMINARY_ANALYSIS_COMPLETE").touch()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
