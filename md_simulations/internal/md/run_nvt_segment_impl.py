#!/usr/bin/env python3
"""Run one fixed-volume NVT segment with transport observables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator


LES_NORM = 90.0474
LES_STRESS_PATCH = "include_les_positions_and_cell_in_symmetric_displacement_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-kind", choices=("OFF", "LES"), required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-source-path", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--stored-precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--resume-dynamics",
        type=Path,
        help="NPZ checkpoint carrying the exact ASE Nose-Hoover state",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--segment", required=True, type=int)
    parser.add_argument("--temperature-K", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestep-fs", default=0.5, type=float)
    parser.add_argument("--equilibration-ps", required=True, type=float)
    parser.add_argument("--production-ps", required=True, type=float)
    parser.add_argument("--ttime-fs", default=10.0, type=float)
    parser.add_argument("--position-every", default=100, type=int)
    parser.add_argument("--velocity-every", default=1, type=int)
    parser.add_argument("--stress-every", default=1, type=int)
    parser.add_argument("--thermo-every", default=1000, type=int)
    parser.add_argument("--reset-velocities-if-missing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def steps_for(duration_ps: float, timestep_fs: float) -> int:
    value = duration_ps * 1000.0 / timestep_fs
    steps = int(round(value))
    if not np.isclose(value, steps):
        raise ValueError("duration is not an integer number of timesteps")
    return steps


def install_les_stress_patch() -> None:
    import mace.modules.extensions as extensions

    if getattr(extensions, "_thermophysical_les_stress_patch", False):
        return
    original_prepare_graph = extensions.prepare_graph

    def prepare_graph_with_les_displacement(
        data,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        lammps_mliap=False,
    ):
        context = original_prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        if lammps_mliap or not (
            compute_virials or compute_stress or compute_displacement
        ):
            return context
        displacement = context.displacement
        if displacement is None:
            return context
        symmetric = 0.5 * (displacement + displacement.transpose(-1, -2))
        original_shape = context.cell.shape
        cells = context.cell.view(-1, 3, 3)
        displaced_cells = cells + torch.matmul(cells, symmetric)
        return context._replace(
            positions=data["positions"],
            cell=displaced_cells.reshape(original_shape),
        )

    extensions.prepare_graph = prepare_graph_with_les_displacement
    extensions._thermophysical_les_stress_patch = True


def load_calculator(args: argparse.Namespace) -> tuple[MACECalculator, dict[str, Any]]:
    actual_hash = sha256(args.model)
    if actual_hash != args.model_sha256:
        raise ValueError("model hash mismatch")
    torch.set_default_dtype(torch.float64)
    if args.model_kind == "LES":
        install_les_stress_patch()
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    stored_dtypes = sorted(
        {
            str(value.dtype).replace("torch.", "")
            for value in model.state_dict().values()
            if torch.is_floating_point(value)
        }
    )
    if stored_dtypes != [args.stored_precision]:
        raise ValueError(f"stored dtype mismatch: {stored_dtypes}")
    model = model.double()
    runtime_dtypes = sorted(
        {
            str(value.dtype).replace("torch.", "")
            for value in model.state_dict().values()
            if torch.is_floating_point(value)
        }
    )
    if runtime_dtypes != ["float64"]:
        raise ValueError(f"runtime dtype mismatch: {runtime_dtypes}")
    les_norm = None
    if args.model_kind == "LES":
        les_norm = float(model.les.ewald.norm_factor)
        if not math.isclose(les_norm, LES_NORM, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("LES normalization changed")
    calculator = MACECalculator(
        models=model,
        device=args.device,
        default_dtype="float64",
        model_type="MACE",
        compute_bec=False,
    )
    return calculator, {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "stored_dtypes": stored_dtypes,
        "runtime_dtypes": runtime_dtypes,
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "residual_lora_state_entries": int(sum("lora_" in key for key in model.state_dict())),
        "les_ewald_norm_factor": les_norm,
        "les_stress_patch": LES_STRESS_PATCH if args.model_kind == "LES" else None,
    }


def validate_atoms(atoms) -> np.ndarray:
    if len(atoms) != 192 or not bool(np.all(atoms.pbc)):
        raise ValueError("expected periodic 64-water input")
    symbols = np.asarray(atoms.get_chemical_symbols())
    oxygen = np.flatnonzero(symbols == "O")
    if len(oxygen) != 64 or np.count_nonzero(symbols == "H") != 128:
        raise ValueError("expected 64 O and 128 H")
    return oxygen


def geometry(atoms) -> dict[str, float]:
    distances = atoms.get_all_distances(mic=True)
    symbols = atoms.get_chemical_symbols()
    oh: list[float] = []
    oxygen: list[int] = []
    for start in range(0, len(atoms), 3):
        group = list(range(start, start + 3))
        o = next(index for index in group if symbols[index] == "O")
        h = [index for index in group if symbols[index] == "H"]
        oxygen.append(o)
        oh.extend(distances[o, h].tolist())
    oo = distances[np.ix_(oxygen, oxygen)].copy()
    np.fill_diagonal(oo, np.inf)
    return {
        "oh_min_A": float(np.min(oh)),
        "oh_max_A": float(np.max(oh)),
        "oo_min_A": float(np.min(oo)),
    }


def main() -> int:
    args = parse_args()
    if args.segment < 1:
        raise ValueError("segment must be positive")
    if (args.segment == 1) == (args.resume_dynamics is not None):
        raise ValueError("segment 1 must start fresh; later segments must resume dynamics")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    equilibration_steps = steps_for(args.equilibration_ps, args.timestep_fs)
    production_steps = steps_for(args.production_ps, args.timestep_fs)
    for interval in (
        args.position_every,
        args.velocity_every,
        args.stress_every,
        args.thermo_every,
    ):
        if interval < 1:
            raise ValueError("sampling intervals must be positive")
    if production_steps % args.position_every or production_steps % args.velocity_every or production_steps % args.stress_every:
        raise ValueError("production steps must be divisible by sampling intervals")

    calculator, model_audit = load_calculator(args)
    if args.resume_dynamics is None:
        atoms = read(args.input)
        oxygen = validate_atoms(atoms)
        atoms.set_masses([15.999 if symbol == "O" else 1.008 for symbol in atoms.get_chemical_symbols()])
        if atoms.get_momenta() is None or not np.any(np.asarray(atoms.get_momenta())):
            if not args.reset_velocities_if_missing:
                raise ValueError("input has no usable momenta")
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.seed)
            )
            Stationary(atoms, preserve_temperature=True)
        atoms.calc = calculator
        if equilibration_steps:
            equilibration = NPT(
                atoms,
                args.timestep_fs * units.fs,
                temperature_K=args.temperature_K,
                ttime=args.ttime_fs * units.fs,
                pfactor=None,
                externalstress=0.0,
            )
            equilibration.run(equilibration_steps)
        write(args.output_dir / "post_equilibration.traj", atoms)
        dynamics = NPT(
            atoms,
            args.timestep_fs * units.fs,
            temperature_K=args.temperature_K,
            ttime=args.ttime_fs * units.fs,
            pfactor=None,
            externalstress=0.0,
        )
    else:
        if equilibration_steps:
            raise ValueError("a resumed production segment cannot equilibrate")
        if not args.resume_dynamics.is_file():
            raise FileNotFoundError(args.resume_dynamics)
        atoms = read(args.input)
        oxygen = validate_atoms(atoms)
        atoms.calc = calculator
        dynamics = NPT(
            atoms,
            args.timestep_fs * units.fs,
            temperature_K=args.temperature_K,
            ttime=args.ttime_fs * units.fs,
            pfactor=None,
            externalstress=0.0,
        )
        state_keys = (
            "eta",
            "eta_past",
            "zeta",
            "zeta_past",
            "zeta_integrated",
            "h",
            "h_past",
            "timeelapsed",
            "q",
            "q_past",
            "q_future",
        )
        with np.load(args.resume_dynamics, allow_pickle=False) as state:
            if set(state.files) != set(state_keys):
                raise ValueError("resumed dynamics checkpoint keys changed")
            for key in state_keys:
                value = state[key]
                setattr(
                    dynamics,
                    key,
                    float(value) if value.ndim == 0 else np.array(value, copy=True),
                )
        dynamics.inv_h = np.linalg.inv(dynamics.h)
        dynamics.initialized = 1
        if not math.isclose(dynamics.dt, args.timestep_fs * units.fs, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("resumed dynamics timestep mismatch")
        if not math.isclose(dynamics.temperature, args.temperature_K * units.kB, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("resumed dynamics temperature mismatch")
        if dynamics.pfactor_given is not None:
            raise ValueError("resumed dynamics is not fixed-volume NVT")
        if not math.isclose(dynamics.ttime, args.ttime_fs * units.fs, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("resumed thermostat time mismatch")
        expected_positions = np.dot(dynamics.q + 0.5, dynamics.h)
        expected_momenta = (
            np.dot(
                dynamics.q_future - dynamics.q_past,
                dynamics.h / (2 * dynamics.dt),
            )
            * dynamics._getmasses()
        )
        if not np.allclose(atoms.cell.array, dynamics.h, rtol=0.0, atol=1.0e-12):
            raise ValueError("resumed cell and dynamics checkpoint differ")
        if not np.allclose(atoms.positions, expected_positions, rtol=0.0, atol=1.0e-10):
            raise ValueError("resumed positions and dynamics checkpoint differ")
        if not np.allclose(atoms.get_momenta(), expected_momenta, rtol=0.0, atol=1.0e-10):
            raise ValueError("resumed momenta and dynamics checkpoint differ")
    initial_cell = np.asarray(atoms.cell.array, dtype=np.float64).copy()
    position_frames = production_steps // args.position_every + 1
    velocity_frames = production_steps // args.velocity_every + 1
    stress_frames = production_steps // args.stress_every + 1
    oxygen_positions = np.lib.format.open_memmap(
        args.output_dir / "oxygen_positions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(position_frames, len(oxygen), 3),
    )
    oxygen_velocities = np.lib.format.open_memmap(
        args.output_dir / "oxygen_velocities.npy",
        mode="w+",
        dtype=np.float32,
        shape=(velocity_frames, len(oxygen), 3),
    )
    stress_components = np.lib.format.open_memmap(
        args.output_dir / "stress_components_eV_A3.npy",
        mode="w+",
        dtype=np.float64,
        shape=(stress_frames, 5),
    )
    position_index = 0
    velocity_index = 0
    stress_index = 0

    thermo_handle = (args.output_dir / "thermo.csv").open("w", newline="", encoding="utf-8")
    thermo_writer = csv.DictWriter(
        thermo_handle,
        fieldnames=("step", "time_ps", "temperature_K", "pressure_bar", "potential_energy_eV", "kinetic_energy_eV"),
    )
    thermo_writer.writeheader()
    thermo_rows = 0

    def record_positions() -> None:
        nonlocal position_index
        oxygen_positions[position_index] = np.asarray(atoms.positions[oxygen], dtype=np.float32)
        position_index += 1

    def record_velocities() -> None:
        nonlocal velocity_index
        oxygen_velocities[velocity_index] = np.asarray(atoms.get_velocities()[oxygen], dtype=np.float32)
        velocity_index += 1

    def record_stress() -> None:
        nonlocal stress_index
        xx, yy, zz, yz, xz, xy = np.asarray(
            atoms.get_stress(include_ideal_gas=True), dtype=np.float64
        )
        stress_components[stress_index] = (
            xy,
            xz,
            yz,
            0.5 * (xx - yy),
            0.5 * (yy - zz),
        )
        stress_index += 1

    def record_thermo() -> None:
        nonlocal thermo_rows
        stress = np.asarray(atoms.get_stress(include_ideal_gas=True), dtype=np.float64)
        thermo_writer.writerow(
            {
                "step": dynamics.nsteps,
                "time_ps": dynamics.nsteps * args.timestep_fs / 1000.0,
                "temperature_K": float(atoms.get_temperature()),
                "pressure_bar": float(-np.mean(stress[:3]) / units.bar),
                "potential_energy_eV": float(atoms.get_potential_energy()),
                "kinetic_energy_eV": float(atoms.get_kinetic_energy()),
            }
        )
        thermo_rows += 1
        if thermo_rows % 100 == 0:
            thermo_handle.flush()

    dynamics.attach(record_positions, interval=args.position_every)
    dynamics.attach(record_velocities, interval=args.velocity_every)
    dynamics.attach(record_stress, interval=args.stress_every)
    dynamics.attach(record_thermo, interval=args.thermo_every)
    started = time.monotonic()
    dynamics.run(production_steps)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    thermo_handle.close()
    oxygen_positions.flush()
    oxygen_velocities.flush()
    stress_components.flush()

    if position_index != position_frames or velocity_index != velocity_frames or stress_index != stress_frames:
        raise RuntimeError("transport observer frame count mismatch")
    if not np.allclose(atoms.cell.array, initial_cell, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("cell changed during fixed-volume NVT")
    final_geometry = geometry(atoms)
    if not (
        final_geometry["oh_min_A"] > 0.5
        and final_geometry["oh_max_A"] < 2.0
        and final_geometry["oo_min_A"] > 1.5
    ):
        raise RuntimeError(f"invalid final geometry: {final_geometry}")
    write(args.output_dir / "final.traj", atoms)
    np.savez(
        args.output_dir / "dynamics_state.npz",
        **{
            key: np.asarray(getattr(dynamics, key))
            for key in (
                "eta",
                "eta_past",
                "zeta",
                "zeta_past",
                "zeta_integrated",
                "h",
                "h_past",
                "timeelapsed",
                "q",
                "q_past",
                "q_future",
            )
        },
    )

    payload = {
        "schema_version": "six-model-nvt-segment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "locked_test_used": False,
        "model_label": args.model_label,
        "model_kind": args.model_kind,
        "model_path": str((args.model_source_path or args.model).resolve()),
        "model_staged_path": str(args.model.resolve()),
        "model_sha256": args.model_sha256,
        "model_audit": model_audit,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "resumed_dynamics_path": (
            str(args.resume_dynamics.resolve()) if args.resume_dynamics else None
        ),
        "resumed_dynamics_sha256": (
            sha256(args.resume_dynamics) if args.resume_dynamics else None
        ),
        "segment": args.segment,
        "temperature_K": args.temperature_K,
        "velocity_seed": args.seed,
        "waters": 64,
        "physical_hydrogen_mass_amu": 1.008,
        "timestep_fs": args.timestep_fs,
        "equilibration_ps": args.equilibration_ps,
        "equilibration_steps": equilibration_steps,
        "production_ps": args.production_ps,
        "production_steps": production_steps,
        "thermostat": "ASE NPT pfactor=None (fixed-volume Nose-Hoover)",
        "thermostat_time_fs": args.ttime_fs,
        "volume_A3": float(atoms.get_volume()),
        "position_every_steps": args.position_every,
        "velocity_every_steps": args.velocity_every,
        "stress_every_steps": args.stress_every,
        "thermo_every_steps": args.thermo_every,
        "position_frames": position_index,
        "velocity_frames": velocity_index,
        "stress_frames": stress_index,
        "production_elapsed_seconds": elapsed,
        "production_steps_per_second": production_steps / elapsed,
        "final_temperature_K": float(atoms.get_temperature()),
        "final_geometry": final_geometry,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "ase": ase.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_procid": os.environ.get("SLURM_PROCID"),
        },
        "outputs": {
            "oxygen_positions": "oxygen_positions.npy",
            "oxygen_velocities": "oxygen_velocities.npy",
            "stress_components": "stress_components_eV_A3.npy",
            "thermo": "thermo.csv",
            "final": "final.traj",
            "dynamics_checkpoint": "dynamics_state.npz",
        },
    }
    atomic_json(args.output_dir / "summary.json", payload)
    (args.output_dir / "COMPLETE").touch()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
