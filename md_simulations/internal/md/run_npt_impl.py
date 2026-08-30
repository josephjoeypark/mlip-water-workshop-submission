#!/usr/bin/env python3
"""Gate or run the configured 1 ns density protocol for one frozen model.

The production path follows the executable protocol released in
ChengUCB/les_fit at commit a886785caa4182a0effb95d95f0e402881adfc4d:
MD_Simulations/Water/NPT/NPT-ase.py uses BFGS to 0.2 eV/A,
5 ps of thermostat-only ASE NPT, then the configured ASE IsotropicMTKNPT run
(1 ns in ``configs/study.yaml``). All models are evaluated in float64 for the
matched comparison.
"""

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
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from mace.calculators import MACECalculator


INPUT_SHA256 = "347fcf41584ca834d1bdb1acc622a9ad9329091ec1d033e423dc29c6ef03ab69"
AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660
EV_PER_ANGSTROM3_TO_KBAR = 1602.1766208
LES_NORM = 90.0474
LES_STRESS_PATCH = "include_les_positions_and_cell_in_symmetric_displacement_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "production"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-kind", choices=("OFF", "LES"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-source-path", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--stored-precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, required=True)
    parser.add_argument("--pressure-bar", type=float, default=1.01325)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--preparation-ps", type=float, default=5.0)
    parser.add_argument("--production-ps", type=float, default=300.0)
    parser.add_argument("--ttime-fs", type=float, default=10.0)
    parser.add_argument("--tdamp-fs", type=float, default=50.0)
    parser.add_argument("--pdamp-fs", type=float, default=500.0)
    parser.add_argument("--tchain", type=int, default=5)
    parser.add_argument("--pchain", type=int, default=5)
    parser.add_argument("--tloop", type=int, default=2)
    parser.add_argument("--ploop", type=int, default=2)
    parser.add_argument("--thermo-every", type=int, default=10)
    parser.add_argument("--trajectory-every", type=int, default=100)
    parser.add_argument(
        "--thermo-flush-every",
        type=int,
        default=1,
        help="Flush thermo.csv after this many records; 1 preserves the released launcher",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    """Ensure the displacement-derived virial includes LES positions/cell."""
    import mace.modules.extensions as extensions

    if getattr(extensions, "_figure7c_les_stress_patch", False):
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
    extensions._figure7c_les_stress_patch = True


def scalar_attribute(model, name: str) -> float | int | str | None:
    value = getattr(model, name, None)
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def load_calculator(args: argparse.Namespace) -> tuple[MACECalculator, dict[str, Any]]:
    actual_hash = sha256(args.model)
    if actual_hash != args.model_sha256:
        raise ValueError(f"model hash mismatch: {actual_hash}")
    if sha256(args.input) != INPUT_SHA256:
        raise ValueError("input hash mismatch")
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
        raise ValueError(
            f"stored dtype mismatch: expected {args.stored_precision}, got {stored_dtypes}"
        )
    model = model.double()
    runtime_dtypes = sorted(
        {
            str(value.dtype).replace("torch.", "")
            for value in model.state_dict().values()
            if torch.is_floating_point(value)
        }
    )
    if runtime_dtypes != ["float64"]:
        raise ValueError(f"runtime dtype is not uniformly float64: {runtime_dtypes}")
    les_norm = None
    if args.model_kind == "LES":
        les_norm = float(model.les.ewald.norm_factor)
        if not math.isclose(les_norm, LES_NORM, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"LES normalization changed: {les_norm}")
    state_keys = list(model.state_dict())
    audit = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "source_sha256": actual_hash,
        "stored_floating_dtypes": stored_dtypes,
        "runtime_floating_dtypes": runtime_dtypes,
        "runtime_conversion": (
            "torch.nn.Module.double()" if args.stored_precision == "float32" else None
        ),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters_at_runtime": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "state_entries": len(state_keys),
        "residual_lora_state_entries": sum("lora_" in key for key in state_keys),
        "r_max_angstrom": scalar_attribute(model, "r_max"),
        "num_interactions": len(getattr(model, "interactions", [])),
        "les_ewald_norm_factor": les_norm,
        "les_stress_patch": LES_STRESS_PATCH if args.model_kind == "LES" else None,
    }
    calculator = MACECalculator(
        models=model,
        device=args.device,
        default_dtype="float64",
        model_type="MACE",
        compute_bec=False,
    )
    return calculator, audit


def validate_atoms(atoms) -> None:
    symbols = atoms.get_chemical_symbols()
    if len(atoms) != 192 or not bool(np.all(atoms.pbc)):
        raise ValueError("expected periodic 192-atom (64-water) input")
    for start in range(0, len(atoms), 3):
        if sorted(symbols[start : start + 3]) != ["H", "H", "O"]:
            raise ValueError("expected contiguous three-atom water records")


def density(atoms) -> float:
    mass = float(np.sum(atoms.get_masses()))
    return mass * AMU_PER_ANGSTROM3_TO_G_PER_CM3 / float(atoms.get_volume())


def geometry(atoms) -> dict[str, float]:
    distances = atoms.get_all_distances(mic=True)
    oh: list[float] = []
    oxygen: list[int] = []
    for start in range(0, len(atoms), 3):
        indices = list(range(start, start + 3))
        oxygen_index = next(index for index in indices if atoms[index].symbol == "O")
        hydrogen = [index for index in indices if atoms[index].symbol == "H"]
        oxygen.append(oxygen_index)
        oh.extend(distances[oxygen_index, hydrogen].tolist())
    oxygen_array = np.asarray(oxygen, dtype=np.int64)
    oo = distances[np.ix_(oxygen_array, oxygen_array)].copy()
    np.fill_diagonal(oo, np.inf)
    return {
        "oh_min_angstrom": float(np.min(oh)),
        "oh_max_angstrom": float(np.max(oh)),
        "oo_min_angstrom": float(np.min(oo)),
    }


def thermo_row(atoms, step: int, timestep_fs: float) -> dict[str, float | int]:
    stress = np.asarray(atoms.get_stress(include_ideal_gas=True), dtype=np.float64)
    return {
        "production_step": step,
        "production_time_ps": step * timestep_fs / 1000.0,
        "temperature_K": float(atoms.get_temperature()),
        "pressure_bar": float(-np.mean(stress[:3]) / units.bar),
        "density_g_per_cm3": density(atoms),
        "volume_angstrom3": float(atoms.get_volume()),
        "potential_energy_eV": float(atoms.get_potential_energy()),
        "kinetic_energy_eV": float(atoms.get_kinetic_energy()),
    }


def static_snapshot(atoms) -> dict[str, Any]:
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = np.asarray(atoms.get_stress(include_ideal_gas=True), dtype=np.float64)
    payload: dict[str, Any] = {
        "temperature_K": float(atoms.get_temperature()),
        "pressure_bar": float(-np.mean(stress[:3]) / units.bar),
        "density_g_per_cm3": density(atoms),
        "volume_angstrom3": float(atoms.get_volume()),
        "potential_energy_eV": float(atoms.get_potential_energy()),
        "maximum_force_eV_per_angstrom": float(
            np.max(np.linalg.norm(forces, axis=1))
        ),
        "total_force_norm_eV_per_angstrom": float(
            np.linalg.norm(np.sum(forces, axis=0))
        ),
    }
    payload.update(geometry(atoms))
    return payload


def initialize_velocities(atoms, temperature_K: float, seed: int) -> None:
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed)
    )
    Stationary(atoms, preserve_temperature=True)


def stress_audit(atoms, strain_fraction: float = 1.0e-5) -> dict[str, float]:
    analytic_stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
    analytic_pressure = float(-np.trace(analytic_stress) / 3.0)
    volume = float(atoms.get_volume())
    energies = []
    for sign in (1.0, -1.0):
        candidate = atoms.copy()
        scale = (1.0 + sign * strain_fraction) ** (1.0 / 3.0)
        candidate.set_cell(candidate.cell.array * scale, scale_atoms=True)
        candidate.calc = atoms.calc
        energies.append(float(candidate.get_potential_energy()))
    finite_difference = -(
        energies[0] - energies[1]
    ) / (2.0 * strain_fraction * volume)
    return {
        "strain_fraction": strain_fraction,
        "analytic_pressure_kbar": analytic_pressure * EV_PER_ANGSTROM3_TO_KBAR,
        "finite_difference_pressure_kbar": finite_difference
        * EV_PER_ANGSTROM3_TO_KBAR,
        "absolute_error_kbar": abs(analytic_pressure - finite_difference)
        * EV_PER_ANGSTROM3_TO_KBAR,
    }


def wrapping_audit(atoms) -> dict[str, float]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    candidate = atoms.copy()
    candidate.positions[:3] += candidate.cell.array[0]
    candidate.calc = atoms.calc
    return {
        "absolute_energy_error_eV": abs(float(candidate.get_potential_energy()) - energy),
        "maximum_force_error_eV_per_angstrom": float(
            np.max(np.abs(np.asarray(candidate.get_forces()) - forces))
        ),
    }


def nve_audit(args: argparse.Namespace, atoms, duration_ps: float = 0.2) -> dict[str, Any]:
    candidate = atoms.copy()
    candidate.calc = atoms.calc
    initialize_velocities(candidate, args.temperature_K, args.seed)
    dynamics = VelocityVerlet(candidate, timestep=args.timestep_fs * units.fs)
    initial = float(candidate.get_total_energy())
    finite = True
    maximum_force = 0.0
    for _ in range(steps_for(duration_ps, args.timestep_fs)):
        dynamics.run(1)
        force = np.asarray(candidate.get_forces(), dtype=np.float64)
        finite = finite and bool(np.all(np.isfinite(force)))
        maximum_force = max(maximum_force, float(np.max(np.linalg.norm(force, axis=1))))
    final = float(candidate.get_total_energy())
    drift = (final - initial) * 1000.0 / 64.0 / duration_ps
    return {
        "duration_ps": duration_ps,
        "finite": bool(finite and math.isfinite(final)),
        "energy_drift_meV_per_water_per_ps": drift,
        "absolute_energy_drift_meV_per_water_per_ps": abs(drift),
        "maximum_force_eV_per_angstrom": maximum_force,
    }


def optimize(atoms, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer = BFGS(
        atoms,
        logfile=str(output_dir / "bfgs.log"),
        trajectory=str(output_dir / "bfgs.traj"),
    )
    converged = bool(optimizer.run(fmax=0.2))
    result = static_snapshot(atoms)
    result.update(
        {
            "optimizer": "ase.optimize.BFGS",
            "target_fmax_eV_per_angstrom": 0.2,
            "steps": int(optimizer.nsteps),
            "converged": converged,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write(output_dir / "optimized.extxyz", atoms, format="extxyz")
    return result


def make_preparation(args: argparse.Namespace, atoms) -> NPT:
    return NPT(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        ttime=args.ttime_fs * units.fs,
        pfactor=None,
        externalstress=0.0,
    )


def make_production(args: argparse.Namespace, atoms) -> IsotropicMTKNPT:
    return IsotropicMTKNPT(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        pressure_au=args.pressure_bar * units.bar,
        tdamp=args.tdamp_fs * units.fs,
        pdamp=args.pdamp_fs * units.fs,
        tchain=args.tchain,
        pchain=args.pchain,
        tloop=args.tloop,
        ploop=args.ploop,
    )


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "ase": ase.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
    }


def common_payload(args: argparse.Namespace, model_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "water-density-npt-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": args.model_label,
        "model_kind": args.model_kind,
        "model_path": str((args.model_source_path or args.model).resolve()),
        "model_staged_path": (
            str(args.model.resolve()) if args.model_source_path is not None else None
        ),
        "model_sha256": sha256(args.model),
        "model_audit": model_audit,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "locked_test_used": False,
        "runtime_precision": "float64",
        "temperature_K": args.temperature_K,
        "pressure_bar": args.pressure_bar,
        "velocity_seed": args.seed,
        "physical_hydrogen_masses": True,
        "environment": environment(),
    }


def run_gate(args: argparse.Namespace, atoms) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calculator, model_audit = load_calculator(args)
    atoms.calc = calculator
    started = time.perf_counter()
    static = static_snapshot(atoms)
    stress = stress_audit(atoms)
    wrapping = wrapping_audit(atoms)
    nve = nve_audit(args, atoms)
    optimized_atoms = atoms.copy()
    optimized_atoms.calc = calculator
    optimization = optimize(optimized_atoms, args.output_dir)
    initialize_velocities(optimized_atoms, args.temperature_K, args.seed)
    preparation = make_preparation(args, optimized_atoms)
    preparation.run(100)
    production = make_production(args, optimized_atoms)
    production.run(200)
    short_mtk = static_snapshot(optimized_atoms)
    short_mtk["finite"] = all(
        math.isfinite(float(value)) for value in short_mtk.values()
    )
    flags = {
        "static_finite": all(math.isfinite(float(value)) for value in static.values()),
        "total_force": static["total_force_norm_eV_per_angstrom"] <= 1.0e-2,
        "isotropic_stress": stress["absolute_error_kbar"] <= 0.05,
        "wrapping_energy": wrapping["absolute_energy_error_eV"] <= 2.0e-2,
        "wrapping_force": wrapping["maximum_force_error_eV_per_angstrom"] <= 2.0e-3,
        "nve_finite": bool(nve["finite"]),
        "optimization_converged": bool(optimization["converged"]),
        "short_mtk_finite": bool(short_mtk["finite"]),
        "short_mtk_density": 0.70 <= short_mtk["density_g_per_cm3"] <= 1.40,
        "short_mtk_geometry": (
            0.5 <= short_mtk["oh_min_angstrom"]
            and short_mtk["oh_max_angstrom"] <= 1.5
            and short_mtk["oo_min_angstrom"] >= 1.8
        ),
        "fused_export_has_no_lora_state": model_audit["residual_lora_state_entries"] == 0,
        "runtime_float64": model_audit["runtime_floating_dtypes"] == ["float64"],
    }
    payload = common_payload(args, model_audit)
    payload.update(
        {
            "scope": "water_density_protocol_fail_closed_gate",
            "static": static,
            "isotropic_stress": stress,
            "wrapping": wrapping,
            "nve_1fs": nve,
            "optimization": optimization,
            "short_mtk": short_mtk,
            "pass_flags": flags,
            "gate_pass": all(flags.values()),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    atomic_json(args.output_dir / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["gate_pass"] else 2


def density_statistics(rows: list[dict[str, Any]], start_ps: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row["production_time_ps"]) >= start_ps]
    values = np.asarray([row["density_g_per_cm3"] for row in selected], dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        return {"start_ps": start_ps, "samples": len(values), "finite": False}
    result: dict[str, Any] = {
        "start_ps": start_ps,
        "samples": len(values),
        "finite": True,
        "mean_g_per_cm3": float(np.mean(values)),
        "sample_sd_g_per_cm3": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }
    times = np.asarray([row["production_time_ps"] for row in selected], dtype=np.float64)
    if times[-1] - times[0] >= 149.0:
        edges = np.linspace(times[0], times[-1] + 1.0e-12, 11)
        block_means = []
        for index in range(10):
            mask = (times >= edges[index]) & (
                (times < edges[index + 1]) if index < 9 else (times <= edges[index + 1])
            )
            if np.any(mask):
                block_means.append(float(np.mean(values[mask])))
        result["ten_block_means_g_per_cm3"] = block_means
        result["ten_block_sem_g_per_cm3"] = (
            float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))
            if len(block_means) > 1
            else None
        )
    return result


def run_production(args: argparse.Namespace, atoms) -> int:
    if args.thermo_flush_every < 1:
        raise ValueError("thermo-flush-every must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    calculator, model_audit = load_calculator(args)
    atoms.calc = calculator
    started = time.perf_counter()
    run_error = None
    rows: list[dict[str, Any]] = []
    completed_preparation_steps = 0
    completed_production_steps = 0
    optimization: dict[str, Any] | None = None
    csv_handle = None
    trajectory = None
    try:
        optimization = optimize(atoms, args.output_dir)
        if not optimization["converged"]:
            raise RuntimeError("BFGS did not converge to fmax=0.2 eV/angstrom")
        initialize_velocities(atoms, args.temperature_K, args.seed)
        preparation = make_preparation(args, atoms)
        preparation_steps = steps_for(args.preparation_ps, args.timestep_fs)
        preparation.run(preparation_steps)
        completed_preparation_steps = preparation.nsteps
        write(args.output_dir / "post_preparation.extxyz", atoms, format="extxyz")

        csv_handle = (args.output_dir / "thermo.csv").open(
            "w", newline="", encoding="utf-8"
        )
        writer: csv.DictWriter | None = None
        trajectory = Trajectory(args.output_dir / "trajectory.traj", "w", atoms)
        dynamics = make_production(args, atoms)

        def record_thermo() -> None:
            nonlocal writer
            row = thermo_row(atoms, dynamics.nsteps, args.timestep_fs)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if len(rows) % args.thermo_flush_every == 0:
                csv_handle.flush()

        def record_trajectory() -> None:
            trajectory.write(atoms)

        dynamics.attach(record_thermo, interval=args.thermo_every)
        dynamics.attach(record_trajectory, interval=args.trajectory_every)
        dynamics.run(steps_for(args.production_ps, args.timestep_fs))
        completed_production_steps = dynamics.nsteps
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception as error:
        run_error = f"{type(error).__name__}: {error}"
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if trajectory is not None:
            trajectory.close()
    elapsed = time.perf_counter() - started
    try:
        write(args.output_dir / "final.extxyz", atoms, format="extxyz")
        final_snapshot = static_snapshot(atoms)
    except Exception as error:
        final_snapshot = None
        if run_error is None:
            run_error = f"{type(error).__name__}: {error}"
    expected_preparation = steps_for(args.preparation_ps, args.timestep_fs)
    expected_production = steps_for(args.production_ps, args.timestep_fs)
    full = density_statistics(rows, 0.0)
    final_150 = density_statistics(rows, max(0.0, args.production_ps - 150.0))
    complete = bool(
        run_error is None
        and completed_preparation_steps == expected_preparation
        and completed_production_steps == expected_production
        and full.get("finite")
        and final_150.get("finite")
    )
    payload = common_payload(args, model_audit)
    payload.update(
        {
            "scope": "water_density_protocol_float64_single_replica",
            "protocol_label": "official_les_fit_NPT-ase_executable_protocol",
            "manuscript_code_discrepancy": (
                "manuscript names Berendsen; released script runs IsotropicMTKNPT"
            ),
            "geometry_optimization": optimization,
            "timestep_fs": args.timestep_fs,
            "preparation_integrator": "ase.md.npt.NPT with pfactor=None",
            "preparation_ps": args.preparation_ps,
            "preparation_ttime_fs": args.ttime_fs,
            "production_integrator": "ase.md.nose_hoover_chain.IsotropicMTKNPT",
            "production_ps": args.production_ps,
            "thermostat_damping_fs": args.tdamp_fs,
            "barostat_damping_fs": args.pdamp_fs,
            "thermostat_chain_length": args.tchain,
            "barostat_chain_length": args.pchain,
            "thermostat_substeps": args.tloop,
            "barostat_substeps": args.ploop,
            "thermo_every_steps": args.thermo_every,
            "thermo_flush_every_records": args.thermo_flush_every,
            "trajectory_every_steps": args.trajectory_every,
            "completed_preparation_steps": completed_preparation_steps,
            "completed_production_steps": completed_production_steps,
            "density_all_production": full,
            "density_last_150ps": final_150,
            "final_snapshot": final_snapshot,
            "elapsed_seconds": elapsed,
            "production_steps_per_second": (
                completed_production_steps / elapsed if elapsed > 0 else None
            ),
            "run_error": run_error,
            "complete": complete,
        }
    )
    atomic_json(args.output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if complete else 2


def main() -> int:
    args = parse_args()
    if len(args.model_sha256) != 64:
        raise ValueError("invalid SHA-256")
    atoms = read(args.input)
    validate_atoms(atoms)
    if args.mode == "gate":
        return run_gate(args, atoms)
    return run_production(args, atoms)


if __name__ == "__main__":
    raise SystemExit(main())
