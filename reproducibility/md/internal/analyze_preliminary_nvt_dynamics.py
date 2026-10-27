#!/usr/bin/env python3
"""Analyze live NVT prefixes using the Montero-de-Hijes transport estimators.

The production trajectories can still be growing.  This program only consumes
complete segments and the part of a partial segment covered by flushed thermo
records.  It never creates campaign completion markers and it records every
source directory and analyzed duration in the output tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "lane_manifest.tsv"
TARGET_PREFIX_PS = 600.0
MD_DT_PS = 0.0005
POSITION_DT_PS = 0.05
CORRELATION_STRIDE = int(os.environ.get("MACE_WATER_CORRELATION_STRIDE", "10"))
if CORRELATION_STRIDE < 1:
    raise ValueError("MACE_WATER_CORRELATION_STRIDE must be positive")
CORRELATION_DT_PS = MD_DT_PS * CORRELATION_STRIDE
VACF_MAX_LAG_PS = 10.0
STRESS_MAX_LAG_PS = 20.0
VACF_PLATEAU_PS = (5.0, 10.0)
VISCOSITY_PLATEAU_PS = (5.0, 10.0)
VISCOSITY_SENSITIVITY_PS = (10.0, 20.0)
MSD_FIT_PS = (10.0, 600.0)
MSD_PLOT_MAX_PS = 1000.0
PREFIX_BLOCK_PS = 100.0
ASE_FS = 0.09822694788464063
KB_J_PER_K = 1.380649e-23
XI_CUBIC = 2.837297
EV_A3_TO_PA = 1.602176634e11
STRESS_LABELS = ("Pxy", "Pxz", "Pyz", "(Pxx-Pyy)/2", "(Pyy-Pzz)/2")
TEMPERATURES = (275, 295, 315, 336)
MODEL_ORDER = (
    "OFF-ZS",
    "OFF-T12-B123200",
    "OFF-D00-B123200",
    "OFF-D11-B123200",
    "LES-ZS",
    "LES-T12-B123200",
    "LES-D00-B123200",
    "LES-D11-B123200",
)
CONDITION_ORDER = ("ZS", "T12", "D00", "D11")
CONDITION_COLORS = {
    "ZS": "#4c78a8",
    "T12": "#f58518",
    "D00": "#54a24b",
    "D11": "#e45756",
}
MONTERO_RPBE = (
    {"temperature_K": 278.0, "diffusion_nm2_ns": 1.484, "viscosity_mPa_s": 1.9,
     "ensemble": "NVT 600 ps; VACF; finite-size corrected"},
    {"temperature_K": 297.0, "diffusion_nm2_ns": 2.768, "viscosity_mPa_s": 0.8,
     "ensemble": "NVT 600 ps; VACF; finite-size corrected"},
    {"temperature_K": 269.5, "diffusion_nm2_ns": 0.689, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE 5-9 ns; MSD"},
    {"temperature_K": 296.0, "diffusion_nm2_ns": 2.065, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE 5-9 ns; MSD"},
    {"temperature_K": 315.0, "diffusion_nm2_ns": 3.661, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE 5-9 ns; MSD"},
)
# Figure 3 markers digitized from the official 7200 x 6000 arXiv source image.
HILPERT_MACE_L2 = (
    {"temperature_K": 300.0, "diffusion_nm2_ns": 2.47, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE MSD; approximately digitized from Fig. 3"},
    {"temperature_K": 325.0, "diffusion_nm2_ns": 4.78, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE MSD; approximately digitized from Fig. 3"},
    {"temperature_K": 350.0, "diffusion_nm2_ns": 7.54, "viscosity_mPa_s": math.nan,
     "ensemble": "NVE MSD; approximately digitized from Fig. 3"},
)


@dataclass(frozen=True)
class SegmentSource:
    segment: int
    path: str
    duration_ps: float
    complete: bool
    velocity_frames: int
    stress_frames: int
    position_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-prefix-ps", type=float, default=TARGET_PREFIX_PS)
    parser.add_argument("--lane", type=int, action="append", dest="lanes")
    parser.add_argument("--model-label", action="append", dest="model_labels")
    parser.add_argument(
        "--comparison-states",
        type=Path,
        help="optional state table from an earlier analysis used to plot numerical changes",
    )
    parser.add_argument(
        "--complete-segments-only",
        action="store_true",
        help=(
            "freeze and analyze each lane's committed checkpoint prefix up to its "
            "requested target; ignore actively growing partial segments"
        ),
    )
    parser.add_argument(
        "--model-target",
        action="append",
        default=[],
        metavar="MODEL=PS",
        help="cap committed-prefix discovery for every lane of one model",
    )
    parser.add_argument(
        "--lane-target",
        action="append",
        default=[],
        metavar="LANE=PS",
        help="override the committed-prefix cap for one lane",
    )
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_target_assignments(values: list[str], value_name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        key, separator, raw_target = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid {value_name} assignment: {value!r}")
        target = float(raw_target)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError(f"invalid {value_name} duration: {value!r}")
        result[key] = target
    return result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_thermo_time(path: Path) -> float:
    last = -math.inf
    if not path.is_file():
        return last
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                value = float(row["time_ps"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                last = max(last, value)
    return last


def candidate_segment(
    lane_root: Path,
    segment: int,
    expected_ps: float,
    allow_partial: bool = True,
) -> SegmentSource | None:
    final = lane_root / f"nvt_segment{segment}"
    if (final / "COMPLETE").is_file() and (final / "summary.json").is_file():
        summary = read_json(final / "summary.json")
        if not summary.get("complete") or int(summary.get("segment", -1)) != segment:
            raise ValueError(f"invalid completed segment metadata: {final}")
        duration = min(float(summary["production_ps"]), expected_ps)
        return SegmentSource(
            segment=segment,
            path=str(final),
            duration_ps=duration,
            complete=True,
            velocity_frames=int(round(duration / MD_DT_PS)) + 1,
            stress_frames=int(round(duration / MD_DT_PS)) + 1,
            position_frames=int(round(duration / POSITION_DT_PS)) + 1,
        )

    if not allow_partial:
        return None

    partials = list(lane_root.glob(f".nvt_segment{segment}.partial.*"))
    scored: list[tuple[float, int, Path]] = []
    for partial in partials:
        last = read_last_thermo_time(partial / "thermo.csv")
        if last >= 0.0:
            scored.append((last, partial.stat().st_mtime_ns, partial))
    if not scored:
        return None
    last, _, partial = max(scored)
    duration = min(last, expected_ps)
    return SegmentSource(
        segment=segment,
        path=str(partial),
        duration_ps=duration,
        complete=False,
        velocity_frames=int(math.floor(duration / MD_DT_PS + 1.0e-8)) + 1,
        stress_frames=int(math.floor(duration / MD_DT_PS + 1.0e-8)) + 1,
        position_frames=int(math.floor(duration / POSITION_DT_PS + 1.0e-8)) + 1,
    )


def discover_segments(
    lane_root: Path,
    target_ps: float,
    allow_partial: bool = True,
) -> list[SegmentSource]:
    sources: list[SegmentSource] = []
    accumulated = 0.0
    for segment in range(1, 7):
        expected = 500.0 if segment == 1 else 100.0
        source = candidate_segment(lane_root, segment, expected, allow_partial=allow_partial)
        if source is None:
            break
        remaining = target_ps - accumulated
        if remaining <= 0.0:
            break
        if source.duration_ps > remaining:
            duration = remaining
            source = SegmentSource(
                segment=source.segment,
                path=source.path,
                duration_ps=duration,
                complete=source.complete,
                velocity_frames=int(math.floor(duration / MD_DT_PS + 1.0e-8)) + 1,
                stress_frames=int(math.floor(duration / MD_DT_PS + 1.0e-8)) + 1,
                position_frames=int(math.floor(duration / POSITION_DT_PS + 1.0e-8)) + 1,
            )
        sources.append(source)
        accumulated += source.duration_ps
        if not source.complete or accumulated >= target_ps - 1.0e-9:
            break
    return sources


def load_thermo(sources: list[SegmentSource]) -> pd.DataFrame:
    tables = []
    offset = 0.0
    for index, source in enumerate(sources):
        path = Path(source.path) / "thermo.csv"
        table = pd.read_csv(path, on_bad_lines="skip")
        table = table[pd.to_numeric(table["time_ps"], errors="coerce") <= source.duration_ps + 1.0e-8].copy()
        table["global_time_ps"] = pd.to_numeric(table["time_ps"], errors="coerce") + offset
        if index:
            table = table[table["time_ps"] > 1.0e-12]
        tables.append(table)
        offset += source.duration_ps
    if not tables:
        raise ValueError("no thermo data")
    return pd.concat(tables, ignore_index=True)


def load_sampled(
    sources: list[SegmentSource], filename: str, frame_field: str, stride: int, dtype: np.dtype
) -> np.ndarray:
    chunks = []
    for index, source in enumerate(sources):
        path = Path(source.path) / filename
        if not path.is_file():
            # A live segment may have atomically changed from partial to final.
            final = Path(source.path).parent / f"nvt_segment{source.segment}" / filename
            if final.is_file():
                path = final
            else:
                raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = min(int(getattr(source, frame_field)), len(array))
        if valid < 2:
            raise ValueError(f"too few valid frames in {path}")
        start = stride if index else 0
        chunk = np.asarray(array[start:valid:stride], dtype=dtype)
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"non-finite data in safe prefix of {path}")
        chunks.append(chunk)
    result = np.concatenate(chunks, axis=0)
    if len(result) < 2 or not np.any(result):
        raise ValueError(f"empty or zero series for {filename}")
    return result


def cumulative_trapezoid(y: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros_like(y, dtype=np.float64)
    if len(y) > 1:
        result[1:] = np.cumsum(0.5 * (y[1:] + y[:-1])) * dt
    return result


def plateau_average(time: np.ndarray, values: np.ndarray, window: tuple[float, float]) -> float:
    mask = (time >= window[0] - 1.0e-12) & (time <= window[1] + 1.0e-12)
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.mean(values[mask]))


def window_diagnostics(time: np.ndarray, values: np.ndarray, window: tuple[float, float]) -> dict[str, float]:
    mask = (time >= window[0]) & (time <= window[1])
    x = time[mask]
    y = values[mask]
    if len(y) < 3:
        return {"mean": math.nan, "range_fraction": math.nan, "slope_fraction": math.nan}
    mean = float(np.mean(y))
    scale = max(abs(mean), 1.0e-30)
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "mean": mean,
        "range_fraction": float(np.ptp(y) / scale),
        "slope_fraction": float(abs(slope) * (window[1] - window[0]) / scale),
    }


def vacf_from_velocities(velocities_A_ps: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(velocities_A_ps, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)  # instantaneous oxygen COM drift
    flat = values.reshape(len(values), -1)
    flat -= flat.mean(axis=0, keepdims=True)
    nframes = len(flat)
    nfft = 1 << (2 * nframes - 1).bit_length()
    power = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for start in range(0, flat.shape[1], 16):
        spectrum = np.fft.rfft(flat[:, start : start + 16], n=nfft, axis=0)
        power += np.sum(spectrum.real * spectrum.real + spectrum.imag * spectrum.imag, axis=1)
    numerator = np.fft.irfft(power, n=nfft)[: max_lag + 1]
    denominator = (nframes - np.arange(max_lag + 1)) * values.shape[1]
    return numerator / denominator


def stress_correlations(stress_pa: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(stress_pa, dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    nframes = len(values)
    nfft = 1 << (2 * nframes - 1).bit_length()
    spectrum = np.fft.rfft(values, n=nfft, axis=0)
    result = np.empty((max_lag + 1, 5, 5), dtype=np.float64)
    counts = nframes - np.arange(max_lag + 1)
    for i in range(5):
        for j in range(5):
            product = spectrum[:, i].conj() * spectrum[:, j]
            result[:, i, j] = np.fft.irfft(product, n=nfft)[: max_lag + 1] / counts
    # C_ij(+t) and C_ji(+t) are equivalent in an isotropic equilibrium liquid.
    return 0.5 * (result + result.transpose(0, 2, 1))


def unwrap_oxygen(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    delta = np.diff(positions, axis=0)
    fractional = delta @ inverse
    delta -= np.rint(fractional) @ cell
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    unwrapped[1:] = positions[0] + np.cumsum(delta, axis=0)
    return unwrapped


def msd_fft(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    values -= values[0:1]
    nframes, atoms, _ = values.shape
    flat = values.reshape(nframes, -1)
    nfft = 1 << (2 * nframes - 1).bit_length()
    spectrum = np.fft.rfft(flat, n=nfft, axis=0)
    autocorrelation = np.fft.irfft(
        spectrum.conj() * spectrum, n=nfft, axis=0
    )[:nframes].sum(axis=1)
    squared = np.einsum("ij,ij->i", flat, flat)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    lags = np.arange(nframes)
    numerator = (
        cumulative[nframes - lags]
        + cumulative[nframes]
        - cumulative[lags]
        - 2.0 * autocorrelation
    )
    return np.maximum(numerator / ((nframes - lags) * atoms), 0.0)


def fit_msd(time_ps: np.ndarray, msd_A2: np.ndarray, window: tuple[float, float]) -> dict[str, float]:
    mask = (time_ps >= window[0]) & (time_ps <= window[1])
    if np.count_nonzero(mask) < 3:
        return {"diffusion_nm2_ns": math.nan, "slope_A2_ps": math.nan, "r_squared": math.nan}
    slope, intercept = np.polyfit(time_ps[mask], msd_A2[mask], 1)
    fitted = slope * time_ps[mask] + intercept
    residual = msd_A2[mask] - fitted
    centered = msd_A2[mask] - np.mean(msd_A2[mask])
    denominator = float(np.dot(centered, centered))
    r_squared = 1.0 - float(np.dot(residual, residual)) / denominator if denominator > 0 else math.nan
    return {
        "diffusion_nm2_ns": float(slope * 10.0 / 6.0),
        "slope_A2_ps": float(slope),
        "r_squared": r_squared,
    }


def yeh_hummer(temperature_K: float, volume_A3: float, viscosity_mPa_s: float) -> float:
    if not math.isfinite(viscosity_mPa_s) or viscosity_mPa_s <= 0.0:
        return math.nan
    length_m = volume_A3 ** (1.0 / 3.0) * 1.0e-10
    correction_m2_s = KB_J_PER_K * temperature_K * XI_CUBIC / (
        6.0 * math.pi * viscosity_mPa_s * 1.0e-3 * length_m
    )
    return correction_m2_s * 1.0e9


def experimental_diffusion(temperature_K: float) -> float:
    """Holz-Heil-Sacco Speedy-Angell fit, returned in nm^2/ns."""
    return 16.35 * ((temperature_K / 215.05) - 1.0) ** 2.063


def experimental_viscosity(temperature_K: float) -> float:
    """IAPWS SR6-08 Eq. 7 at 0.1 MPa, returned in mPa s."""
    reduced = temperature_K / 300.0
    coefficients = ((280.68, -1.9), (511.45, -7.7), (61.131, -19.6), (0.45903, -40.0))
    # Equation gives micro-Pa s; divide by 1000 to obtain mPa s.
    return sum(a * reduced**b for a, b in coefficients) / 1000.0


def volume_for_lane(lane_root: Path, sources: list[SegmentSource]) -> tuple[float, np.ndarray]:
    from ase.io import read

    first = Path(sources[0].path)
    start = first / "nvt_start.traj"
    if not start.is_file():
        start = lane_root / "nvt_segment1" / "nvt_start.traj"
    atoms = read(start, index=0)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    cell_volume = float(abs(np.linalg.det(cell)))
    volume = cell_volume
    summary = first / "summary.json"
    npt_analysis = lane_root / "npt" / "npt_analysis.json"
    if summary.is_file():
        volume = float(read_json(summary)["volume_A3"])
    elif npt_analysis.is_file():
        volume = float(read_json(npt_analysis)["representative_volume_A3"])
    if not math.isclose(cell_volume, volume, rel_tol=2.0e-6, abs_tol=2.0e-3):
        raise ValueError(f"cell/metadata volume mismatch in {lane_root}")
    return volume, cell


def block_estimates(
    velocities: np.ndarray,
    stress_pa: np.ndarray,
    temperature_K: float,
    volume_A3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_frames = int(round(PREFIX_BLOCK_PS / CORRELATION_DT_PS))
    blocks = min(len(velocities), len(stress_pa)) // block_frames
    times: list[float] = []
    d_values: list[float] = []
    eta_values: list[float] = []
    vacf_lag = int(round(VACF_MAX_LAG_PS / CORRELATION_DT_PS))
    stress_lag = int(round(STRESS_MAX_LAG_PS / CORRELATION_DT_PS))
    vacf_time = np.arange(vacf_lag + 1) * CORRELATION_DT_PS
    stress_time = np.arange(stress_lag + 1) * CORRELATION_DT_PS
    volume_m3 = volume_A3 * 1.0e-30
    for block in range(blocks):
        start = block * block_frames
        stop = start + block_frames
        vacf = vacf_from_velocities(velocities[start:stop], vacf_lag)
        running_d = cumulative_trapezoid(vacf, CORRELATION_DT_PS) * (10.0 / 3.0)
        corr = stress_correlations(stress_pa[start:stop], stress_lag)
        diagonal = np.trace(corr, axis1=1, axis2=2) / 5.0
        running_eta = (
            volume_m3
            / (KB_J_PER_K * temperature_K)
            * cumulative_trapezoid(diagonal, CORRELATION_DT_PS * 1.0e-12)
            * 1000.0
        )
        times.append((block + 1) * PREFIX_BLOCK_PS)
        d_values.append(plateau_average(vacf_time, running_d, VACF_PLATEAU_PS))
        eta_values.append(plateau_average(stress_time, running_eta, VISCOSITY_PLATEAU_PS))
    if not times:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(times), np.cumsum(d_values) / np.arange(1, blocks + 1), np.cumsum(eta_values) / np.arange(1, blocks + 1)


def analyze_lane(
    lane: dict[str, str],
    output_dir: str,
    target_ps: float,
    allow_partial: bool = True,
) -> dict[str, Any]:
    lane_id = int(lane["lane_id"])
    lane_root = Path(lane["output_dir"])
    lane_output = Path(output_dir) / "lane_curves" / f"lane_{lane_id:02d}.npz"
    try:
        sources = discover_segments(lane_root, target_ps, allow_partial=allow_partial)
        if not sources:
            raise ValueError("no NVT segment with a flushed prefix")
        analyzed_ps = float(sum(source.duration_ps for source in sources))
        if analyzed_ps < 100.0:
            raise ValueError(f"only {analyzed_ps:.1f} ps are safely visible")

        thermo = load_thermo(sources)
        temperatures = pd.to_numeric(thermo["temperature_K"], errors="coerce").to_numpy()
        temperatures = temperatures[np.isfinite(temperatures)]
        if len(temperatures) < 2:
            raise ValueError("insufficient finite thermo temperatures")
        mean_temperature = float(np.mean(temperatures))
        target_temperature = float(lane["temperature_K"])
        if abs(mean_temperature - target_temperature) > 20.0:
            raise ValueError("mean NVT temperature differs from target by more than 20 K")

        volume_A3, cell = volume_for_lane(lane_root, sources)
        velocities = load_sampled(
            sources,
            "oxygen_velocities.npy",
            "velocity_frames",
            CORRELATION_STRIDE,
            np.float32,
        )
        velocities *= ASE_FS * 1000.0
        stress_pa = load_sampled(
            sources,
            "stress_components_eV_A3.npy",
            "stress_frames",
            CORRELATION_STRIDE,
            np.float64,
        )
        stress_pa *= EV_A3_TO_PA
        positions = load_sampled(
            sources,
            "oxygen_positions.npy",
            "position_frames",
            1,
            np.float64,
        )
        positions = unwrap_oxygen(positions, cell)

        vacf_max = min(int(round(VACF_MAX_LAG_PS / CORRELATION_DT_PS)), len(velocities) - 2)
        vacf_time = np.arange(vacf_max + 1) * CORRELATION_DT_PS
        vacf = vacf_from_velocities(velocities, vacf_max)
        running_d = cumulative_trapezoid(vacf, CORRELATION_DT_PS) * (10.0 / 3.0)
        diffusion_pbc = plateau_average(vacf_time, running_d, VACF_PLATEAU_PS)
        d_diag = window_diagnostics(vacf_time, running_d, VACF_PLATEAU_PS)

        stress_max = min(int(round(STRESS_MAX_LAG_PS / CORRELATION_DT_PS)), len(stress_pa) - 2)
        stress_time = np.arange(stress_max + 1) * CORRELATION_DT_PS
        stress_corr = stress_correlations(stress_pa, stress_max)
        stress_diagonal = np.trace(stress_corr, axis1=1, axis2=2) / 5.0
        running_eta = (
            volume_A3 * 1.0e-30
            / (KB_J_PER_K * mean_temperature)
            * cumulative_trapezoid(stress_diagonal, CORRELATION_DT_PS * 1.0e-12)
            * 1000.0
        )
        viscosity = plateau_average(stress_time, running_eta, VISCOSITY_PLATEAU_PS)
        viscosity_10_20 = plateau_average(stress_time, running_eta, VISCOSITY_SENSITIVITY_PS)
        eta_diag = window_diagnostics(stress_time, running_eta, VISCOSITY_PLATEAU_PS)

        scale = np.sqrt(np.maximum(np.diag(stress_corr[0]), 1.0e-300))
        normalized_cross = stress_corr / (scale[None, :, None] * scale[None, None, :])
        unexpected_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5) if (i, j) != (3, 4)]
        short = stress_time <= 2.0
        unexpected_rms = float(
            np.sqrt(np.mean([normalized_cross[short, i, j] ** 2 for i, j in unexpected_pairs]))
        )
        expected_diag_pair_zero = float(normalized_cross[0, 3, 4])

        msd_time = np.arange(len(positions)) * POSITION_DT_PS
        msd = msd_fft(positions)
        msd_fit = fit_msd(msd_time, msd, MSD_FIT_PS)
        msd_windows = {}
        for start, stop in (
            (5.0, 20.0),
            (20.0, 50.0),
            (50.0, 100.0),
            (100.0, 200.0),
            MSD_FIT_PS,
        ):
            if stop <= msd_time[-1] + 1.0e-8:
                msd_windows[f"{int(start)}_{int(stop)}ps"] = fit_msd(msd_time, msd, (start, stop))["diffusion_nm2_ns"]

        correction = yeh_hummer(mean_temperature, volume_A3, viscosity)
        prefix_time, prefix_d, prefix_eta = block_estimates(
            velocities, stress_pa, mean_temperature, volume_A3
        )
        prefix_correction = np.asarray(
            [yeh_hummer(mean_temperature, volume_A3, value) for value in prefix_eta]
        )
        prefix_d_corrected = prefix_d + prefix_correction

        lane_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            lane_output,
            vacf_time_ps=vacf_time.astype(np.float32),
            vacf_A2_ps2=vacf.astype(np.float32),
            vacf_normalized=(vacf / vacf[0]).astype(np.float32),
            running_diffusion_pbc_nm2_ns=running_d.astype(np.float32),
            stress_time_ps=stress_time.astype(np.float32),
            stress_diagonal_Pa2=stress_diagonal.astype(np.float32),
            stress_cross_normalized=normalized_cross.astype(np.float32),
            running_viscosity_mPa_s=running_eta.astype(np.float32),
            msd_time_ps=msd_time.astype(np.float32),
            msd_A2=msd.astype(np.float32),
            prefix_time_ps=prefix_time.astype(np.float32),
            prefix_diffusion_pbc_nm2_ns=prefix_d.astype(np.float32),
            prefix_viscosity_mPa_s=prefix_eta.astype(np.float32),
            prefix_diffusion_corrected_nm2_ns=prefix_d_corrected.astype(np.float32),
        )
        return {
            "status": "ok",
            "lane_id": lane_id,
            "model_label": lane["model_label"],
            "backbone": lane["backbone"],
            "condition": lane["condition"],
            "temperature_K": int(lane["temperature_K"]),
            "md_replica": int(lane["md_replica"]),
            "md_seed": int(lane["md_seed"]),
            "analyzed_ps": analyzed_ps,
            "target_prefix_ps": target_ps,
            "complete_600ps": analyzed_ps >= 600.0 - 1.0e-9,
            "reached_target_ps": analyzed_ps >= target_ps - 1.0e-9,
            "source_segments": json.dumps([asdict(source) for source in sources], separators=(",", ":")),
            "mean_temperature_K": mean_temperature,
            "volume_A3": volume_A3,
            "box_length_A": volume_A3 ** (1.0 / 3.0),
            "diffusion_vacf_pbc_nm2_ns": diffusion_pbc,
            "diffusion_vacf_plateau_range_fraction": d_diag["range_fraction"],
            "diffusion_vacf_plateau_slope_fraction": d_diag["slope_fraction"],
            "viscosity_mPa_s": viscosity,
            "viscosity_10_20ps_mPa_s": viscosity_10_20,
            "viscosity_plateau_range_fraction": eta_diag["range_fraction"],
            "viscosity_plateau_slope_fraction": eta_diag["slope_fraction"],
            "stress_unexpected_cross_rms_0_2ps": unexpected_rms,
            "stress_diagonal_pair_rho_t0": expected_diag_pair_zero,
            "yeh_hummer_nm2_ns": correction,
            "diffusion_vacf_corrected_nm2_ns": finite_size_corrected_diffusion(
                diffusion_pbc, correction
            ),
            "diffusion_msd_pbc_nm2_ns": msd_fit["diffusion_nm2_ns"],
            "diffusion_msd_corrected_nm2_ns": finite_size_corrected_diffusion(
                msd_fit["diffusion_nm2_ns"], correction
            ),
            "diffusion_msd_r_squared": msd_fit["r_squared"],
            "diffusion_msd_windows_json": json.dumps(msd_windows, separators=(",", ":")),
            "experimental_diffusion_nm2_ns": experimental_diffusion(target_temperature),
            "experimental_viscosity_mPa_s": experimental_viscosity(target_temperature),
            "curve_file": str(lane_output.relative_to(Path(output_dir))),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "lane_id": lane_id,
            "model_label": lane.get("model_label", ""),
            "backbone": lane.get("backbone", ""),
            "condition": lane.get("condition", ""),
            "temperature_K": int(lane.get("temperature_K", 0)),
            "md_replica": int(lane.get("md_replica", 0)),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, 0
    sem = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan
    return float(np.mean(array)), sem, len(array)


def finite_size_corrected_diffusion(diffusion_pbc: float, yeh_hummer: float) -> float:
    """Apply the cubic-box Yeh–Hummer shift to any long-time D estimator."""
    return float(diffusion_pbc + yeh_hummer)


def aggregate_states(lanes: pd.DataFrame) -> pd.DataFrame:
    numeric = (
        "analyzed_ps",
        "mean_temperature_K",
        "volume_A3",
        "diffusion_vacf_pbc_nm2_ns",
        "viscosity_mPa_s",
        "viscosity_10_20ps_mPa_s",
        "yeh_hummer_nm2_ns",
        "diffusion_vacf_corrected_nm2_ns",
        "diffusion_msd_pbc_nm2_ns",
        "diffusion_msd_corrected_nm2_ns",
        "diffusion_msd_r_squared",
        "stress_unexpected_cross_rms_0_2ps",
        "stress_diagonal_pair_rho_t0",
        "diffusion_vacf_plateau_range_fraction",
        "viscosity_plateau_range_fraction",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in lanes.groupby(["model_label", "backbone", "condition", "temperature_K"], sort=False):
        model, backbone, condition, temperature = keys
        row: dict[str, Any] = {
            "model_label": model,
            "backbone": backbone,
            "condition": condition,
            "temperature_K": int(temperature),
            "replicas": len(group),
            "complete_600ps_replicas": int(group["complete_600ps"].sum()),
            "analyzed_min_ps": float(group["analyzed_ps"].min()),
            "analyzed_max_ps": float(group["analyzed_ps"].max()),
            "experimental_diffusion_nm2_ns": experimental_diffusion(float(temperature)),
            "experimental_viscosity_mPa_s": experimental_viscosity(float(temperature)),
        }
        for column in numeric:
            mean, sem, valid = mean_sem(group[column])
            row[column] = mean
            row[f"{column}_sem"] = sem
            row[f"{column}_n"] = valid
        row["diffusion_corrected_percent_error"] = 100.0 * (
            row["diffusion_vacf_corrected_nm2_ns"] / row["experimental_diffusion_nm2_ns"] - 1.0
        )
        row["viscosity_percent_error"] = 100.0 * (
            row["viscosity_mPa_s"] / row["experimental_viscosity_mPa_s"] - 1.0
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["model_label"] = pd.Categorical(result["model_label"], MODEL_ORDER, ordered=True)
    return result.sort_values(["model_label", "temperature_K"]).reset_index(drop=True)


def paired_shifts(lanes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone in ("OFF", "LES"):
        subset = lanes[lanes["backbone"] == backbone]
        for left, right, name in (("D00", "D11", "D11-D00"), ("ZS", "T12", "T12-ZS")):
            a = subset[subset["condition"] == left]
            b = subset[subset["condition"] == right]
            merged = a.merge(b, on=["temperature_K", "md_replica"], suffixes=("_left", "_right"))
            for temperature, group in merged.groupby("temperature_K"):
                row: dict[str, Any] = {
                    "backbone": backbone,
                    "contrast": name,
                    "temperature_K": int(temperature),
                    "paired_replicas": len(group),
                }
                for column in (
                    "diffusion_vacf_corrected_nm2_ns",
                    "diffusion_vacf_pbc_nm2_ns",
                    "viscosity_mPa_s",
                    "diffusion_msd_pbc_nm2_ns",
                ):
                    delta = group[f"{column}_right"] - group[f"{column}_left"]
                    mean, sem, valid = mean_sem(delta)
                    row[column + "_shift"] = mean
                    row[column + "_shift_sem"] = sem
                    row[column + "_shift_n"] = valid
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["contrast", "backbone", "temperature_K"])


def compare_states(states: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    keys = ["model_label", "temperature_K"]
    metrics = (
        "diffusion_vacf_corrected_nm2_ns",
        "diffusion_msd_pbc_nm2_ns",
        "viscosity_mPa_s",
    )
    current = states[keys + list(metrics)].copy()
    baseline = reference[keys + list(metrics)].copy()
    merged = current.merge(baseline, on=keys, suffixes=("_current", "_reference"), validate="one_to_one")
    if len(merged) != len(current):
        raise ValueError(
            f"comparison table covers {len(merged)}/{len(current)} selected model-temperature states"
        )
    for metric in metrics:
        current_column = f"{metric}_current"
        reference_column = f"{metric}_reference"
        merged[f"{metric}_delta"] = merged[current_column] - merged[reference_column]
        merged[f"{metric}_percent_delta"] = 100.0 * (
            merged[current_column] / merged[reference_column] - 1.0
        )
    merged["model_label"] = pd.Categorical(merged["model_label"], MODEL_ORDER, ordered=True)
    return merged.sort_values(["model_label", "temperature_K"]).reset_index(drop=True)


def load_curve(row: pd.Series, output_dir: Path) -> dict[str, np.ndarray]:
    with np.load(output_dir / row["curve_file"], allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def grouped_curve(
    group: pd.DataFrame, output_dir: Path, x_name: str, y_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = [load_curve(row, output_dir) for _, row in group.iterrows()]
    x = np.unique(np.concatenate([curve[x_name] for curve in curves]))
    stack = np.full((len(curves), len(x)), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        curve_x = curve[x_name]
        positions = np.searchsorted(x, curve_x)
        if not np.array_equal(x[positions], curve_x):
            raise ValueError(f"incompatible {x_name} grids")
        stack[index, positions] = curve[y_name]

    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    mean = np.divide(
        np.nansum(stack, axis=0),
        count,
        out=np.full(len(x), np.nan),
        where=count > 0,
    )
    squared_deviation = np.nansum((stack - mean) ** 2, axis=0)
    sem = np.divide(
        np.sqrt(squared_deviation / np.maximum(count - 1, 1)),
        np.sqrt(count),
        out=np.full(len(x), np.nan),
        where=count > 1,
    )
    return x, mean, sem


def grouped_curve_with_scalar_offset(
    group: pd.DataFrame,
    output_dir: Path,
    x_name: str,
    y_name: str,
    offset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = [load_curve(row, output_dir) for _, row in group.iterrows()]
    x = np.unique(np.concatenate([curve[x_name] for curve in curves]))
    stack = np.full((len(curves), len(x)), np.nan, dtype=np.float64)
    for index, ((_, row), curve) in enumerate(zip(group.iterrows(), curves)):
        curve_x = curve[x_name]
        positions = np.searchsorted(x, curve_x)
        if not np.array_equal(x[positions], curve_x):
            raise ValueError(f"incompatible {x_name} grids")
        stack[index, positions] = curve[y_name] + float(row[offset_name])

    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    mean = np.divide(
        np.nansum(stack, axis=0),
        count,
        out=np.full(len(x), np.nan),
        where=count > 0,
    )
    squared_deviation = np.nansum((stack - mean) ** 2, axis=0)
    sem = np.divide(
        np.sqrt(squared_deviation / np.maximum(count - 1, 1)),
        np.sqrt(count),
        out=np.full(len(x), np.nan),
        where=count > 1,
    )
    return x, mean, sem


def present_models(frame: pd.DataFrame) -> tuple[str, ...]:
    present = {str(value) for value in frame["model_label"].dropna().tolist()}
    return tuple(model for model in MODEL_ORDER if model in present)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_coverage(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    summary = lanes.groupby("model_label", observed=False)["analyzed_ps"].agg(["min", "mean", "max"])
    summary = summary.reindex(models)
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.errorbar(
        summary["mean"], y,
        xerr=np.vstack((summary["mean"] - summary["min"], summary["max"] - summary["mean"])),
        fmt="o", color="#3b6ea8", capsize=4,
    )
    ax.axvline(600.0, color="black", ls="--", lw=1, label="paper-matched 600 ps")
    ax.set(yticks=y, yticklabels=summary.index, xlabel="Safely visible NVT prefix per lane (ps)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("NVT coverage frozen at analysis time (range over 12 lanes/model)")
    save_figure(fig, output_dir, "coverage_by_model")


def plot_transport(states: pd.DataFrame, output_dir: Path, quantity: str) -> None:
    is_diffusion = quantity == "diffusion"
    column = "diffusion_vacf_corrected_nm2_ns" if is_diffusion else "viscosity_mPa_s"
    sem_column = column + "_sem"
    ylabel = r"$D$ (nm$^2$ ns$^{-1}$)" if is_diffusion else r"$\eta$ (mPa s)"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    temperature_grid = np.linspace(273.15, 360.0, 300)
    reference = (
        np.asarray([experimental_diffusion(t) for t in temperature_grid])
        if is_diffusion
        else np.asarray([experimental_viscosity(t) for t in temperature_grid])
    )
    for ax, backbone in zip(axes, ("OFF", "LES")):
        ax.plot(temperature_grid, reference, color="black", lw=2, label="experiment")
        for condition in CONDITION_ORDER:
            group = states[(states["backbone"] == backbone) & (states["condition"] == condition)]
            if group.empty:
                continue
            ax.errorbar(
                group["temperature_K"], group[column], yerr=group[sem_column], marker="o",
                color=CONDITION_COLORS[condition], capsize=3, label=condition,
            )
        if is_diffusion:
            ax.scatter(
                [item["temperature_K"] for item in MONTERO_RPBE if math.isfinite(item["viscosity_mPa_s"])],
                [item["diffusion_nm2_ns"] for item in MONTERO_RPBE if math.isfinite(item["viscosity_mPa_s"])],
                marker="*", s=100, color="#7a5195", label="Montero RPBE NVT",
            )
            ax.scatter(
                [item["temperature_K"] for item in HILPERT_MACE_L2],
                [item["diffusion_nm2_ns"] for item in HILPERT_MACE_L2],
                marker="D", s=38, facecolors="none", edgecolors="#8c6d31", label="Hilpert MACE L2 (digitized)",
            )
        else:
            ax.scatter(
                [item["temperature_K"] for item in MONTERO_RPBE if math.isfinite(item["viscosity_mPa_s"])],
                [item["viscosity_mPa_s"] for item in MONTERO_RPBE if math.isfinite(item["viscosity_mPa_s"])],
                marker="*", s=100, color="#7a5195", label="Montero RPBE NVT",
            )
        ax.set(title=f"{backbone} backbone", xlabel="Temperature (K)")
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Finite-size-corrected VACF diffusion" if is_diffusion else "Green–Kubo shear viscosity")
    save_figure(fig, output_dir, f"{quantity}_versus_temperature")


def plot_diffusion_estimator_comparison(states: pd.DataFrame, output_dir: Path) -> None:
    """Show the identical lane-wise finite-size shift applied to VACF and MSD."""
    models = present_models(states)
    short_labels = {
        "OFF-ZS": "OFF-ZS",
        "OFF-T12-B123200": "OFF-T12",
        "OFF-D00-B123200": "OFF-D00",
        "OFF-D11-B123200": "OFF-D11",
        "LES-ZS": "LES-ZS",
        "LES-T12-B123200": "LES-T12",
        "LES-D00-B123200": "LES-D00",
        "LES-D11-B123200": "LES-D11",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    for ax, temperature in zip(axes.ravel(), TEMPERATURES):
        selected = states[states["temperature_K"] == temperature].set_index("model_label")
        for index, model in enumerate(models):
            row = selected.loc[model]
            x_vacf = index - 0.12
            x_msd = index + 0.12
            vacf_raw = float(row["diffusion_vacf_pbc_nm2_ns"])
            vacf_corrected = float(row["diffusion_vacf_corrected_nm2_ns"])
            msd_raw = float(row["diffusion_msd_pbc_nm2_ns"])
            msd_corrected = float(row["diffusion_msd_corrected_nm2_ns"])
            ax.plot([x_vacf, x_vacf], [vacf_raw, vacf_corrected], color="#4c78a8", lw=1.2)
            ax.plot([x_msd, x_msd], [msd_raw, msd_corrected], color="#f58518", lw=1.2)
            ax.errorbar(
                x_vacf, vacf_raw,
                yerr=row["diffusion_vacf_pbc_nm2_ns_sem"],
                marker="o", mfc="white", mec="#4c78a8", color="#4c78a8", capsize=2,
            )
            ax.errorbar(
                x_vacf, vacf_corrected,
                yerr=row["diffusion_vacf_corrected_nm2_ns_sem"],
                marker="o", mfc="#4c78a8", mec="#4c78a8", color="#4c78a8", capsize=2,
            )
            ax.errorbar(
                x_msd, msd_raw,
                yerr=row["diffusion_msd_pbc_nm2_ns_sem"],
                marker="s", mfc="white", mec="#f58518", color="#f58518", capsize=2,
            )
            ax.errorbar(
                x_msd, msd_corrected,
                yerr=row["diffusion_msd_corrected_nm2_ns_sem"],
                marker="s", mfc="#f58518", mec="#f58518", color="#f58518", capsize=2,
            )
        ax.axhline(experimental_diffusion(temperature), color="black", ls="--", lw=1.3)
        ax.set(title=f"{temperature} K", ylabel=r"$D$ (nm$^2$ ns$^{-1}$)")
        ax.set_xticks(range(len(models)), [short_labels[model] for model in models], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].scatter([], [], marker="o", color="#4c78a8", label="VACF")
    axes[0, 0].scatter([], [], marker="s", color="#f58518", label="MSD")
    axes[0, 0].scatter([], [], marker="o", facecolors="white", edgecolors="0.3", label=r"raw $D_{PBC}$")
    axes[0, 0].scatter([], [], marker="o", color="0.3", label=r"corrected $D_\infty$")
    axes[0, 0].plot([], [], color="black", ls="--", label="experiment")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Finite-size correction of VACF and NVT-MSD diffusion", y=1.05)
    save_figure(fig, output_dir, "vacf_msd_finite_size_comparison")


def plot_vacf(lanes: pd.DataFrame, output_dir: Path) -> None:
    selected = lanes[lanes["temperature_K"] == 295]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for row, backbone in enumerate(("OFF", "LES")):
        for condition in CONDITION_ORDER:
            group = selected[(selected["backbone"] == backbone) & (selected["condition"] == condition)]
            if group.empty:
                continue
            time, vacf, vacf_sem = grouped_curve(group, output_dir, "vacf_time_ps", "vacf_normalized")
            axes[row, 0].plot(time, vacf, color=CONDITION_COLORS[condition], label=condition)
            axes[row, 0].fill_between(time, vacf - vacf_sem, vacf + vacf_sem, color=CONDITION_COLORS[condition], alpha=0.15)
            time, running, running_sem = grouped_curve(
                group, output_dir, "vacf_time_ps", "running_diffusion_pbc_nm2_ns"
            )
            correction = float(group["yeh_hummer_nm2_ns"].mean())
            axes[row, 1].plot(time, running + correction, color=CONDITION_COLORS[condition], label=condition)
            axes[row, 1].fill_between(
                time, running + correction - running_sem, running + correction + running_sem,
                color=CONDITION_COLORS[condition], alpha=0.15,
            )
        axes[row, 0].set(xlim=(0, 2.0), ylabel=f"{backbone}\n$C_{{VV}}(t)/C_{{VV}}(0)$")
        axes[row, 1].axhline(experimental_diffusion(295), color="black", ls="--", lw=1)
        axes[row, 1].set(xlim=(0, 10.0), ylabel=f"{backbone}\ncorrected running $D$")
        for ax in axes[row]:
            ax.axhline(0.0, color="0.5", lw=0.6)
            ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Correlation time (ps)")
    axes[1, 1].set_xlabel("VACF integration limit (ps)")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("295 K oxygen VACF and its Green–Kubo diffusion integral")
    save_figure(fig, output_dir, "vacf_and_running_diffusion_295K")


def plot_vacf_and_corrected_running_diffusion_by_temperature(
    lanes: pd.DataFrame,
    output_dir: Path,
) -> None:
    models = present_models(lanes)
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2), sharey="row")
    for column, temperature in enumerate(TEMPERATURES):
        selected = lanes[lanes["temperature_K"] == temperature]
        for model in models:
            group = selected[selected["model_label"] == model]
            condition = str(group.iloc[0]["condition"])
            linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
            time, vacf, vacf_sem = grouped_curve(
                group,
                output_dir,
                "vacf_time_ps",
                "vacf_normalized",
            )
            vacf_mask = time <= 2.0
            axes[0, column].plot(
                time[vacf_mask],
                vacf[vacf_mask],
                color=CONDITION_COLORS[condition],
                ls=linestyle,
                lw=1.2,
                label=model,
            )
            axes[0, column].fill_between(
                time[vacf_mask],
                vacf[vacf_mask] - vacf_sem[vacf_mask],
                vacf[vacf_mask] + vacf_sem[vacf_mask],
                color=CONDITION_COLORS[condition],
                alpha=0.10,
            )
            time, running, running_sem = grouped_curve_with_scalar_offset(
                group,
                output_dir,
                "vacf_time_ps",
                "running_diffusion_pbc_nm2_ns",
                "yeh_hummer_nm2_ns",
            )
            running_mask = time <= 10.0
            axes[1, column].plot(
                time[running_mask],
                running[running_mask],
                color=CONDITION_COLORS[condition],
                ls=linestyle,
                lw=1.2,
                label=model,
            )
            axes[1, column].fill_between(
                time[running_mask],
                running[running_mask] - running_sem[running_mask],
                running[running_mask] + running_sem[running_mask],
                color=CONDITION_COLORS[condition],
                alpha=0.10,
            )
        axes[0, column].axhline(0.0, color="0.5", lw=0.6)
        axes[0, column].set(title=f"{temperature} K", xlim=(0.0, 2.0), ylim=(-0.25, 1.05))
        axes[1, column].axhline(experimental_diffusion(temperature), color="black", lw=1.3)
        axes[1, column].axvspan(*VACF_PLATEAU_PS, color="0.7", alpha=0.16)
        axes[1, column].set(xlim=(0.0, 10.0), ylim=(0.0, 8.5))
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[0, column].set_xlabel("Velocity-correlation lag (ps)")
        axes[1, column].set_xlabel("VACF integration limit (ps)")
    axes[0, 0].set_ylabel(r"$C_{VV}(t)/C_{VV}(0)$")
    axes[1, 0].set_ylabel(r"Corrected running $D$ (nm$^2$/ns)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Oxygen velocity VACF and finite-size-corrected Green–Kubo diffusion", y=1.08)
    fig.subplots_adjust(top=0.84, wspace=0.12, hspace=0.32)
    save_figure(fig, output_dir, "vacf_and_corrected_running_diffusion_by_temperature")


def plot_stress_cross(lanes: pd.DataFrame, output_dir: Path) -> None:
    selected = lanes[lanes["temperature_K"] == 295]
    models = present_models(lanes)
    columns = min(3, len(models))
    rows = int(math.ceil(len(models) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 3.6 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes.ravel(), models):
        group = selected[selected["model_label"] == model]
        curves = [load_curve(row, output_dir) for _, row in group.iterrows()]
        length = min(len(curve["stress_time_ps"]) for curve in curves)
        time = curves[0]["stress_time_ps"][:length]
        cross = np.mean([curve["stress_cross_normalized"][:length] for curve in curves], axis=0)
        for i in range(5):
            for j in range(i + 1, 5):
                special = (i, j) == (3, 4)
                ax.plot(
                    time, cross[:, i, j],
                    color="#7a5195" if special else "#4c78a8",
                    lw=1.8 if special else 0.75,
                    alpha=1.0 if special else 0.55,
                )
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set(xlim=(0, 2.0), ylim=(-0.65, 0.3), title=model)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(models):]:
        ax.remove()
    for ax in axes[-1]:
        ax.set_xlabel("Correlation time (ps)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$C_{ij}(t)/[C_{ii}(0)C_{jj}(0)]^{1/2}$")
    fig.text(
        0.5, 0.01,
        "Purple: expected cross-correlation of the two non-orthogonal diagonal combinations; blue: other nine pairs",
        ha="center", fontsize=9,
    )
    fig.suptitle("295 K pairwise cross-correlations of the five stored traceless stress components")
    fig.subplots_adjust(bottom=0.10, top=0.90)
    save_figure(fig, output_dir, "stress_cross_correlations_295K")


def plot_running_viscosity(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for ax, temperature in zip(axes.ravel(), TEMPERATURES):
        selected = lanes[lanes["temperature_K"] == temperature]
        for model in models:
            group = selected[selected["model_label"] == model]
            time, eta, _ = grouped_curve(group, output_dir, "stress_time_ps", "running_viscosity_mPa_s")
            condition = str(group.iloc[0]["condition"])
            linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
            ax.plot(time, eta, color=CONDITION_COLORS[condition], ls=linestyle, lw=1.2, label=model)
        ax.axhline(experimental_viscosity(temperature), color="black", lw=1.3, label="IAPWS")
        ax.axvspan(*VISCOSITY_PLATEAU_PS, color="0.7", alpha=0.18)
        ax.set(title=f"{temperature} K", xlim=(0, 20))
        ax.set_yscale("symlog", linthresh=0.25)
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Stress-correlation integration limit (ps)")
    axes[1, 1].set_xlabel("Stress-correlation integration limit (ps)")
    axes[0, 0].set_ylabel(r"Running $\eta$ (mPa s)")
    axes[1, 0].set_ylabel(r"Running $\eta$ (mPa s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Green–Kubo viscosity convergence (OFF solid; LES dashed)", y=1.08)
    save_figure(fig, output_dir, "running_viscosity_by_temperature")


def plot_viscosity_prefix_convergence(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    max_duration = float(lanes["analyzed_ps"].max())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, sharey=True)
    for ax, temperature in zip(axes.ravel(), TEMPERATURES):
        selected = lanes[lanes["temperature_K"] == temperature]
        for model in models:
            group = selected[selected["model_label"] == model]
            condition = str(group.iloc[0]["condition"])
            linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
            for _, lane in group.iterrows():
                curve = load_curve(lane, output_dir)
                ax.plot(
                    curve["prefix_time_ps"],
                    curve["prefix_viscosity_mPa_s"],
                    color=CONDITION_COLORS[condition],
                    ls=linestyle,
                    lw=0.65,
                    alpha=0.22,
                )
            time, eta, eta_sem = grouped_curve(
                group,
                output_dir,
                "prefix_time_ps",
                "prefix_viscosity_mPa_s",
            )
            common_duration = float(group["analyzed_ps"].min())
            mask = time <= common_duration
            ax.plot(
                time[mask],
                eta[mask],
                color=CONDITION_COLORS[condition],
                ls=linestyle,
                lw=1.35,
                marker="o",
                ms=2.8,
                label=model,
            )
            ax.fill_between(
                time[mask],
                eta[mask] - eta_sem[mask],
                eta[mask] + eta_sem[mask],
                color=CONDITION_COLORS[condition],
                alpha=0.10,
            )
        ax.axhline(experimental_viscosity(temperature), color="black", lw=1.3, label="IAPWS")
        ax.set(
            title=f"{temperature} K",
            xlim=(0.0, max_duration),
            ylim=(0.25, 5.0),
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Accumulated NVT trajectory (ps)")
    axes[1, 1].set_xlabel("Accumulated NVT trajectory (ps)")
    axes[0, 0].set_ylabel(r"Cumulative $\eta$ (mPa s)")
    axes[1, 0].set_ylabel(r"Cumulative $\eta$ (mPa s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Full-duration cumulative viscosity convergence (OFF solid; LES dashed)", y=1.08)
    save_figure(fig, output_dir, "viscosity_prefix_convergence_by_temperature")


def plot_diffusion_prefix_convergence(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    max_duration = float(lanes["analyzed_ps"].max())
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, sharey=True)
    for ax, temperature in zip(axes.ravel(), TEMPERATURES):
        selected = lanes[lanes["temperature_K"] == temperature]
        for model in models:
            group = selected[selected["model_label"] == model]
            condition = str(group.iloc[0]["condition"])
            linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
            for _, lane in group.iterrows():
                curve = load_curve(lane, output_dir)
                ax.plot(
                    curve["prefix_time_ps"],
                    curve["prefix_diffusion_corrected_nm2_ns"],
                    color=CONDITION_COLORS[condition],
                    ls=linestyle,
                    lw=0.65,
                    alpha=0.22,
                )
            time, diffusion, diffusion_sem = grouped_curve(
                group,
                output_dir,
                "prefix_time_ps",
                "prefix_diffusion_corrected_nm2_ns",
            )
            common_duration = float(group["analyzed_ps"].min())
            mask = time <= common_duration
            ax.plot(
                time[mask],
                diffusion[mask],
                color=CONDITION_COLORS[condition],
                ls=linestyle,
                lw=1.35,
                marker="o",
                ms=2.8,
                label=model,
            )
            ax.fill_between(
                time[mask],
                diffusion[mask] - diffusion_sem[mask],
                diffusion[mask] + diffusion_sem[mask],
                color=CONDITION_COLORS[condition],
                alpha=0.10,
            )
        ax.axhline(experimental_diffusion(temperature), color="black", lw=1.3, label="experiment")
        ax.set(
            title=f"{temperature} K",
            xlim=(0.0, max_duration),
            ylim=(0.5, 10.0),
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Accumulated NVT trajectory (ps)")
    axes[1, 1].set_xlabel("Accumulated NVT trajectory (ps)")
    axes[0, 0].set_ylabel(r"Cumulative corrected $D$ (nm$^2$/ns)")
    axes[1, 0].set_ylabel(r"Cumulative corrected $D$ (nm$^2$/ns)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Full-duration corrected VACF-diffusion convergence (OFF solid; LES dashed)", y=1.08)
    save_figure(fig, output_dir, "diffusion_prefix_convergence_by_temperature")


def plot_msd(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for ax, temperature in zip(axes.ravel(), TEMPERATURES):
        selected = lanes[lanes["temperature_K"] == temperature]
        for model in models:
            group = selected[selected["model_label"] == model]
            condition = str(group.iloc[0]["condition"])
            linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
            for _, lane in group.iterrows():
                curve = load_curve(lane, output_dir)
                lane_time = curve["msd_time_ps"]
                lane_msd = curve["msd_A2"]
                lane_mask = lane_time <= MSD_PLOT_MAX_PS
                ax.plot(
                    lane_time[lane_mask],
                    lane_msd[lane_mask],
                    color=CONDITION_COLORS[condition],
                    ls=linestyle,
                    lw=0.65,
                    alpha=0.22,
                )
            time, msd, _ = grouped_curve(group, output_dir, "msd_time_ps", "msd_A2")
            common_duration = float(group["analyzed_ps"].min())
            mask = time <= min(MSD_PLOT_MAX_PS, common_duration)
            ax.plot(time[mask], msd[mask], color=CONDITION_COLORS[condition], ls=linestyle, lw=1.2, label=model)
        reference_time = np.linspace(0.0, MSD_PLOT_MAX_PS, 1001)
        # D [nm2/ns] -> slope 6D/10 in A2/ps.
        ax.plot(
            reference_time,
            0.6 * experimental_diffusion(temperature) * reference_time,
            color="black", lw=1.4, label="experiment: 6Dt",
        )
        ax.axvspan(*MSD_FIT_PS, color="0.7", alpha=0.16)
        ax.set(title=f"{temperature} K", xlim=(0.0, MSD_PLOT_MAX_PS), ylim=(0.0, None))
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Lag time (ps)")
    axes[1, 1].set_xlabel("Lag time (ps)")
    axes[0, 0].set_ylabel(r"Oxygen MSD ($\AA^2$)")
    axes[1, 0].set_ylabel(r"Oxygen MSD ($\AA^2$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Full available NVT oxygen MSD (10–600 ps fit)", y=1.08)
    save_figure(fig, output_dir, "oxygen_msd_by_temperature")


def plot_prefix_convergence(lanes: pd.DataFrame, output_dir: Path) -> None:
    models = present_models(lanes)
    selected = lanes[lanes["temperature_K"] == 295]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for model in models:
        group = selected[selected["model_label"] == model]
        condition = str(group.iloc[0]["condition"])
        linestyle = "-" if str(group.iloc[0]["backbone"]) == "OFF" else "--"
        for ax, y_name in zip(
            axes, ("prefix_diffusion_corrected_nm2_ns", "prefix_viscosity_mPa_s")
        ):
            time, value, sem = grouped_curve(group, output_dir, "prefix_time_ps", y_name)
            ax.plot(time, value, color=CONDITION_COLORS[condition], ls=linestyle, marker="o", label=model)
            ax.fill_between(time, value - sem, value + sem, color=CONDITION_COLORS[condition], alpha=0.10)
    axes[0].axhline(experimental_diffusion(295), color="black", lw=1.3)
    axes[1].axhline(experimental_viscosity(295), color="black", lw=1.3)
    axes[0].set(ylabel=r"Corrected $D$ (nm$^2$/ns)", xlabel="Trajectory prefix (ps)")
    axes[1].set(ylabel=r"$\eta$ (mPa s)", xlabel="Trajectory prefix (ps)")
    axes[1].set_yscale("symlog", linthresh=0.25)
    for ax in axes:
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("295 K cumulative 100 ps-block convergence (OFF solid; LES dashed)", y=1.12)
    save_figure(fig, output_dir, "prefix_convergence_295K")


def plot_shifts(shifts: pd.DataFrame, output_dir: Path) -> None:
    contrasts = tuple(
        contrast for contrast in ("D11-D00", "T12-ZS")
        if contrast in set(shifts["contrast"].tolist())
    )
    fig, axes = plt.subplots(
        2,
        len(contrasts),
        figsize=(6.0 * len(contrasts), 8.0),
        sharex=True,
        squeeze=False,
    )
    for column_index, contrast in enumerate(contrasts):
        subset = shifts[shifts["contrast"] == contrast]
        for backbone, marker in (("OFF", "o"), ("LES", "s")):
            group = subset[subset["backbone"] == backbone]
            axes[0, column_index].errorbar(
                group["temperature_K"], group["diffusion_vacf_corrected_nm2_ns_shift"],
                yerr=group["diffusion_vacf_corrected_nm2_ns_shift_sem"], marker=marker,
                capsize=3, label=backbone,
            )
            axes[1, column_index].errorbar(
                group["temperature_K"], group["viscosity_mPa_s_shift"],
                yerr=group["viscosity_mPa_s_shift_sem"], marker=marker,
                capsize=3, label=backbone,
            )
        axes[0, column_index].set_title(contrast)
        axes[1, column_index].set_xlabel("Temperature (K)")
        for ax in axes[:, column_index]:
            ax.axhline(0.0, color="black", lw=0.8)
            ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"Shift in corrected $D$ (nm$^2$/ns)")
    axes[1, 0].set_ylabel(r"Shift in $\eta$ (mPa s)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Paired-seed adaptation shifts (right condition minus left condition)")
    save_figure(fig, output_dir, "paired_model_shifts")


def plot_state_changes(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharex=True)
    panels = (
        ("diffusion_vacf_corrected_nm2_ns_delta", r"$\Delta D_{corr}$ (nm$^2$/ns)"),
        ("diffusion_msd_pbc_nm2_ns_delta", r"$\Delta D_{MSD}$ (nm$^2$/ns)"),
        ("viscosity_mPa_s_delta", r"$\Delta \eta$ (mPa s)"),
    )
    for model in present_models(comparison):
        group = comparison[comparison["model_label"] == model]
        condition = "ZS" if model.endswith("ZS") else ("T12" if "T12" in model else "D00")
        linestyle = "-" if model.startswith("OFF") else "--"
        for ax, (column, ylabel) in zip(axes, panels):
            ax.plot(
                group["temperature_K"],
                group[column],
                color=CONDITION_COLORS[condition],
                linestyle=linestyle,
                marker="o",
                label=model,
            )
            ax.set(ylabel=ylabel, xlabel="Temperature (K)")
    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.8)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Change from the common 600 ps analysis to lane-specific committed prefixes", y=1.15)
    save_figure(fig, output_dir, "change_from_600ps")


def markdown_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def markdown_mean_sem(mean: Any, sem: Any, digits: int = 3) -> str:
    if not math.isfinite(float(mean)):
        return "—"
    if math.isfinite(float(sem)):
        return f"{float(mean):.{digits}f} ± {float(sem):.{digits}f}"
    return f"{float(mean):.{digits}f}"


def dataframe_markdown(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def model_scores(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in states.groupby("model_label", observed=True):
        d_ratio = group["diffusion_vacf_corrected_nm2_ns"] / group["experimental_diffusion_nm2_ns"]
        e_ratio = group["viscosity_mPa_s"] / group["experimental_viscosity_mPa_s"]
        d_valid = d_ratio[(d_ratio > 0) & np.isfinite(d_ratio)]
        e_valid = e_ratio[(e_ratio > 0) & np.isfinite(e_ratio)]
        d_score = float(np.sqrt(np.mean(np.log(d_valid) ** 2))) if len(d_valid) else math.nan
        e_score = float(np.sqrt(np.mean(np.log(e_valid) ** 2))) if len(e_valid) else math.nan
        combined = math.sqrt((d_score**2 + e_score**2) / 2.0) if math.isfinite(d_score) and math.isfinite(e_score) else math.nan
        rows.append({"model_label": str(model), "diffusion_log_rmse": d_score, "viscosity_log_rmse": e_score, "combined_log_rmse": combined})
    return pd.DataFrame(rows).sort_values("combined_log_rmse")


def write_report(
    output_dir: Path,
    lanes: pd.DataFrame,
    states: pd.DataFrame,
    shifts: pd.DataFrame,
    started_utc: str,
    finished_utc: str,
) -> None:
    coverage_rows = []
    for model in MODEL_ORDER:
        group = lanes[lanes["model_label"] == model]
        coverage_rows.append([
            model,
            f"{group['analyzed_ps'].min():.1f}–{group['analyzed_ps'].max():.1f}",
            f"{group['analyzed_ps'].median():.1f}",
            f"{int(group['complete_600ps'].sum())}/12",
        ])

    state_rows = []
    for _, row in states.iterrows():
        state_rows.append([
            str(row["model_label"]),
            int(row["temperature_K"]),
            f"{row['analyzed_min_ps']:.1f}–{row['analyzed_max_ps']:.1f}",
            markdown_mean_sem(row["diffusion_vacf_pbc_nm2_ns"], row["diffusion_vacf_pbc_nm2_ns_sem"]),
            markdown_mean_sem(row["yeh_hummer_nm2_ns"], row["yeh_hummer_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_vacf_corrected_nm2_ns"], row["diffusion_vacf_corrected_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_msd_pbc_nm2_ns"], row["diffusion_msd_pbc_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_msd_corrected_nm2_ns"], row["diffusion_msd_corrected_nm2_ns_sem"]),
            markdown_number(row["experimental_diffusion_nm2_ns"]),
            markdown_mean_sem(row["viscosity_mPa_s"], row["viscosity_mPa_s_sem"]),
            markdown_number(row["experimental_viscosity_mPa_s"]),
        ])

    shift_rows = []
    for _, row in shifts.iterrows():
        shift_rows.append([
            row["contrast"], row["backbone"], int(row["temperature_K"]), int(row["paired_replicas"]),
            markdown_mean_sem(
                row["diffusion_vacf_corrected_nm2_ns_shift"],
                row["diffusion_vacf_corrected_nm2_ns_shift_sem"],
            ),
            markdown_mean_sem(row["viscosity_mPa_s_shift"], row["viscosity_mPa_s_shift_sem"]),
        ])

    scores = model_scores(states)
    score_rows = [
        [index + 1, row["model_label"], markdown_number(row["diffusion_log_rmse"]),
         markdown_number(row["viscosity_log_rmse"]), markdown_number(row["combined_log_rmse"])]
        for index, (_, row) in enumerate(scores.iterrows())
    ]
    best = str(scores.iloc[0]["model_label"]) if len(scores) else "undetermined"
    complete_count = int(lanes["complete_600ps"].sum())
    eta_unstable = int(np.count_nonzero(states["viscosity_plateau_range_fraction"] > 0.5))
    d_unstable = int(np.count_nonzero(states["diffusion_vacf_plateau_range_fraction"] > 0.1))
    vacf_msd_relative = np.abs(
        lanes["diffusion_vacf_pbc_nm2_ns"] - lanes["diffusion_msd_pbc_nm2_ns"]
    ) / lanes["diffusion_vacf_pbc_nm2_ns"]
    viscosity_window_relative = np.abs(
        lanes["viscosity_mPa_s"] - lanes["viscosity_10_20ps_mPa_s"]
    ) / lanes["viscosity_mPa_s"]
    t12_shifts = shifts[shifts["contrast"] == "T12-ZS"]
    d11_shifts = shifts[shifts["contrast"] == "D11-D00"]

    report = fr"""# Preliminary NVT dynamics of the 96-trajectory water campaign

_Frozen analysis: {finished_utc}. Source-prefix discovery began at {started_utc}._

## Executive reading

This is a real early-prefix calculation over all **96 lanes**, not a projection from NpT speed. Exactly **{complete_count}/96** lanes had a complete 600 ps NVT prefix when each worker froze its inputs. The other lanes use only flushed, explicitly recorded prefixes; their exact durations are in the coverage table and `lanes.csv`. Therefore the numerical ranking is **preliminary** and must be regenerated after every lane has at least 600 ps (and again at the campaign endpoint of 1 ns).

On a simple equal-weight log-error score against temperature-matched experimental diffusion and viscosity, the current best combined screen is **{best}**. This score is descriptive, not a model-selection verdict: the models do not all yet have equal trajectory length, viscosity is noisy, and the 64-water Yeh–Hummer correction uses the same noisy viscosity estimate.

The prespecified plateau gates flag **{d_unstable}/32** state means with more than 10% VACF-integral variation over 5–10 ps and **{eta_unstable}/32** with more than 50% viscosity-integral variation over 5–10 ps. That contrast is expected: each VACF frame contributes 64 oxygen velocities, while each stress frame contributes only five components.

## Main preliminary signals

- **{best}** has the lowest current combined relative-error score against experiment. This is a frozen-prefix screen, not a final model-selection result.
- T12−ZS raises corrected diffusion in **{int((t12_shifts['diffusion_vacf_corrected_nm2_ns_shift'] > 0).sum())}/{len(t12_shifts)}** backbone/temperature comparisons (range **{t12_shifts['diffusion_vacf_corrected_nm2_ns_shift'].min():.3f} to {t12_shifts['diffusion_vacf_corrected_nm2_ns_shift'].max():.3f} nm²/ns**) and lowers viscosity in **{int((t12_shifts['viscosity_mPa_s_shift'] < 0).sum())}/{len(t12_shifts)}** (range **{t12_shifts['viscosity_mPa_s_shift'].min():.3f} to {t12_shifts['viscosity_mPa_s_shift'].max():.3f} mPa s**).
- D11−D00 is more temperature dependent: it lowers corrected diffusion in **{int((d11_shifts['diffusion_vacf_corrected_nm2_ns_shift'] < 0).sum())}/{len(d11_shifts)}** comparisons, while the point estimates raise viscosity in **{int((d11_shifts['viscosity_mPa_s_shift'] > 0).sum())}/{len(d11_shifts)}**. The 275 K LES viscosity difference is much smaller than its replica SEM, so signs are descriptive rather than significance tests.

## Numerical sanity checks

- All **{len(lanes)}/96** lane estimates are finite and have positive diffusion and viscosity.
- VACF and NVT-MSD finite-box diffusion differ by a median **{100.0 * vacf_msd_relative.median():.1f}%** across lanes; the 95th percentile is **{100.0 * vacf_msd_relative.quantile(0.95):.1f}%**. Every 10–600 ps MSD fit has **R² ≥ {lanes['diffusion_msd_r_squared'].min():.5f}**.
- The 5–10 and 10–20 ps viscosity estimates differ by a median **{100.0 * viscosity_window_relative.median():.1f}%** and at worst **{100.0 * viscosity_window_relative.max():.1f}%**.
- The expected zero-lag correlation between the non-orthogonal diagonal stress combinations averages **{lanes['stress_diagonal_pair_rho_t0'].mean():.4f}**. The largest lane-level RMS among the other nine cross-component pairs over 0–2 ps is **{lanes['stress_unexpected_cross_rms_0_2ps'].max():.4f}**.

![Coverage by model](coverage_by_model.png)

## What the two papers did, and what is matched here

| Item | Montero de Hijes et al. | Hilpert & Kresse | This early screen |
| --- | --- | --- | --- |
| Main diffusion estimator | Oxygen VACF in 600 ps NVT; separate 5–9 ns NVE oxygen MSD | Oxygen MSD after NpT → NVT equilibration → selected-energy NVE start | Oxygen VACF from the visible NVT prefix; NVT oxygen MSD is a diagnostic |
| Time step / H mass | 0.5 fs / physical H for MACE comparison | 0.5 fs / 1 amu for diffusion | 0.5 fs / 1.008 amu |
| System size | 512 waters for MACE; 128 for broad BPNNP validation | 64/63-water training; diffusion figure from RPBE-D3 MACE protocol | 64 waters |
| Viscosity | Five traceless stress autocorrelations, 600 ps NVT | Not reported | Same five-component Green–Kubo estimator |
| Finite-size handling | Yeh–Hummer using viscosity from the same NVT run | Figure reports NVE MSD values; paper does not tabulate a viscosity correction | Raw `D_PBC` and Yeh–Hummer-corrected `D` both reported |
| Rotational dynamics | OH second-rank correlation and integral | Not part of the reported dynamics section | **Unavailable:** H-coordinate time series were not stored |

The campaign matches the Montero high-frequency observable set unusually well: the stored stress columns are exactly `Pxy`, `Pxz`, `Pyz`, `(Pxx-Pyy)/2`, and `(Pyy-Pzz)/2`. Velocities and stresses were recorded every 0.5 fs; this analysis uses a prespecified {CORRELATION_DT_PS * 1000.0:g} fs stride, which still resolves the short VACF/stress decay while making 96-lane FFT analysis tractable. A future final analysis should repeat a 0.5/1/2/5 fs stride sensitivity on representative lanes.

The main protocol mismatch is ensemble for the MSD. Hilpert and the long-time branch of Montero use NVE trajectories, while the MSD below comes from thermostatted NVT and is fit over 10–600 ps. It is a convergence/cross-check plot, not a paper-equivalent long-time diffusion endpoint. The full 0–1000 ps axis is shown, but each curve ends at its actual available duration; no curve is extrapolated. At the longest lags, fewer time origins contribute, especially for the shorter prefixes.

## Prefix coverage

{dataframe_markdown(['Model', 'lane range (ps)', 'median (ps)', 'complete 600 ps'], coverage_rows)}

The range is over the 12 temperature/replica lanes for each model. Partial segments are accepted only through the final readable `thermo.csv` time; preallocated zero-filled array tails are never included.

## Diffusion

Primary values use

$$
D_{{\rm PBC}}=\frac13\int_0^\infty C_{{VV}}(t)\,dt,
\qquad
D_\infty=D_{{\rm PBC}}+\Delta D_{{\rm YH}},
\qquad
\Delta D_{{\rm YH}}=\frac{{k_B T\,2.837297}}{{6\pi\eta L}}.
$$

The reported VACF estimate is the mean of the running integral over 5–10 ps. The same lane-wise `ΔD_YH`, calculated from that lane's NVT temperature, box length, and viscosity, is added to both the VACF and 10–600 ps NVT-MSD estimates. This is appropriate because periodic hydrodynamic drag affects the long-time diffusion coefficient, independent of whether it is estimated from VACF or MSD. Error bars are SEM over the three MD replicas, not time origins. The experimental curve is the Holz–Heil–Sacco PFG-NMR/tracer fit at the exact campaign temperature.

![Diffusion versus temperature](diffusion_versus_temperature.png)

![Raw and finite-size-corrected VACF/MSD diffusion](vacf_msd_finite_size_comparison.png)

![VACF and running diffusion](vacf_and_running_diffusion_295K.png)

The all-temperature panel below shows the normalized oxygen velocity VACF in its top row and its finite-size-corrected Green–Kubo integral in the bottom row. The gray 5–10 ps integration window is the reporting window. Each lane's constant Yeh–Hummer shift is added to its running finite-box integral before replica averaging, so the early-lag corrected curve is a convergence diagnostic rather than an instantaneous diffusion coefficient.

![All-temperature velocity VACF and corrected running diffusion](vacf_and_corrected_running_diffusion_by_temperature.png)

The cumulative panel below uses successive 100 ps velocity blocks. At every prefix, finite-box VACF diffusion and viscosity are cumulatively averaged and that prefix's viscosity supplies its Yeh–Hummer correction. Its trajectory-duration axis starts at zero and the first defined estimate is at 100 ps. Faint lines are individual replicas; bold means stop at the shortest replica duration for each state.

![Full-duration cumulative corrected VACF diffusion](diffusion_prefix_convergence_by_temperature.png)

![NVT oxygen MSD](oxygen_msd_by_temperature.png)

## Viscosity and stress cross-correlations

The paper-matched estimator is

$$
C_\eta(t)=\frac15\sum_\alpha\langle p_\alpha(t')p_\alpha(t'+t)\rangle,
\qquad
\eta=\frac{{V}}{{k_B T}}\int_0^\infty C_\eta(t)\,dt.
$$

This implementation subtracts each component's finite-trajectory mean before correlation. The table uses the 5–10 ps mean of the running integral and records 10–20 ps as a sensitivity in `lanes.csv`/`states.csv`. The experimental line is IAPWS SR6-08 Eq. 7 at 0.1 MPa.

![Viscosity versus temperature](viscosity_versus_temperature.png)

![Running viscosity](running_viscosity_by_temperature.png)

The running-integral panel above resolves Green–Kubo convergence versus stress-correlation lag. The complementary full-duration panel below shows how the reported 5–10 ps plateau estimate changes as 100 ps trajectory blocks accumulate. Its axis begins at zero, but the first statistically defined estimate is at 100 ps. Faint lines are individual replicas through their maximum committed duration; bold means stop at the shortest replica duration for each state.

![Full-duration cumulative viscosity](viscosity_prefix_convergence_by_temperature.png)

The pairwise graph below is important for interpreting the stored basis. The last two diagonal combinations are linearly independent but not orthogonal. For an isotropic liquid their normalized zero-lag cross-correlation is expected to be about **−0.5**, so the purple curve is not evidence of a defect. The other nine pairs should fluctuate around zero. `lanes.csv` reports their RMS over 0–2 ps as `stress_unexpected_cross_rms_0_2ps`.

![Stress cross-correlations](stress_cross_correlations_295K.png)

## All state values

`D_VACF,PBC` and `D_MSD,PBC` are finite-periodic-box values. `ΔD_YH` is calculated separately for every lane before replica averaging and is applied identically to both estimators. `D_VACF,∞` remains the primary paper-matched result; `D_MSD,∞` is an NVT cross-check, not a replacement for a multi-nanosecond NVE MSD calculation. Each simulated entry is mean ± replica SEM.

{dataframe_markdown(['Model', 'T (K)', 'prefix (ps)', 'D_VACF,PBC', 'ΔD_YH', 'D_VACF,∞', 'D_MSD,PBC', 'D_MSD,∞', 'D_exp', 'η', 'η_exp'], state_rows)}

Units: diffusion in nm²/ns; viscosity in mPa s.

## Reported paper values

Montero et al. Table II reports RPBE-D3/zero-damping MACE NVT results of **D = 1.484 nm²/ns, η = 1.9 mPa s at 278 K** and **D = 2.768 nm²/ns, η = 0.8 mPa s at 297 K**. Those diffusion values are explicitly finite-size corrected. Their separate NVE/MSD values are 0.689 at 269.5 K, 2.065 at 296 K, and 3.661 nm²/ns at 315 K, without viscosities.

The paper did not leave the NVE/MSD branch raw because MSD is exempt from finite-size effects. Its 600 ps NVT branch supplied both the high-frequency oxygen VACF and a state-matched Green–Kubo viscosity, so the authors could apply Yeh–Hummer there. They did not compute viscosity for the separate 5–9 ns NVE branch and reported pressure offsets up to roughly ±350 bar; they therefore left those MSD values uncorrected and estimated that they were about 15% low. In this campaign, VACF, MSD, and viscosity all come from the same NVT lane, so reporting raw and corrected values for both estimators is internally consistent.

Hilpert & Kresse do not tabulate Fig. 3 numerically. Reproducible marker-centroid digitization of the official arXiv source image gives approximate MACE-L2 NVE/MSD values **2.47 at 300 K, 4.78 at 325 K, and 7.54 nm²/ns at 350 K**. These approximations are plotted as open diamonds and should not be treated as more precise than roughly two decimals.

Neither paper curve is a strict ground truth for these eight models: architectures, training data, equilibrium densities, system sizes, and (for the MSD) ensembles differ. The temperature-matched experimental curves are the cleanest absolute reference. The paper points are protocol/context checks.

## Paired model shifts

The same MD seed is paired across conditions. Positive values mean the right-hand condition in the contrast increases the property.

{dataframe_markdown(['Contrast', 'Backbone', 'T (K)', 'pairs', 'ΔD_corr', 'Δη'], shift_rows)}

![Paired shifts](paired_model_shifts.png)

![Prefix convergence](prefix_convergence_295K.png)

The “phase-diagram shift” discussed by Montero et al. is specifically a shift relative to each functional's melting temperature. This campaign has not yet measured melting points for the adapted models, so the paired differences above are **model/adaptation shifts at fixed absolute T**, not a demonstrated `T − T_m` collapse.

## Provisional experimental-error ranking

The score is `sqrt(mean(log(simulation/experiment)^2))` over the four temperatures; lower is better. It weights relative diffusion and viscosity errors equally and is shown only to summarize this early screen.

{dataframe_markdown(['Rank', 'Model', 'D log-RMSE', 'η log-RMSE', 'combined'], score_rows)}

## What can and cannot be concluded yet

- VACF diffusion is already much better conditioned than viscosity, but unequal prefixes and a 64-water cell still prevent a final model ranking.
- A stable-looking viscosity plateau can be accidental. Require agreement of the 5–10 and 10–20 ps windows, replica consistency, and cumulative-prefix stability before accepting a value.
- Yeh–Hummer couples the diffusion correction to the noisiest observable. Negative or non-finite provisional viscosities produce no correction and remain visible as missing values rather than being patched.
- The 64-water box is smaller than Montero's 512-water MACE box. A publication claim needs a size series or at least a ≥512-water check for shortlisted models.
- The requested 10–600 ps MSD fit includes progressively fewer displacement time origins at large lag. Its replica SEM and fit-window sensitivity, not R² alone, must be used to judge stability.
- The stored oxygen-only position series cannot reconstruct `C_OH(t)` or the second-rank rotational time. Future runs must store molecular orientation vectors or hydrogen coordinates at high frequency.
- Hilpert's NVE/MSD construction and Montero's 5–9 ns NVE branch still require a separate long NVE extension. The present NVT MSD is explicitly diagnostic.
- The final 1 ns report should rerun this exact frozen analysis, add stride sensitivity, enforce plateau gates, and retain all three paired replicas.

## Reproducibility and files

- Lane-level numbers and exact source paths: [`lanes.csv`](lanes.csv)
- Three-replica state summaries: [`states.csv`](states.csv)
- Paired contrasts: [`paired_shifts.csv`](paired_shifts.csv)
- Experimental and paper references: [`reference_values.csv`](reference_values.csv)
- Machine-readable metadata: [`analysis_summary.json`](analysis_summary.json)
- Per-lane curves used in every graph: [`lane_curves/`](lane_curves/)
- Analysis program: [`../analyze_preliminary_nvt_dynamics.py`](../analyze_preliminary_nvt_dynamics.py)

## Sources

1. P. Montero de Hijes, L. Neubeck, G. Kresse, and C. Dellago, [“Dynamical properties of ab initio water from machine-learning potentials,” arXiv:2606.18163 (2026)](https://arxiv.org/abs/2606.18163).
2. T. Hilpert and G. Kresse, [“Accurate Thermophysical Properties of Water using Machine-Learned Potentials,” arXiv:2601.21103 (2026)](https://arxiv.org/abs/2601.21103).
3. M. Holz, S. R. Heil, and A. Sacco, [“Temperature-dependent self-diffusion coefficients of water and six selected molecular liquids,” PCCP 2, 4740 (2000)](https://pubs.rsc.org/doi/b005319h). The fit used here is `D = 1.635×10⁻⁸[(T/215.05)−1]^2.063 m²/s`, valid from 273–373 K with a reported ≤1% fit error.
4. IAPWS, [Supplementary Release on Properties of Liquid Water at 0.1 MPa, SR6-08(2011)](https://www.iapws.org/relguide/LiquidWater.pdf), Eq. 7 for viscosity.
"""
    (output_dir / "PRELIMINARY_NVT_DYNAMICS.md").write_text(report, encoding="utf-8")


def write_variable_prefix_report(
    output_dir: Path,
    lanes: pd.DataFrame,
    states: pd.DataFrame,
    shifts: pd.DataFrame,
    comparison: pd.DataFrame | None,
    started_utc: str,
    finished_utc: str,
) -> None:
    models = present_models(lanes)
    coverage_rows = []
    for model in models:
        group = lanes[lanes["model_label"] == model]
        coverage_rows.append([
            model,
            f"{group['analyzed_ps'].min():.1f}–{group['analyzed_ps'].max():.1f}",
            f"{group['analyzed_ps'].median():.1f}",
            f"{int(group['complete_600ps'].sum())}/{len(group)}",
        ])

    state_rows = []
    for _, row in states.iterrows():
        state_rows.append([
            str(row["model_label"]),
            int(row["temperature_K"]),
            f"{row['analyzed_min_ps']:.1f}–{row['analyzed_max_ps']:.1f}",
            markdown_mean_sem(row["diffusion_vacf_pbc_nm2_ns"], row["diffusion_vacf_pbc_nm2_ns_sem"]),
            markdown_mean_sem(row["yeh_hummer_nm2_ns"], row["yeh_hummer_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_vacf_corrected_nm2_ns"], row["diffusion_vacf_corrected_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_msd_pbc_nm2_ns"], row["diffusion_msd_pbc_nm2_ns_sem"]),
            markdown_mean_sem(row["diffusion_msd_corrected_nm2_ns"], row["diffusion_msd_corrected_nm2_ns_sem"]),
            markdown_number(row["experimental_diffusion_nm2_ns"]),
            markdown_mean_sem(row["viscosity_mPa_s"], row["viscosity_mPa_s_sem"]),
            markdown_number(row["experimental_viscosity_mPa_s"]),
        ])

    shift_rows = []
    for _, row in shifts.iterrows():
        shift_rows.append([
            row["contrast"], row["backbone"], int(row["temperature_K"]), int(row["paired_replicas"]),
            markdown_mean_sem(
                row["diffusion_vacf_corrected_nm2_ns_shift"],
                row["diffusion_vacf_corrected_nm2_ns_shift_sem"],
            ),
            markdown_mean_sem(row["viscosity_mPa_s_shift"], row["viscosity_mPa_s_shift_sem"]),
        ])

    scores = model_scores(states)
    score_rows = [
        [index + 1, row["model_label"], markdown_number(row["diffusion_log_rmse"]),
         markdown_number(row["viscosity_log_rmse"]), markdown_number(row["combined_log_rmse"])]
        for index, (_, row) in enumerate(scores.iterrows())
    ]
    best = str(scores.iloc[0]["model_label"]) if len(scores) else "undetermined"
    eta_unstable = int(np.count_nonzero(states["viscosity_plateau_range_fraction"] > 0.5))
    d_unstable = int(np.count_nonzero(states["diffusion_vacf_plateau_range_fraction"] > 0.1))
    vacf_msd_relative = np.abs(
        lanes["diffusion_vacf_pbc_nm2_ns"] - lanes["diffusion_msd_pbc_nm2_ns"]
    ) / lanes["diffusion_vacf_pbc_nm2_ns"]
    viscosity_window_relative = np.abs(
        lanes["viscosity_mPa_s"] - lanes["viscosity_10_20ps_mPa_s"]
    ) / lanes["viscosity_mPa_s"]
    t12_shifts = shifts[shifts["contrast"] == "T12-ZS"]
    prefix_counts = lanes["analyzed_ps"].value_counts().sort_index()
    prefix_summary = ", ".join(
        f"{int(count)} lanes at {duration:.0f} ps" for duration, count in prefix_counts.items()
    )
    positive_count = int(np.count_nonzero(
        (lanes["diffusion_vacf_pbc_nm2_ns"] > 0.0) & (lanes["viscosity_mPa_s"] > 0.0)
    ))
    state_correction_fraction = (
        states["yeh_hummer_nm2_ns"] / states["diffusion_vacf_pbc_nm2_ns"]
    )

    comparison_section = ""
    if comparison is not None:
        comparison_rows = []
        for _, row in comparison.iterrows():
            comparison_rows.append([
                str(row["model_label"]),
                int(row["temperature_K"]),
                markdown_number(row["diffusion_vacf_corrected_nm2_ns_delta"]),
                markdown_number(row["diffusion_msd_pbc_nm2_ns_delta"]),
                markdown_number(row["viscosity_mPa_s_delta"]),
            ])
        comparison_section = fr"""
## Change from the common 600 ps report

The deltas below are `current unequal-prefix value − common-600-ps value`. They show how the estimators moved when each lane used its selected additional committed data.

![Change from 600 ps](change_from_600ps.png)

{dataframe_markdown(['Model', 'T (K)', 'ΔD_corr', 'ΔD_MSD,PBC', 'Δη'], comparison_rows)}

The machine-readable current, reference, absolute-delta, and percentage-delta columns are in [`comparison_to_600ps.csv`](comparison_to_600ps.csv).
"""

    report = fr"""# Unequal committed-prefix NVT dynamics for six water models

_Frozen analysis: {finished_utc}. Source-prefix discovery began at {started_utc}._

## Executive reading

This calculation covers **{len(lanes)} lanes = {len(models)} models × 4 temperatures × 3 replicas**. It deliberately does **not** truncate trajectories to a common duration. Each lane uses the committed prefix selected from the supplied coverage table: {prefix_summary}. Only atomically committed checkpoint segments are included; active partial arrays and later campaign progress are excluded.

All **{int(lanes['complete_600ps'].sum())}/{len(lanes)}** lanes reach at least 600 ps. The lane-specific 700/800/900 ps selection uses more available data than the common-600-ps report, but it makes this a convergence snapshot rather than a controlled equal-duration model ranking.

On a simple equal-weight log-error score against temperature-matched experimental diffusion and viscosity, the lowest current combined score is **{best}**. The ranking remains descriptive because trajectory durations differ, viscosity is noisy, and the cell contains only 64 waters.

The plateau gates flag **{d_unstable}/{len(states)}** state means with more than 10% VACF-integral variation over 5–10 ps and **{eta_unstable}/{len(states)}** with more than 50% viscosity-integral variation over 5–10 ps.

## Main signals

- T12−ZS raises corrected diffusion in **{int((t12_shifts['diffusion_vacf_corrected_nm2_ns_shift'] > 0).sum())}/{len(t12_shifts)}** backbone/temperature comparisons and lowers viscosity in **{int((t12_shifts['viscosity_mPa_s_shift'] < 0).sum())}/{len(t12_shifts)}**.
- Across state means, `ΔD_YH` spans **{states['yeh_hummer_nm2_ns'].min():.3f}–{states['yeh_hummer_nm2_ns'].max():.3f} nm²/ns** and has a median size of **{100.0 * state_correction_fraction.median():.1f}%** of raw VACF diffusion. The large shift reflects the 64-water box and inherits uncertainty from viscosity.
- The retained LES-ZS branch uses saved late-NPT endpoints for 11 lanes. It is included to compare directly with the earlier 600 ps report, but one corrected strict LES-ZS branch must replace it in the final analysis.
- The 700/800/900 ps differences reflect campaign progress, not different physical protocols.

## Numerical sanity checks

- **{positive_count}/{len(lanes)}** lane estimates have positive finite diffusion and viscosity.
- VACF and NVT-MSD finite-box diffusion differ by a median **{100.0 * vacf_msd_relative.median():.1f}%**; the 95th percentile is **{100.0 * vacf_msd_relative.quantile(0.95):.1f}%**. Every 10–600 ps MSD fit has **R² ≥ {lanes['diffusion_msd_r_squared'].min():.5f}**.
- The 5–10 and 10–20 ps viscosity estimates differ by a median **{100.0 * viscosity_window_relative.median():.1f}%** and at worst **{100.0 * viscosity_window_relative.max():.1f}%**.
- The expected zero-lag correlation between the two non-orthogonal diagonal stress combinations averages **{lanes['stress_diagonal_pair_rho_t0'].mean():.4f}**.

![Coverage by model](coverage_by_model.png)

## Prefix coverage

{dataframe_markdown(['Model', 'lane range (ps)', 'median (ps)', 'at least 600 ps'], coverage_rows)}

## Diffusion

The finite-box VACF estimate is integrated over 5–10 ps. The NVT-MSD estimate is the 10–600 ps oxygen-MSD slope. Both finite-box estimates receive the same lane-wise Yeh–Hummer correction,

$$
D_\infty=D_{{\rm PBC}}+\Delta D_{{\rm YH}},
\qquad
\Delta D_{{\rm YH}}=\frac{{k_B T\,2.837297}}{{6\pi\eta L}},
$$

using that lane's NVT temperature, box length, and Green–Kubo viscosity. Periodic hydrodynamic drag affects the diffusion coefficient itself, so the correction is independent of whether `D_PBC` was estimated from VACF or MSD. Error bars are SEM over the three MD replicas.

The MSD graph uses a linear 0–1000 ps axis and displays every available lag. Faint lines show individual replicas through their committed 700, 800, or 900 ps endpoints; bold means stop at the shortest available replica for that model and temperature so the averaging denominator never changes mid-curve. Nothing is extended to 1000 ps. The 10–600 ps fit is shaded. Because the number of displacement time origins decreases with lag, the high-lag end is intrinsically noisier and is retained here at the requested weighting.

![Diffusion versus temperature](diffusion_versus_temperature.png)

![Raw and finite-size-corrected VACF/MSD diffusion](vacf_msd_finite_size_comparison.png)

![VACF and running diffusion](vacf_and_running_diffusion_295K.png)

The all-temperature panel below shows the normalized oxygen velocity VACF in its top row and its finite-size-corrected Green–Kubo integral in the bottom row. The gray 5–10 ps integration window is the reporting window. Each lane's constant Yeh–Hummer shift is added to its running finite-box integral before replica averaging, so the early-lag corrected curve is a convergence diagnostic rather than an instantaneous diffusion coefficient.

![All-temperature velocity VACF and corrected running diffusion](vacf_and_corrected_running_diffusion_by_temperature.png)

The cumulative panel below uses successive 100 ps velocity blocks. At every prefix, finite-box VACF diffusion and viscosity are cumulatively averaged and that prefix's viscosity supplies its Yeh–Hummer correction. Its trajectory-duration axis starts at zero and the first defined estimate is at 100 ps. Faint lines are individual replicas; bold means stop at the shortest replica duration for each state.

![Full-duration cumulative corrected VACF diffusion](diffusion_prefix_convergence_by_temperature.png)

![NVT oxygen MSD](oxygen_msd_by_temperature.png)

## Viscosity and stress diagnostics

The Green–Kubo viscosity averages five stored traceless stress autocorrelations. The reported value is the 5–10 ps running-integral mean; 10–20 ps is retained as a sensitivity check.

![Viscosity versus temperature](viscosity_versus_temperature.png)

![Running viscosity](running_viscosity_by_temperature.png)

The running-integral panel above resolves Green–Kubo convergence versus stress-correlation lag. The complementary full-duration panel below shows how the reported 5–10 ps plateau estimate changes as 100 ps trajectory blocks accumulate. Its axis begins at zero, but the first statistically defined estimate is at 100 ps. Faint lines are individual replicas through their maximum committed duration; bold means stop at the shortest replica duration for each state.

![Full-duration cumulative viscosity](viscosity_prefix_convergence_by_temperature.png)

![Stress cross-correlations](stress_cross_correlations_295K.png)

## All state values

`D_VACF,∞` remains the primary paper-matched result. `D_MSD,∞` is a same-state NVT cross-check; it is not equivalent to the separate multi-nanosecond NVE endpoint used in the reference papers.

{dataframe_markdown(['Model', 'T (K)', 'prefix (ps)', 'D_VACF,PBC', 'ΔD_YH', 'D_VACF,∞', 'D_MSD,PBC', 'D_MSD,∞', 'D_exp', 'η', 'η_exp'], state_rows)}

Units: diffusion in nm²/ns; viscosity in mPa s.

## Why Montero corrected VACF but not its NVE MSD

This was a data-availability and state-matching choice, not a distinction between the two diffusion estimators. Montero de Hijes et al. calculated oxygen VACF and Green–Kubo viscosity from the same 600 ps NVT trajectories, so a state-matched Yeh–Hummer correction was available for the VACF result. Their independent 5–9 ns NVE trajectories targeted long-time MSD, but viscosity was not calculated for those runs and their pressures could differ from ambient by up to roughly ±350 bar. The paper therefore left the NVE/MSD values uncorrected and estimated that the 512-water finite-box values were about 15% low.

Here, VACF, MSD, and viscosity are all extracted from each same NVT lane. Applying its single `ΔD_YH` to both raw estimators is therefore internally consistent and makes their agreement a useful diagnostic. A persistent VACF–MSD discrepancy after the identical shift would implicate the VACF integration window, MSD fit window, unwrapping, thermostat influence, or sampling—not finite size.

{comparison_section}

## Paired model shifts

The same MD seed is paired across ZS and T12 conditions. Positive values mean T12 increases the property. These pairs share seeds but are not duration matched when their selected prefixes differ.

{dataframe_markdown(['Contrast', 'Backbone', 'T (K)', 'pairs', 'ΔD_corr', 'Δη'], shift_rows)}

![Paired shifts](paired_model_shifts.png)

![Prefix convergence](prefix_convergence_295K.png)

## Provisional experimental-error ranking

The score is `sqrt(mean(log(simulation/experiment)^2))` over four temperatures, weighting relative diffusion and viscosity errors equally.

{dataframe_markdown(['Rank', 'Model', 'D log-RMSE', 'η log-RMSE', 'combined'], score_rows)}

## Interpretation limits

- Unequal prefixes prevent a controlled duration-matched model ranking; use the delta figure to assess sensitivity to the extra data.
- The 64-water box is smaller than the 512-water MACE transport box used by Montero de Hijes et al.
- NVT oxygen MSD is not the separate long-time NVE diffusion endpoint used in the reference workflows.
- The requested 10–600 ps MSD fit includes progressively fewer displacement time origins at large lag. Its larger replica SEM and fit-window sensitivity must be considered alongside R².
- The corrected NVT-MSD value uses the same noisy lane-wise viscosity as corrected VACF; their finite-size uncertainties are therefore correlated rather than independent.
- Hydrogen positions were not stored at high frequency, so OH rotational dynamics cannot be reconstructed.
- Final conclusions require 1 ns for all lanes and replacement of retained mixed-end LES-ZS by a corrected strict branch.

## Reproducibility files

- [`lanes.csv`](lanes.csv): lane-level estimates and exact segment paths
- [`input_snapshot.csv`](input_snapshot.csv): frozen per-lane durations and committed segment selection
- [`states.csv`](states.csv): three-replica state summaries
- [`paired_shifts.csv`](paired_shifts.csv): paired T12−ZS contrasts
- [`comparison_to_600ps.csv`](comparison_to_600ps.csv): changes from the common 600 ps analysis
- [`analysis_summary.json`](analysis_summary.json): machine-readable run metadata
- [`analyze_preliminary_nvt_dynamics.py`](analyze_preliminary_nvt_dynamics.py): frozen analysis program

## Sources

1. P. Montero de Hijes, L. Neubeck, G. Kresse, and C. Dellago, [“Dynamical properties of ab initio water from machine-learning potentials,” arXiv:2606.18163 (2026)](https://arxiv.org/abs/2606.18163).
2. T. Hilpert and G. Kresse, [“Accurate Thermophysical Properties of Water using Machine-Learned Potentials,” arXiv:2601.21103 (2026)](https://arxiv.org/abs/2601.21103).
"""
    (output_dir / "VARIABLE_PREFIX_NVT_DYNAMICS.md").write_text(report, encoding="utf-8")


def write_references(output_dir: Path) -> None:
    rows = []
    for temperature in TEMPERATURES:
        rows.append({
            "source": "Holz-Heil-Sacco experimental fit",
            "temperature_K": temperature,
            "diffusion_nm2_ns": experimental_diffusion(temperature),
            "viscosity_mPa_s": math.nan,
            "notes": "temperature-matched experiment",
        })
        rows.append({
            "source": "IAPWS SR6-08 Eq. 7",
            "temperature_K": temperature,
            "diffusion_nm2_ns": math.nan,
            "viscosity_mPa_s": experimental_viscosity(temperature),
            "notes": "0.1 MPa reference correlation",
        })
    rows.extend({"source": "Montero RPBE-D3/zd MACE", **item, "notes": item.pop("ensemble")} for item in [dict(x) for x in MONTERO_RPBE])
    rows.extend({"source": "Hilpert RPBE-D3 MACE L2", **item, "notes": item.pop("ensemble")} for item in [dict(x) for x in HILPERT_MACE_L2])
    pd.DataFrame(rows).to_csv(output_dir / "reference_values.csv", index=False)


def main() -> int:
    args = parse_args()
    started = utc_now()
    if args.output_dir.exists():
        complete = args.output_dir / "ANALYSIS_COMPLETE"
        if complete.exists() or not args.overwrite_incomplete:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with args.manifest.open(encoding="utf-8") as handle:
        lanes = list(csv.DictReader(handle, delimiter="\t"))
    if args.lanes:
        selected = set(args.lanes)
        lanes = [lane for lane in lanes if int(lane["lane_id"]) in selected]
    if args.model_labels:
        requested_models = set(args.model_labels)
        unknown = requested_models - set(MODEL_ORDER)
        if unknown:
            raise ValueError(f"unknown model labels: {sorted(unknown)}")
        lanes = [lane for lane in lanes if lane["model_label"] in requested_models]
    if not lanes:
        raise ValueError("no lanes selected")

    model_targets = parse_target_assignments(args.model_target, "model target")
    lane_targets = parse_target_assignments(args.lane_target, "lane target")
    unknown_target_models = set(model_targets) - set(MODEL_ORDER)
    if unknown_target_models:
        raise ValueError(f"unknown model targets: {sorted(unknown_target_models)}")
    selected_lane_ids = {str(lane["lane_id"]) for lane in lanes}
    unknown_target_lanes = set(lane_targets) - selected_lane_ids
    if unknown_target_lanes:
        raise ValueError(f"lane targets are not selected: {sorted(unknown_target_lanes)}")

    frozen_targets: dict[int, float] = {}
    snapshot_rows: list[dict[str, Any]] = []
    if args.complete_segments_only:
        for lane in lanes:
            lane_id = int(lane["lane_id"])
            requested_duration = lane_targets.get(
                str(lane_id),
                model_targets.get(lane["model_label"], args.target_prefix_ps),
            )
            sources = discover_segments(
                Path(lane["output_dir"]),
                requested_duration,
                allow_partial=False,
            )
            duration = float(sum(source.duration_ps for source in sources))
            if duration < requested_duration - 1.0e-9:
                raise ValueError(
                    f"lane {lane_id} has {duration:.1f} committed ps, below requested "
                    f"{requested_duration:.1f} ps"
                )
            frozen_targets[lane_id] = duration
            snapshot_rows.append({
                "lane_id": lane_id,
                "model_label": lane["model_label"],
                "temperature_K": int(lane["temperature_K"]),
                "md_replica": int(lane["md_replica"]),
                "requested_ps": requested_duration,
                "committed_ps": duration,
                "source_segments": json.dumps(
                    [asdict(source) for source in sources], separators=(",", ":")
                ),
            })
        pd.DataFrame(snapshot_rows).sort_values("lane_id").to_csv(
            args.output_dir / "input_snapshot.csv", index=False
        )
    else:
        frozen_targets = {int(lane["lane_id"]): args.target_prefix_ps for lane in lanes}

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_lane,
                lane,
                str(args.output_dir),
                frozen_targets[int(lane["lane_id"])],
                not args.complete_segments_only,
            ): int(lane["lane_id"])
            for lane in lanes
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:02d}/{len(futures):02d}] lane {result['lane_id']:02d}: "
                f"{result['status']}"
                + (f" ({result.get('analyzed_ps', math.nan):.1f} ps)" if result["status"] == "ok" else ""),
                flush=True,
            )

    raw = pd.DataFrame(results).sort_values("lane_id")
    raw.to_csv(args.output_dir / "lanes_all_attempts.csv", index=False)
    errors = raw[raw["status"] != "ok"]
    if len(errors):
        (args.output_dir / "ERRORS.txt").write_text(
            "\n\n".join(f"lane {row.lane_id}:\n{row.error}" for row in errors.itertuples()),
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(errors)} lane analyses failed; see ERRORS.txt")

    lane_table = raw.drop(columns=["status", "error"])
    lane_table.to_csv(args.output_dir / "lanes.csv", index=False)
    states = aggregate_states(lane_table)
    states.to_csv(args.output_dir / "states.csv", index=False)
    shifts = paired_shifts(lane_table)
    shifts.to_csv(args.output_dir / "paired_shifts.csv", index=False)
    comparison = None
    if args.comparison_states:
        comparison = compare_states(states, args.comparison_states)
        comparison.to_csv(args.output_dir / "comparison_to_600ps.csv", index=False)
    write_references(args.output_dir)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_coverage(lane_table, args.output_dir)
    plot_transport(states, args.output_dir, "diffusion")
    plot_transport(states, args.output_dir, "viscosity")
    plot_diffusion_estimator_comparison(states, args.output_dir)
    plot_vacf(lane_table, args.output_dir)
    plot_vacf_and_corrected_running_diffusion_by_temperature(lane_table, args.output_dir)
    plot_stress_cross(lane_table, args.output_dir)
    plot_running_viscosity(lane_table, args.output_dir)
    plot_viscosity_prefix_convergence(lane_table, args.output_dir)
    plot_diffusion_prefix_convergence(lane_table, args.output_dir)
    plot_msd(lane_table, args.output_dir)
    plot_prefix_convergence(lane_table, args.output_dir)
    plot_shifts(shifts, args.output_dir)
    if comparison is not None:
        plot_state_changes(comparison, args.output_dir)

    finished = utc_now()
    if args.complete_segments_only:
        write_variable_prefix_report(
            args.output_dir,
            lane_table,
            states,
            shifts,
            comparison,
            started,
            finished,
        )
        report_path = args.output_dir / "VARIABLE_PREFIX_NVT_DYNAMICS.md"
    else:
        write_report(args.output_dir, lane_table, states, shifts, started, finished)
        report_path = args.output_dir / "PRELIMINARY_NVT_DYNAMICS.md"
    shutil.copy2(Path(__file__), args.output_dir / "analyze_preliminary_nvt_dynamics.py")
    summary = {
        "schema_version": "preliminary-nvt-dynamics-v3",
        "started_utc": started,
        "finished_utc": finished,
        "manifest": str(args.manifest.resolve()),
        "manifest_lane_count": len(lanes),
        "selected_models": list(present_models(lane_table)),
        "successful_lane_count": len(lane_table),
        "complete_600ps_lane_count": int(lane_table["complete_600ps"].sum()),
        "target_prefix_ps": args.target_prefix_ps,
        "duration_policy": (
            "lane-specific committed prefix selected by explicit model/lane caps; "
            "active partial segments excluded"
            if args.complete_segments_only
            else "safely visible prefix including flushed partial segments"
        ),
        "input_snapshot": "input_snapshot.csv" if args.complete_segments_only else None,
        "comparison_states": str(args.comparison_states.resolve()) if args.comparison_states else None,
        "model_targets_ps": model_targets,
        "lane_targets_ps": lane_targets,
        "estimators": {
            "correlation_stride_fs": CORRELATION_DT_PS * 1000.0,
            "vacf_plateau_ps": VACF_PLATEAU_PS,
            "viscosity_plateau_ps": VISCOSITY_PLATEAU_PS,
            "viscosity_sensitivity_ps": VISCOSITY_SENSITIVITY_PS,
            "msd_fit_ps": MSD_FIT_PS,
            "finite_size_corrected_estimators": ["VACF", "NVT-MSD"],
            "primary_diffusion_estimator": "finite-size-corrected VACF",
            "prefix_block_ps": PREFIX_BLOCK_PS,
            "oxygen_com_drift_removed": True,
            "stress_component_time_means_removed": True,
        },
        "limitations": [
            "unequal per-lane committed prefixes",
            "64-water finite-size screen",
            "NVT MSD is not long-time NVE diffusion",
            "VACF and NVT-MSD finite-size corrections share the same noisy lane-wise viscosity",
            "LES-ZS uses retained mixed-end NPT starting states for 11 lanes",
            "OH rotational correlation unavailable because H positions were not stored",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ANALYSIS_COMPLETE").touch()
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
