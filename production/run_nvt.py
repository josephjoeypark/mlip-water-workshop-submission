#!/usr/bin/env python3
"""Run the continuous 1 ns NVT transport trajectory in six segments."""

import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True); parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--replica", type=int, choices=(1, 2, 3), required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/study.yaml"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); root = config_path.parent; path = lambda value: (root / value).resolve()
    if args.model not in config["models"] or args.temperature not in config["md"]["temperatures_K"]: parser.error("model or temperature is not in study.yaml")
    model, nvt = config["models"][args.model], config["md"]["nvt"]
    base = path(config["paths"]["output"]); lane = base / "nvt" / args.model / f"T{args.temperature}" / f"replica{args.replica}"
    previous_atoms = base / "npt" / args.model / f"T{args.temperature}" / f"replica{args.replica}/representative.traj"
    previous_state = None; seed = args.temperature * 10000 + args.replica * 1000 + 729
    for segment, duration in enumerate(nvt["production_segments_ps"], 1):
        output = lane / f"segment{segment}"
        command = [sys.executable, path(config["paths"]["implementation"]) / "md/run_nvt_segment_impl.py", "--model-label", f"{model['foundation']} / {model['condition']}",
            "--model-kind", model["kind"], "--model", path(model["path"]), "--model-sha256", model["sha256"], "--stored-precision", model["stored_precision"],
            "--input", previous_atoms, "--output-dir", output, "--segment", segment, "--temperature-K", args.temperature, "--seed", seed, "--device", args.device,
            "--timestep-fs", nvt["timestep_fs"], "--equilibration-ps", nvt["equilibration_ps"] if segment == 1 else 0, "--production-ps", duration,
            "--ttime-fs", nvt["thermostat_time_fs"], "--position-every", nvt["position_every_steps"], "--velocity-every", nvt["velocity_every_steps"],
            "--stress-every", nvt["stress_every_steps"], "--thermo-every", nvt["thermo_every_steps"]] + (["--resume-dynamics", previous_state] if previous_state else [])
        command = [str(item) for item in command]; print("$ " + " ".join(map(shlex.quote, command)))
        if not args.dry_run and not (output / "COMPLETE").is_file(): subprocess.run(command, check=True)
        previous_atoms, previous_state = output / "final.traj", output / "dynamics_state.npz"


if __name__ == "__main__": main()
