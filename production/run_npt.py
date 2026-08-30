#!/usr/bin/env python3
"""Run one 1 ns NPT density lane or its preflight gate."""

import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--replica", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--mode", choices=("gate", "production"), default="production")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/study.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); root = config_path.parent
    path = lambda value: (root / value).resolve()
    if args.model not in config["models"] or args.temperature not in config["md"]["temperatures_K"]: parser.error("model or temperature is not in study.yaml")
    model, md, npt = config["models"][args.model], config["md"], config["md"]["npt"]
    seed = args.temperature * 10000 + args.replica * 1000 + 729
    section = "npt" if args.mode == "production" else "gates/npt"
    output = path(config["paths"]["output"]) / section / args.model / f"T{args.temperature}" / f"replica{args.replica}"
    command = [sys.executable, path(config["paths"]["implementation"]) / "md/run_npt_impl.py", "--mode", args.mode,
        "--model-label", f"{model['foundation']} / {model['condition']}", "--model-kind", model["kind"], "--model", path(model["path"]),
        "--model-sha256", model["sha256"], "--stored-precision", model["stored_precision"], "--input", path(md["input"]), "--output-dir", output,
        "--temperature-K", args.temperature, "--pressure-bar", md["pressure_bar"], "--seed", seed, "--device", args.device,
        "--timestep-fs", npt["timestep_fs"], "--preparation-ps", npt["preparation_ps"], "--production-ps", npt["production_ps"],
        "--ttime-fs", npt["thermostat_time_fs"], "--tdamp-fs", npt["thermostat_damping_fs"], "--pdamp-fs", npt["barostat_damping_fs"],
        "--thermo-every", npt["thermo_every_steps"], "--trajectory-every", npt["trajectory_every_steps"]]
    command = [str(item) for item in command]; print("$ " + " ".join(map(shlex.quote, command)))
    if not args.dry_run: subprocess.run(command, check=True)


if __name__ == "__main__": main()

