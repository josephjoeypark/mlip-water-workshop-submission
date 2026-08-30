#!/usr/bin/env python3
"""Compute density/RDF and select the representative NPT frame."""

import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True); parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--replica", type=int, choices=(1, 2, 3), required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "configs/study.yaml"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); root = config_path.parent; path = lambda value: (root / value).resolve()
    if args.model not in config["models"] or args.temperature not in config["md"]["temperatures_K"]: parser.error("model or temperature is not in study.yaml")
    npt = config["md"]["npt"]; run = path(config["paths"]["output"]) / "npt" / args.model / f"T{args.temperature}" / f"replica{args.replica}"
    command = [sys.executable, path(config["paths"]["md_implementation"]) / "md/analyze_npt_impl.py", "--run-dir", run,
        "--model-sha256", config["models"][args.model]["sha256"], "--temperature-K", args.temperature,
        "--analysis-start-ps", npt["analysis_start_ps"], "--timestep-fs", npt["timestep_fs"], "--trajectory-every", npt["trajectory_every_steps"]]
    command = [str(item) for item in command]; print("$ " + " ".join(map(shlex.quote, command)))
    if not args.dry_run: subprocess.run(command, check=True)


if __name__ == "__main__": main()

