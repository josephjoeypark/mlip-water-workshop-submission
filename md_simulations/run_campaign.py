#!/usr/bin/env python3
"""Run or print every lane for one NPT/NVT campaign stage."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("npt", "analyze-npt", "nvt"))
    parser.add_argument("--lane-index", type=int, help="select one 0-based lane (useful for scheduler arrays)")
    parser.add_argument("--execute", action="store_true", help="execute; otherwise only print commands")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/study.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    lanes = [(model, temperature, replica) for model in config["models"]
             for temperature in config["md"]["temperatures_K"]
             for replica in range(1, config["md"]["replicas"] + 1)]
    if args.lane_index is not None:
        if not 0 <= args.lane_index < len(lanes):
            parser.error(f"lane index must be 0--{len(lanes) - 1}")
        lanes = [lanes[args.lane_index]]
    script = {"npt": Path(__file__).with_name("run_npt.py"), "analyze-npt": Path(__file__).parent / "analysis" / "density_rdf.py", "nvt": Path(__file__).with_name("run_nvt.py")}[args.stage]
    for model, temperature, replica in lanes:
        command = [sys.executable, script, "--model", model,
                   "--temperature", str(temperature), "--replica", str(replica)]
        if args.stage != "analyze-npt":
            command += ["--device", args.device]
        print("$ " + " ".join(map(str, command)), flush=True)
        if args.execute:
            subprocess.run([str(value) for value in command], check=True)


if __name__ == "__main__":
    main()
