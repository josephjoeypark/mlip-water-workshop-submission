#!/usr/bin/env python3
"""Train and export one G1-G2 or G1-G6 adaptation."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", choices=("maceoff23m", "maceles-off"), required=True)
    parser.add_argument("--condition", choices=("g1-g2", "g1-g6"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs/study.yaml")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent
    path = lambda value: (root / value).resolve()
    settings = config["training"]
    seed = settings["seed"] if args.seed is None else args.seed
    device = args.device or settings["device"]
    foundation_id = settings["foundation_model"][args.foundation]
    foundation = path(config["models"][foundation_id]["path"])
    foundation_name = config["models"][foundation_id]["foundation"]
    output = path(config["paths"]["output"]) / "training" / f"{args.foundation}-{args.condition}-seed{seed}"
    internal = path(config["paths"]["implementation"]) / "training"
    commands = []

    if args.condition == "g1-g2":
        lane = settings["g1-g2"]
        run_name = f"{args.foundation}-g1-g2-seed{seed}"
        env = os.environ.copy()
        env.update({
            "FOUNDATION_MODEL": str(foundation), "TRAIN_FILE": str(path(lane["train"])),
            "VALID_FILE": str(path(lane["valid"])), "TEST_FILE": str(path(lane["test"])),
            "RUN_NAME": run_name, "OUTPUT_DIR": str(output),
            "MODEL_KIND": "MACE" if args.foundation == "maceoff23m" else "MACELES",
            "SEED": str(seed), "EPOCHS": str(lane["epochs_to_update_123200"]), "DEVICE": device,
        })
        train = ["bash", internal / "g1-g2/train_lora.sh"]
        checkpoint = output / "checkpoints" / f"{run_name}_run-{seed}_epoch-{lane['reported_epoch_index']}.pt"
        fused = output / f"{args.foundation}-g1-g2.model"
        export = [sys.executable, internal / "g1-g2/export_model.py", "--foundation", foundation,
                  "--checkpoint", checkpoint, "--expected-update", settings["reported_update"],
                  "--display-name", f"{foundation_name} / G1-G2", "--output", fused,
                  "--metadata", output / "export.json"]
        evaluate = [sys.executable, Path(__file__).with_name("evaluate_clusters.py"),
                    "--model", fused, "--data", path(lane["evaluation"]),
                    "--model-label", f"{foundation_name} / G1-G2", "--device", device,
                    "--seed", seed, "--optimizer-updates", settings["reported_update"],
                    "--output", output / "evaluation.csv"]
        commands = [(train, env), (export, None), (evaluate, None)]
    else:
        lane = settings["g1-g6"]
        train, valid, source = path(lane["train"]), path(lane["valid"]), path(lane["source_manifest"])
        audit, calibration = output / "data_audit.json", output / "calibration.json"
        scripts = internal / "g1-g6"
        file_slug = "maceoff23m" if args.foundation == "maceoff23m" else "maceles-off"
        fused = output / "export" / f"{file_slug}-g1-g6-seed{seed}-fused.model"
        commands = [
            ([sys.executable, scripts / "audit_data.py", "--train", train, "--valid", valid, "--source-manifest", source, "--output", audit], None),
            ([sys.executable, scripts / "calibrate.py", "--foundation", foundation, "--train", train, "--source-manifest", source, "--output", calibration, "--seed", seed, "--device", device], None),
            ([sys.executable, scripts / "train_impl.py", "--foundation", foundation, "--condition", "G1-G6", "--train", train, "--valid", valid, "--source-manifest", source, "--calibration", calibration, "--output-dir", output, "--seed", seed, "--max-steps", settings["reported_update"], "--device", device] + (["--resume"] if args.resume else []), None),
            ([sys.executable, scripts / "evaluate.py", "--foundation", foundation, "--backbone", config["models"][foundation_id]["kind"], "--condition", "G1-G6", "--checkpoint", output / "final_ema.pt", "--expected-step", settings["reported_update"], "--valid", valid, "--source-manifest", source, "--output-dir", output / "export", "--seed", seed, "--device", device], None),
            ([sys.executable, Path(__file__).with_name("evaluate_clusters.py"), "--model", fused,
              "--data", path(lane["evaluation"]), "--model-label", f"{foundation_name} / G1-G6",
              "--seed", seed, "--optimizer-updates", settings["reported_update"],
              "--device", device, "--output", output / "evaluation.csv"], None),
        ]

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for command, env in commands:
        command = [str(item) for item in command]
        shown = " ".join(map(shlex.quote, command))
        print("$ " + shown)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
