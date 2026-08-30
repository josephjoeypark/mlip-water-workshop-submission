#!/usr/bin/env python3
"""Train one G1-G6 adaptation at a fixed optimizer budget.

The archived implementation also retains its internal counterfactual objective
tokens so the historical checkpoint path remains reconstructable.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PilotCollator, PilotDataset, batch_to_device, sha256
from loss import CONDITIONS, condition_objective, objective_terms
from model import load_model, load_trainable_state, trainable_state


PRODUCTION_BUDGETS = (39_600, 123_200, 246_400, 369_600)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class TrainableEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.state = trainable_state(model)

    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, average in self.state.items():
                average.mul_(self.decay).add_(
                    parameters[name].detach().cpu(), alpha=1.0 - self.decay
                )

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module) -> Iterator[None]:
        original = trainable_state(model)
        load_trainable_state(model, self.state)
        try:
            yield
        finally:
            load_trainable_state(model, original)


def validate_primary(
    model: torch.nn.Module,
    dataset: PilotDataset,
    collator: PilotCollator,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    sums = {"total_energy": 0.0, "total_force": 0.0, "primary": 0.0}
    count = 0
    model.eval()
    for host_batch in loader:
        batch = batch_to_device(host_batch, device)
        current = model(
            batch["clusters"],
            training=False,
            compute_counterfactual=False,
        )
        terms = objective_terms(current, batch["reference"], condition="D00", scales={})
        records = len(host_batch["records"])
        for name in sums:
            sums[name] += float(terms[name].detach().cpu()) * records
        count += records
    return {name: value / count for name, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--condition", choices=(*CONDITIONS, "G1-G6"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=39600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--valid-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-8)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-records", type=int, default=0)
    args = parser.parse_args()
    internal_condition = "D00" if args.condition == "G1-G6" else args.condition
    if "test" in args.train.name.lower() or "test" in args.valid.name.lower():
        raise ValueError("locked-test inputs are forbidden")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    source = json.loads(args.source_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if not source.get("passed") or not calibration.get("passed"):
        raise ValueError("data or calibration gate did not pass")
    for split, path in (("train", args.train), ("valid", args.valid)):
        if sha256(path) != source["products"][split]["sha256"]:
            raise ValueError(f"{split} hash differs from gated source")
    if calibration["foundation_sha256"] != sha256(args.foundation):
        raise ValueError("foundation differs from calibration")
    if calibration["train_sha256"] != sha256(args.train):
        raise ValueError("training data differ from calibration")

    model = load_model(args.foundation, seed=args.seed, device=device)
    initial_state = trainable_state(model)
    initial_hash = state_sha256(initial_state)
    if initial_hash != calibration["initial_trainable_sha256"]:
        raise ValueError("initial trainable state differs from calibration")
    trainability = model.audit_trainability()
    train_dataset = PilotDataset(args.train)
    valid_dataset = PilotDataset(
        args.valid,
        null_size_means=train_dataset.audit["null_size_energy_means_eV"],
    )
    if args.smoke_records:
        train_dataset.records = train_dataset.records[: args.smoke_records]
        valid_dataset.records = valid_dataset.records[: min(args.smoke_records, len(valid_dataset))]
    collator = PilotCollator(
        atomic_numbers=model.trunk.atomic_numbers.detach().cpu().tolist(),
        cutoff=float(model.trunk.r_max),
    )
    scales = calibration["scales"]
    if internal_condition == "D00":
        lambda_energy = 0.0
        lambda_force = 0.0
    else:
        lane_calibration = calibration["calibrations"][internal_condition]
        lambda_energy = float(lane_calibration["lambda_energy"])
        lambda_force = float(lane_calibration["lambda_force"])

    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    ema = TrainableEMA(model, args.ema_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    start_epoch = 0
    global_step = 0
    resume_from_step = 0
    best_valid = math.inf
    best_step = -1
    if args.resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["initial_trainable_sha256"] != initial_hash:
            raise ValueError("resume initialization differs")
        load_trainable_state(model, saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        ema.state = {name: value.clone() for name, value in saved["ema_state"].items()}
        start_epoch = int(saved["epoch"])
        global_step = int(saved["global_step"])
        best_valid = float(saved["best_valid_primary"])
        best_step = int(saved["best_step"])
        torch.set_rng_state(saved["torch_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        resume_from_step = global_step

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    if not args.smoke_records:
        if (len(train_dataset), args.batch_size) != (8800, 2):
            raise ValueError("production must use the frozen 8,800-record G1-G6 data and batch size 2")
        if args.max_steps not in PRODUCTION_BUDGETS:
            raise ValueError(f"production budget must be one of {PRODUCTION_BUDGETS}")
        if args.max_steps % steps_per_epoch != 0:
            raise ValueError("production budget must end at a complete deterministic epoch")
    if global_step > args.max_steps:
        raise ValueError(f"resume checkpoint step {global_step} exceeds target step {args.max_steps}")
    need_counterfactual = internal_condition != "D00"
    log_path = args.output_dir / "metrics.jsonl"
    start_time = time.monotonic()
    training_seconds = 0.0
    for epoch in range(start_epoch, math.ceil(args.max_steps / steps_per_epoch)):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collator,
        )
        model.train()
        sums = {
            name: 0.0
            for name in (
                "total_energy",
                "total_force",
                "primary",
                "interaction_energy_aux",
                "interaction_force_aux",
                "objective",
            )
        }
        count = 0
        for host_batch in loader:
            if global_step >= args.max_steps:
                break
            batch = batch_to_device(host_batch, device)
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            current = model(
                batch["clusters"],
                batch["monomers"] if need_counterfactual else None,
                batch["monomer_owner"] if need_counterfactual else None,
                training=True,
                compute_counterfactual=need_counterfactual,
            )
            terms = objective_terms(
                current,
                batch["reference"],
                condition=internal_condition,
                scales=scales,
            )
            objective = condition_objective(
                terms,
                condition=internal_condition,
                lambda_energy=lambda_energy,
                lambda_force=lambda_force,
            )
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("non-finite objective")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            ema.update(model)
            training_seconds += time.monotonic() - step_start
            records = len(host_batch["records"])
            for name in terms:
                sums[name] += float(terms[name].detach().cpu()) * records
            sums["objective"] += float(objective.detach().cpu()) * records
            count += records
            global_step += 1

        with ema.applied(model):
            valid = validate_primary(
                model,
                valid_dataset,
                collator,
                device=device,
                batch_size=args.valid_batch_size,
            )
        if valid["primary"] < best_valid:
            best_valid = valid["primary"]
            best_step = global_step
            atomic_save(
                args.output_dir / "best_valid.pt",
                {
                    "schema_version": "single-mace-counterfactual-ema-v1",
                    "kind": "best_validation_ema",
                    "condition": args.condition,
                    "seed": args.seed,
                    "global_step": global_step,
                    "valid_primary": best_valid,
                    "initial_trainable_sha256": initial_hash,
                    "trainable_state": {name: value.clone() for name, value in ema.state.items()},
                },
            )
        checkpoint = {
            "schema_version": "single-mace-counterfactual-resume-v1",
            "condition": args.condition,
            "seed": args.seed,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_valid_primary": best_valid,
            "best_step": best_step,
            "initial_trainable_sha256": initial_hash,
            "trainable_state": trainable_state(model),
            "ema_state": {name: value.clone() for name, value in ema.state.items()},
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_save(checkpoint_path, checkpoint)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": {name: value / count for name, value in sums.items()},
            "valid_ema": valid,
            "best_valid_primary": best_valid,
            "best_step": best_step,
            "elapsed_seconds": time.monotonic() - start_time,
            "training_loop_seconds": training_seconds,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if global_step >= args.max_steps:
            break
    if global_step != args.max_steps:
        raise RuntimeError(f"completed {global_step}, expected {args.max_steps}")
    atomic_save(
        args.output_dir / "final_ema.pt",
        {
            "schema_version": "single-mace-counterfactual-ema-v1",
            "kind": "fixed_update_final_ema",
            "condition": args.condition,
            "seed": args.seed,
            "global_step": global_step,
            "initial_trainable_sha256": initial_hash,
            "trainable_state": {name: value.clone() for name, value in ema.state.items()},
        },
    )
    protocol = {
        "schema_version": "single-mace-counterfactual-protocol-v1",
        "passed": True,
        "condition": args.condition,
        "seed": args.seed,
        "foundation": str(args.foundation.resolve()),
        "foundation_sha256": sha256(args.foundation),
        "train_sha256": sha256(args.train),
        "valid_sha256": sha256(args.valid),
        "calibration_sha256": sha256(args.calibration),
        "global_step": global_step,
        "resume_from_step": resume_from_step,
        "steps_completed_this_run": global_step - resume_from_step,
        "epochs": global_step / steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "validation_batch_size": args.valid_batch_size,
        "optimizer": "AdamW_AMSGrad",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "ema_decay": args.ema_decay,
        "dtype": "float64",
        "lora_rank": 32,
        "lora_alpha": 32,
        "total_path": "ordinary single MACE cluster energy and its gradient",
        "counterfactual_path": "training/evaluation only",
        "lambda_energy": lambda_energy,
        "lambda_force": lambda_force,
        "best_valid_primary": best_valid,
        "best_step": best_step,
        "initial_trainable_sha256": initial_hash,
        "trainability": trainability,
        "seconds_per_optimizer_step_this_run": (
            training_seconds / (global_step - resume_from_step)
            if global_step > resume_from_step
            else None
        ),
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    (args.output_dir / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    main()
