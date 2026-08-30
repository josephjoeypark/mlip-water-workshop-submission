#!/usr/bin/env python3
"""Fuse one fixed-update MACE LoRA checkpoint for deployment."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from mace.modules.lora import inject_LoRAs, merge_lora_weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    steps = sorted({int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values() if "step" in state})
    if steps != [args.expected_update]:
        raise ValueError(f"checkpoint optimizer update is {steps}, expected {args.expected_update}")

    native = inject_LoRAs(torch.load(args.foundation, map_location="cpu", weights_only=False), rank=32, alpha=32.0)
    incompatible = native.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    native = native.double().eval()
    fused = merge_lora_weights(copy.deepcopy(native), inplace=True).double().eval()
    native_lora = [name for name, _ in native.named_parameters() if "lora_A" in name or "lora_B" in name]
    fused_lora = [name for name, _ in fused.named_parameters() if "lora_A" in name or "lora_B" in name]
    if not native_lora or fused_lora:
        raise ValueError("LoRA injection or fusion audit failed")
    if hasattr(native, "les") and float(native.les.ewald.norm_factor) != float(fused.les.ewald.norm_factor):
        raise ValueError("LES Ewald configuration changed during fusion")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fused, args.output)
    result = {
        "display_name": args.display_name,
        "optimizer_updates": steps[0],
        "foundation_sha256": sha256(args.foundation),
        "checkpoint_sha256": sha256(args.checkpoint),
        "output": args.output.name,
        "output_sha256": sha256(args.output),
        "parameters": sum(value.numel() for value in fused.parameters()),
        "residual_lora_tensors": fused_lora,
    }
    args.metadata.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

