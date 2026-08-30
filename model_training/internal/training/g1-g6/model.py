"""One deployable MACE with training-only counterfactual monomer evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from mace.modules.lora import inject_LoRAs


def owner_sum(values: torch.Tensor, owner: torch.Tensor, graphs: int) -> torch.Tensor:
    result = values.new_zeros((graphs, *values.shape[1:]))
    result.index_add_(0, owner, values)
    return result


def load_model(
    foundation: Path | str,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> "CounterfactualMACE":
    torch.manual_seed(int(seed))
    trunk = torch.load(Path(foundation), map_location="cpu", weights_only=False)
    trunk = inject_LoRAs(trunk.to(dtype=dtype), rank=32, alpha=32)
    model = CounterfactualMACE(trunk).to(device=device, dtype=dtype)
    model.audit_trainability()
    return model


class CounterfactualMACE(torch.nn.Module):
    """The cluster prediction is the only total/deployment path."""

    def __init__(self, trunk: torch.nn.Module) -> None:
        super().__init__()
        self.trunk = trunk

    def audit_trainability(self) -> dict[str, Any]:
        names = [name for name, value in self.named_parameters() if value.requires_grad]
        if not names or not all("lora_A" in name or "lora_B" in name for name in names):
            raise RuntimeError("only official LoRA tensors may be trainable")
        return {
            "trainable_names": names,
            "trainable_tensors": len(names),
            "trainable_parameters": int(
                sum(value.numel() for value in self.parameters() if value.requires_grad)
            ),
            "total_parameters": int(sum(value.numel() for value in self.parameters())),
        }

    def forward(
        self,
        cluster_data: Mapping[str, torch.Tensor],
        monomer_data: Mapping[str, torch.Tensor] | None = None,
        monomer_owner: torch.Tensor | None = None,
        *,
        training: bool,
        compute_counterfactual: bool,
    ) -> dict[str, torch.Tensor | None]:
        cluster = self.trunk(
            cluster_data,
            training=training,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        result: dict[str, torch.Tensor | None] = {
            "energy": cluster["energy"],
            "forces": cluster["forces"],
            "one_body_energy": None,
            "one_body_forces": None,
            "interaction_energy": None,
            "interaction_forces": None,
        }
        if not compute_counterfactual:
            return result
        if monomer_data is None or monomer_owner is None:
            raise ValueError("counterfactual evaluation requires monomer data and owners")
        graphs = int(cluster_data["ptr"].numel() - 1)
        monomers = int(monomer_data["ptr"].numel() - 1)
        if monomer_owner.shape != (monomers,):
            raise ValueError("monomer owner shape changed")
        monomer = self.trunk(
            monomer_data,
            training=training,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        one_energy = owner_sum(monomer["energy"].unsqueeze(-1), monomer_owner, graphs).squeeze(-1)
        if cluster["forces"].shape != monomer["forces"].shape:
            raise RuntimeError("cluster and aligned monomer force shapes differ")
        one_forces = monomer["forces"]
        result.update(
            {
                "one_body_energy": one_energy,
                "one_body_forces": one_forces,
                "interaction_energy": cluster["energy"] - one_energy,
                "interaction_forces": cluster["forces"] - one_forces,
            }
        )
        return result


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def load_trainable_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(state) != expected:
        raise ValueError("trainable checkpoint keys differ")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))
