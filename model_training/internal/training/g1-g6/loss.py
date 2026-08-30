"""Primary total-E/F and calibrated counterfactual auxiliary losses."""

from __future__ import annotations

from typing import Mapping

import torch


CONDITIONS = ("D00", "D11", "D11N")


def objective_terms(
    prediction: Mapping[str, torch.Tensor | None],
    reference: Mapping[str, torch.Tensor],
    *,
    condition: str,
    scales: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    energy = prediction["energy"]
    forces = prediction["forces"]
    if energy is None or forces is None:
        raise ValueError("total energy/forces are required")
    total_energy = torch.mean(
        torch.square((energy - reference["total_energy"]) / reference["num_atoms"])
    )
    total_force = torch.mean(torch.square(forces - reference["total_forces"]))
    primary = total_energy + 100.0 * total_force
    zero = primary.new_zeros(())
    if condition == "D00":
        return {
            "total_energy": total_energy,
            "total_force": total_force,
            "primary": primary,
            "interaction_energy_aux": zero,
            "interaction_force_aux": zero,
        }
    int_energy = prediction["interaction_energy"]
    int_forces = prediction["interaction_forces"]
    if int_energy is None or int_forces is None:
        raise ValueError("counterfactual interaction predictions are required")
    prefix = "null_" if condition == "D11N" else ""
    target_energy = reference[f"{prefix}interaction_energy"]
    target_forces = reference[f"{prefix}interaction_forces"]
    energy_aux = torch.mean(
        torch.square(
            ((int_energy - target_energy) / reference["num_atoms"])
            / float(scales[f"{condition}_energy_eV_per_atom"])
        )
    )
    force_aux = torch.mean(
        torch.square(
            (int_forces - target_forces)
            / float(scales[f"{condition}_force_eV_per_A"])
        )
    )
    return {
        "total_energy": total_energy,
        "total_force": total_force,
        "primary": primary,
        "interaction_energy_aux": energy_aux,
        "interaction_force_aux": force_aux,
    }


def gradient_vector(
    objective: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]


def vector_norm(values: list[torch.Tensor]) -> float:
    total = values[0].new_zeros(())
    for value in values:
        total = total + value.square().sum()
    result = float(torch.sqrt(total).cpu())
    if not result > 0.0:
        raise FloatingPointError(f"invalid gradient norm {result}")
    return result


def calibrated_lambdas(
    terms: Mapping[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    *,
    target_fraction: float = 0.075,
) -> dict[str, float]:
    if not 0.05 <= target_fraction <= 0.10:
        raise ValueError("target fraction must be in [0.05, 0.10]")
    primary = gradient_vector(terms["primary"], parameters, retain_graph=True)
    energy = gradient_vector(
        terms["interaction_energy_aux"], parameters, retain_graph=True
    )
    force = gradient_vector(
        terms["interaction_force_aux"], parameters, retain_graph=False
    )
    primary_norm = vector_norm(primary)
    energy_norm = vector_norm(energy)
    force_norm = vector_norm(force)
    balanced = [
        energy_value / energy_norm + force_value / force_norm
        for energy_value, force_value in zip(energy, force, strict=True)
    ]
    balanced_norm = vector_norm(balanced)
    common = target_fraction * primary_norm / balanced_norm
    lambda_energy = common / energy_norm
    lambda_force = common / force_norm
    combined = [
        lambda_energy * energy_value + lambda_force * force_value
        for energy_value, force_value in zip(energy, force, strict=True)
    ]
    achieved = vector_norm(combined) / primary_norm
    return {
        "target_fraction": target_fraction,
        "achieved_combined_fraction": achieved,
        "primary_gradient_norm": primary_norm,
        "energy_aux_gradient_norm": energy_norm,
        "force_aux_gradient_norm": force_norm,
        "balanced_combined_gradient_norm": balanced_norm,
        "lambda_energy": lambda_energy,
        "lambda_force": lambda_force,
    }


def condition_objective(
    terms: Mapping[str, torch.Tensor],
    *,
    condition: str,
    lambda_energy: float,
    lambda_force: float,
) -> torch.Tensor:
    result = terms["primary"]
    if condition != "D00":
        result = (
            result
            + float(lambda_energy) * terms["interaction_energy_aux"]
            + float(lambda_force) * terms["interaction_force_aux"]
        )
    return result
