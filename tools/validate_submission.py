#!/usr/bin/env python3
"""Validate the portable workshop artifact without importing ML packages."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONS = {"MACEOFF23(M)", "MACELES-OFF"}
CONDITIONS = {"Baseline", "G1-G2", "G1-G6"}
MODELS = {f"{foundation} / {condition}" for foundation in FOUNDATIONS for condition in CONDITIONS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extxyz_records(path: Path) -> int:
    records = 0
    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                return records
            if not line.strip():
                continue
            atoms = int(line)
            if not handle.readline():
                raise ValueError(f"truncated comment in {path}")
            for _ in range(atoms):
                if not handle.readline():
                    raise ValueError(f"truncated atoms in {path}")
            records += 1


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_training() -> None:
    for lane in ("g1-g2", "g1-g6"):
        directory = ROOT / "model_training/data/molecular_clusters_1000" / lane
        manifest = json.loads((directory / "manifest.json").read_text())
        for item in manifest["products"].values():
            path = directory / item["path"]
            assert path.is_file(), path
            assert sha256(path) == item["sha256"], f"checksum: {path}"
            count = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.suffix == ".jsonl" else extxyz_records(path)
            assert count == item["records"], f"record count: {path} ({count})"


def validate_tables() -> None:
    figure1 = rows(ROOT / "model_training/data/force_energy_seed17.csv")
    assert {row["model_label"] for row in figure1} == MODELS
    assert {row["backbone"] for row in figure1} == FOUNDATIONS
    assert {row["condition"] for row in figure1} == CONDITIONS
    adapted = [row for row in figure1 if row["condition"] != "Baseline"]
    assert {int(float(row["optimizer_updates"])) for row in adapted} == {123200}

    density = rows(ROOT / "md_simulations/data/published_results/density.csv")
    transport = rows(ROOT / "md_simulations/data/published_results/transport_0p5fs.csv")
    assert {row["model"] for row in density} == MODELS and len(density) == 24
    assert {row["model_label"] for row in transport} == MODELS and len(transport) == 24
    assert {float(row["analyzed_min_ps"]) for row in transport} == {700.0}

    bec = rows(ROOT / "model_training/data/bec_frame0/les_bec_64water_frame0.csv")
    les_models = {f"MACELES-OFF / {condition}" for condition in CONDITIONS}
    assert len(bec) == 576 and {row["model"] for row in bec} == les_models
    assert sum(row["element"] == "H" for row in bec) == 384
    assert sum(row["element"] == "O" for row in bec) == 192
    table2 = rows(ROOT / "model_training/data/bec_frame0/bec_comparison.csv")
    assert len(table2) == 4 and {row["source_or_model"] for row in table2[1:]} == les_models
    upstream_hash = "347fcf41584ca834d1bdb1acc622a9ad9329091ec1d033e423dc29c6ef03ab69"
    md_frame = ROOT / "md_simulations/data/initial_configuration/liquid-64.xyz"
    bec_frame = ROOT / "model_training/data/bec_frame0/liquid-64.xyz"
    assert sha256(md_frame) == upstream_hash, "64-water source hash"
    assert sha256(bec_frame) == upstream_hash, "BEC frame hash"
    assert md_frame.read_bytes() == bec_frame.read_bytes(), "MD and BEC frames differ"


def validate_layout() -> None:
    required = (
        "README.md", "RESULTS.md", "FIGURES.md", "DATA_CARD.md", "REPRODUCIBILITY.md",
        "MODEL_ACCESS.md",
        "configs/study.yaml", "models/manifest.json", "model_training/train.py",
        "md_simulations/run_npt.py", "md_simulations/analysis/density_rdf.py", "md_simulations/run_nvt.py",
        "md_simulations/analysis/diffusion.py", "md_simulations/analysis/viscosity.py",
        "md_simulations/data/initial_configuration/SOURCE.md",
        "paper/build_figures.py", "paper/build_tables.py",
        "paper/figures/figure1_molecular_transfer.pdf", "paper/figures/figure2_equilibrium_structure.pdf",
        "paper/figures/figure3_transport_relaxation.pdf",
    )
    for name in required:
        path = ROOT / name
        assert path.is_file() and path.stat().st_size > 0, name
    assert not any(path.is_symlink() for path in ROOT.rglob("*")), "symlinks are not portable"
    included = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and "runs" not in path.parts)
    assert included < 100 * 1024 * 1024, f"artifact is {included / 1024**2:.1f} MiB"


def validate_anonymity_and_names() -> None:
    identity = re.compile(r"/global/|/pscratch/|jpark25|joseph|berkeley\.edu", re.I)
    legacy = re.compile(r"\b(?:OFF|LES)-(?:ZS|T12|D00)(?:-B123200)?\b")
    text_suffixes = {".md", ".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".tsv", ".txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes or ".git" in path.parts:
            continue
        # The later corrected release retains internal archived model tokens
        # and has its own checksum, provenance and numerical validator.
        if path.relative_to(ROOT).parts[0] in {"reproducibility", "runs", ".venv-reproduce"}:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not identity.search(text), f"identity or absolute HPC path in {path.relative_to(ROOT)}"
        relative = path.relative_to(ROOT)
        allowed_legacy = relative == Path("PROVENANCE.md") or relative == Path("models/manifest.json") or "internal" in relative.parts
        if not allowed_legacy:
            assert not legacy.search(text), f"legacy model name in {relative}"


def validate_checksums() -> None:
    checksum_file = ROOT / "SHA256SUMS"
    assert checksum_file.is_file(), "SHA256SUMS is missing"
    for line in checksum_file.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = ROOT / name.lstrip("* ")
        assert path.is_file() and sha256(path) == expected, f"released checksum: {name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", action="store_true", help="also verify the immutable release manifest")
    args = parser.parse_args()
    validate_layout(); validate_training(); validate_tables(); validate_anonymity_and_names()
    if args.checksums:
        validate_checksums()
    print("PASS: layout, size, split hashes/counts, paper tables, names, and anonymity")


if __name__ == "__main__":
    main()
