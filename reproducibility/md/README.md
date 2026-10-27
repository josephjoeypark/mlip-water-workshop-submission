# Corrected 2 ns molecular dynamics

The four programs in `internal/` are copied byte for byte from the corrected
production campaign. `run_nvt_segment.py` is the validated October restart
correction, which restores the exact saved momenta and thermostat state.
The LES runners verify native single-cell strain and do not apply the
historical duplicate strain patch. Source hashes are in [the manifest](../manifest.json).

The exact 168 model/temperature/training-seed/velocity-seed identities are in
[`configs/lanes.csv`](../configs/lanes.csv). The initial 64-water input is
[`input/liquid-64.xyz`](input/liquid-64.xyz). Use this exact file: it has a
different byte hash from the older input elsewhere in the repository.

Install the [production dependencies](../environment/requirements-production.txt)
and obtain the [checksum-identified models](../models/manifest.json). Put them
under `reproducibility/models/` using the manifest filenames, or pass
`--models-dir`. The launcher selects the included frozen source snapshot.

From the repository root, print the commands for one lane:

```bash
python reproducibility/md/run_lane.py --lane 0 --stage gate
python reproducibility/md/run_lane.py --lane 0 --stage npt
python reproducibility/md/run_lane.py --lane 0 --stage analyze-npt
python reproducibility/md/run_lane.py --lane 0 --stage nvt --segment 1
python reproducibility/md/run_lane.py --lane 0 --stage nvt --segment 2
```

Add `--execute` within an appropriate compute allocation. Complete the gate,
NPT, and NPT analysis before starting NVT. Run NVT segments **1 through 16
in order**: 500 ps, followed by fifteen 100 ps continuations, totaling 2 ns.
Every continuation uses the preceding `final.traj` and `dynamics_state.npz`.
The settings are read from [`protocol_2ns.json`](../configs/protocol_2ns.json).
The default output is `runs/corrected-2ns-md/lane_NNN/`.

No simulation or scheduler submission is part of the compact figure rebuild.
Re-running MD requires the model binaries and compute resources. The bundled
CSV/NPZ results are sufficient for all included figures.

## Analyze trajectories

`analyze_npt.py` calculates density and RDF and chooses the representative-volume
frame for NVT initialization. Its launcher command above uses 500–1,000 ps.

[`analyze_transport.py`](analyze_transport.py) applies the frozen VACF/MSD
diffusion and five-channel Green–Kubo viscosity estimators with the completed
campaign's 16-segment discovery. It sets correlation stride to 1, keeps training
seeds separate, checks each segment's checkpoint identity and corrected LES
stress implementation, and supports all four archived analysis windows:

```bash
python reproducibility/md/analyze_transport.py --window all
# In a CPU allocation, once trajectories are available:
python reproducibility/md/analyze_transport.py --window all --execute
```

Pass `--runs-dir` to relocate the MD outputs and `--output` for a new analysis
directory. `--window` accepts `1000`, `1500`, `2000`, `1000:2000`, or `all`;
`--lane N` restricts work to selected lanes. The default is a dry run. Full
168-lane runs produce per-lane, per-seed, and 24-state summaries using the same
aggregation as [`build_results.py`](../scripts/build_results.py).

The historical estimator module in `internal/` is preserved unchanged. Its
standalone segment discovery stops at 1 ns and its historical summary does
not distinguish training seeds. Use the portable wrapper above for the 2 ns
study; it extends discovery to segment 16, starts the second nanosecond at
segment 7, and keeps the original estimators unchanged.

The released compact tables also include the frozen 1,000/1,500/2,000 ps
prefixes and disjoint 1,000–2,000 ps analysis. Figure reproduction reads these
native-cadence scalar estimates and their compact plot arrays directly.
