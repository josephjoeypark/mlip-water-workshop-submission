#!/usr/bin/env python3
"""Print, or execute in an allocation, one corrected MD lane/stage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SECTION = Path(__file__).resolve().parents[1]
INTERNAL = SECTION / 'md/internal'


def commands(args):
    protocol = json.loads((SECTION / 'configs/protocol_2ns.json').read_text())
    with (SECTION / 'configs/lanes.csv').open() as handle:
        row = next(r for r in csv.DictReader(handle) if int(r['lane_id']) == args.lane)
    model = args.models_dir.resolve() / (row['model_id'] + '.model')
    root = args.runs_dir.resolve() / f'lane_{args.lane:03d}'
    label = row['backbone'] + '-' + row['condition'] + ('' if row['condition'] == 'ZS' else '-B123200')
    common = ['--model-label', label, '--model-kind', row['backbone'], '--model', str(model),
              '--model-sha256', row['model_sha256'], '--stored-precision', row['stored_precision'],
              '--temperature-K', row['temperature_K'], '--seed', row['md_seed'], '--device', args.device]
    if args.stage in ('gate', 'npt'):
        values = protocol['npt']
        command = [sys.executable, str(INTERNAL / 'run_density.py'), '--mode', 'gate' if args.stage == 'gate' else 'production'] + common
        command += ['--input', str(SECTION / 'md/input/liquid-64.xyz'), '--output-dir', str(root / args.stage)]
        for name in ('pressure_bar','timestep_fs','preparation_ps','production_ps','ttime_fs','tdamp_fs','pdamp_fs',
                     'tchain','pchain','tloop','ploop','thermo_every','trajectory_every','thermo_flush_every'):
            command += ['--'+name.replace('_','-'), str(values[name])]
    elif args.stage == 'analyze-npt':
        command = [sys.executable, str(INTERNAL / 'analyze_npt.py'), '--run-dir', str(root / 'npt'),
                   '--model-sha256', row['model_sha256'], '--temperature-K', row['temperature_K'],
                   '--analysis-start-ps', str(protocol['npt']['analysis_start_ps']), '--timestep-fs', '1',
                   '--trajectory-every', '100', '--rdf-stride', '10', '--blocks', '10']
    else:
        segment = args.segment
        values = protocol['nvt']
        previous = root / f'nvt_segment{segment-1}'
        source = root / 'npt/representative.traj' if segment == 1 else previous / 'final.traj'
        command = [sys.executable, str(INTERNAL / 'run_nvt_segment.py')] + common
        command += ['--input', str(source), '--output-dir', str(root / f'nvt_segment{segment}'),
                    '--segment', str(segment), '--protocol-id', protocol['schema_version'],
                    '--production-ps', str(values['production_segments_ps'][segment-1])]
        for name in ('timestep_fs','equilibration_ps','ttime_fs','position_every','velocity_every','stress_every','thermo_every'):
            command += ['--'+name.replace('_','-'), str(values[name])]
        if segment > 1:
            command += ['--resume-dynamics', str(previous / 'dynamics_state.npz')]
    return command, row, model, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', type=int, choices=range(168), required=True, metavar='0..167')
    parser.add_argument('--stage', choices=['gate','npt','analyze-npt','nvt'], required=True)
    parser.add_argument('--segment', type=int, choices=range(1,17), default=1, metavar='1..16')
    parser.add_argument('--models-dir', type=Path, default=SECTION / 'models')
    parser.add_argument('--runs-dir', type=Path, default=SECTION.parent / 'runs/corrected-2ns-md')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true', help='execute inside a suitable compute allocation')
    args = parser.parse_args()
    command, row, model, root = commands(args)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if args.stage != 'analyze-npt':
        if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != row['model_sha256']:
            parser.error('Supply the exact checkpoint identified in models/manifest.json.')
    if args.stage in ('npt', 'nvt'):
        gate = root / 'gate/gate.json'
        if not gate.is_file():
            parser.error('Run this lane with --stage gate --execute first.')
        audit = json.loads(gate.read_text())
        if audit.get('gate_pass') is not True or audit.get('model_sha256') != row['model_sha256']:
            parser.error('Checkpoint gate is failed or does not match this model.')
        if row['backbone'] == 'LES' and audit.get('model_audit', {}).get('les_stress_patch') != 'native_mace_les_single_cell_strain_v1':
            parser.error('Gate does not certify corrected native LES stress.')
    env = os.environ.copy()
    vendor = SECTION / 'md/vendor'
    env['PYTHONPATH'] = os.pathsep.join(str(vendor / name) for name in ('compat','mace','les'))
    env.update(MACE_WATER_CORRELATION_STRIDE='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
