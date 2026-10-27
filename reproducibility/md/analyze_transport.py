#!/usr/bin/env python3
"""Apply frozen native-cadence estimators to all 16 segments, retaining seeds."""
import argparse
import csv
from functools import partial
import json
import math
import os
from pathlib import Path
import sys

SECTION = Path(__file__).resolve().parents[1]
WINDOWS = {'1000': (0,1000), '1500': (0,1500), '2000': (0,2000), '1000:2000': (1000,2000)}


def discover(lane_root, target_ps, allow_partial=False, first_segment=1):
    """The completed campaign's 16-segment discovery with portable imports."""
    if allow_partial:
        raise ValueError('Corrected transport analysis requires committed segments')
    sources, accumulated = [], 0.0
    for segment in range(first_segment, 17):
        if accumulated >= target_ps - 1e-9:
            break
        expected = 500.0 if segment == 1 else 100.0
        source = historical.candidate_segment(lane_root, segment, expected, allow_partial=False)
        if source is None:
            break
        remaining = target_ps - accumulated
        if source.duration_ps > remaining:
            source = historical.SegmentSource(segment=segment, path=source.path,
                duration_ps=remaining, complete=True,
                velocity_frames=int(math.floor(remaining / historical.MD_DT_PS + 1e-8)) + 1,
                stress_frames=int(math.floor(remaining / historical.MD_DT_PS + 1e-8)) + 1,
                position_frames=int(math.floor(remaining / historical.POSITION_DT_PS + 1e-8)) + 1)
        sources.append(source)
        accumulated += source.duration_ps
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=SECTION.parent / 'runs/corrected-2ns-md')
    parser.add_argument('--output', type=Path, default=SECTION.parent / 'runs/transport-native-2ns')
    parser.add_argument('--window', choices=[*WINDOWS, 'all'], default='2000')
    parser.add_argument('--lane', type=int, action='append', choices=range(168), metavar='0..167')
    parser.add_argument('--execute', action='store_true', help='run raw-trajectory FFT analysis inside a CPU allocation')
    args = parser.parse_args()
    with (SECTION / 'configs/lanes.csv').open() as handle:
        lanes = list(csv.DictReader(handle))
    if args.lane:
        lanes = [row for row in lanes if int(row['lane_id']) in args.lane]
    windows = list(WINDOWS.values()) if args.window == 'all' else [WINDOWS[args.window]]
    print(json.dumps(dict(lanes=len(lanes), windows_ps=windows, runs_dir=str(args.runs_dir.resolve()),
                         output=str(args.output.resolve()), correlation_stride=1, execute=args.execute), indent=2))
    if not args.execute:
        return
    os.environ['MACE_WATER_CORRELATION_STRIDE'] = '1'
    sys.path.insert(0, str(SECTION / 'md/internal'))
    sys.path.insert(0, str(SECTION / 'scripts'))
    global historical
    import analyze_preliminary_nvt_dynamics as historical
    import numpy as np
    import pandas as pd
    from build_results import aggregate, METRICS
    for start, end in windows:
        name = f'prefix{end}ps' if start == 0 else f'window{start}_{end}ps'
        output = args.output.resolve() / name
        if output.exists():
            raise FileExistsError(f'{output}: choose a new output directory')
        output.mkdir(parents=True)
        historical.discover_segments = partial(discover, first_segment=1 if start == 0 else 7)
        records = []
        for original in lanes:
            row = dict(original)
            row['output_dir'] = str(args.runs_dir.resolve() / ('lane_' + row['lane_id'].zfill(3)))
            row['model_label'] = row['backbone'] + '-' + row['condition'] + ('' if row['condition']=='ZS' else '-B123200')
            sources = discover(Path(row['output_dir']), end-start, first_segment=1 if start==0 else 7)
            if sum(source.duration_ps for source in sources) != end-start:
                raise ValueError('Incomplete requested window for lane ' + row['lane_id'])
            for source in sources:
                summary = json.loads((Path(source.path)/'summary.json').read_text())
                if summary['model_sha256'] != row['model_sha256']:
                    raise ValueError('Checkpoint identity mismatch in ' + source.path)
                if row['backbone']=='LES' and summary['model_audit']['les_stress_patch'] != 'native_mace_les_single_cell_strain_v1':
                    raise ValueError('Uncorrected LES stress in ' + source.path)
            result = historical.analyze_lane(row, str(output), end-start, allow_partial=False)
            if result.get('status')!='ok' or not result.get('reached_target_ps') or result['analyzed_ps']!=end-start:
                raise ValueError(result.get('error', 'Incomplete result'))
            npt = json.loads((Path(row['output_dir'])/'npt/npt_analysis.json').read_text())
            result.update(training_seed=int(row['training_seed']) if row['training_seed'] else None,
                          model_sha256=row['model_sha256'], paper_condition=row['paper_condition'],
                          optimizer_updates=int(row['optimizer_updates']) if row['optimizer_updates'] else None,
                          analysis_window_start_ps=start,analysis_window_end_ps=end,
                          density_g_per_cm3=npt['density_mean_g_per_cm3'],data_origin='corrected',
                          stress_implementation='native_mace_les_single_cell_strain_v1' if row['backbone']=='LES' else None)
            result = {k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in result.items()}
            (output / f'lane_{int(row["lane_id"]):03d}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            records.append(result)
            print(f'Completed {name}, lane {row["lane_id"]}',flush=True)
        frame = pd.DataFrame(records).sort_values('lane_id')
        frame.to_csv(output/'lanes.csv',index=False)
        if len(frame)==168:
            aggregate(frame).to_csv(output/'model_summary.csv',index=False)
            seeds = frame.groupby(['backbone','condition','training_seed','temperature_K'],dropna=False)[METRICS].agg(['mean','std'])
            seeds.columns = ['_'.join(column) for column in seeds.columns]
            seeds.reset_index().to_csv(output/'seed_means_and_md_sd.csv',index=False)
        (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(complete=True,lanes=len(frame),window_ps=[start,end],correlation_stride=1))+'\n')


if __name__ == '__main__':
    main()
