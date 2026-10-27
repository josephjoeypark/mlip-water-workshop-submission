#!/usr/bin/env python3
"""Validate released inputs and optionally compare a fresh numerical rebuild."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
FROZEN = RELEASE / 'results' / '2ns'
KEYS = ['backbone', 'condition', 'temperature_K']
METRICS = ['diffusion_vacf_pbc_nm2_ns', 'diffusion_vacf_corrected_nm2_ns',
           'diffusion_msd_pbc_nm2_ns', 'diffusion_msd_corrected_nm2_ns',
           'viscosity_mPa_s', 'viscosity_10_20ps_mPa_s', 'yeh_hummer_nm2_ns', 'density_g_per_cm3']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksums():
    count = 0
    for line in (RELEASE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = RELEASE / name
        require(path.is_file() and sha(path) == expected, 'Release checksum mismatch: ' + name)
        count += 1
    manifest = json.loads((RELEASE / 'manifest.json').read_text())
    for item in manifest['files']:
        require(sha(RELEASE / item['path']) == item['sha256'], 'Source manifest mismatch: ' + item['path'])
    for item in manifest['repository_dependencies']:
        require(sha(RELEASE.parent / item['path']) == item['sha256'], 'Repository dependency mismatch: ' + item['path'])
    return count


def validate_inputs():
    count = checksums()
    protocol = json.loads((RELEASE / 'configs/protocol_2ns.json').read_text())
    model_list = json.loads((RELEASE / 'models/manifest.json').read_text())['models']
    require(len(model_list) == 14, 'Expected 14 model identities')
    models = {m['fused_model_sha256']: m for m in model_list}
    require(len(models) == 14, 'Model hashes must be distinct')
    require(sum(protocol['nvt']['production_segments_ps']) == 2000, 'NVT duration mismatch')
    require(sha(RELEASE / 'md/input/liquid-64.xyz') == protocol['input_sha256'], 'MD input changed')
    configured = pd.read_csv(RELEASE / 'configs/lanes.csv').sort_values('lane_id').reset_index(drop=True)
    require(len(configured) == 168 and configured.lane_id.tolist() == list(range(168)), 'Lane manifest coverage')
    complete = pd.read_csv(FROZEN / 'data/trajectory_completion.csv').sort_values('lane_id').reset_index(drop=True)
    require(len(complete) == 168 and complete.lane_id.tolist() == list(range(168)), 'Completion coverage')
    require((complete.npt_ps == 1000).all() and (complete.nvt_ps == 2000).all(), 'Incomplete trajectory')
    require(complete.exact_final_restart_passed.astype(str).str.lower().eq('true').all(), 'Archived restart audit failed')
    for directory, start, end in [('prefix1000ps', 0, 1000), ('prefix1500ps', 0, 1500),
                                   ('prefix2000ps', 0, 2000), ('window1000_2000ps', 1000, 2000)]:
        frame = pd.read_csv(FROZEN / 'data' / directory / 'lanes.csv').sort_values('lane_id').reset_index(drop=True)
        require(len(frame) == 168 and frame.lane_id.tolist() == list(range(168)), directory + ': lane coverage')
        require(frame.status.eq('ok').all() and frame.reached_target_ps.astype(str).str.lower().eq('true').all(), directory + ': incomplete results')
        require(frame.analyzed_ps.eq(end - start).all(), directory + ': analyzed duration')
        require(frame.analysis_window_start_ps.eq(start).all() and frame.analysis_window_end_ps.eq(end).all(), directory + ': window mismatch')
        require(np.isfinite(frame[METRICS].to_numpy(float)).all(), directory + ': nonfinite values')
        require(not frame.duplicated(KEYS + ['training_seed', 'md_replica']).any(), directory + ': duplicate replicas')
        for key in KEYS + ['training_seed', 'md_replica', 'md_seed', 'model_sha256']:
            pd.testing.assert_series_equal(frame[key], configured[key], check_dtype=False, check_names=False)
        require(len(frame.groupby(KEYS)) == 24, directory + ': state coverage')
        for (backbone, condition, temp), group in frame.groupby(KEYS):
            require(temp in protocol['temperatures_K'], 'Unexpected temperature')
            require(set(group.md_replica) == {1, 2, 3}, 'MD replica identities')
            if condition == 'ZS':
                require(len(group) == 3 and group.training_seed.isna().all(), 'Baseline counted as a trained seed')
            else:
                require(len(group) == 9 and set(group.training_seed) == {17, 29, 43}, 'Training-seed coverage')
                require(group.groupby('training_seed').size().eq(3).all(), 'Replicas per training seed')
                require(group.optimizer_updates.eq(123200).all(), 'Checkpoint update mismatch')
            if backbone == 'LES':
                require(group.data_origin.eq('corrected').all(), 'Historical LES data in corrected results')
            for row in group.itertuples():
                model = models[row.model_sha256]
                require(model['backbone'] == row.backbone, 'Wrong model backbone')
                require(model['condition'] == row.paper_condition, 'Wrong model condition')
                expected_seed = model['training_seed']
                require(pd.isna(row.training_seed) if expected_seed is None else row.training_seed == expected_seed,
                        'Wrong checkpoint training seed')
        shift = 2.837297 * 1.380649e-23 * frame.mean_temperature_K / (6 * np.pi * frame.viscosity_mPa_s * 1e-3 * (frame.volume_A3 * 1e-30)**(1/3)) * 1e9
        np.testing.assert_allclose(frame.yeh_hummer_nm2_ns, shift, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(frame.diffusion_vacf_corrected_nm2_ns, frame.diffusion_vacf_pbc_nm2_ns + shift, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(complete.model_sha256, configured.model_sha256)
    from aggregate_rdf import aggregate_rdf
    rdf_replicas = pd.read_csv(FROZEN / 'data/rdf_295K_per_replica.csv')
    require(rdf_replicas.lane_id.nunique() == 42 and len(rdf_replicas) == 12600, 'RDF replica coverage')
    identities = rdf_replicas.drop_duplicates('lane_id').set_index('lane_id')
    expected_identities = configured[configured.temperature_K == 295].set_index('lane_id')
    for key in ['backbone','condition','training_seed','temperature_K','md_replica']:
        pd.testing.assert_series_equal(identities[key].sort_index(), expected_identities[key].sort_index(), check_dtype=False)
    rdf, rdf_seeds = aggregate_rdf(rdf_replicas)
    pd.testing.assert_frame_equal(rdf, pd.read_csv(FROZEN / 'data/rdf_295K.csv'), check_dtype=False, rtol=1e-10, atol=1e-12)
    molecular = pd.read_csv(FROZEN / 'data/cluster_rmse_all_seeds.csv')
    fitted = molecular[molecular.training_seed.notna()]
    require(set(fitted.training_seed) == {17, 29, 43}, 'Molecular training-seed coverage')
    require(set(fitted.cluster_group) == {'G1','G2','G3','G4','G5','G6'}, 'Molecular cluster-group coverage')
    require(fitted.groupby(['model','condition','training_seed']).size().eq(6).all(), 'Molecular groups per seed')
    for directory in ('curves', 'stress_diagnostics'):
        paths = sorted((FROZEN / 'data' / directory).glob('lane_*.npz'))
        require([p.name for p in paths] == [f'lane_{i:03d}.npz' for i in range(168)], directory + ': curve coverage')
        for path in paths:
            with np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    values = arrays[name]
                    if np.issubdtype(values.dtype, np.number):
                        require(np.isfinite(values).all(), str(path) + ': nonfinite ' + name)
    return dict(passed=True, checksummed_files=count, trajectories=168, model_identities=14,
                model_temperature_states=24, seed_temperature_states=56,
                analyzed_windows=4, finite_size_correction_checks=672,
                curve_files=336, rdf_replicas=42, rdf_model_seed_groups=14, historical_les_excluded=True,
                scope='Compact released data, identities, coverage, hashes, and lane-wise correction; no raw MD restart or native-curve integration was rerun.')


def compare_rebuild(output):
    compared = []
    # These independent frozen tables catch aggregation or estimator changes.
    for name in ('rdf_295K.csv', 'model_summary_2ns.csv', 'seed_means_and_md_sd.csv', 'finite_size_covariance.csv',
                 'density_reference_comparison.csv', 'density_four_temperature_errors.csv',
                 'molecular_multineighbor_per_seed_macro_rmse.csv', 'stress_decomposition_lanes.csv',
                 'block_length_statistics.csv', 'lane_lag_sensitivity.csv', 'model_lag_sensitivity.csv',
                 'transport_reference_errors.csv'):
        expected = pd.read_csv(FROZEN / 'data' / name)
        actual = pd.read_csv(output / 'data' / name)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
        compared.append(name)
    from pypdf import PdfReader
    captions = (FROZEN / 'FIGURE_CAPTIONS.txt').read_text().strip().split('\n\n')
    figure_names = ['figure1_structure','figure2_transport','figureA1_molecular','figureS1_raw_transport',
                    'figureS2_stress_short','figureS3_stress_tail','figureS4_green_kubo_50ps','figureS5_green_kubo_20ps',
                    'figureS6_vacf','figureS7_running_diffusion','figureS8_disjoint_viscosity','figureS9_disjoint_diffusion',
                    'figureS10_block_length','figureS11_prefix_length_295K']
    require(len(captions) == len(figure_names), 'Caption count mismatch')
    for name in figure_names:
        require(len(PdfReader(output / 'figures' / (name+'.pdf')).pages) == 1, 'Missing figure: ' + name)
        require((output / 'figures' / (name+'.png')).stat().st_size > 0, 'Missing preview: ' + name)
    require(len(PdfReader(output / 'figures/all_24_states.pdf').pages) == 24, 'State appendix page count')
    require(len(PdfReader(output / 'paper_figures.pdf').pages) == 3, 'Paper figure PDF page count')
    return dict(passed=True, numerical_tables_compared=compared, figure_pairs=14, state_appendix_pages=24,
                rtol=1e-10, atol=1e-12, scope='Freshly rebuilt compact-data outputs compared with frozen numerical results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='also validate this completed rebuild')
    args = parser.parse_args()
    report = validate_inputs()
    if args.output:
        report['rebuild'] = compare_rebuild(args.output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
