#!/usr/bin/env python3
"""Rebuild the corrected 2 ns figures from portable compact inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify import FROZEN, RELEASE, compare_rebuild, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RELEASE.parent / 'runs/reproducibility-2ns')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == RELEASE or RELEASE in output.parents or output in RELEASE.parents:
        parser.error('Choose an output directory outside the released reproducibility section.')
    marker = output / '.reproducibility-output.json'
    if output.exists() and any(output.iterdir()) and not marker.exists():
        parser.error('Output is not empty and was not created by this script; choose a new directory.')
    report = validate_inputs()
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'schema_version': 'water-corrected-2ns-generated-v1'}) + '\n')
    shutil.copytree(FROZEN / 'data', output / 'data', dirs_exist_ok=True)
    (output / 'figures').mkdir(exist_ok=True)
    (output / 'validation').mkdir(exist_ok=True)
    # These are fresh compact-data checks, not copied claims about raw MD.
    for name in ('input_audit.json', 'stress_diagnostics.json'):
        (output / 'validation' / name).write_text(json.dumps(report, indent=2) + '\n')
    env = os.environ.copy()
    env.update(WATER_REPRO_OUTPUT=str(output), MPLCONFIGDIR=str(output / '.matplotlib'),
               MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for script in ('build_results.py', 'build_diagnostics.py', 'compare_transport_reference.py', 'build_report.py'):
        print('Running ' + script, flush=True)
        subprocess.run([sys.executable, '-B', str(RELEASE / 'scripts' / script)], env=env, check=True)
    comparison = compare_rebuild(output)
    (output / 'validation/portable_reproduction.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print('PASS: corrected 2 ns tables and all 14 figures reproduced in ' + str(output))


if __name__ == '__main__':
    main()
