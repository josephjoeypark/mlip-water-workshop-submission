"""Portable locations for the corrected 2 ns figure builders."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

RELEASE = Path(__file__).resolve().parents[1]
FROZEN = RELEASE / 'results' / '2ns'
OUT = Path(os.environ.get('WATER_REPRO_OUTPUT', RELEASE.parent / 'runs' / 'reproducibility-2ns')).resolve()
DATA = OUT / 'data'
FIG = OUT / 'figures'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
