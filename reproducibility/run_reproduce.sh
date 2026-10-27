#!/usr/bin/env bash
set -euo pipefail
section_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -B "${section_dir}/scripts/reproduce.py" "$@"
