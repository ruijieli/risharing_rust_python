#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
algorithm="${1:-ppo}"
config_path="${2:-config.toml}"

cd "${project_dir}"
export PYO3_PYTHON="$(command -v python)"
"${script_dir}/build-python-extension.sh"
python -m training.train --config "${config_path}" --algorithm "${algorithm}"
