#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
algorithm="${1:-}"
config_path="${2:-config.toml}"

cd "${project_dir}"
export PYO3_PYTHON="$(command -v python)"
"${script_dir}/build-python-extension.sh"
args=(--config "${config_path}")
if [[ -n "${algorithm}" ]]; then
    args+=(--algorithm "${algorithm}")
fi
python -m training.train "${args[@]}"
