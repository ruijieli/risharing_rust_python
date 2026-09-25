#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
algorithm="${1:-}"
config="${2:-config.toml}"

cd "${project_dir}"
"${script_dir}/build-python-extension.sh"
args=(--config "${config}")
if [[ -n "${algorithm}" ]]; then
    args+=(--algorithm "${algorithm}")
fi
python -m training.evaluate "${args[@]}"
