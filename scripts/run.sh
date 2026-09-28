#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
    echo "错误：当前环境找不到 Python。请先执行 conda activate RL。" >&2
    exit 1
fi

export PYO3_PYTHON="$(command -v python)"
python_lib_dir="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [[ -n "${python_lib_dir}" ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH="${python_lib_dir}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
fi
cd "${project_dir}"
echo "请使用 ./scripts/train.sh 或 ./scripts/evaluate.sh 运行项目。" >&2
exit 2
