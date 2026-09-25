#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "${project_dir}"

python_bin="$(command -v python)"
export PYO3_PYTHON="${python_bin}"
python_lib_dir="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
export DYLD_FALLBACK_LIBRARY_PATH="${python_lib_dir}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"

if [[ "$(uname -s)" == "Darwin" ]]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup"
fi
cargo build --release --lib --no-default-features --features python-extension
extension_suffix="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
cp "target/release/lib_rust_core.dylib" "_rust_core${extension_suffix}"
echo "Python Rust extension: ${project_dir}/_rust_core${extension_suffix}"
