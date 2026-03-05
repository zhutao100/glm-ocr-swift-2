#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate `examples/result/` by batch-running the Swift CLI against `examples/source/`.

Usage:
  scripts/run_examples.sh [options]

Options:
  -c, --configuration debug|release   SwiftPM build config (default: release)
  --source-dir <path>                Input dir (default: examples/source)
  --output-dir <path>                Output dir (default: examples/result)
EOF
}

config="release"
source_dir="examples/source"
output_dir="examples/result"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configuration)
      config="${2:-}"; shift 2 ;;
    --source-dir)
      source_dir="${2:-}"; shift 2 ;;
    --output-dir)
      output_dir="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$config" != "debug" && "$config" != "release" ]]; then
  echo "Invalid configuration: $config (expected 'debug' or 'release')" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

echo "swift build -c $config --product GlmOCRCLI"
bin_path="$(swift build -c "$config" --product GlmOCRCLI --show-bin-path)"

mlx_lib="$bin_path/mlx.metallib"
if [[ ! -f "$mlx_lib" ]]; then
  echo "mlx.metallib missing at: $mlx_lib"
  echo "scripts/build_mlx_metallib.sh -c $config"
  scripts/build_mlx_metallib.sh -c "$config"
fi

echo "$bin_path/GlmOCRCLI --source-dir $source_dir --output-dir $output_dir"
"$bin_path/GlmOCRCLI" \
  --source-dir "$source_dir" \
  --output-dir "$output_dir"

echo "OK: wrote $output_dir"
