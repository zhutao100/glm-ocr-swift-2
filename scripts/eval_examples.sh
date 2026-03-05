#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

if [[ ! -f "tools/example_eval/pyproject.toml" ]]; then
  echo "tools/example_eval is missing. Initialize submodules:" >&2
  echo "  git submodule update --init --recursive tools/example_eval" >&2
  exit 2
fi

if [[ ! -d "examples/result" ]]; then
  echo "examples/result/ is missing. Generate it first:" >&2
  echo "  scripts/run_examples.sh" >&2
  exit 2
fi

exec uv run --project tools/example_eval example-eval evaluate --repo-root . "$@"
