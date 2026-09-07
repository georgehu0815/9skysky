#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
rlx_root="$(cd -- "$script_dir/.." && pwd -P)"
workspace_root="$(cd -- "$rlx_root/.." && pwd -P)"
python_bin="${MICRODUCK_PYTHON:-/usr/local/bin/python3.12}"
microduck_local="${MICRODUCK_LOCAL_DIR:-$workspace_root/microduck_local}"
checkpoint="$rlx_root/runs/dance/dance.safetensors"
seed=1
check_only=false

usage() {
  cat <<'USAGE'
Open an RLX MicroDuck dance checkpoint in the native MuJoCo viewer.

Usage:
  scripts/view-microduck-dance.sh [options]

Options:
  --checkpoint PATH  Checkpoint to load (default: runs/dance/dance.safetensors)
  --seed INTEGER     Deterministic viewer seed (default: 1)
  --check            Validate the environment without opening the viewer
  -h, --help         Show this help

Environment variables:
  MICRODUCK_PYTHON     Framework-linked Python 3.12 executable
  MICRODUCK_LOCAL_DIR  Path to the microduck_local checkout

Examples:
  scripts/view-microduck-dance.sh
  scripts/view-microduck-dance.sh --checkpoint runs/dance-smoke/dance.safetensors
  scripts/view-microduck-dance.sh --seed 7
USAGE
}

while (($# > 0)); do
  case "$1" in
    --checkpoint)
      (($# >= 2)) || { printf 'error: --checkpoint requires a path\n' >&2; exit 2; }
      checkpoint="$2"
      shift 2
      ;;
    --seed)
      (($# >= 2)) || { printf 'error: --seed requires an integer\n' >&2; exit 2; }
      seed="$2"
      shift 2
      ;;
    --check)
      check_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$seed" =~ ^[0-9]+$ ]] || { printf 'error: --seed must be a non-negative integer\n' >&2; exit 2; }

if [[ "$checkpoint" != /* ]]; then
  checkpoint="$rlx_root/$checkpoint"
fi

[[ "$(uname -s)" == "Darwin" ]] || {
  printf 'error: this launcher uses the macOS mjpython viewer\n' >&2
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  printf 'error: uv is not installed or is not on PATH\n' >&2
  exit 1
}
[[ -x "$python_bin" ]] || {
  printf 'error: framework-linked Python 3.12 is not executable: %s\n' "$python_bin" >&2
  exit 1
}
[[ -f "$rlx_root/examples/ppo_microduck_dance.py" ]] || {
  printf 'error: RLX dance example is missing under %s\n' "$rlx_root" >&2
  exit 1
}
[[ -f "$microduck_local/pyproject.toml" ]] || {
  printf 'error: microduck_local checkout not found: %s\n' "$microduck_local" >&2
  exit 1
}
[[ -f "$checkpoint" ]] || {
  printf 'error: checkpoint does not exist: %s\n' "$checkpoint" >&2
  printf 'train it first or pass --checkpoint PATH\n' >&2
  exit 1
}
[[ -f "$checkpoint.json" ]] || {
  printf 'error: checkpoint sidecar does not exist: %s.json\n' "$checkpoint" >&2
  exit 1
}

run_in_overlay() {
  env -u VIRTUAL_ENV UV_PYTHON_PREFERENCE=only-system \
    uv run --isolated --no-project \
      --python "$python_bin" \
      --with-editable "$rlx_root" \
      --with-editable "$microduck_local" \
      "$@"
}

cd -- "$rlx_root"

if $check_only; then
  run_in_overlay python -c \
    'import mlx, microduck_local, mujoco, onnxruntime, rlx, sys, sysconfig; assert sys.version_info[:2] == (3, 12); assert sysconfig.get_config_var("PYTHONFRAMEWORK"); print("viewer environment OK:", sys.executable)'
  run_in_overlay mjpython examples/ppo_microduck_dance.py view --help >/dev/null
  printf 'checkpoint OK: %s\n' "$checkpoint"
  printf 'mjpython launcher OK\n'
  exit 0
fi

printf 'Opening MicroDuck dance viewer\n'
printf 'checkpoint: %s\n' "$checkpoint"
printf 'seed: %s\n' "$seed"
printf 'Close the MuJoCo window or press Ctrl+C here to stop.\n'

run_in_overlay mjpython examples/ppo_microduck_dance.py view \
  --checkpoint "$checkpoint" \
  --seed "$seed"
