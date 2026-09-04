#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_RESTART="$ROOT/.claude/skills/restart-servers/restart.sh"
VIEWER="$ROOT/duck-viewer"
VIEWER_LOG="${TMPDIR:-/tmp}/microduck-viewer.log"
VIEWER_PID="${TMPDIR:-/tmp}/microduck-viewer.pid"

[[ -x "$BACKEND_RESTART" ]] || {
  printf 'Missing backend restart script: %s\n' "$BACKEND_RESTART" >&2
  exit 1
}
[[ -f "$VIEWER/package.json" ]] || {
  printf 'Missing viewer package: %s\n' "$VIEWER/package.json" >&2
  exit 1
}

printf '[1/3] Restarting duck-lab with first-gait and alpha_walking...\n'
bash "$BACKEND_RESTART" --fresh \
  runs/first-gait \
  ../microduck/policies/alpha_walking.onnx

printf '[2/3] Installing viewer dependencies...\n'
npm --prefix "$VIEWER" install

printf '[3/3] Starting duck-viewer on http://127.0.0.1:63317 ...\n'
nohup npm --prefix "$VIEWER" run dev >"$VIEWER_LOG" 2>&1 &
echo "$!" >"$VIEWER_PID"

for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:63317/ >/dev/null 2>&1; then
    printf 'duck-lab:    http://127.0.0.1:8788\n'
    printf 'duck-viewer: http://127.0.0.1:63317\n'
    printf 'viewer log:  %s\n' "$VIEWER_LOG"
    exit 0
  fi
  sleep 1
done

printf 'duck-viewer failed to become ready. Last log lines:\n' >&2
tail -20 "$VIEWER_LOG" >&2 || true
exit 1
