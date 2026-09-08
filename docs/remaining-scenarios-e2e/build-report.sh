#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

cd "$REPO_ROOT"
python3 "$ROOT/generate-report.py" "$@"

cd "$ROOT"
pandoc REPORT.md --from=gfm --to=html5 --standalone \
  --toc --toc-depth=2 \
  --metadata title="Microduck Remaining Scenarios - End-to-End Evidence" \
  --css=print.css --output=REPORT.html
weasyprint REPORT.html REPORT.pdf
pdfinfo REPORT.pdf
