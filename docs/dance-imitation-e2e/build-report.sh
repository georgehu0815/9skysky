#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pandoc "$ROOT/REPORT.md" --from=gfm --to=html5 --standalone \
  --toc --toc-depth=2 --metadata title="Microduck Dance Imitation — End-to-End Evidence" \
  --css=print.css --output="$ROOT/REPORT.html"
weasyprint "$ROOT/REPORT.html" "$ROOT/REPORT.pdf"
pdfinfo "$ROOT/REPORT.pdf"
