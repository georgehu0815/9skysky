#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/microduck-studio-experiments-guide.md"
CSS="$ROOT/print.css"
HTML="$ROOT/microduck-studio-experiments-guide.html"
OUTPUT="$ROOT/microduck-studio-experiments-guide.pdf"

for tool in pandoc weasyprint; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'Missing prerequisite: %s\n' "$tool" >&2
    exit 1
  }
done

pandoc "$SOURCE" \
  --from=gfm+yaml_metadata_block \
  --to=html5 \
  --standalone \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --css="$CSS" \
  --metadata title="Microduck Studio Experiments Guide" \
  --output="$HTML"

weasyprint "$HTML" "$OUTPUT"

test -s "$OUTPUT"
printf 'Built %s\n' "$OUTPUT"
