#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/microduck-studio-dance-guide.md"
CSS="$ROOT/print.css"
HTML="$ROOT/microduck-studio-dance-guide.html"
OUTPUT="$ROOT/microduck-studio-dance-guide.pdf"
DIAGRAM_SOURCE="$ROOT/diagrams/dance-training-sequence.mmd"
DIAGRAM_OUTPUT="$ROOT/images/dance-training-sequence.png"
DIAGRAM_SVG="$ROOT/images/dance-training-sequence.svg"
MERMAID_CONFIG="$ROOT/diagrams/mermaid-config.json"

for tool in pandoc weasyprint mmdc; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'Missing prerequisite: %s\n' "$tool" >&2
    exit 1
  }
done

mmdc \
  --input "$DIAGRAM_SOURCE" \
  --output "$DIAGRAM_OUTPUT" \
  --configFile "$MERMAID_CONFIG" \
  --backgroundColor white \
  --width 3200 \
  --scale 2

mmdc \
  --input "$DIAGRAM_SOURCE" \
  --output "$DIAGRAM_SVG" \
  --configFile "$MERMAID_CONFIG" \
  --backgroundColor transparent \
  --width 3200

dimensions="$(sips -g pixelWidth -g pixelHeight "$DIAGRAM_OUTPUT" 2>/dev/null |
  awk '/pixelWidth:/{w=$2}/pixelHeight:/{h=$2}END{print w "x" h}')"
width="${dimensions%x*}"
[[ "$width" =~ ^[0-9]+$ && "$width" -ge 3000 ]] || {
  printf 'Sequence diagram is below 3000 px: %s\n' "$dimensions" >&2
  exit 1
}

pandoc "$SOURCE" \
  --from=gfm+yaml_metadata_block \
  --to=html5 \
  --standalone \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --css="$CSS" \
  --metadata title="Microduck Studio Dance Guide" \
  --output="$HTML"

weasyprint "$HTML" "$OUTPUT"

test -s "$OUTPUT"
printf 'Built %s with sequence diagram %s\n' "$OUTPUT" "$dimensions"
