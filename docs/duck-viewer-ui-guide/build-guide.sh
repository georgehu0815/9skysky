#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EN_SOURCE="$ROOT/duck-viewer-ui-guide.md"
ZH_SOURCE="$ROOT/duck-viewer-ui-guide.zh-CN.md"
DIAGRAM_CONFIG="$ROOT/diagrams/mermaid-config.json"
EN_DIAGRAM_SOURCE="$ROOT/diagrams/architecture.mmd"
ZH_DIAGRAM_SOURCE="$ROOT/diagrams/architecture.zh-CN.mmd"
EN_DIAGRAM_SVG="$ROOT/diagrams/architecture.svg"
ZH_DIAGRAM_SVG="$ROOT/diagrams/architecture.zh-CN.svg"
EN_DIAGRAM_PNG="$ROOT/diagrams/architecture.png"
ZH_DIAGRAM_PNG="$ROOT/diagrams/architecture.zh-CN.png"
EN_COVER="$ROOT/cover-en.tex"
ZH_COVER="$ROOT/cover-zh-CN.tex"
EN_OUTPUT="$ROOT/duck-viewer-ui-guide.pdf"
ZH_OUTPUT="$ROOT/duck-viewer-ui-guide.zh-CN.pdf"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Missing prerequisite: %s\n' "$1" >&2
    exit 1
  }
}

require mmdc
require pandoc
require xelatex

for input in "$EN_SOURCE" "$ZH_SOURCE" "$DIAGRAM_CONFIG" "$EN_DIAGRAM_SOURCE" "$ZH_DIAGRAM_SOURCE" "$EN_COVER" "$ZH_COVER"; do
  [[ -f "$input" ]] || { printf 'Missing build input: %s\n' "$input" >&2; exit 1; }
done

render_diagram() {
  local source="$1"
  local svg_output="$2"
  local png_output="$3"

  printf 'Rendering %s...\n' "$(basename "$svg_output")"
  mmdc --input "$source" --output "$svg_output" \
    --configFile "$DIAGRAM_CONFIG" \
    --backgroundColor transparent \
    --width 1800 --height 1500

  printf 'Rendering %s...\n' "$(basename "$png_output")"
  mmdc --input "$source" --output "$png_output" \
    --configFile "$DIAGRAM_CONFIG" \
    --backgroundColor white \
    --width 1800 --height 1500 --scale 3

  local dimensions
  dimensions="$(sips -g pixelWidth -g pixelHeight "$png_output" 2>/dev/null | awk '/pixelWidth:/{w=$2}/pixelHeight:/{h=$2}END{print w "x" h}')"
  local width="${dimensions%x*}"
  [[ "$width" =~ ^[0-9]+$ && "$width" -ge 3000 ]] || {
    printf 'Diagram PNG is below the required 3000 px width: %s (%s)\n' "$png_output" "$dimensions" >&2
    exit 1
  }
  printf 'Created %s (%s)\n' "$(basename "$png_output")" "$dimensions"
}

build_pdf() {
  local source="$1"
  local cover="$2"
  local output="$3"
  local header="$4"
  shift 4
  printf 'Building %s...\n' "$(basename "$output")"
  pandoc "$source" \
    --from=gfm+yaml_metadata_block \
    --to=pdf \
    --pdf-engine=xelatex \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --shift-heading-level-by=-1 \
    --syntax-highlighting=tango \
    --resource-path="$ROOT:$ROOT/images:$ROOT/diagrams" \
    --include-in-header="$cover" \
    --metadata=link-citations:true \
    --variable=colorlinks=true \
    --variable=linkcolor=blue \
    --variable=urlcolor=blue \
    --variable=geometry:margin=0.8in \
    --variable=fontsize=11pt \
    --variable=documentclass=article \
    --variable=papersize=letter \
    --variable=linestretch=1.08 \
    --variable=graphics=true \
    --variable=header-includes:"\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{$header}\\fancyhead[R]{2026-09-04}\\fancyfoot[C]{\\thepage}\\setlength{\\headheight}{14pt}\\usepackage{float}\\floatplacement{figure}{H}\\usepackage{longtable}\\usepackage{booktabs}" \
    "$@" \
    --output "$output"
  [[ -s "$output" ]] || { printf 'PDF was not created or is empty: %s\n' "$output" >&2; exit 1; }
}

render_diagram "$EN_DIAGRAM_SOURCE" "$EN_DIAGRAM_SVG" "$EN_DIAGRAM_PNG"
render_diagram "$ZH_DIAGRAM_SOURCE" "$ZH_DIAGRAM_SVG" "$ZH_DIAGRAM_PNG"
build_pdf "$EN_SOURCE" "$EN_COVER" "$EN_OUTPUT" 'Duck Viewer UI Guide'
build_pdf "$ZH_SOURCE" "$ZH_COVER" "$ZH_OUTPUT" 'Duck Viewer UI 操作指南' \
  --variable=mainfont:'PingFang SC'

printf 'Built:\n  %s\n  %s\n' "$EN_OUTPUT" "$ZH_OUTPUT"
