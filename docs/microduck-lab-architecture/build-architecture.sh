#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGRAM_DIR="$ROOT/diagrams"
CONFIG="$DIAGRAM_DIR/mermaid-config.json"
EN_SOURCE="$ROOT/microduck-lab-architecture.md"
ZH_SOURCE="$ROOT/microduck-lab-architecture.zh-CN.md"
EN_COVER="$ROOT/cover-en.tex"
ZH_COVER="$ROOT/cover-zh-CN.tex"
EN_PDF="$ROOT/microduck-lab-architecture.pdf"
ZH_PDF="$ROOT/microduck-lab-architecture.zh-CN.pdf"
DIAGRAMS=(system-context runtime-components training-flow runtime-sequence)

require() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Missing prerequisite: %s\n' "$1" >&2
    exit 1
  }
}

require mmdc
require pandoc
require xelatex
require sips

for input in "$CONFIG" "$EN_SOURCE" "$ZH_SOURCE" "$EN_COVER" "$ZH_COVER"; do
  [[ -f "$input" ]] || { printf 'Missing build input: %s\n' "$input" >&2; exit 1; }
done

render_diagram() {
  local source="$1"
  local stem="${source%.mmd}"
  local svg="$stem.svg"
  local png="$stem.png"
  local scale=1
  case "$(basename "$source")" in
    system-context*|runtime-components*) scale=4 ;;
    training-flow*) scale=2 ;;
    runtime-sequence*) scale=2 ;;
  esac

  printf 'Rendering %s\n' "$(basename "$source")"
  mmdc --input "$source" --output "$svg" \
    --configFile "$CONFIG" --backgroundColor transparent \
    --width 1900 --height 1350
  mmdc --input "$source" --output "$png" \
    --configFile "$CONFIG" --backgroundColor white \
    --width 1900 --height 1350 --scale "$scale"

  if grep -qi '<foreignObject' "$svg"; then
    printf 'SVG contains forbidden foreignObject: %s\n' "$svg" >&2
    exit 1
  fi
  local dimensions width
  dimensions="$(sips -g pixelWidth -g pixelHeight "$png" 2>/dev/null | awk '/pixelWidth:/{w=$2}/pixelHeight:/{h=$2}END{print w "x" h}')"
  width="${dimensions%x*}"
  [[ "$width" =~ ^[0-9]+$ && "$width" -ge 3000 ]] || {
    printf 'PNG width is below 3000 px: %s (%s)\n' "$png" "$dimensions" >&2
    exit 1
  }
  printf 'Created %s and %s (%s)\n' "$(basename "$svg")" "$(basename "$png")" "$dimensions"
}

for name in "${DIAGRAMS[@]}"; do
  for source in "$DIAGRAM_DIR/$name.mmd" "$DIAGRAM_DIR/$name.zh-CN.mmd"; do
    [[ -f "$source" ]] || { printf 'Missing diagram source: %s\n' "$source" >&2; exit 1; }
    render_diagram "$source"
  done
done

build_pdf() {
  local source="$1"
  local cover="$2"
  local output="$3"
  local header="$4"
  local mainfont="$5"

  printf 'Building %s\n' "$(basename "$output")"
  pandoc "$source" \
    --from=gfm+yaml_metadata_block \
    --to=pdf \
    --pdf-engine=xelatex \
    --toc --toc-depth=3 --number-sections \
    --shift-heading-level-by=-1 \
    --syntax-highlighting=tango \
    --resource-path="$ROOT:$DIAGRAM_DIR" \
    --include-in-header="$cover" \
    --metadata=author:'George Hu' \
    --metadata=date:'2026-09-04' \
    --variable=colorlinks=true \
    --variable=linkcolor=blue \
    --variable=urlcolor=blue \
    --variable=geometry:margin=0.78in \
    --variable=fontsize=11pt \
    --variable=documentclass=article \
    --variable=papersize=letter \
    --variable=linestretch=1.06 \
    --variable=graphics=true \
    --variable=mainfont:"$mainfont" \
    --variable=monofont:"$mainfont" \
    --variable=header-includes:"\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{$header}\\fancyhead[R]{2026-09-04}\\fancyfoot[C]{\\thepage}\\setlength{\\headheight}{14pt}\\usepackage{float}\\floatplacement{figure}{H}\\usepackage{longtable}\\usepackage{booktabs}\\usepackage{microtype}" \
    --output "$output"
  [[ -s "$output" ]] || { printf 'PDF was not created: %s\n' "$output" >&2; exit 1; }
}

build_pdf "$EN_SOURCE" "$EN_COVER" "$EN_PDF" 'Microduck Lab Architecture' 'Helvetica'
build_pdf "$ZH_SOURCE" "$ZH_COVER" "$ZH_PDF" 'Microduck Lab 架构设计' 'PingFang SC'

printf 'Built:\n  %s\n  %s\n' "$EN_PDF" "$ZH_PDF"
