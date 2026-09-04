#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK="$SCRIPT_DIR/microduck-lab-student-book.md"
DIAGRAM_DIR="$SCRIPT_DIR/diagrams"
COVER_BACKGROUND="$SCRIPT_DIR/cover-background.png"
COVER_SVG="$SCRIPT_DIR/cover.svg"
COVER_PNG="$SCRIPT_DIR/cover.png"
PDF="$SCRIPT_DIR/microduck-lab-student-book.pdf"
DOCX="$SCRIPT_DIR/microduck-lab-student-book.docx"

fail() {
  printf 'build-book: %s\n' "$*" >&2
  exit 1
}

command -v pandoc >/dev/null 2>&1 || fail "pandoc is required. Install it and retry."
command -v xelatex >/dev/null 2>&1 || fail "xelatex is required for PDF output. Install a TeX distribution and retry."
command -v rsvg-convert >/dev/null 2>&1 || fail "rsvg-convert is required for cover and diagram conversion."
command -v unzip >/dev/null 2>&1 || fail "unzip is required for DOCX cover-page generation."
command -v zip >/dev/null 2>&1 || fail "zip is required for DOCX cover-page generation."
command -v perl >/dev/null 2>&1 || fail "perl is required for DOCX cover-page generation."
[[ -f "$BOOK" ]] || fail "book source not found: $BOOK"
[[ -f "$COVER_BACKGROUND" ]] || fail "cover screenshot not found: $COVER_BACKGROUND"
[[ -f "$COVER_SVG" ]] || fail "cover design not found: $COVER_SVG"

for name in system-architecture module-architecture learning-loop; do
  [[ -f "$DIAGRAM_DIR/$name.svg" ]] || fail "missing diagram: $DIAGRAM_DIR/$name.svg. Render the Mermaid sources first."
done

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/microduck-book.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

printf 'Rendering screenshot cover...\n'
rsvg-convert --width 1275 --height 1650 "$COVER_SVG" -o "$COVER_PNG"

cat >"$TMP_DIR/cover.tex" <<EOF_TEX
\usepackage{eso-pic}
\renewcommand{\maketitle}{%
  \newgeometry{margin=0pt}%
  \thispagestyle{empty}%
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \includegraphics[width=\paperwidth,height=\paperheight]{$COVER_PNG}%
    }%
  }%
  \null\clearpage%
  \restoregeometry%
}
EOF_TEX

printf 'Building PDF with screenshot cover and vector SVG diagrams...\n'
pandoc "$BOOK" \
  --from=markdown+yaml_metadata_block+tex_math_dollars \
  --toc --toc-depth=3 --number-sections \
  --syntax-highlighting=tango \
  --resource-path="$SCRIPT_DIR:$DIAGRAM_DIR" \
  --pdf-engine=xelatex \
  --include-in-header="$TMP_DIR/cover.tex" \
  -V geometry:margin=0.82in \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -o "$PDF"

printf 'Preparing cover and 600 DPI-class PNG diagrams for DOCX...\n'
for svg in "$DIAGRAM_DIR"/*.svg; do
  base="$(basename "${svg%.svg}")"
  rsvg-convert --zoom 6.25 --keep-aspect-ratio "$svg" -o "$TMP_DIR/$base.png"
done
mkdir -p "$TMP_DIR/diagrams"
cp "$TMP_DIR"/*.png "$TMP_DIR/diagrams/"
cp "$COVER_PNG" "$TMP_DIR/cover.png"
perl -0pe '
  s/^title:.*$/title: |\n  ![](cover.png){width=7in}/m;
  s/^author:.*\n//m;
  s/^classoption:.*\n//m;
  s#\(diagrams/([^)]+)\.svg\)#(diagrams/$1.png)#g;
' "$BOOK" >"$TMP_DIR/book-docx.md"

printf 'Building DOCX...\n'
pandoc "$TMP_DIR/book-docx.md" \
  --from=markdown+yaml_metadata_block+tex_math_dollars \
  --toc --toc-depth=3 --number-sections \
  --syntax-highlighting=tango \
  --resource-path="$TMP_DIR:$SCRIPT_DIR" \
  -V papersize=letter \
  -V geometry:margin=0.82in \
  -o "$DOCX"

printf 'Separating the DOCX cover from its table of contents...\n'
DOCX_PACKAGE_DIR="$TMP_DIR/docx-package"
mkdir -p "$DOCX_PACKAGE_DIR"
unzip -q "$DOCX" -d "$DOCX_PACKAGE_DIR"
perl -0pi -e '$count = s{(<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents")}{<w:p><w:r><w:br w:type="page" /></w:r></w:p>$1}; die "TOC marker not found\n" unless $count == 1' "$DOCX_PACKAGE_DIR/word/document.xml"
perl -0pi -e '$count = s{(<w:pStyle w:val="TOCHeading" />)}{$1<w:pageBreakBefore />}; die "TOC heading not found\n" unless $count == 1' "$DOCX_PACKAGE_DIR/word/document.xml"
perl -0pi -e 's{<dc:title>.*?</dc:title>}{<dc:title>Microduck Lab Student Book</dc:title>}; s{<dc:creator>.*?</dc:creator>}{<dc:creator>George Hu</dc:creator>}' "$DOCX_PACKAGE_DIR/docProps/core.xml"
rm "$DOCX"
pushd "$DOCX_PACKAGE_DIR" >/dev/null
zip -q -r "$DOCX" .
popd >/dev/null

printf 'Built:\n  %s\n  %s\n' "$PDF" "$DOCX"
