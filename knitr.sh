#!/usr/bin/env bash
# Knit an .Rnw document and compile it with LuaLaTeX on macOS,
# while fixing the DYLD environment so that numba/llvmlite work
# inside reticulate.

set -euo pipefail

# ---------------- 0. user input -------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <file.Rnw>" >&2
  exit 1
fi
RNW_FILE=$1
[[ -f $RNW_FILE ]] || { echo "Error: '$RNW_FILE' not found." >&2 ; exit 1; }

RNW_PATH=$(cd "$(dirname "$RNW_FILE")" && pwd)
RNW_BASE=$(basename "$RNW_FILE" .Rnw)
TEX_FILE="$RNW_PATH/$RNW_BASE.tex"

# --------------- 2. Knit (Rnw → tex) --------------
echo "Knitting $RNW_FILE → $TEX_FILE"
Rscript --vanilla -e "knitr::knit('$RNW_FILE', output='$TEX_FILE')"

# --------------- 3. LuaLaTeX (2 passes)------------
echo "Running LuaLaTeX (pass 1)…"
lualatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >/dev/null
echo "Running LuaLaTeX (pass 2)…"
lualatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >/dev/null

echo "Success – produced ${RNW_BASE}.pdf"
