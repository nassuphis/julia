#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input.pdf> <output.pdf>"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: file '$INPUT' not found."
    exit 1
fi


gs \
 -sDEVICE=pdfwrite \
 -dCompatibilityLevel=1.5 \
 -dPDFSETTINGS=/prepress \
 -dNOPAUSE -dQUIET -dBATCH \
 -sOutputFile="$OUTPUT" \
 "$INPUT"

