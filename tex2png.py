#!/usr/bin/env python

import subprocess
import pathlib
import pyvips as vips


def latex_pdf_to_png(
    formula: str,
    out_png: str = "formula.png",
    border_pt: int = 2,
):
    """
    LaTeX math -> formula.tex / formula.pdf -> formula.png via pyvips.
    No thresholding, no bitdepth tricks, just raw rasterization.
    """
    cwd = pathlib.Path(".").resolve()
    tex_path = cwd / "formula.tex"
    pdf_path = cwd / "formula.pdf"

    # --- write LaTeX file ---
    template = r"""\documentclass[border=__BORDER__pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\begin{document}
\[
__FORMULA__
\]
\end{document}
"""
    tex_source = (
        template
        .replace("__BORDER__", str(border_pt))
        .replace("__FORMULA__", formula)
    )
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"wrote {tex_path}")

    # --- run pdflatex ---
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        print("pdflatex returned", proc.returncode)
        print(proc.stdout)

    if not pdf_path.exists():
        raise RuntimeError("pdflatex failed; no formula.pdf produced")

    print(f"wrote {pdf_path}")

    # --- load PDF with vips (first page) ---
    img = vips.Image.new_from_file(str(pdf_path), page=0,dpi=600)
    print(f"vips image: {img.width}x{img.height}, bands={img.bands}, fmt={img.format}")

    rgb = img.extract_band(0).bandjoin([
        img.extract_band(1),
        img.extract_band(2),
    ])

    # convert to grayscale
    gray = rgb.colourspace("b-w")

    # threshold at midrange
    mx = gray.max()
    thresh = mx / 2.0 if mx > 0 else 0
    bilevel = (gray > thresh).ifthenelse(255, 0).cast("uchar")

    print(f"vips image: {bilevel.width}x{bilevel.height}, bands={bilevel.bands}, fmt={bilevel.format}")
    # --- save as plain PNG ---
    bilevel.write_to_file(
        out_png,
        compression=1, effort=1, filter="none",
        interlace=False, strip=True, bitdepth=1,
    )

    print(f"wrote {out_png}")


if __name__ == "__main__":
    latex_pdf_to_png(
        r"\frac{(z^{3}+c)\,(z+5c)^{5}(z-3ic)^{5}}{"
        r"(z^{5}+(2.5+i)z^{4}+(1.5-i)z^{3}+(-0.5+4i)z^{2}+z-1+3i)\,(z^{3}-5c)^{15}}",
        out_png="formula.png",
    )
