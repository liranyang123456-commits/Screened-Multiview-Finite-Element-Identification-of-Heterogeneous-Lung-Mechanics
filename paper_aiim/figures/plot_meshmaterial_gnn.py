# -*- coding: utf-8 -*-
"""
MeshMaterialGNN (+ screened FEM) — PlotNeuralNet diagram.

Code reference: models/lung_mesh_material_gnn.py
Library: PlotNeuralNet (https://github.com/HarisIqbal88/PlotNeuralNet); set PLOTNEURALNET_DIR

Usage (from this directory):
    python plot_meshmaterial_gnn.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PLOTNEURALNET_DIR = Path(os.environ.get("PLOTNEURALNET_DIR", "external_tools/PlotNeuralNet-master"))
sys.path.insert(0, str(PLOTNEURALNET_DIR))

from pycore.tikzeng import (  # noqa: E402
    to_Conv,
    to_Pool,
    to_SoftMax,
    to_Sum,
    to_begin,
    to_connection,
    to_cor,
    to_end,
    to_generate,
    to_head,
    to_skip,
)


def build_arch() -> list[str]:
    """Compact single-row schematic with parallel streams and parallel heads."""
    h, d = 14, 12
    # ~1.5× former inter-module gap (0.70 → 1.05).
    gap = "1.05,0,0"
    gap_fuse = "1.08,0,0"
    # Clear gap between the two plates inside Enc / MP×4.
    gap_sub = "0.85,0,0"
    return [
        to_head(str(PLOTNEURALNET_DIR).replace("\\", "/")),
        to_cor(),
        to_begin(),
        # Dynamic temporal stream (parallel to static; no Enc→Dyn edge).
        to_Conv(
            "dyn",
            "7",
            14,
            offset="(0,0,0)",
            to="(0,0,0)",
            height=h,
            depth=d,
            width=1.05,
            caption="Dyn",
        ),
        to_Conv(
            "tok",
            "7",
            128,
            offset=f"({gap})",
            to="(dyn-east)",
            height=h,
            depth=d,
            width=0.95,
            caption="Tok",
        ),
        to_connection("dyn", "tok"),
        to_Conv(
            "tr",
            "7",
            128,
            offset=f"({gap})",
            to="(tok-east)",
            height=h,
            depth=d,
            width=1.15,
            caption="Trans$\\times$1",
        ),
        to_connection("tok", "tr"),
        to_Pool(
            "vatt",
            offset=f"({gap})",
            to="(tr-east)",
            height=7,
            depth=7,
            width=1.0,
            opacity=0.55,
            caption="Attn",
        ),
        to_connection("tr", "vatt"),
        # Static encoder: two plates with a small internal gap.
        to_Conv(
            "static",
            "N",
            5,
            offset=f"({gap})",
            to="(vatt-east)",
            height=h,
            depth=d,
            width=1.05,
            caption="Static $x$",
        ),
        to_Conv(
            "enca",
            "N",
            128,
            offset=f"({gap})",
            to="(static-east)",
            height=h,
            depth=d,
            width=1.0,
            caption=" ",
        ),
        to_connection("static", "enca"),
        to_Conv(
            "encb",
            "N",
            128,
            offset=f"({gap_sub})",
            to="(enca-east)",
            height=h,
            depth=d,
            width=1.0,
            caption="Enc",
        ),
        to_connection("enca", "encb"),
        # Residual fuse: Attn (skip over Static/Enc) + Enc.
        to_Sum("fuse", offset=f"({gap_fuse})", to="(encb-east)", radius=1.2, opacity=0.65),
        to_connection("encb", "fuse"),
        to_skip("vatt", "fuse", pos=1.15),
        # Message passing: two plates with a small internal gap (label MP×4).
        to_Conv(
            "mpa",
            "N",
            128,
            offset=f"({gap_fuse})",
            to="(fuse-east)",
            height=h,
            depth=d,
            width=1.1,
            caption=" ",
        ),
        to_connection("fuse", "mpa"),
        to_Conv(
            "mpb",
            "N",
            128,
            offset=f"({gap_sub})",
            to="(mpa-east)",
            height=h,
            depth=d,
            width=1.1,
            caption="MP$\\times$4",
        ),
        to_connection("mpa", "mpb"),
        to_Pool(
            "pool",
            offset=f"({gap})",
            to="(mpb-east)",
            height=7,
            depth=6,
            width=0.95,
            opacity=0.55,
            caption="Pool",
        ),
        to_connection("mpb", "pool"),
        to_Conv(
            "pp",
            1,
            128,
            offset=f"({gap})",
            to="(pool-east)",
            height=7,
            depth=6,
            width=0.95,
            caption="Proj",
        ),
        to_connection("pool", "pp"),
        # Parallel heads (minimal vertical fork; no Global→Node edge).
        to_SoftMax(
            "ghead",
            r"$\theta_g$",
            offset="(1.05,2.0,0)",
            to="(pp-east)",
            width=1.05,
            height=1.9,
            depth=7,
            caption="Global",
        ),
        to_SoftMax(
            "nhead",
            r"$\theta_n$",
            offset="(1.05,-2.0,0)",
            to="(pp-east)",
            width=1.05,
            height=1.9,
            depth=7,
            caption="Node",
        ),
        r"\draw [connection] (pp-east) -- node {\midarrow} (ghead-west);",
        # Node reads post-MP states (not the pooled global vector).
        r"\draw [copyconnection] (mpb-east) -- ++(0,-1.35,0) -| node {\copymidarrow} (nhead-west);",
        # Prior merges both heads, then screened FEM.
        to_Conv(
            "prior",
            1,
            8,
            offset="(1.55,-2.0,0)",
            to="(ghead-east)",
            height=7,
            depth=6,
            width=1.05,
            caption="$\\theta_0{+}R$",
        ),
        r"\draw [connection] (ghead-east) -- node {\midarrow} (prior-west);",
        r"\draw [connection] (nhead-east) -| node {\midarrow} (prior-south);",
        to_SoftMax(
            "fem",
            r"$\mathcal{F}$",
            offset=f"({gap})",
            to="(prior-east)",
            width=0.95,
            height=1.9,
            depth=6,
            caption="FEM",
        ),
        to_connection("prior", "fem"),
        to_SoftMax(
            "mapj",
            r"$\mathcal{J}$",
            offset=f"({gap})",
            to="(fem-east)",
            width=0.95,
            height=1.9,
            depth=6,
            caption="MAP",
        ),
        to_connection("fem", "mapj"),
        to_Pool(
            "delta",
            offset=f"({gap})",
            to="(mapj-east)",
            height=7,
            depth=6,
            width=0.95,
            opacity=0.6,
            caption="$\\Delta$",
        ),
        to_connection("mapj", "delta"),
        to_SoftMax(
            "dec",
            "out",
            offset=f"({gap})",
            to="(delta-east)",
            width=1.05,
            height=1.9,
            depth=7,
            caption="Retain",
        ),
        to_connection("delta", "dec"),
        # The diagram is ~9.3 in wide and is printed at \linewidth (6.5-6.9 in),
        # i.e. scaled by ~0.7. With the 12pt base class \small is 11 pt in the
        # source, so every legend/label line stays >= 7 pt on the page.
        r"""
\node[anchor=south west, align=left, font=\small\bfseries, yshift=3pt]
at (current bounding box.north west) {
MeshMaterialGNN with screened FEM
};
% Legend (block meanings only; no caption-style prose).
\node[anchor=north west, align=left, font=\small, inner sep=4pt,
  draw=black!55, fill=white, rounded corners=1pt, text width=23.1cm]
at ([yshift=-4pt]current bounding box.south west) {
\textbf{Legend}\quad
\fcolorbox{black}{\ConvColor}{\rule{0pt}{5pt}\rule{5pt}{0pt}} Feature/MLP (incl.\ Enc, MP plates) \quad
\fcolorbox{black}{\PoolColor}{\rule{0pt}{5pt}\rule{5pt}{0pt}} Pool/Attn/Screen \quad
\fcolorbox{black}{\SoftmaxColor}{\rule{0pt}{5pt}\rule{5pt}{0pt}} Heads/FEM/Decision \quad
\textcolor{rgb:blue,5;green,15}{$\boldsymbol{+}$} residual fuse\\[0.3em]
\textbf{Flow:} Static$\|$Dyn $\oplus$ MP$\times$4;\;
Pool$\to$Global;\; MP$\to$Node (parallel);\;
$\theta_0{+}R\to\mathcal{F}$ (FEM) $\to\mathcal{J}$ (MAP objective) $\to\Delta\to$Retain.\\[0.2em]
\textbf{Blocks:} Enc $=$ Lin+SiLU+LN;\;
Attn $=$ view/load;\;
Pool $=$ mean$\|$max.\quad
\textbf{Outputs:} Global $\log E_b,\log\rho,\mathbf{c},r$;\;
Node $\log E_i,s_i$, partition.
};
""",
        to_end(),
    ]


def compile_tex(tex_path: Path) -> Path:
    layers = (PLOTNEURALNET_DIR / "layers").resolve()
    env = os.environ.copy()
    env["TEXINPUTS"] = str(layers) + "//;" + env.get("TEXINPUTS", "")
    # amsmath for \xrightarrow in the legend.
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] if proc.stdout else "")
        sys.stderr.write(proc.stderr[-2000:] if proc.stderr else "")
        raise RuntimeError(f"pdflatex failed for {tex_path}")
    return tex_path.with_suffix(".pdf")


def export_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(png_path))
        doc.close()
        image = Image.open(png_path)
        image.save(png_path, dpi=(dpi, dpi))
        return
    except Exception as exc:  # noqa: BLE001
        print(f"PyMuPDF PNG export failed ({exc}); trying magick/pdftoppm.")

    for cmd in (
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        ["magick", "-density", str(dpi), str(pdf_path), str(png_path)],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and png_path.exists():
                return
        except FileNotFoundError:
            continue
    raise RuntimeError(f"Could not export PNG for {pdf_path}")


def main() -> Path:
    out_tex = CURRENT_DIR / "meshmaterial_gnn.tex"
    # Inject amsmath after documentclass for \xrightarrow in the legend.
    sink = open(os.devnull, "w", encoding="utf-8")
    old_out, sys.stdout = sys.stdout, sink
    try:
        to_generate(build_arch(), str(out_tex))
    finally:
        sys.stdout = old_out
        sink.close()

    text = out_tex.read_text(encoding="utf-8")
    # 12pt base: the Box.sty labels use \small / normalsize, so after the
    # ~0.7x scaling to \linewidth they print at >= 7 pt (10pt base gave ~5 pt).
    text = text.replace(
        r"\documentclass[border=8pt, multi, tikz]{standalone}",
        r"\documentclass[12pt, border=8pt, multi, tikz]{standalone}",
        1,
    )
    if r"\usepackage{amsmath}" not in text:
        text = text.replace(
            r"\documentclass[12pt, border=8pt, multi, tikz]{standalone}",
            r"\documentclass[12pt, border=8pt, multi, tikz]{standalone}"
            + "\n"
            + r"\usepackage{amsmath}",
            1,
        )
    out_tex.write_text(text, encoding="utf-8")
    print("Wrote:", out_tex)

    pdf = compile_tex(out_tex)
    print("Compiled:", pdf)

    target_pdf = CURRENT_DIR / "fig_network_fem.pdf"
    target_png = CURRENT_DIR / "fig_network_fem.png"
    shutil.copy2(pdf, target_pdf)
    export_png(target_pdf, target_png)
    print("Copied:", target_pdf)
    print("Wrote:", target_png)

    for suffix in (".aux", ".log", ".vscodeLog"):
        aux = out_tex.with_suffix(suffix)
        if aux.exists():
            aux.unlink()
    return target_pdf


if __name__ == "__main__":
    main()
