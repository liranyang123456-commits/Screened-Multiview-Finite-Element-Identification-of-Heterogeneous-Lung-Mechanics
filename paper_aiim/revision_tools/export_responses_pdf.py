"""Export Response_Reviewer[1-4].md to PDF with an embedded Unicode TrueType font.

The base-14 Times fonts used previously cannot encode Greek letters or
typographic dashes, so reviewer quotes such as "nu = 0.45" were transliterated.
This exporter embeds Times New Roman (regular/bold/italic TrueType files from
C:\\Windows\\Fonts, matching the .docx body font) through PyMuPDF's TextWriter,
so Greek letters, en/em dashes, curly quotes and ellipses are rendered as
written. Only the Mathematical Alphanumeric Symbols block (U+1D400-U+1D7FF,
e.g. mathematical-italic nu/n inside reviewer quotes) is folded to the plain
letter with NFKC, because no system TrueType font covers that block.
"""
from pathlib import Path
import re
import unicodedata
import fitz

base = Path(__file__).resolve().parent
source = base / "responses"
target = base / "responses_pdf_final"
target.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_FILES = {
    "regular": FONT_DIR / "times.ttf",
    "bold": FONT_DIR / "timesbd.ttf",
    "italic": FONT_DIR / "timesi.ttf",
}
for path in FONT_FILES.values():
    if not path.is_file():
        raise FileNotFoundError(f"Unicode TrueType font not found: {path}")
FONTS = {key: fitz.Font(fontfile=str(path)) for key, path in FONT_FILES.items()}

PAGE_W, PAGE_H = 612.0, 792.0
MARGIN_X, MARGIN_Y = 60.0, 56.0
WIDTH = PAGE_W - 2 * MARGIN_X


def normalize(text: str) -> str:
    """Fold mathematical-alphanumeric letters (U+1D400-U+1D7FF) to plain letters."""
    out = []
    for ch in text:
        if 0x1D400 <= ord(ch) <= 0x1D7FF:
            out.append(unicodedata.normalize("NFKC", ch))
        else:
            out.append(ch)
    return "".join(out)


def check_glyphs(text: str, font: fitz.Font, where: str) -> None:
    missing = sorted({ch for ch in text if ord(ch) > 31 and not font.has_glyph(ord(ch))})
    if missing:
        raise ValueError(f"{where}: font {font.name} lacks glyphs for {[hex(ord(c)) for c in missing]}")


for md in sorted(source.glob("Response_Reviewer[1-4].md")) + sorted(source.glob("Response_to_Editor.md")):
    doc = fitz.open()
    state = {"page": doc.new_page(width=PAGE_W, height=PAGE_H), "y": MARGIN_Y}

    def new_page():
        state["page"] = doc.new_page(width=PAGE_W, height=PAGE_H)
        state["y"] = MARGIN_Y

    def draw(text, size=10.2, style="regular", color=(0.08, 0.08, 0.08), after=6, indent=0):
        font = FONTS[style]
        text = normalize(text)
        check_glyphs(text, font, md.name)
        words = text.split()
        if not words:
            state["y"] += after
            return
        max_width = WIDTH - indent
        lines, line = [], ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if font.text_length(candidate, fontsize=size) <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        line_height = size * 1.42
        required = len(lines) * line_height + after
        # Keep short paragraphs together; let long paragraphs flow across pages
        # (a paragraph taller than the text area would otherwise leave a page
        # almost empty).
        if state["y"] + required > PAGE_H - MARGIN_Y and required <= (PAGE_H - 2 * MARGIN_Y) * 0.6:
            new_page()
        writer = fitz.TextWriter(state["page"].rect)
        for ln in lines:
            if state["y"] + line_height > PAGE_H - MARGIN_Y:
                writer.write_text(state["page"], color=color)
                new_page()
                writer = fitz.TextWriter(state["page"].rect)
            writer.append((MARGIN_X + indent, state["y"] + size), ln, font=font, fontsize=size)
            state["y"] += line_height
        writer.write_text(state["page"], color=color)
        state["y"] += after

    for raw in md.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line == "---":
            state["y"] += 3
        elif line.startswith("# "):
            draw(line[2:], 16, "bold", (0.12, 0.25, 0.39), 11)
        elif line.startswith("## "):
            draw(line[3:], 12.2, "bold", (0.12, 0.25, 0.39), 7)
        elif line.startswith("### "):
            draw(line[4:], 11, "bold", (0.12, 0.25, 0.39), 6)
        elif line.startswith("> "):
            draw(line[2:].replace("**", ""), 9.7, "italic", (0.32, 0.32, 0.32), 8, 12)
        else:
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", line).replace("`", "")
            draw(cleaned)

    out = target / f"{md.stem}.pdf"
    doc.set_metadata({"title": md.stem.replace("_", " "), "author": "ARRAY-D-26-05373 revision response"})
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    check = fitz.open(out)
    assert check.page_count and check[0].get_text().strip(), f"Invalid PDF: {out}"
    print(f"{out}: {check.page_count} pages, {out.stat().st_size} bytes")
