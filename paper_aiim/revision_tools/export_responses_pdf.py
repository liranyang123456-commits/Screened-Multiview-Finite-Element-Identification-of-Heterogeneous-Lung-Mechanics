"""Export Response_Reviewer[1-4].md and Response_to_Editor.md to PDF with an embedded Unicode TrueType font.

The base-14 Times fonts used previously cannot encode Greek letters or
typographic dashes, so reviewer quotes such as "nu = 0.45" were transliterated.
This exporter embeds Times New Roman (regular/bold/italic TrueType files from
C:\\Windows\\Fonts, matching the .docx body font) through PyMuPDF's TextWriter,
so Greek letters, en/em dashes, curly quotes and ellipses are rendered as
written. Only the Mathematical Alphanumeric Symbols block (U+1D400-U+1D7FF,
e.g. mathematical-italic nu/n inside reviewer quotes) is folded to the plain
letter with NFKC, because no system TrueType font covers that block.

Headings, the quoted-comment labels, and the labels "Author Response:" /
"Location of revisions:" are kept with the text that follows them: if fewer
than MIN_FOLLOW_LINES lines of the following paragraph would fit on the same
page, a new page is started before the heading or label. The paragraph that
directly follows such a heading or label may break across pages, keeping at
least MIN_FOLLOW_LINES lines with the label and MIN_CARRY_LINES on the next
page; a paragraph too short for that is kept whole together with its label.
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
BOTTOM = PAGE_H - MARGIN_Y
# Paragraphs up to this height are moved to the next page as a whole instead of
# being split (a paragraph taller than the text area would otherwise leave a
# page almost empty, so longer paragraphs flow across pages).
KEEP_TOGETHER = (PAGE_H - 2 * MARGIN_Y) * 0.6
MIN_FOLLOW_LINES = 3
MIN_CARRY_LINES = 2
SPACE = 3.0
HEAD_COLOR = (0.12, 0.25, 0.39)
QUOTE_COLOR = (0.32, 0.32, 0.32)
BODY_COLOR = (0.08, 0.08, 0.08)
LABELS = {"Author Response:", "Location of revisions:"}
QUOTE_LABELS = {"Reviewer's Comment:", "Editor's Comment:"}


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


def wrap(text: str, font: fitz.Font, size: float, max_width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = word if not line else f"{line} {word}"
        if font.text_length(candidate, fontsize=size) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def parse(md: Path) -> list:
    """One block per Markdown line; None marks a blank line or separator."""
    blocks = []
    for raw in md.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line == "---":
            blocks.append(None)
            continue
        if line.startswith("# "):
            text, size, style, color, after, indent, keep = line[2:], 16, "bold", HEAD_COLOR, 11, 0, True
        elif line.startswith("## "):
            text, size, style, color, after, indent, keep = line[3:], 12.2, "bold", HEAD_COLOR, 7, 0, True
        elif line.startswith("### "):
            text, size, style, color, after, indent, keep = line[4:], 11, "bold", HEAD_COLOR, 6, 0, True
        elif line.startswith("> "):
            text = line[2:].replace("**", "")
            size, style, color, after, indent = 9.7, "italic", QUOTE_COLOR, 8, 12
            keep = text.strip() in QUOTE_LABELS
        else:
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", line).replace("`", "")
            size, style, color, after, indent = 10.2, "regular", BODY_COLOR, 6, 0
            keep = text.strip() in LABELS
        font = FONTS[style]
        text = normalize(text)
        check_glyphs(text, font, md.name)
        lines = wrap(text, font, size, WIDTH - indent)
        if not lines:
            blocks.append(None)
            continue
        lh = size * 1.42
        blocks.append({"lines": lines, "size": size, "font": font, "color": color, "after": after,
                       "indent": indent, "keep": keep, "lh": lh, "full": len(lines) * lh + after})
    return blocks


def follow_need(blocks: list, i: int) -> float:
    """Space that must remain below block i so that it is not stranded at a page bottom."""
    gap, j = 0.0, i + 1
    while j < len(blocks) and blocks[j] is None:
        gap += SPACE
        j += 1
    if j >= len(blocks):
        return 0.0
    nxt = blocks[j]
    if nxt["keep"]:
        return gap + nxt["full"] + follow_need(blocks, j)
    if len(nxt["lines"]) < MIN_FOLLOW_LINES + MIN_CARRY_LINES:
        return gap + nxt["full"]
    return gap + MIN_FOLLOW_LINES * nxt["lh"]


def split_point(b: dict, y: float) -> int | None:
    """Line index at which a paragraph that follows a heading/label breaks to the next page."""
    n = len(b["lines"])
    if y + b["full"] <= BOTTOM:
        return None
    fit = int((BOTTOM - y) // b["lh"])
    if n - fit < MIN_CARRY_LINES:
        fit = n - MIN_CARRY_LINES
    return max(fit, 0)


for md in sorted(source.glob("Response_Reviewer[1-4].md")) + sorted(source.glob("Response_to_Editor.md")):
    doc = fitz.open()
    blocks = parse(md)
    state = {"page": doc.new_page(width=PAGE_W, height=PAGE_H), "y": MARGIN_Y, "drawn": False}

    def new_page():
        state["page"] = doc.new_page(width=PAGE_W, height=PAGE_H)
        state["y"] = MARGIN_Y
        state["drawn"] = False

    after_keep = False
    for i, b in enumerate(blocks):
        if b is None:
            state["y"] += SPACE
            continue
        split = None
        if state["drawn"]:
            if b["keep"]:
                if state["y"] + b["full"] + follow_need(blocks, i) > BOTTOM:
                    new_page()
            elif after_keep:
                split = split_point(b, state["y"])
            elif state["y"] + b["full"] > BOTTOM and b["full"] <= KEEP_TOGETHER:
                new_page()
        after_keep = b["keep"]
        writer = fitz.TextWriter(state["page"].rect)
        for k, ln in enumerate(b["lines"]):
            if state["y"] + b["lh"] > BOTTOM or (k and k == split):
                writer.write_text(state["page"], color=b["color"])
                new_page()
                writer = fitz.TextWriter(state["page"].rect)
            writer.append((MARGIN_X + b["indent"], state["y"] + b["size"]), ln, font=b["font"], fontsize=b["size"])
            state["y"] += b["lh"]
            state["drawn"] = True
        writer.write_text(state["page"], color=b["color"])
        state["y"] += b["after"]

    out = target / f"{md.stem}.pdf"
    doc.set_metadata({"title": md.stem.replace("_", " "), "author": "ARRAY-D-26-05373 revision response"})
    doc.subset_fonts()
    doc.save(out, garbage=4, deflate=True)
    check = fitz.open(out)
    assert check.page_count and check[0].get_text().strip(), f"Invalid PDF: {out}"
    print(f"{out}: {check.page_count} pages, {out.stat().st_size} bytes")
