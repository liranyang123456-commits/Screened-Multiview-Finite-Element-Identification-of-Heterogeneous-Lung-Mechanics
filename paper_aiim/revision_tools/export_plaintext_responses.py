"""Export each point-by-point response as ONE plain-text document for pasting.

Editorial Manager accepts a plain-text response per reviewer. This script converts
Response_Reviewer[1-4].md (and Response_to_Editor.md) into plain text: no Markdown
markup, one paragraph per line, blank lines between paragraphs, and only ASCII
characters (typographic quotes, dashes, Greek letters and math symbols are
transliterated) so that the text survives any web form. The wording is identical
to the uploaded Response PDFs.

Output: responses/plain_text/Response_Reviewer<n>_plain.txt (and Response_to_Editor_plain.txt)
"""
from pathlib import Path
import re
import unicodedata

base = Path(__file__).resolve().parent
source = base / "responses"
target = source / "plain_text"
target.mkdir(parents=True, exist_ok=True)

ASCII_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": " - ", "\u2212": "-", "\u2011": "-",
    "\u00d7": "x", "\u2264": "<=", "\u2265": ">=", "\u00b1": "+/-",
    "\u2192": "->", "\u2190": "<-", "\u2026": "...", "\u2248": "~",
    "\u2208": " in ", "\u00b0": " deg", "\u00a0": " ", "\u00ad": "",
    "\u03bd": "nu", "\u03c1": "rho", "\u03bc": "mu", "\u03bb": "lambda",
    "\u03c3": "sigma", "\u0394": "Delta", "\u03b4": "delta", "\u03b8": "theta",
    "\u03c0": "pi", "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
    "\u2032": "'", "\u00b2": "^2", "\u00b3": "^3", "\u221a": "sqrt",
}


def to_ascii(text: str) -> str:
    out = []
    for ch in text:
        if 0x1D400 <= ord(ch) <= 0x1D7FF:
            ch = unicodedata.normalize("NFKC", ch)
        out.append(ASCII_MAP.get(ch, ch))
    text = "".join(out)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if ord(ch) < 128)
    return re.sub(r"[ \t]+", " ", text)


def plain(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return text.strip()


def convert(md: Path) -> str:
    lines = md.read_text(encoding="utf-8-sig").splitlines()
    paras: list[str] = []
    quote: list[str] = []

    def flush_quote():
        if quote:
            paras.append("Reviewer's comment:")
            paras.append('"' + " ".join(quote) + '"')
            quote.clear()

    for raw in lines:
        line = raw.strip()
        if line.startswith(">"):
            body = plain(line.lstrip(">").strip())
            if body.lower().startswith(("reviewer's comment:", "editor's comment:")):
                continue
            if body:
                quote.append(body)
            continue
        flush_quote()
        if not line:
            continue
        if line == "---":
            paras.append("-" * 60)
            continue
        if line.startswith("# "):
            paras.append(plain(line[2:]).upper())
            continue
        if line.startswith("### "):
            paras.append(plain(line[4:]))
            continue
        if line.startswith("## "):
            paras.append(plain(line[3:]))
            continue
        if line.startswith("**Author Response:**"):
            paras.append("Response:")
            rest = plain(line.split("**Author Response:**", 1)[1])
            if rest:
                paras.append(rest)
            continue
        if line.startswith("**Location of revisions:**"):
            rest = plain(line.split("**Location of revisions:**", 1)[1])
            paras.append("Location of revisions:" + (" " + rest if rest else ""))
            continue
        cleaned = plain(line)
        if paras and paras[-1] == "Location of revisions:":
            paras[-1] = "Location of revisions: " + cleaned
        else:
            paras.append(cleaned)
    flush_quote()
    # drop a trailing separator
    while paras and set(paras[-1]) == {"-"}:
        paras.pop()
    return to_ascii("\n\n".join(paras)) + "\n"


report = []
for md in sorted(source.glob("Response_Reviewer[1-4].md")) + sorted(source.glob("Response_to_Editor.md")):
    text = convert(md)
    out = target / f"{md.stem}_plain.txt"
    out.write_text(text, encoding="utf-8", newline="\n")
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})
    report.append(f"{out.name}: {len(text):,} characters, {len(text.split()):,} words, non-ASCII left: {non_ascii or 'none'}")
print("\n".join(report))
