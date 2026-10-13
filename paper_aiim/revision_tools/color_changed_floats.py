"""Color new or changed tables and equations blue in a latexdiff output.

latexdiff is run with tabular/longtable treated as pictures and with math markup
off, so a changed table body or a new equation appears in its revised form without
colour. This post-processor compares the tables and display equations of the
revised source with those of the submitted source and inserts \\color{blue} at the
start of every table float, every centred tabular outside a float, and every
display equation whose content is new or changed. Unchanged ones stay black.
Commented-out (deleted) material is left untouched.

usage: color_changed_floats.py DIFF_TEX NEW_FLAT_TEX BASELINE_TEX
"""
import re
import sys
from pathlib import Path

diff_path, new_path, base_path = (Path(a) for a in sys.argv[1:4])


def inline_inputs(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    def repl(m):
        name = m.group(1)
        target = path.parent / (name if name.endswith(".tex") else name + ".tex")
        return target.read_text(encoding="utf-8") if target.exists() else m.group(0)

    return re.sub(r"\\input\{([^}]*)\}", repl, text)


diff = diff_path.read_text(encoding="utf-8")
new = inline_inputs(new_path)
base = inline_inputs(base_path)


def commented(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    return "%" in text[line_start:pos]


def blocks(text: str, begin_pat: str, end_pat: str):
    """Yield (body_start, body_end, env_start, env_end) for uncommented environments."""
    ends = [m for m in re.finditer(end_pat, text) if not commented(text, m.start())]
    for b in re.finditer(begin_pat, text):
        if commented(text, b.start()):
            continue
        e = next((e for e in ends if e.start() > b.end()), None)
        if e is not None:
            yield b.end(), e.start(), b.start(), e.end()


TABLE = (r"\\begin\{table\*?\}(\[[^\]]*\])?", r"\\end\{table\*?\}")
TABULAR = (r"\\begin\{tabularx?\}", r"\\end\{tabularx?\}")
EQUATION = (r"\\begin\{(?:equation|align|gather|multline)\*?\}", r"\\end\{(?:equation|align|gather|multline)\*?\}")


def norm(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace(r"\ifrevision\color{blue}\fi", "")
    s = re.sub(r"\\rev\{", "{", s)
    s = re.sub(r"\\DIF(add|del)(begin|end)(FL)?", "", s)
    s = re.sub(r"[{}\s]", "", s)
    return s


def key_of(body: str) -> str:
    m = re.search(r"\\label\{([^}]*)\}", body)
    if m:
        return "label:" + m.group(1)
    m = re.search(r"\\caption\{(.{0,80})", body, re.S)
    return "caption:" + norm(m.group(1)) if m else "body:" + norm(body)[:160]


base_tables = {key_of(base[a:b]): norm(base[a:b]) for a, b, *_ in blocks(base, *TABLE)}
changed_tables = {key_of(new[a:b]) for a, b, *_ in blocks(new, *TABLE)
                  if base_tables.get(key_of(new[a:b])) != norm(new[a:b])}
base_tabulars = {norm(base[a:b]) for a, b, *_ in blocks(base, *TABULAR)}
base_eqs = {norm(base[a:b]) for a, b, *_ in blocks(base, *EQUATION)}
new_eqs = {norm(new[a:b]) for a, b, *_ in blocks(new, *EQUATION)} - base_eqs

inserts = []  # (position, text)
float_spans = []
n_tab = n_ctr = n_eq = 0
for a, b, start, _ in blocks(diff, *TABLE):
    float_spans.append((start, b))
    if key_of(diff[a:b]) in changed_tables:
        inserts.append((a, "\\color{blue}"))
        n_tab += 1
for a, b, start, _ in blocks(diff, *TABULAR):
    if any(s <= start < e for s, e in float_spans):
        continue
    before = diff[:start]
    in_center = before.rfind(r"\begin{center}") > before.rfind(r"\end{center}")
    if in_center and norm(diff[a:b]) not in base_tabulars:
        inserts.append((start, "\\color{blue}"))
        n_ctr += 1
for a, b, start, end in blocks(diff, *EQUATION):
    if norm(diff[a:b]) in new_eqs:
        inserts.append((end, "\\endgroup"))
        inserts.append((start, "\\begingroup\\color{blue}"))
        n_eq += 1
for p, text in sorted(inserts, key=lambda t: t[0], reverse=True):
    diff = diff[:p] + text + diff[p:]
diff_path.write_text(diff, encoding="utf-8")
print(f"{diff_path.name}: blue table floats {n_tab}/{len(base_tables)} baseline, "
      f"{len(changed_tables)} changed in revision; blue centred tabulars {n_ctr}; blue equations {n_eq}")
