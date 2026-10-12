"""Post-process a latexdiff output file: move \\path{...} out of \\DIFadd{} /
\\DIFdel{} (and the FL variants) so that long repository paths stay breakable.

ulem's \\uwave / \\sout set their argument word by word and cannot break inside
\\path{...}, which pushes long result paths past the right margin in the
marked-up supplement. The path keeps the change colour (blue/red) but is set
outside the wavy-underline/strike-out wrapper, where the url package can break
it at '/', '_' and '.' as in the clean build.
"""
import io, sys

WRAPPERS = ("\\DIFaddFL{", "\\DIFdelFL{", "\\DIFadd{", "\\DIFdel{")
PATHCMD = "\\path{"


def match_brace(s, i):
    """s[i] == '{'; return index of the matching '}' (handles nesting and \\{ \\})."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced braces at %d" % i)


def split_inner(inner, wrapper):
    color = "blue" if "add" in wrapper else "red"
    out, pos, changed = [], 0, False
    while True:
        k = inner.find(PATHCMD, pos)
        if k < 0:
            break
        end = match_brace(inner, k + len(PATHCMD) - 1)
        before = inner[pos:k]
        if before:
            out.append(wrapper + before + "}")
        out.append("{\\color{%s}%s}" % (color, inner[k:end + 1]))
        pos = end + 1
        changed = True
    if not changed:
        return None
    rest = inner[pos:]
    if rest:
        out.append(wrapper + rest + "}")
    return "".join(out)


def process(text):
    i, out, n = 0, [], 0
    while i < len(text):
        hit = None
        for w in WRAPPERS:
            if text.startswith(w, i):
                hit = w
                break
        if hit is None:
            out.append(text[i])
            i += 1
            continue
        open_brace = i + len(hit) - 1
        close = match_brace(text, open_brace)
        inner = text[open_brace + 1:close]
        if PATHCMD in inner:
            # process nested wrappers inside first (rare), then split
            replaced = split_inner(inner, hit)
            if replaced is not None:
                out.append(replaced)
                n += 1
                i = close + 1
                continue
        out.append(text[i:close + 1])
        i = close + 1
    return "".join(out), n


for p in sys.argv[1:]:
    t = io.open(p, encoding="utf-8").read()
    new, n = process(t)
    io.open(p, "w", encoding="utf-8", newline="\n").write(new)
    print(f"{p}: moved \\path out of {n} change wrappers")
