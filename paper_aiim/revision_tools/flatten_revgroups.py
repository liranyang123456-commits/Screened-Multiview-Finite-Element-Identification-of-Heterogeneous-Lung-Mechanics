"""Remove the {\\ifrevision\\color{blue}\\fi ... } grouping wrappers (and bare
\\ifrevision\\color{blue}\\fi lines) from a copy of a .tex file so that latexdiff
does not split groups across change chunks. Content is unchanged (\\revisionfalse)."""
import sys, io
src, dst = sys.argv[1], sys.argv[2]
txt = io.open(src, encoding='utf-8').read()
OPEN = '{\\ifrevision\\color{blue}\\fi'
BARE = '\\ifrevision\\color{blue}\\fi'
out = []
i = 0
n = len(txt)
removed_groups = 0
removed_bare = 0
# We operate on the raw string. Find each OPEN that starts a line.
def line_start(pos):
    return pos == 0 or txt[pos-1] == '\n'
# collect positions of openers that begin a line and are followed by newline
openers = []
pos = 0
while True:
    k = txt.find(OPEN, pos)
    if k < 0: break
    end = k + len(OPEN)
    if line_start(k) and (end >= n or txt[end] in '\r\n'):
        openers.append(k)
    pos = end
# find matching close brace for each opener
closers = []
for k in openers:
    depth = 0
    j = k
    while j < n:
        c = txt[j]
        if c == '\\':
            j += 2; continue
        if c == '%':
            # skip comment to end of line
            while j < n and txt[j] != '\n': j += 1
            continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                closers.append(j); break
        j += 1
    else:
        raise SystemExit('unmatched group at %d' % k)
remove = set()
for k, j in zip(openers, closers):
    # remove opener line (including its newline)
    e = k + len(OPEN)
    while e < n and txt[e] in '\r\n':
        e += 1
        if txt[e-1] == '\n': break
    for p in range(k, e): remove.add(p)
    remove.add(j)
    # if the closer is alone on its line, also remove the newline
    ls = txt.rfind('\n', 0, j) + 1
    le = txt.find('\n', j)
    if le < 0: le = n
    if txt[ls:j].strip() == '' and txt[j+1:le].strip() == '':
        for p in range(ls, le+1 if le < n else le): remove.add(p)
    removed_groups += 1
# bare lines
pos = 0
while True:
    k = txt.find(BARE, pos)
    if k < 0: break
    e = k + len(BARE)
    if line_start(k) and (e >= n or txt[e] in '\r\n'):
        ee = e
        while ee < n and txt[ee] in '\r\n':
            ee += 1
            if txt[ee-1] == '\n': break
        for p in range(k, ee): remove.add(p)
        removed_bare += 1
    pos = e
res = ''.join(c for idx, c in enumerate(txt) if idx not in remove)
# unwrap \rev{...} -> ... (content unchanged; avoids \ifrevision inside ulem's \uwave in the diff)
removed_rev = 0
while True:
    k = res.find('\\rev{')
    if k < 0: break
    # make sure it's not \revision...
    j = k + len('\\rev{')
    depth = 1
    while j < len(res) and depth > 0:
        c = res[j]
        if c == '\\': j += 2; continue
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        j += 1
    inner = res[k+len('\\rev{'):j-1]
    res = res[:k] + inner + res[j:]
    removed_rev += 1
io.open(dst, 'w', encoding='utf-8', newline='\n').write(res)
print(f"{src}: removed {removed_groups} revision groups, {removed_bare} bare conditionals, {removed_rev} \\rev wrappers -> {dst}")
