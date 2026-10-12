import io, re
ROOT = r".\paper_aiim\array_submission\revision_outputs\supplementary"
clean = ROOT + r"\supplementary_clean.tex"
marked = ROOT + r"\supplementary_markedup_source.tex"
with io.open(clean, encoding='utf-8') as f:
    txt = f.read()
# strip REVISION-BLOCK comments
lines = [l for l in txt.split('\n') if not l.startswith('% REVISION-BLOCK')]
txt = '\n'.join(lines)
assert txt.count('\\revisionfalse') == 1, txt.count('\\revisionfalse')
with io.open(clean, 'w', encoding='utf-8', newline='\n') as f:
    f.write(txt)
m = txt.replace('% Designated supplement source (clean).', '% Designated supplement source (marked-up).', 1)
m = m.replace('\\revisionfalse', '\\revisiontrue', 1)
with io.open(marked, 'w', encoding='utf-8', newline='\n') as f:
    f.write(m)
# verify identity except the two lines
a = txt.split('\n'); b = m.split('\n')
diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
print('lines', len(a), len(b), 'differing lines:', len(diff))
for d in diff: print(d)
