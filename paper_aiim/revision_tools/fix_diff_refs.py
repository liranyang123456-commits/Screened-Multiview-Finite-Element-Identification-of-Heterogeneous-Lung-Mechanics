"""Replace \\ref{label} for labels that exist only in the OLD (submitted) version
with the old printed number, so deleted text in the latexdiff PDF does not show '??'."""
import sys, re, io
diff_tex, old_aux, new_aux = sys.argv[1], sys.argv[2], sys.argv[3]
def labels(aux):
    t = io.open(aux, encoding='latin-1').read()
    return {m.group(1): m.group(2) for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', t)}
old = labels(old_aux); new = labels(new_aux)
t = io.open(diff_tex, encoding='utf-8').read()
missing = sorted(set(re.findall(r'\\ref\{([^}]*)\}', t)) - set(new))
n = 0
for lab in missing:
    if lab in old:
        t, k = re.subn(r'\\ref\{' + re.escape(lab) + r'\}', old[lab], t)
        n += k
        print(f"  {lab}: old number {old[lab]} substituted x{k}")
    else:
        print(f"  {lab}: not in old aux either (left as is)")
io.open(diff_tex, 'w', encoding='utf-8', newline='\n').write(t)
print("substitutions:", n)
