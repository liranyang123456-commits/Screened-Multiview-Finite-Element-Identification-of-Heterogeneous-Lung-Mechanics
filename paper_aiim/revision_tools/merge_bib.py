import re, io
ROOT = r".\paper_aiim\array_submission\revision_outputs"
FROZEN = r".\frozen_array_20260824\paper_aiim\array_submission\manuscript"
new = io.open(ROOT + r"\references.bib", encoding='utf-8').read()
old = io.open(FROZEN + r"\references.bib", encoding='utf-8').read()
newkeys = set(re.findall(r'@\w+\{([^,]+),', new))
extra = []
for m in re.finditer(r'@\w+\{([^,]+),.*?\n\}', old, flags=re.S):
    if m.group(1) not in newkeys:
        extra.append(m.group(0))
print("old-only bib keys added for diff build:", [re.match(r'@\w+\{([^,]+),', e).group(1) for e in extra])
io.open(ROOT + r"\build_diff\references.bib", 'w', encoding='utf-8').write(new + "\n\n% --- entries only in the submitted (Aug-2026) version, kept so deleted citations resolve in the diff ---\n" + "\n\n".join(extra) + "\n")
