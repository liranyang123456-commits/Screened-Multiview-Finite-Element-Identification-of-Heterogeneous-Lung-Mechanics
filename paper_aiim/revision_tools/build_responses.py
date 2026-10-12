from pathlib import Path
import re
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor

src_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

for md_path in sorted(src_dir.glob('Response_Reviewer[1-4].md')):
    text = md_path.read_text(encoding='utf-8-sig')
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for stylename, size in [('Title', 16), ('Heading 1', 13), ('Heading 2', 11.5)]:
        style = doc.styles[stylename]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(31, 78, 121)
    if '|' in text[:80]:
        text = re.sub(r'^\s*\|+', '', text)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('# '):
            doc.add_heading(stripped[2:].strip(), level=0)
            continue
        if stripped.startswith('## '):
            doc.add_heading(stripped[3:].strip(), level=1)
            continue
        if stripped.startswith('### '):
            doc.add_heading(stripped[4:].strip(), level=2)
            continue
        if stripped.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.18)
            p.paragraph_format.right_indent = Inches(0.18)
            r = p.add_run(stripped[2:].replace('  ', ' '))
            r.italic = True
            r.font.color.rgb = RGBColor(90, 90, 90)
            continue
        if stripped == '---':
            continue
        p = doc.add_paragraph()
        parts = re.split(r'(\*\*.*?\*\*|`[^`]+`)', stripped)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                p.add_run(part[2:-2]).bold = True
            elif part.startswith('`') and part.endswith('`'):
                run = p.add_run(part[1:-1])
                run.font.name = 'Consolas'
                run.font.size = Pt(9.5)
            else:
                p.add_run(part)
    output = out_dir / f'{md_path.stem}.docx'
    doc.save(output)
    print(output)
