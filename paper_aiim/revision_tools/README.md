
# Revision build helpers

Scripts used to build the revised manuscript package (not needed to
reproduce any number):

- `build_all.ps1` - compiles the clean and marked-up main text and
  supplement with `pdflatex` / `bibtex` and reports undefined references.
- `make_diff.ps1`, `run_latexdiff.py`, `flatten_revgroups.py`,
  `fix_diff_refs.py`, `fix_diff_paths.py`, `merge_bib.py`, `sync_supp.py` -
  the `latexdiff` pipeline that produced the marked-up PDFs from the
  submitted (frozen) and revised sources.
- `build_responses.py` - converts the Markdown point-by-point responses
  (four reviewers and the editor) to `.docx`; `export_responses_pdf.py` -
  renders them to PDF with an embedded Unicode TrueType font (PyMuPDF);
  `export_platform_replies.py` - writes one plain-text reply block per
  reviewer comment for the journal's online peer review platform.

They operate on the manuscript sources (`main.tex`, `supplementary/*.tex`,
`responses/*.md`, the frozen August-2026 submission), which are not part of
this code repository, and they reference local tool installations (`perl`
and `latexdiff-so` from Git for Windows / MiKTeX, the Windows Times New
Roman TrueType files). Adjust `$ROOT`, `$FROZEN`, `PERL`, `LD` and `FONT_DIR`
for another machine. Machine-local paths of the development tree were
scrubbed to `.` at release time.
