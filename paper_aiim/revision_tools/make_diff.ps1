$ErrorActionPreference = "Continue"
$ROOT = ".\paper_aiim\array_submission\revision_outputs"
$FROZEN = ".\frozen_array_20260824\paper_aiim\array_submission"
$PY = "python"
$T = $PSScriptRoot
# Options kept from the first marked-up build (flatten, no math markup, section
# commands excluded, \path safe) plus the QA fixes: --graphics-markup=none,
# \placeholderfigure no longer declared safe (so old+new figures are not both
# rendered), and tabular/longtable bodies treated as PICTUREENV so changed
# tables are replaced whole instead of being word-diffed.
$opts = @("--flatten", "--graphics-markup=none", "--math-markup=0",
          "--append-safecmd=path", "--exclude-textcmd=section,subsection,paragraph",
          "--config", "PICTUREENV=(?:picture|DIFnomarkup|tabular|longtable)[[:alnum:]*@]*")
# NB: Git's MSYS perl strips backslashes from arguments ([\w\d*@]* arrived as
# [wd*@]* and tabularx was not matched), hence the POSIX class above.

# ---- main ----
Set-Location $ROOT
New-Item -ItemType Directory -Force -Path build_diff\frozen_main | Out-Null
& $PY "$T\flatten_revgroups.py" main.tex build_diff\main_new_flat.tex
Push-Location "$FROZEN\manuscript"
cmd /c "pdflatex -interaction=nonstopmode -output-directory=$ROOT\build_diff\frozen_main main.tex > $ROOT\build_diff\frozen_main\pass1.stdout 2>&1"
Pop-Location
& $PY "$T\run_latexdiff.py" build_diff\main_diff.tex build_diff\main_diff.latexdiff.err "$FROZEN\manuscript\main.tex" build_diff\main_new_flat.tex @opts
& $PY "$T\merge_bib.py"
& $PY "$T\fix_diff_refs.py" build_diff\main_diff.tex build_diff\frozen_main\main.aux build_final\main_clean.aux
& $PY "$T\fix_diff_paths.py" build_diff\main_diff.tex
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\main_diff.tex > build_diff\main_diff.pass1.stdout 2>&1"
Push-Location build_diff
cmd /c "bibtex main_diff > main_diff.bibtex.stdout 2>&1"
Pop-Location
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\main_diff.tex > build_diff\main_diff.pass2.stdout 2>&1"
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\main_diff.tex > build_diff\main_diff.pass3.stdout 2>&1"
"main_diff pdflatex exit: $LASTEXITCODE"

# ---- supplement ----
Set-Location "$ROOT\supplementary"
New-Item -ItemType Directory -Force -Path build_diff\frozen_supp | Out-Null
& $PY "$T\flatten_revgroups.py" supplementary_clean.tex build_diff\supplementary_new_flat.tex
Push-Location "$FROZEN\supplementary"
cmd /c "pdflatex -interaction=nonstopmode -output-directory=$ROOT\supplementary\build_diff\frozen_supp supplementary.tex > $ROOT\supplementary\build_diff\frozen_supp\pass1.stdout 2>&1"
Pop-Location
Copy-Item ion_geometry_failure_table.tex build_diff\ion_geometry_failure_table.tex -Force
& $PY "$T\run_latexdiff.py" build_diff\supplementary_diff.tex build_diff\supplementary_diff.latexdiff.err "$FROZEN\supplementary\supplementary.tex" build_diff\supplementary_new_flat.tex @opts
& $PY "$T\fix_diff_refs.py" build_diff\supplementary_diff.tex build_diff\frozen_supp\supplementary.aux build_final\supplementary_clean.aux
& $PY "$T\fix_diff_paths.py" build_diff\supplementary_diff.tex
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\supplementary_diff.tex > build_diff\supplementary_diff.pass1.stdout 2>&1"
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\supplementary_diff.tex > build_diff\supplementary_diff.pass2.stdout 2>&1"
cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_diff build_diff\supplementary_diff.tex > build_diff\supplementary_diff.pass3.stdout 2>&1"
"supp_diff pdflatex exit: $LASTEXITCODE"
