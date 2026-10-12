$ErrorActionPreference = "Continue"
$ROOT = ".\paper_aiim\array_submission\revision_outputs"
Set-Location $ROOT
Copy-Item references.bib build_final\references.bib -Force
function RunTex($job, $src) {
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass1.stdout 2>&1"
  Push-Location build_final
  cmd /c "bibtex $job > $job.bibtex.stdout 2>&1"
  Pop-Location
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass2.stdout 2>&1"
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass3.stdout 2>&1"
  $log = Get-Content "build_final\$job.log" -Raw
  $errs = ([regex]::Matches($log, '(?m)^! .*$') | ForEach-Object { $_.Value })
  $undef = ([regex]::Matches($log, 'LaTeX Warning: (Reference|Citation) `[^'']*'' on page \d+ undefined') | ForEach-Object { $_.Value })
  $pages = [regex]::Match($log, 'Output written on .*? \((\d+) pages').Groups[1].Value
  $undefrefs = if ($log -match 'There were undefined references') { 'YES' } else { 'no' }
  "== $job : pages=$pages errors=$($errs.Count) undefined-refs=$undefrefs undef-items=$($undef.Count)"
  $errs | Select-Object -First 5
  $undef | Select-Object -First 10
}
RunTex "main_clean" "main.tex"
RunTex "main_markedup" "main_markedup.tex"

Set-Location "$ROOT\supplementary"
function RunSupp($job, $src) {
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass1.stdout 2>&1"
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass2.stdout 2>&1"
  cmd /c "pdflatex -interaction=nonstopmode -output-directory=build_final -jobname=$job $src > build_final\$job.pass3.stdout 2>&1"
  $log = Get-Content "build_final\$job.log" -Raw
  $errs = ([regex]::Matches($log, '(?m)^! .*$') | ForEach-Object { $_.Value })
  $undef = ([regex]::Matches($log, 'LaTeX Warning: (Reference|Citation) `[^'']*'' on page \d+ undefined') | ForEach-Object { $_.Value })
  $pages = [regex]::Match($log, 'Output written on .*? \((\d+) pages').Groups[1].Value
  $undefrefs = if ($log -match 'There were undefined references') { 'YES' } else { 'no' }
  "== $job : pages=$pages errors=$($errs.Count) undefined-refs=$undefrefs undef-items=$($undef.Count)"
  $errs | Select-Object -First 5
  $undef | Select-Object -First 10
}
RunSupp "supplementary_clean" "supplementary_clean.tex"
RunSupp "supplementary_markedup" "supplementary_markedup_source.tex"
