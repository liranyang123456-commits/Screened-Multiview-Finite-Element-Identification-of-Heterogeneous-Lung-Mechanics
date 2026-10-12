"""Run latexdiff-so (Git for Windows perl) with an explicit argument list and
write its stdout/stderr to files as raw bytes (no console re-encoding).

usage: run_latexdiff.py OUT.tex ERR.txt OLD.tex NEW.tex [latexdiff options...]
"""
import subprocess, sys, os
PERL = r"C:\Program Files\Git\usr\bin\perl.exe"
LD = r"D:\Ctex\MiKTeX\scripts\latexdiff\latexdiff-so"
out, err, old, new, *opts = sys.argv[1:]
env = dict(os.environ)
env["PATH"] = r"C:\Program Files\Git\usr\bin;" + env.get("PATH", "")
res = subprocess.run([PERL, LD, *opts, old, new], capture_output=True, env=env)
open(out, "wb").write(res.stdout)
open(err, "wb").write(res.stderr)
print(f"latexdiff exit {res.returncode}; {len(res.stdout)} bytes -> {out}; stderr {len(res.stderr)} bytes")
sys.exit(res.returncode)
