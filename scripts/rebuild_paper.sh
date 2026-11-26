#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

if [ ! -f paper/research_paper.tex ]; then
  echo "[paper] Private manuscript is not present in this checkout."
  exit 1
fi

echo "[paper] Regenerating artifact-driven manuscript inputs"
./run_py scripts/build_paper_artifacts.py

if [ -f logs/all_subjects_20260928/results.json ]; then
  ./run_py scripts/build_subject_comparison_paper_artifacts.py
fi
if [ -f docs/research/finalization_20261125/comparison.json ]; then
  ./run_py scripts/build_finalization_report.py
fi

./run_py scripts/build_subject_release_paper.py

echo "[paper] Recompiling research_paper.tex"
cd "$REPO_ROOT/paper"
latexmk -pdf -interaction=nonstopmode research_paper.tex

echo "[paper] Done"
echo "[paper] PDF: $REPO_ROOT/paper/research_paper.pdf"
