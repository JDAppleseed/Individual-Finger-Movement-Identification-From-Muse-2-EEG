#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

echo "[paper] Regenerating artifact-driven manuscript inputs"
python3 scripts/build_paper_artifacts.py

# Optional local research inputs are not part of the public manuscript bundle.
if [ -f docs/research/finalization_20261125/comparison.json ]; then
  python3 scripts/build_concise_paper_figures.py
  if [ -f docs/research/finalization_20261125/datasets.json ]; then
    python3 scripts/build_paper_support_analysis.py
  fi
fi

echo "[paper] Recompiling research_paper.tex"
cd "$REPO_ROOT/paper"
latexmk -C research_paper.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode research_paper.tex

echo "[paper] Done"
echo "[paper] PDF: $REPO_ROOT/paper/research_paper.pdf"
