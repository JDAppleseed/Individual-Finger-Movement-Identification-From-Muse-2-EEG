#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

echo "[paper] Regenerating artifact-driven manuscript inputs"
python3 scripts/build_paper_artifacts.py

echo "[paper] Recompiling research_paper.tex"
cd "$REPO_ROOT/paper"
latexmk -C research_paper.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode research_paper.tex

echo "[paper] Done"
echo "[paper] PDF: $REPO_ROOT/paper/research_paper.pdf"
