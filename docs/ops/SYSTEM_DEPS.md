# System Dependencies

These packages are optional. You do not need them for normal UI or CLI use.

Install them only if you need LaTeX/PDF report generation or other non-PyPI tooling.

## LaTeX For Reports

### macOS (MacTeX/BasicTeX)

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install geometry amsmath amsfonts graphics booktabs hyperref microtype xcolor titlesec tocloft parskip setspace float latexmk
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk
```
