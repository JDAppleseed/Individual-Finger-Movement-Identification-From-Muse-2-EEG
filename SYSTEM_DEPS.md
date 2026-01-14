# System Dependencies

These dependencies are **not** available on PyPI and must be installed with system
package managers. They are required only for generating LaTeX/PDF reports and
front-end tooling, not for core model training/evaluation.

## LaTeX (reports)

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

## Node tooling (demo_app)

```bash
cd demo_app
npm install
```
