# Thesis (Persian, XePersian)

Draft of the bachelor's thesis. Chapters are in `chapters/`; every number,
table and chart comes from `scripts/make_reports.py`:

- `../tables/*.tex` — tables, and `numbers.tex` (macros quoted in the text)
- `../data/*.csv` — chart data for pgfplots
- `../figures/*.png` — radar chips from the console

Rebuild them after any new result:

```bash
.venv-eval/Scripts/python -u scripts/make_reports.py
```

## Build

1. Install a TeX distribution with XeLaTeX (MiKTeX installs missing packages
   on demand).
2. Put the Estedad fonts in `fonts/`: `Estedad-Regular.ttf` and
   `Estedad-Bold.ttf` from https://github.com/aminabedi68/Estedad/releases
   (SIL Open Font License; not committed).
3. From this folder, run XeLaTeX three times (lists of tables/figures and
   references settle on the later runs):

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

To check pages by eye without a PDF viewer:

```bash
.venv-docs/Scripts/python scripts/render_pdf.py reports/thesis/main.pdf --out <folder> --pages 1-3
```

## Typography rules

- Numbers: Latin digits with Persian separators (`٫` decimal, `٬` thousands)
  inside `\LRE{...}`; the digit font draws them in Persian and the order is kept.
- Intervals: «a تا b» (brackets inside `\LRE` are mirrored).
- Latin terms: `\lr{...}`. Persian text inside TikZ/pgfplots nodes: `\rl{...}`
  (nodes are left to right).
- The document is a «پروژه کارشناسی» (title page in `main.tex`); call it
  «گزارش» or «پروژه» in the text, not «پایان‌نامه».
