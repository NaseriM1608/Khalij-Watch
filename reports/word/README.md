# Bachelor's project report in Word

The retained `template.docx` is the university's template. The Word report is the
current report; `reports/thesis/` is the earlier LaTeX draft. Body text uses
B Nazanin 14 and Times New Roman 12. The English abstract and title page are
included. The examiner's name is left blank.

## Build

From the repository root on this Windows host:

```powershell
.venv-docs/Scripts/python scripts/build_report_docx.py --charts --pdf
.venv-docs/Scripts/python scripts/render_pdf.py reports/word/build/report.pdf --out reports/word/build/render --dpi 130
```

`--charts` compiles the six plots and two diagrams using the installed XeLaTeX
and rasterises them at 300 dpi. Existing CSVs and report figures supply the
measurements and chips. `--pdf` uses installed Microsoft Word to update fields,
save the document, and export the PDF. No detector, training job, or remote
data query is part of this build. Review every rendered page after changes.
The `build/` directory is ignored by Git.

Word export uses a fresh staging filename and publishes the DOCX/PDF pair only
after export succeeds, preserving the previous pair if Word fails.
If Word returns without a PDF, the helper reopens the saved document and retries
once, then fails explicitly if the output is still missing.

## Sources and notation

- `content/meta.md`: titles, names, dates, abstracts, keywords, abbreviations.
- `content/01-*.md` through `05-*.md`: chapters; the conclusion is unnumbered.
- `content/references.md`: verified bibliography; only cited works are printed.
- `content/diagrams/*.tex`: TikZ bodies included by the chart builder.
- `research/*.md`: source verification notes and the methods audit.
- `reports/tables/numbers.tex`: existing numerical macros used in the prose.

Markup supports chapter and section headings, paragraph text, lists, citation
keys such as `[@paolo2024]`, number macros such as `{{testFOneModel}}`, and
cross-references such as `[[sec:evaluation]]`. Figure, table and equation blocks
have an `id`. Figures use a registered image name after `chart:` or `image:`.
The `short` attribute supplies a concise list entry. Equation syntax is the
explicit subset in `scripts/report_docx/omml.py`; unsupported commands fail.

## Interpretation and reproducibility

Scores are agreement with GFW, which also contributed to the model's training
labels. Product-level hashing keeps a frame's chips together but does not block
regions, dates, vessels, or all acquisition groups. The report gives both the
original 40-area-pass comparison and the expanded 92-area-pass comparison,
preserves confidence bounds below zero, and documents the changed validation
selection rule. Iranian and satellite-transfer results use the frozen model.

Some exact numbers are copied from the verified methods notes and recorded
run summaries rather than numerical macros, especially confidence intervals,
the historical fusion snapshot, and the split audit. Rebuilding typesetting
does not recompute these analyses. The report explains gaps in training-code
provenance, data versions, calibration records, and exploratory scripts.
Fresh AIS fusion must not silently update this report's historical numbers.

## Local checks

```powershell
.venv-eval/Scripts/python -m pytest tests/test_report_docx.py tests/test_report_docx_content.py tests/test_report_docx_export.py -q
uv run --offline --no-project --with ruff ruff check scripts/report_docx scripts/build_report_docx.py tests/test_report_docx*.py
```

The full repository checks use the separate environments documented in
`handoff.md`. No package download is required for these commands.
