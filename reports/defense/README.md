# Persian defense-study guides

These two guides explain the completed khalij-watch project for the student's
study and presentation. They do not replace or revise the supervisor-approved
university report. No new training, labeling, detection, data collection or
scientific evaluation is part of this work.

- `phases.md`: the actual work in phases 0–8, inputs, implementation decisions,
  outputs, evidence, measured results, limits, and a 10–15 minute presentation
  narrative. The October 13 operational fusion snapshot is separate from the
  accepted report's historical results.
- `technical.md`: 96 questions with short answers and detailed explanations,
  CNN operations, the EfficientNet-B0/U-Net architecture, point outputs rather
  than learned boxes, targets, losses, training history, validation, deployment,
  metrics and methodological limits. Worked educational examples are explicitly
  distinguished from project measurements.
- `meta.json`: Persian cover metadata.
- `phases_professor.md`: a separate formal Persian account of the completed
  phases for presenting to the professor. It retains the methods, results and
  limitations, removes private study/rehearsal instructions and repository
  pointers, and leaves both original study PDFs unchanged. At the user's
  request it has no project nickname, inline citations or bibliography.

The sources use the existing report markup. The study guides use the verified bibliography in
`reports/word/content/references.md`. Existing diagrams and the saved Kish chip
are reused. Current-code architecture checks explain structure; they are not
new trained-model evaluations and do not prove exact reproduction of the
historical Kaggle environment.

## Build and check

Run from the repository root with a Python environment containing the existing
report-builder dependencies. Microsoft Word and the already-built report charts
are required. No package download is needed in the configured local workspace.

```powershell
# Use the Codex bundled Python runtime in this workspace.
python scripts/build_defense_guides.py
# Or rebuild one guide:
python scripts/build_defense_guides.py --only technical
# Build the separate professor copy without rebuilding the study PDFs:
python scripts/build_defense_guides.py --only phases_professor

.venv-docs/Scripts/python scripts/render_pdf.py reports/defense/build/phases.pdf --out reports/defense/build/render-phases --dpi 130
.venv-docs/Scripts/python scripts/render_pdf.py reports/defense/build/technical.pdf --out reports/defense/build/render-technical --dpi 130
.venv-eval/Scripts/python -m pytest tests/test_report_docx.py tests/test_report_docx_content.py tests/test_report_docx_export.py tests/test_report_docx_defense.py -q
uv run --offline --no-project --with ruff ruff check scripts/report_docx scripts/build_report_docx.py scripts/build_defense_guides.py tests/test_report_docx*.py
```

The study PDFs are `build/phases.pdf` and `build/technical.pdf`. The additional
professor copy is `build/phases_professor.pdf`; it is selected explicitly with
`--only phases_professor`. The default command still builds only the two study
guides. Intermediate Word
files, render images, architecture-check output and all build artifacts are
ignored by Git. Inspect every PDF page after changes, including equations,
multi-page tables, references and mixed Persian/Latin text. The guide builder
uses compact study styles and does not alter the original report renderer,
template or accepted report content.

Further project improvements or experiments require an explicit user request.
Remaining work is studying and rehearsing the accepted project, and responding
to requested presentation corrections.
