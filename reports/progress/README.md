# Persian progress presentations

Four separate 12-slide academic presentations cover phases 1/1.5/2, 3/4,
5/6 and 7/8. The phase grouping supports the supervisor's fortnightly progress
presentations; it does not invent fortnightly development dates. The content
uses existing measurements, numeric tables, real stored SAR imagery, and
Persian speaker notes. There is no project nickname or visible bibliography.
Evidence locations and qualifications are retained in each slide's notes.

The current delivery is eight files under `build/deliverables/`: one PPTX and
one PDF for each stem `phases_1_1-5_2`, `phases_3_4`, `phases_5_6` and
`phases_7_8`. Each has 12 slides. The first two use finalized revision v12;
the latter two use v10. `build/verified_deliverables.json` records hashes,
page counts and validation receipts. All 48 PDF pages were visually reviewed;
the final wording/order changes were rechecked, and unchanged pages compared
pixel-identical. Use the PDFs for predictable Persian display. PPTX content,
native tables/charts and Persian speaker notes remain editable; native
PowerPoint rendering was not available on this machine.

`slides.mjs` is the canonical content. `build_decks.mjs` creates editable native
PowerPoint tables, three native data charts with embedded workbook snapshots,
images and text using the bundled `@oai/artifact-tool`. `rtl_pptx.py` adds
explicit Persian paragraph metadata before the required package, layout,
font, editable-evidence and first-party-import finalization checks.

Outputs and validation receipts are ignored under `build/`. No font files,
model weights, raw data or build outputs belong in a commit. B Nazanin is a
locally installed design font; use the PDF companions when that font is not
available on the presentation computer.

## Build and render

Use the Node and Python paths returned by Codex's workspace dependency tool.
The bundled runtime needs no download. The build directory's `node_modules`
junction points to the bundled Node packages. The authoring marker from the
presentation skill must run once immediately before an authoring operation.

```powershell
# Use a new revision because the finalizer preserves existing outputs.
$env:REVISION='v13'
# Optional: one deck only; omit ONLY to build all four.
$env:ONLY='phases_1_1-5_2'
& '<bundled Node>' reports/progress/build_decks.mjs
.venv-docs/Scripts/python scripts/render_pdf.py reports/progress/build/output/phases_1_1-5_2_v13.pdf --out reports/progress/build/phases_1_1-5_2_v13-pdf --dpi 96
```

## Rendering limitation and fallback

The bundled Artifact Tool's preview renderer reverses Persian word order, and
its PDF export fails because an internal `#standard-fonts/Helvetica` module is
missing. PowerPoint's COM registration exists on this device but its binary
cannot be activated (`80070002`). No dependency or Office installation was
attempted. The editable PPTX therefore has structural/import validation and
explicit logical-order RTL text, but was not executed in native PowerPoint.

`html_pdf.mjs` renders a companion from the same canonical content through the
installed Chrome print engine. These PDFs retain real text, embedded fonts,
numeric tables, chart data and original source images with correct Persian
ordering. Their page images are the visual QA authority on this machine.
The PDF chart layout is equivalent in data, not a native PowerPoint screenshot.
Every final PDF page must be rendered and inspected before delivery.

Historical research results and the accepted university report stay fixed.
The dated 29 October motion-estimation addition is separate from phases 7/8
and must describe only implemented and measured behavior. Do not present a
short-term estimate as a live satellite detection or confirmed dark ship.
