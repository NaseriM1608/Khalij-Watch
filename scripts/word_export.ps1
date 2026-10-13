# Update every field (contents, lists of figures and tables, page numbers) in a .docx,
# save it, and export a PDF, using the installed Microsoft Word.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/word_export.ps1 -In <docx> -Pdf <pdf>
param([Parameter(Mandatory = $true)][string]$In, [Parameter(Mandatory = $true)][string]$Pdf)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.Options.SaveNormalPrompt = $false
$exportLog = $Pdf + '.export.log'
function Report-Stage([string]$stage) {
  ((Get-Date).ToString('o') + ' ' + $stage) | Add-Content -LiteralPath $exportLog -Encoding UTF8
}
try {
  Report-Stage 'opening'
  $doc = $word.Documents.Open((Resolve-Path $In).Path, $false, $false)
  Report-Stage 'updating fields'
  $doc.Fields.Update() | Out-Null
  Report-Stage 'updating contents'
  foreach ($toc in $doc.TablesOfContents) {
    $toc.Update() | Out-Null
    $toc.Range.ParagraphFormat.Alignment = 2
    $toc.Range.ParagraphFormat.ReadingOrder = 0
    $toc.Range.ParagraphFormat.SpaceAfter = 2
    $toc.Range.ParagraphFormat.LineSpacingRule = 0
    $toc.Range.Font.NameBi = 'B Nazanin'
    $toc.Range.Font.NameAscii = 'Times New Roman'
    $toc.Range.Font.Size = 12
    $toc.Range.Font.SizeBi = 12
  }
  Report-Stage 'repaginating'
  $doc.Repaginate()
  foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() | Out-Null }
  Report-Stage 'saving'
  $doc.Save()
  Report-Stage 'exporting PDF'
  $doc.ExportAsFixedFormat($Pdf, 17)
  if (-not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
    # Word can return successfully without writing on the first export of a
    # newly assembled package. Reopen the saved, field-updated document once.
    Report-Stage 'retrying PDF from saved document'
    $doc.Close(0)
    $doc = $word.Documents.Open((Resolve-Path $In).Path, $false, $false)
    $doc.ExportAsFixedFormat($Pdf, 17)
  }
  if (-not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
    throw "Word returned without creating the PDF: $Pdf"
  }
  "pages: " + $doc.ComputeStatistics(2)
  $doc.Close(0)
  Report-Stage 'complete'
} catch {
  Report-Stage ('FAILED: ' + $_.Exception.Message)
  throw
} finally {
  $word.Quit()
}
