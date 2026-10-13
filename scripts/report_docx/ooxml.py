"""WordprocessingML building blocks (strings), in the template's conventions.

Persian runs carry ``<w:rtl/>`` (B Nazanin, right-to-left); Latin runs carry none
(Times New Roman). Sizes come from the paragraph style unless given here.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from scripts.report_docx.text import split_scripts

EMU_PER_CM = 360000
TEXT_WIDTH_TWIPS = 9185  # A4 21 cm minus the template's 2.5 cm right and 2.3 cm left margins


def run(
    t: str,
    *,
    rtl: bool,
    bold: bool = False,
    size: float | None = None,
    font: str | None = None,
    style: str | None = None,
) -> str:
    props = []
    if style:
        props.append(f'<w:rStyle w:val="{style}"/>')
    if font:
        props.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>')
    if bold:
        props.append("<w:b/><w:bCs/>")
    if size:
        half = round(size * 2)
        props.append(f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/>')
    if rtl:
        props.append("<w:rtl/>")
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def text_runs(t: str, *, bold: bool = False, size: float | None = None) -> str:
    """Runs for mixed Persian/Latin text: Persian pieces right-to-left, Latin pieces not."""
    return "".join(
        run(piece, rtl=rtl, bold=bold, size=size) for piece, rtl in split_scripts(t) if piece
    )


def latin_runs(
    t: str, *, bold: bool = False, italic: bool = False, size: float | None = None
) -> str:
    props = ("<w:b/><w:bCs/>" if bold else "") + ("<w:i/><w:iCs/>" if italic else "")
    if size:
        props += f'<w:sz w:val="{round(size * 2)}"/><w:szCs w:val="{round(size * 2)}"/>'
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def paragraph(
    runs: str,
    *,
    style: str | None = None,
    align: str | None = None,
    rtl: bool = True,
    before: int | None = None,
    after: int | None = None,
    line: int | None = None,
    indent: str = "",
    keep_next: bool = False,
    page_break_before: bool = False,
    tabs: str = "",
    extra: str = "",
) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        ppr.append("<w:keepNext/>")
    if page_break_before:
        ppr.append("<w:pageBreakBefore/>")
    if tabs:
        ppr.append(f"<w:tabs>{tabs}</w:tabs>")
    spacing = "".join(
        f' w:{k}="{v}"'
        for k, v in (("before", before), ("after", after), ("line", line))
        if v is not None
    )
    if spacing:
        rule = ' w:lineRule="auto"' if line is not None else ""
        ppr.append(f"<w:spacing{spacing}{rule}/>")
    if indent:
        ppr.append(indent)
    ppr.append("<w:bidi/>" if rtl else '<w:bidi w:val="0"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(extra)
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{runs}</w:p>"


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def field(code: str, placeholder: str = "") -> str:
    """A complex field (TOC, PAGE, ...); Word computes it when fields are updated."""
    return (
        '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> {escape(code)} </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        + run(placeholder or " ", rtl=True)
        + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def inline_image(rid: str, cx: int, cy: int, pid: int, name: str) -> str:
    """An inline picture (EMU sizes) referring to an image relationship."""
    return (
        '<w:r><w:drawing xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{pid}" name="{escape(name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{pid}" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
    )


def _border(side: str, size: int) -> str:
    return f'<w:{side} w:val="single" w:sz="{size}" w:space="0" w:color="000000"/>'


def table(rows: list[list[str]], *, header_rows: int = 1, borders: bool = True) -> str:
    """A right-to-left table; ``rows`` hold ready-made paragraph XML per cell.

    Booktabs-like rules: a heavy line above and below, a light one under the header.
    """
    tbl_borders = (
        f"<w:tblBorders>{_border('top', 12)}{_border('bottom', 12)}</w:tblBorders>"
        if borders
        else '<w:tblBorders><w:top w:val="nil"/><w:bottom w:val="nil"/><w:insideH w:val="nil"/>'
        '<w:insideV w:val="nil"/><w:left w:val="nil"/><w:right w:val="nil"/></w:tblBorders>'
    )
    out = [
        "<w:tbl><w:tblPr>"
        '<w:bidiVisual/><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/>'
        f"{tbl_borders}"
        '<w:tblLayout w:type="autofit"/>'
        '<w:tblCellMar><w:left w:w="85" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
    ]
    for r, cells in enumerate(rows):
        is_header = r < header_rows
        trpr = (
            "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>"
            if is_header
            else "<w:trPr><w:cantSplit/></w:trPr>"
        )
        out.append(f"<w:tr>{trpr}")
        for cell in cells:
            tcpr = (
                f'<w:tcPr><w:tcBorders>{_border("bottom", 6)}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                if is_header and borders and r == header_rows - 1
                else '<w:tcPr><w:vAlign w:val="center"/></w:tcPr>'
            )
            out.append(f"<w:tc>{tcpr}{cell}</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)
