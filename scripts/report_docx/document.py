"""Assemble the report .docx on the university template.

Keeps the template's package (styles, theme, logos, footers, settings) and
writes a new body: Persian title page, بسم‌الله, juror approval, student
declaration, copyright, abstract, contents and lists (Word fields), the
abbreviations, the chapters (one section each, with the template's running
header: page number, short title, chapter name), references, English abstract
and English title page. Page numbers: front matter in Persian letters at the
bottom centre, chapters in digits at the top, title pages unnumbered.
"""

from __future__ import annotations

import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from scripts.report_docx import ooxml
from scripts.report_docx.markup import Block
from scripts.report_docx.render import Chapter, Renderer
from scripts.report_docx.text import fa_digits

FONT_FA = "B Nazanin"
FONT_EN = "Times New Roman"
# Template relationships kept as they are.
LOGO_RIDS = (
    "rId8",
    "rId13",
    "rId14",
    "rId15",
    "rId26",
)  # title, approval, declaration, copyright, English
BISMILLAH_RID = "rId9"
FOOTER_EMPTY, FOOTER_LETTERS, HEADER_EMPTY = "rId10", "rId12", "rId11"
PAGE = '<w:pgSz w:w="11907" w:h="16839" w:code="9"/>'
MARGINS = '<w:pgMar w:top="1418" w:right="1418" w:bottom="1134" w:left="1304" w:header="720" w:footer="720" w:gutter="0"/>'

EXTRA_STYLES = """
<w:style w:type="paragraph" w:customStyle="1" w:styleId="FigureCaption"><w:name w:val="FigureCaption"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:before="60" w:after="200" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
<w:rPr><w:sz w:val="20"/><w:szCs w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableCaption"><w:name w:val="TableCaption"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:spacing w:before="200" w:after="60" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
<w:rPr><w:sz w:val="20"/><w:szCs w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText"><w:name w:val="TableText"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
<w:rPr><w:sz w:val="20"/><w:szCs w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="FootnoteText"><w:name w:val="footnote text"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="18"/><w:szCs w:val="20"/></w:rPr></w:style>
<w:style w:type="character" w:styleId="FootnoteReference"><w:name w:val="footnote reference"/><w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>
"""


@dataclass(frozen=True)
class Meta:
    title_fa: str
    title_en: str
    short_title: str
    author_fa: str
    author_en: str
    supervisor_fa: str
    supervisor_en: str
    date_fa: str
    date_en: str
    abstract_fa: str
    keywords_fa: str
    abstract_en: str
    keywords_en: str
    abbreviations: list[tuple[str, str]] = field(default_factory=list)
    faculty_fa: str = "دانشکده فنی و مهندسی"
    department_fa: str = "بخش مهندسی کامپیوتر"


def png_size(data: bytes) -> tuple[int, int]:
    """Pixel width and height from a PNG header."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", data[16:24])


def section(*, header: str, footer: str, fmt: str | None = None, start: int | None = None) -> str:
    num = ""
    if fmt or start:
        attrs = (f' w:fmt="{fmt}"' if fmt else "") + (f' w:start="{start}"' if start else "")
        num = f"<w:pgNumType{attrs}/>"
    return (
        f'<w:sectPr><w:headerReference w:type="default" r:id="{header}"/>'
        f'<w:footerReference w:type="default" r:id="{footer}"/>'
        '<w:footnotePr><w:numRestart w:val="eachPage"/></w:footnotePr>'
        f'{PAGE}{MARGINS}{num}<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )


def section_paragraph(sect: str) -> str:
    """An empty paragraph that closes a section."""
    return f"<w:p><w:pPr><w:bidi/>{sect}</w:pPr></w:p>"


def fa(t: str, size: float, bold: bool = False) -> str:
    return ooxml.run(t, rtl=True, bold=bold, size=size, font=FONT_FA)


def en(t: str, size: float, bold: bool = False) -> str:
    return ooxml.run(t, rtl=False, bold=bold, size=size, font=FONT_EN)


def centered(runs: str, *, before: int = 0, after: int = 0, rtl: bool = True) -> str:
    return ooxml.paragraph(runs, align="center", before=before, after=after, line=240, rtl=rtl)


class Assembler:
    def __init__(self, template: Path) -> None:
        self.template = template
        with zipfile.ZipFile(template) as z:
            self.parts = {i.filename: z.read(i.filename) for i in z.infolist()}
        self.media: list[tuple[str, str, bytes]] = []  # (rid, part name, data)
        self.headers: list[tuple[str, str, str]] = []  # (rid, part name, xml)
        self._pid = 1

    # --- packaging ------------------------------------------------------------

    def add_image(self, name: str, data: bytes) -> tuple[str, int, int]:
        rid = f"rIdRepImg{len(self.media) + 1}"
        self.media.append((rid, f"word/media/rep_{name}.png", data))
        w, h = png_size(data)
        return rid, w, h

    def _header_xml(self, short_title: str, chapter_label: str) -> str:
        """The template's chapter header (header3.xml) with this chapter's texts."""
        xml = self.parts["word/header3.xml"].decode("utf-8")
        xml = re.sub(r'w:cs="2\s+Lotus"', f'w:cs="{FONT_FA}"', xml)
        paras = re.findall(r"<w:p [^>]*>.*?</w:p>|<w:p>.*?</w:p>", xml, re.S)
        titled = next(p for p in paras if "pBdr" in p)
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", titled, re.S).group(0)
        runs = (
            ooxml.run(short_title, rtl=True, size=12)
            + "<w:r><w:rPr><w:rtl/></w:rPr><w:tab/></w:r><w:r><w:rPr><w:rtl/></w:rPr><w:tab/></w:r>"
            + ooxml.run(chapter_label, rtl=True, size=12)
        )
        return xml.replace(titled, f"<w:p>{ppr}{runs}</w:p>")

    def add_header(self, short_title: str, chapter_label: str) -> str:
        n = len(self.headers) + 1
        rid = f"rIdRepHdr{n}"
        self.headers.append(
            (rid, f"word/header_rep{n}.xml", self._header_xml(short_title, chapter_label))
        )
        return rid

    def _logo(self, rid: str, width_cm: float = 3.0) -> str:
        target = re.search(
            rf'Id="{rid}"[^>]*Target="([^"]+)"', self.parts["word/_rels/document.xml.rels"].decode()
        )
        w, h = png_size(self.parts["word/" + target.group(1)])
        self._pid += 1
        cx = round(width_cm * ooxml.EMU_PER_CM)
        return ooxml.inline_image(rid, cx, round(cx * h / w), self._pid, f"logo{self._pid}")

    # --- front matter ---------------------------------------------------------

    def _faculty(self, m: Meta, logo_rid: str) -> str:
        return (
            centered(self._logo(logo_rid), after=60)
            + centered(fa(m.faculty_fa, 15, bold=True))
            + centered(fa(m.department_fa, 15, bold=True), after=360)
        )

    def title_page(self, m: Meta) -> str:
        return (
            centered(self._logo(LOGO_RIDS[0], 3.2), before=0, after=60)
            + centered(fa(m.faculty_fa, 16, bold=True))
            + centered(fa(m.department_fa, 16, bold=True), after=900)
            + centered(fa("گزارش پروژه کارشناسی", 14, bold=True), after=700)
            + centered(fa("عنوان:", 12), after=120)
            + centered(ooxml.text_runs(m.title_fa, bold=True, size=17), after=900)
            + centered(fa("نگارش:", 12, bold=True), after=120)
            + centered(fa(m.author_fa, 14, bold=True), after=600)
            + centered(fa("استاد راهنما:", 12), after=120)
            + centered(fa(m.supervisor_fa, 14, bold=True), after=1200)
            + centered(fa(m.date_fa, 14, bold=True))
        )

    def bismillah(self) -> str:
        self._pid += 1
        cx = round(12 * ooxml.EMU_PER_CM)
        return ooxml.page_break() + centered(
            ooxml.inline_image(BISMILLAH_RID, cx, round(cx * 499 / 600), self._pid, "bismillah"),
            before=3000,
        )

    def approval(self, m: Meta) -> str:
        body = (
            fa("هیأت داوران پس از مطالعه گزارش با عنوان ", 14)
            + ooxml.text_runs(m.title_fa, bold=True, size=14)
            + fa(" به نگارش ", 14)
            + fa(m.author_fa, 14, bold=True)
            + fa(
                "، پروژه انجام‌شده را برای اخذ درجه کارشناسی رشته مهندسی کامپیوتر مورد تأیید قرار دادند.",
                14,
            )
        )
        rows = [
            [
                ooxml.paragraph(fa(f"استاد راهنما: {m.supervisor_fa}", 13), align="right"),
                ooxml.paragraph(fa("امضاء:", 13), align="right"),
            ],
            [
                ooxml.paragraph(
                    fa("استاد داور: دکتر ..............................", 13), align="right"
                ),
                ooxml.paragraph(fa("امضاء:", 13), align="right"),
            ],
        ]
        return (
            ooxml.page_break()
            + self._faculty(m, LOGO_RIDS[1])
            + centered(fa("تأییدیه هیأت داوران", 16, bold=True), after=240)
            + ooxml.paragraph(body, line=360, after=600)
            + ooxml.table(rows, header_rows=0, borders=False)
        )

    def declaration(self, m: Meta) -> str:
        intro = (
            fa("اینجانب ", 14)
            + fa(m.author_fa, 14, bold=True)
            + fa(
                " دانشجوی دوره کارشناسی رشته مهندسی کامپیوتر دانشکده فنی و مهندسی دانشگاه شهید باهنر کرمان گواهی می‌نمایم که پروژه ارائه‌شده در این گزارش با عنوان:",
                14,
            )
        )
        rest = (
            fa("با راهنمایی استاد محترم جناب آقای/سرکار خانم ", 14)
            + fa(m.supervisor_fa, 14, bold=True)
            + fa(
                " توسط شخص اینجانب انجام شده است. صحت و اصالت مطالب نگارش‌شده در این پروژه مورد تأیید اینجانب می‌باشد. "
                "در مورد استفاده از کار سایر محققین به مرجع مورد استفاده اشاره شده است. همچنین، گواهی می‌نمایم که مطالب "
                "مندرج در گزارش پروژه تاکنون برای دریافت هیچ نوع مدرک یا امتیازی توسط اینجانب یا فرد دیگری در هیچ جا ارائه "
                "نشده است و در تدوین متن پروژه چارچوب مصوب دانشگاه به طور کامل رعایت شده است.",
                14,
            )
        )
        return (
            ooxml.page_break()
            + self._faculty(m, LOGO_RIDS[2])
            + centered(fa("اظهارنامه دانشجو", 16, bold=True), after=240)
            + ooxml.paragraph(intro, line=360, after=120)
            + centered(ooxml.text_runs(m.title_fa, bold=True, size=14), after=120)
            + ooxml.paragraph(rest, line=360, after=600)
            + ooxml.paragraph(
                fa("امضاء دانشجو:", 12, bold=True), indent='<w:ind w:left="5000"/>', after=240
            )
            + ooxml.paragraph(fa("تاریخ:", 12, bold=True), indent='<w:ind w:left="5000"/>')
        )

    def copyright(self, m: Meta) -> str:
        items = [
            "حق چاپ و تکثیر این اثر متعلق به نویسنده و استاد راهنمای آن می‌باشد. هرگونه تصویربرداری از کل یا بخشی از آن تنها با موافقت استاد راهنما مجاز می‌باشد.",
            "کلیه حقوق معنوی این اثر متعلق به دانشگاه شهید باهنر کرمان می‌باشد و بدون اجازه کتبی دانشگاه به شخص ثالث قابل واگذاری نیست.",
            "استفاده از اطلاعات و نتایج موجود در این اثر بدون ذکر مرجع مجاز نمی‌باشد.",
        ]
        paras = "".join(
            ooxml.paragraph(
                fa(f"{fa_digits(str(k))}. ", 14) + fa(t, 14),
                line=360,
                after=120,
                indent='<w:ind w:left="567" w:hanging="340"/>',
            )
            for k, t in enumerate(items, start=1)
        )
        return (
            ooxml.page_break()
            + self._faculty(m, LOGO_RIDS[3])
            + centered(fa("حق طبع، نشر و مالکیت نتایج", 16, bold=True), after=360)
            + paras
        )

    def abstract(self, m: Meta, renderer: Renderer) -> str:
        return (
            ooxml.page_break()
            + ooxml.paragraph(fa("چکیده", 14, bold=True), after=240)
            + ooxml.paragraph(renderer.inline(m.abstract_fa), style="NewParagraph", after=360)
            + ooxml.paragraph(
                fa("واژه‌های کلیدی: ", 14, bold=True) + ooxml.text_runs(m.keywords_fa, size=14)
            )
        )

    @staticmethod
    def _list_page(title: str, code: str) -> str:
        head = ooxml.paragraph(
            fa("عنوان", 13, bold=True)
            + "<w:r><w:rPr><w:rtl/></w:rPr><w:tab/></w:r>"
            + fa("صفحه", 13, bold=True),
            tabs=f'<w:tab w:val="left" w:pos="{ooxml.TEXT_WIDTH_TWIPS - 1000}"/>',
            after=120,
            extra='<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="auto"/></w:pBdr>',
        )
        return (
            ooxml.page_break()
            + ooxml.paragraph(fa(title, 14, bold=True), after=240)
            + head
            + ooxml.paragraph(ooxml.field(code, "برای ساختن فهرست، فیلدها را به‌روز کنید."))
        )

    def contents(self) -> str:
        return (
            self._list_page("فهرست مطالب", 'TOC \\o "1-3" \\h \\z \\u')
            + self._list_page("فهرست شکل‌ها", 'TOC \\h \\z \\t "FigureCaption,4"')
            + self._list_page("فهرست جدول‌ها", 'TOC \\h \\z \\t "TableCaption,4"')
        )

    def abbreviations(self, m: Meta) -> str:
        rows = [
            [
                ooxml.paragraph(fa("عنوان", 12, bold=True), style="TableText", align="right"),
                ooxml.paragraph(fa("علامت اختصاری", 12, bold=True), style="TableText"),
            ]
        ]
        rows += [
            [
                ooxml.paragraph(ooxml.text_runs(full, size=11), style="TableText", align="right"),
                ooxml.paragraph(en(abbr, 10), style="TableText"),
            ]
            for abbr, full in m.abbreviations
        ]
        return (
            ooxml.page_break()
            + ooxml.paragraph(fa("فهرست علائم و اختصارات", 14, bold=True), after=240)
            + ooxml.table(rows)
        )

    # --- back matter ----------------------------------------------------------

    def english_pages(self, m: Meta) -> tuple[str, str]:
        def p(runs: str, align: str = "both", before: int = 0, after: int = 120) -> str:
            return ooxml.paragraph(
                runs, rtl=False, align=align, before=before, after=after, line=276
            )

        abstract = (
            p(en("Abstract", 14, bold=True), align="left", after=240)
            + "".join(p(en(par, 12)) for par in m.abstract_en.split("\n\n"))
            + p(en("Keywords: ", 12, bold=True) + en(m.keywords_en, 12), before=240)
        )
        title = (
            centered(self._logo(LOGO_RIDS[4], 3.2), after=360, rtl=False)
            + centered(en("Shahid Bahonar University of Kerman", 14), rtl=False)
            + centered(en("Faculty of Technical and Engineering", 14), after=720, rtl=False)
            + centered(en("A Report Submitted in Partial Fulfillment of the", 13), rtl=False)
            + centered(
                en(
                    "Requirements for the Degree of Bachelor of Science (B.Sc.) in Computer Engineering",
                    13,
                ),
                after=900,
                rtl=False,
            )
            + centered(en(m.title_en, 16, bold=True), after=900, rtl=False)
            + centered(en("By:", 12), rtl=False)
            + centered(en(m.author_en, 12, bold=True), after=480, rtl=False)
            + centered(en("Supervisor:", 12), rtl=False)
            + centered(en(m.supervisor_en, 12, bold=True), after=900, rtl=False)
            + centered(en(m.date_en, 12, bold=True), rtl=False)
        )
        return abstract, title

    # --- whole document -------------------------------------------------------

    def body(self, meta: Meta, blocks: list[Block], renderer: Renderer) -> str:
        def chapter_end(ch: Chapter) -> str:
            first = ch is renderer.chapters[0]
            rid = self.add_header(meta.short_title, ch.label)
            return section_paragraph(
                section(header=rid, footer=FOOTER_EMPTY, start=1 if first else None)
            )

        renderer.section_end = chapter_end
        front_sect = section(header=HEADER_EMPTY, footer=FOOTER_EMPTY)
        letters_sect = section(
            header=HEADER_EMPTY, footer=FOOTER_LETTERS, fmt="arabicAlpha", start=1
        )
        chapters = renderer.render(blocks)
        last = chapter_end(renderer.chapters[-1])
        abstract_en, title_en = self.english_pages(meta)
        return (
            self.title_page(meta)
            + self.bismillah()
            + section_paragraph(front_sect)
            + self.approval(meta).removeprefix(ooxml.page_break())
            + self.declaration(meta)
            + self.copyright(meta)
            + self.abstract(meta, renderer)
            + self.contents()
            + self.abbreviations(meta)
            + section_paragraph(letters_sect)
            + chapters
            + last
            + abstract_en
            + section_paragraph(section(header=HEADER_EMPTY, footer=FOOTER_EMPTY))
            + title_en
            + section(header=HEADER_EMPTY, footer=FOOTER_EMPTY)
        )

    def write(
        self, out: Path, body_xml: str, footnotes: list[tuple[int, str]], renderer: Renderer
    ) -> None:
        doc = self.parts["word/document.xml"].decode("utf-8")
        root = re.match(r".*?<w:body>", doc, re.S).group(0)
        document = f"{root}{body_xml}</w:body></w:document>"

        styles = self.parts["word/styles.xml"].decode("utf-8")
        styles = re.sub(r'w:cs="2\s+Lotus"', f'w:cs="{FONT_FA}"', styles)

        styles = styles.replace("</w:styles>", EXTRA_STYLES + "</w:styles>")

        notes = self.parts["word/footnotes.xml"].decode("utf-8")
        entries = []
        for fid, note in footnotes:
            is_latin = not re.search("[؀-ۿ]", note)
            runs = en(note, 10) if is_latin else renderer.inline(note, size=10)
            direction = '<w:bidi w:val="0"/><w:jc w:val="left"/>' if is_latin else "<w:bidi/>"
            entries.append(
                f'<w:footnote w:id="{fid}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/>{direction}</w:pPr>'
                '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>'
                f'<w:r><w:t xml:space="preserve"> </w:t></w:r>{runs}</w:p></w:footnote>'
            )
        notes = notes.replace("</w:footnotes>", "".join(entries) + "</w:footnotes>")

        rels = self.parts["word/_rels/document.xml.rels"].decode("utf-8")
        img_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        hdr_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
        new_rels = "".join(
            f'<Relationship Id="{rid}" Type="{img_type}" Target="{escape(name.removeprefix("word/"))}"/>'
            for rid, name, _ in self.media
        ) + "".join(
            f'<Relationship Id="{rid}" Type="{hdr_type}" Target="{name.removeprefix("word/")}"/>'
            for rid, name, _ in self.headers
        )
        rels = rels.replace("</Relationships>", new_rels + "</Relationships>")

        types = self.parts["[Content_Types].xml"].decode("utf-8")
        hdr_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"
        types = types.replace(
            "</Types>",
            "".join(
                f'<Override PartName="/{name}" ContentType="{hdr_ct}"/>'
                for _, name, _ in self.headers
            )
            + "</Types>",
        )
        if 'Extension="png"' not in types:
            types = types.replace(
                "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>'
            )

        replaced = {
            "word/document.xml": document,
            "word/styles.xml": styles,
            "word/footnotes.xml": notes,
            "word/_rels/document.xml.rels": rels,
            "[Content_Types].xml": types,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in self.parts.items():
                z.writestr(name, replaced[name].encode("utf-8") if name in replaced else data)
            for _, name, data in self.media:
                z.writestr(name, data)
            for _, name, xml in self.headers:
                z.writestr(name, xml.encode("utf-8"))
