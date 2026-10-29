"""Add explicit Persian paragraph metadata absent from Artifact Tool's facade."""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET
from zipfile import ZipFile

target = Path(sys.argv[1])
ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
with ZipFile(target) as source:
    parts = {info.filename: source.read(info.filename) for info in source.infolist()}
for name, data in parts.items():
    if not name.startswith(("ppt/slides/slide", "ppt/notesSlides/notesSlide")) or not name.endswith(".xml"):
        continue
    root = ET.fromstring(data)
    for p in root.findall(".//a:p", ns):
        ppr = p.find("a:pPr", ns)
        if ppr is None:
            ppr = ET.Element(f"{{{ns['a']}}}pPr")
            p.insert(0, ppr)
        ppr.set("rtl", "1")
        for rpr in p.findall(".//a:rPr", ns) + p.findall("a:endParaRPr", ns):
            rpr.set("lang", "fa-IR")
            rpr.set("altLang", "en-US")
            cs = rpr.find("a:cs", ns)
            if cs is None:
                cs = ET.SubElement(rpr, f"{{{ns['a']}}}cs")
            cs.set("typeface", "B Nazanin")
    parts[name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
with NamedTemporaryFile(dir=target.parent, suffix=".pptx", delete=False) as tmp:
    staging = Path(tmp.name)
with ZipFile(staging, "w") as output:
    for name, data in parts.items():
        output.writestr(name, data)
staging.replace(target)
