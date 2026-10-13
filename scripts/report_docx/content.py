"""Metadata and reference-list files of the report.

meta.md:        ``key: value`` lines, ``:::name`` ... ``:::`` blocks for long texts,
                and an ``:::abbreviations`` block of ``ABBR | Persian`` lines.
references.md:  one ``key | formatted reference`` line per work; ``%%`` comments.
"""

from __future__ import annotations


def parse_meta(src: str) -> dict:
    meta: dict = {}
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("%%"):
            continue
        if line.startswith(":::"):
            name = line[3:].strip()
            body = []
            while i < len(lines) and lines[i].strip() != ":::":
                body.append(lines[i])
                i += 1
            i += 1
            if name == "abbreviations":
                meta[name] = [tuple(s.strip() for s in b.split("|", 1)) for b in body if b.strip()]
            else:
                meta[name] = "\n".join(body).strip()
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"expected 'key: value', got {line!r}")
        meta[key.strip()] = value.strip()
    return meta


def parse_references(src: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith("%%"):
            continue
        key, _, text = line.partition("|")
        key = key.strip()
        if key in refs:
            raise ValueError(f"duplicated reference key {key}")
        refs[key] = text.strip()
    return refs
