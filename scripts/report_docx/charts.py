"""The report's charts as standalone pgfplots files, rendered to PNG for Word.

Tick labels and value labels are written here in Persian digits (B Nazanin has
them; pgfplots would print Latin ones), with the decimal separator «٫» taken
from Times New Roman because B Nazanin lacks that glyph. Data come from
reports/data/*.csv, written by scripts/make_reports.py.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from scripts.report_docx.text import LATIN, fa_digits

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{pgfplots}
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\pgfplotsset{compat=1.18}
\usepackage{xepersian}
\settextfont{B Nazanin}
\setlatintextfont{Times New Roman}
\newfontfamily\sepfont{Times New Roman}
\pgfplotsset{every axis/.append style={font=\small}, tick label style={font=\small},
  legend style={font=\small}, grid style={gray!25}}
\begin{document}
"""
END = "\n\\end{document}\n"
SEPARATOR = '{\\sepfont\\char"066B\\relax}'
MINUS = '{\\sepfont\\char"2212\\relax}'  # B Nazanin lacks the minus sign too


def num(x: float, decimals: int = 2) -> str:
    """A number for a chart label: Persian digits, «٫» from Times New Roman, «−» for minus."""
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    out = fa_digits(whole) + (SEPARATOR + fa_digits(frac) if frac else "")
    return (MINUS if x < 0 and float(s) != 0 else "") + out


def ticks(values: list[float], decimals: int) -> str:
    positions = ",".join(f"{v:g}" for v in values)
    labels = ",".join("{" + num(v, decimals) + "}" for v in values)
    return f"{positions}", labels


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sweep(data: Path) -> str:
    xs, xl = ticks([0.1 * k for k in range(1, 10)], 1)
    ys, yl = ticks([0.2, 0.4, 0.6, 0.8, 1.0], 1)
    path = (data / "threshold_sweep.csv").as_posix()
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=13cm, height=7cm, xmin=0.05, xmax=0.9, ymin=0.2, ymax=1,
  xtick={{{xs}}}, xticklabels={{{xl}}}, ytick={{{ys}}}, yticklabels={{{yl}}},
  xlabel={{\rl{{آستانه}}}}, ylabel={{\rl{{مقدار}}}}, grid=major,
  legend pos=south west, legend cell align=right, table/col sep=comma]
\addplot[blue!70!black, thick, mark=*, mark size=1.3pt] table[x=threshold, y=f1] {{{path}}};
\addlegendentry{{\rl{{اف‌وان (همان خانه)}}}}
\addplot[teal, thick, mark=square*, mark size=1.3pt] table[x=threshold, y=f1nb] {{{path}}};
\addlegendentry{{\rl{{اف‌وان (همسایه)}}}}
\addplot[orange!80!black, thick, dashed, mark=triangle*, mark size=1.5pt] table[x=threshold, y=unmatched] {{{path}}};
\addlegendentry{{\rl{{بازیابی بدون AIS (همان خانه)}}}}
\draw[red!70!black, densely dotted, thick] (axis cs:0.6,0.2) -- (axis cs:0.6,1);
\end{{axis}}
\end{{tikzpicture}}"""


def label_bias(data: Path) -> str:
    rows = {r["detector"]: r for r in _rows(data / "label_bias.csv")}
    names = {"CFAR": r"\rl{CFAR}", "run1": r"\rl{مدل نخست}", "run2": r"\rl{مدل نهایی}"}
    ys, yl = ticks([0, 0.25, 0.5, 0.75, 1.0], 2)

    def series(col: str) -> str:
        return " ".join(
            f"({k},{float(rows[k][col]):.4f}) [{num(float(rows[k][col]))}]" for k in names
        )

    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=11cm, height=7cm, ybar, bar width=18pt, ymin=0, ymax=1.1,
  symbolic x coords={{CFAR,run1,run2}}, xtick=data, xticklabels={{{",".join("{" + v + "}" for v in names.values())}}},
  ytick={{{ys}}}, yticklabels={{{yl}}}, ylabel={{\rl{{بازیابی (همان خانه)}}}},
  enlarge x limits=0.25, area legend, legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=2}},
  nodes near coords, point meta=explicit symbolic, every node near coord/.append style={{font=\footnotesize}}]
\addplot[fill=blue!45] coordinates {{{series("matched")}}};
\addlegendentry{{\rl{{با AIS}}}}
\addplot[fill=orange!65] coordinates {{{series("unmatched")}}};
\addlegendentry{{\rl{{بدون AIS}}}}
\end{{axis}}
\end{{tikzpicture}}"""


def generalisation(data: Path) -> str:
    rows = _rows(data / "generalisation.csv")
    names = [
        r"\rl{آزمون امارات}",
        r"\rl{امارات ۲۰۲۶}",
        r"\rl{امارات S1C/D}",
        r"\rl{ایران S1A}",
        r"\rl{ایران S1C/D}",
    ]
    ys, yl = ticks([-0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20], 2)
    points = {"all": [], "shore_1000m": []}
    for k, r in enumerate(rows):
        x = k // 2 + (0.15 if r["stratum"] == "shore_1000m" else -0.15)
        d, lo, hi = float(r["diff"]), float(r["low"]), float(r["high"])
        points[r["stratum"]].append(f"({x:.2f},{d:.4f}) +- (0,{max(hi - d, lo - d, key=abs):.4f})")

    def plot(key: str, style: str, legend: str) -> str:
        coords = []
        for k, r in enumerate(x for x in rows if x["stratum"] == key):
            x = k + (0.15 if key == "shore_1000m" else -0.15)
            d, lo, hi = float(r["diff"]), float(r["low"]), float(r["high"])
            coords.append(f"({x:.2f},{d:.4f}) += (0,{hi - d:.4f}) -= (0,{d - lo:.4f})")
        return (
            f"\\addplot[{style}, only marks, error bars/.cd, y dir=both, y explicit] coordinates {{{' '.join(coords)}}};\n"
            f"\\addlegendentry{{\\rl{{{legend}}}}}"
        )

    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=15cm, height=7.5cm, xmin=-0.6, xmax=4.6, ymin=-0.045, ymax=0.22,
  xtick={{0,1,2,3,4}}, xticklabels={{{",".join("{" + n + "}" for n in names)}}},
  ytick={{{ys}}}, yticklabels={{{yl}}}, ylabel={{\rl{{اف‌وان مدل منهای CFAR}}}}, grid=major,
  legend pos=north west, legend cell align=right]
{plot("all", "blue!70!black, mark=*", "امتیازدهی استاندارد")}
{plot("shore_1000m", "orange!80!black, mark=square*", "بیش از ۱ کیلومتر از خشکی")}
\draw[gray, thin] (axis cs:-0.6,0) -- (axis cs:4.6,0);
\end{{axis}}
\end{{tikzpicture}}"""


def dark_margin(data: Path) -> str:
    rows = _rows(data / "dark_margin.csv")
    xl = ",".join("{" + fa_digits(r["margin"]) + "}" for r in rows)
    ys, yl = ticks([0, 2, 4, 6, 8], 0)
    test = " ".join(f"({k},{float(r['test']):.2f})" for k, r in enumerate(rows))
    s1cd = " ".join(f"({k},{float(r['s1cd']):.2f})" for k, r in enumerate(rows))
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=12cm, height=7cm, xmin=0, xmax={len(rows) - 1}, ymin=0, ymax=8.5,
  xtick={{0,...,{len(rows) - 1}}}, xticklabels={{{xl}}}, ytick={{{ys}}}, yticklabels={{{yl}}},
  xlabel={{\rl{{حاشیه دروازه (متر)}}}}, ylabel={{\rl{{نرخ تاریک (درصد)}}}}, grid=major,
  legend pos=north east, legend cell align=right]
\addplot[blue!70!black, thick, mark=*] coordinates {{{test}}};
\addlegendentry{{\rl{{آزمون S1A ۲۰۲۴ تا ۲۰۲۵}}}}
\addplot[orange!80!black, thick, mark=square*] coordinates {{{s1cd}}};
\addlegendentry{{\rl{{S1C/S1D سپتامبر ۲۰۲۶}}}}
\end{{axis}}
\end{{tikzpicture}}"""


def ais_trend(data: Path) -> str:
    rows = _rows(data / "ais_trend.csv")
    months = [0, 6, 12, 18, 24, 30]
    xl = ",".join("{" + fa_digits(rows[i]["month"]) + "}" for i in months)
    xs = ",".join(str(i) for i in months)
    ys, yl = ticks([0, 0.2, 0.4, 0.6, 0.8], 1)
    hs, hl = ticks([0, 50, 100, 150], 0)
    west = " ".join(f"({r['i']},{r['west']})" for r in rows)
    east = " ".join(f"({r['i']},{r['east']})" for r in rows)
    hours = " ".join(f"({r['i']},{r['hours']})" for r in rows)
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[name=top, width=15cm, height=6cm, xmin=0, xmax=32, ymin=0, ymax=0.85,
  xtick={{{xs}}}, xticklabels={{{xl}}}, ytick={{{ys}}}, yticklabels={{{yl}}},
  ylabel={{\rl{{سهم تطبیق‌یافته با AIS}}}}, grid=major, legend pos=south west, legend cell align=right]
\addplot[blue!70!black, thick, mark=*, mark size=1pt] coordinates {{{west}}};
\addlegendentry{{\rl{{غرب ۵۶ درجه شرقی}}}}
\addplot[orange!80!black, thick, dashed, mark=square*, mark size=1pt] coordinates {{{east}}};
\addlegendentry{{\rl{{شرق ۵۶ درجه شرقی}}}}
\end{{axis}}
\begin{{axis}}[at={{(top.south west)}}, anchor=north west, yshift=-1.2cm, width=15cm, height=5cm,
  xmin=0, xmax=32, ymin=0, ymax=170, xtick={{{xs}}}, xticklabels={{{xl}}},
  ytick={{{hs}}}, yticklabels={{{hl}}}, ylabel={{\rl{{هزار ساعت-شناور در روز}}}}, grid=major]
\addplot[teal, thick, mark=*, mark size=1pt] coordinates {{{hours}}};
\end{{axis}}
\end{{tikzpicture}}"""


def study_map(data: Path) -> str:
    xs = list(range(48, 61, 2))
    ys = list(range(23, 31))
    xl = ",".join("{" + fa_digits(str(v)) + "}" for v in xs)
    yl = ",".join("{" + fa_digits(str(v)) + "}" for v in ys)
    coast, boxes = (data / "coast.csv").as_posix(), (data / "boxes.csv").as_posix()
    labels = [
        (51.6, 26.9, "خلیج فارس", r"\normalsize"),
        (58.6, 24.3, "دریای عمان", r"\normalsize"),
        (56.25, 27.45, "بندرعباس", r"\footnotesize"),
        (50.3, 29.65, "خارک", r"\footnotesize"),
        (53.98, 26.15, "کیش", r"\footnotesize"),
        (57.8, 25.2, "جاسک", r"\footnotesize"),
        (56.95, 24.95, "فجیره", r"\footnotesize"),
        (55.2, 25.05, "دبی", r"\footnotesize"),
        (52.55, 27.75, "عسلویه", r"\footnotesize"),
        (51.2, 28.85, "بوشهر", r"\footnotesize"),
    ]
    nodes = "\n".join(
        f"\\node[font={size}] at (axis cs:{x},{y}) {{\\rl{{{t}}}}};" for x, y, t, size in labels
    )
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=16cm, height=13cm, xmin=47.5, xmax=60, ymin=23, ymax=30.6,
  unit vector ratio=0.895 1, unbounded coords=jump, axis background/.style={{fill=blue!4}},
  xtick={{{",".join(map(str, xs))}}}, xticklabels={{{xl}}}, ytick={{{",".join(map(str, ys))}}}, yticklabels={{{yl}}},
  xlabel={{\rl{{طول جغرافیایی (درجه شرقی)}}}}, ylabel={{\rl{{عرض جغرافیایی (درجه شمالی)}}}}, table/col sep=comma]
\addplot[gray!60!black, thin] table[x=lon, y=lat] {{{coast}}};
\addplot[red!80!black, thick] table[x=lon, y=lat] {{{boxes}}};
{nodes}
\end{{axis}}
\end{{tikzpicture}}"""


CHARTS: dict[str, Callable[[Path], str]] = {
    "map": study_map,
    "sweep": sweep,
    "labelbias": label_bias,
    "generalisation": generalisation,
    "margin": dark_margin,
    "aistrend": ais_trend,
}


def latin_in_labels(body: str) -> str:
    """Wrap Latin words inside \\rl{...} labels in \\lr{...}: B Nazanin has no Latin glyphs."""

    def fix(m: re.Match) -> str:
        return "\\rl{" + LATIN.sub(lambda w: "\\lr{" + w.group(0) + "}", m.group(1)) + "}"

    return re.sub(r"\\rl\{([^{}]*)\}", fix, body)


def build(
    out_dir: Path, data: Path, xelatex: Path, extra: dict[str, str] | None = None, dpi: int = 300
) -> dict[str, Path]:
    """Write, compile and rasterise every chart; returns {name: png path}."""
    import pypdfium2 as pdfium  # .venv-docs only

    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: make(data) for name, make in CHARTS.items()} | (extra or {})
    pngs = {}
    for name, body in sources.items():
        tex = out_dir / f"{name}.tex"
        tex.write_text(PREAMBLE + latin_in_labels(body) + END, encoding="utf-8")
        result = subprocess.run(
            [str(xelatex), "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=out_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            log = (out_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")
            error = next((line for line in log.splitlines() if line.startswith("!")), "see log")
            raise RuntimeError(f"chart {name} failed: {error}")
        pdf = pdfium.PdfDocument(str(out_dir / f"{name}.pdf"))
        png = out_dir / f"{name}.png"
        pdf[0].render(scale=dpi / 72).to_pil().save(png, dpi=(dpi, dpi))
        pdf.close()
        pngs[name] = png
    return pngs
