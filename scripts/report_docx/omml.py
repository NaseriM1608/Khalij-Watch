"""A small LaTeX subset to Office Math (OMML), so the report's equations are editable in Word.

Supported: letters, digits and operators; ``x_{..}``, ``x^{..}``; ``\\frac{}{}``,
``\\sqrt{}``, ``\\hat{}``, ``\\bar{}``, ``\\text{}``, ``\\mathrm{}``,
``\\operatorname{}``; ``\\sum`` with limits; ``\\begin{cases} a & b \\\\ c & d
\\end{cases}``; ``\\left``/``\\right`` (dropped, the delimiter stays); Greek
letters and a few symbols. Anything else raises, so a typo cannot slip through
as literal text.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

_SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "omega": "ω",
    "Delta": "Δ",
    "Sigma": "Σ",
    "Omega": "Ω",
    "cdot": "·",
    "times": "×",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "approx": "≈",
    "in": "∈",
    "pm": "±",
    "infty": "∞",
    "to": "→",
    "ldots": "…",
    "mid": "|",
    "lvert": "|",
    "rvert": "|",
    "cap": "∩",
    "cup": "∪",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "quad": " ",
    ",": " ",
    ";": " ",
    "{": "{",
    "}": "}",
    "%": "%",
}
_FUNCTIONS = {"log", "ln", "exp", "max", "min", "arg", "sin", "cos", "tan"}
_ACCENTS = {"hat": "̂", "bar": "̅", "tilde": "̃"}


def _run(t: str, upright: bool = False) -> str:
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    return f'<m:r>{sty}<m:t xml:space="preserve">{escape(t)}</m:t></m:r>'


class _Parser:
    def __init__(self, src: str) -> None:
        self.tokens = self._tokenize(src)
        self.i = 0

    @staticmethod
    def _tokenize(src: str) -> list[str]:
        tokens, i = [], 0
        while i < len(src):
            c = src[i]
            if c == "\\":
                if src.startswith("\\\\", i):
                    tokens.append("\\\\")
                    i += 2
                    continue
                j = i + 1
                while j < len(src) and src[j].isalpha():
                    j += 1
                if j == i + 1:  # \, \; \{ \} \%
                    j = i + 2
                tokens.append(src[i:j])
                i = j
            elif c.isspace():
                i += 1
            else:
                tokens.append(c)
                i += 1
        return tokens

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of equation")
        self.i += 1
        return token

    def raw_group(self) -> str:
        """The literal text of a {...} group (for \\text)."""
        if self.take() != "{":
            raise ValueError("expected '{'")
        depth, parts = 1, []
        while True:
            t = self.take()
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    return " ".join(parts) if any(len(p) > 1 for p in parts) else "".join(parts)
            parts.append(t.lstrip("\\") if t.startswith("\\") and t[1:] in _SYMBOLS else t)

    def group(self) -> str:
        """A {...} group or a single atom, as OMML."""
        if self.peek() == "{":
            self.take()
            out = self.expr(stop={"}"})
            self.take()
            return out
        return self.atom()

    def expr(self, stop: set[str]) -> str:
        out = []
        while self.peek() is not None and self.peek() not in stop:
            out.append(self.atom())
        return "".join(out)

    def scripts(self, base: str) -> str:
        sub = sup = None
        while self.peek() in ("_", "^"):
            if self.take() == "_":
                sub = self.group()
            else:
                sup = self.group()
        if sub is not None and sup is not None:
            return (
                f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            )
        if sub is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
        if sup is not None:
            return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
        return base

    def cases(self) -> str:
        rows, cells, cell = [], [], []
        while True:
            t = self.peek()
            if t is None:
                raise ValueError("unclosed cases")
            if t == "\\end":
                self.take()
                self.raw_group()
                break
            if t == "&":
                self.take()
                cells.append("".join(cell))
                cell = []
            elif t == "\\\\":
                self.take()
                cells.append("".join(cell))
                rows.append(cells)
                cells, cell = [], []
            else:
                cell.append(self.atom())
        if cell or cells:
            cells.append("".join(cell))
            rows.append(cells)
        gap = _run("  ", upright=True)
        arr = "".join(f"<m:e>{gap.join(r)}</m:e>" for r in rows)
        return (
            '<m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val=""/></m:dPr>'
            f"<m:e><m:eqArr>{arr}</m:eqArr></m:e></m:d>"
        )

    def atom(self) -> str:
        t = self.take()
        if t == "{":
            out = self.expr(stop={"}"})
            self.take()
            return self.scripts(out)
        if not t.startswith("\\"):
            if t in ("_", "^", "}", "&"):
                raise ValueError(f"unexpected {t!r}")
            return self.scripts(_run(t, upright=not t.isalpha()))
        name = t[1:]
        if name == "frac":
            num, den = self.group(), self.group()
            return self.scripts(f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>")
        if name == "sqrt":
            arg = self.group()
            return self.scripts(
                f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{arg}</m:e></m:rad>'
            )
        if name in _ACCENTS:
            arg = self.group()
            return self.scripts(
                f'<m:acc><m:accPr><m:chr m:val="{_ACCENTS[name]}"/></m:accPr><m:e>{arg}</m:e></m:acc>'
            )
        if name in ("text", "mathrm", "operatorname"):
            return self.scripts(_run(self.raw_group(), upright=True))
        if name == "sum":
            sub = sup = ""
            while self.peek() in ("_", "^"):
                if self.take() == "_":
                    sub = self.group()
                else:
                    sup = self.group()
            body = self.atom() if self.peek() not in (None, "}") else ""
            return (
                '<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr>'
                f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>"
            )
        if name == "begin":
            env = self.raw_group()
            if env != "cases":
                raise ValueError(f"unknown environment {env!r}")
            return self.cases()
        if name in ("left", "right"):
            return self.atom() if self.peek() not in (None, ".") else (self.take() and "")
        if name in _FUNCTIONS:
            return self.scripts(_run(name, upright=True))
        if name in _SYMBOLS:
            return self.scripts(_run(_SYMBOLS[name], upright=not _SYMBOLS[name].isalpha()))
        raise ValueError(f"unknown command \\{name}")


def to_omml(src: str) -> str:
    """One equation as an ``<m:oMath>`` element."""
    parser = _Parser(src)
    body = parser.expr(stop=set())
    return f"<m:oMath>{body}</m:oMath>"
