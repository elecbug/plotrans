"""Offline MathML for OCR LaTeX; no browser scripts or remote fonts required."""

from __future__ import annotations

from html import escape
import re
import xml.etree.ElementTree as ET

from latex2mathml.converter import convert


_CURRENCY = r"\s*\d[\d,.]*(?:[kKmMbBtT])?\s*(?i:USD|EUR|GBP|CAD|AUD|dollars?|million|billion)\b"
LATEX = re.compile(r"(?<!\\)\$\$([\s\S]+?)\$\$|(?<!\\)\$(?!\$)(?!" + _CURRENCY + r")([^$\n]+?)\$(?!\$)|\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]")
_TAGS = set("math mrow mi mo mn mtext mspace mfrac mroot msqrt msup msub msubsup mover munder munderover mtable mtr mtd mlabeledtr mfenced menclose mstyle mpadded mphantom none mmultiscripts mprescripts semantics annotation".split())
_ATTRS = set("display mathvariant stretchy fence separator lspace rspace accent accentunder movablelimits largeop columnalign rowalign columnspacing rowspacing columnlines rowlines frame framespacing equalrows equalcolumns align rowspan columnspan notation width height depth voffset linethickness bevelled scriptlevel displaystyle encoding".split())
ET.register_namespace("", "http://www.w3.org/1998/Math/MathML")


def expression(match: re.Match) -> str:
    return next(group for group in match.groups() if group is not None).strip()


def mathml(latex: str, display: bool = False) -> str:
    # The converter accepts some incomplete input; reject incomplete groups
    # before conversion instead of displaying a plausible but truncated formula.
    depth = 0
    for token in re.findall(r"\\.|[{}]", latex):
        if token == "{": depth += 1
        elif token == "}": depth -= 1
        if depth < 0: raise ValueError("Unbalanced LaTeX group")
    if depth or len(re.findall(r"\\left\b", latex)) != len(re.findall(r"\\right\b", latex)):
        raise ValueError("Incomplete LaTeX expression")
    root = ET.fromstring(convert(latex, display="block" if display else "inline"))
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] not in _TAGS:
            raise ValueError("Unsupported MathML element")
        node.attrib = {key: value for key, value in node.attrib.items() if key in _ATTRS}
    rendered = ET.tostring(root, encoding="unicode")
    # Markdown exports can be opened by other CommonMark renderers. Encode
    # operator characters that would otherwise start emphasis inside MathML.
    return rendered.replace(">*<", ">&#42;<").replace(">_<", ">&#95;<")


def valid_math(text: str) -> bool:
    try:
        for match in LATEX.finditer(text):
            mathml(expression(match), bool(match.group(1) or match.group(4)))
        # A lone dollar in a transcription usually means it was cut mid-formula.
        return len(re.findall(r"(?<!\\)\$", text)) % 2 == 0
    except Exception:
        return False


def render_math(text: str) -> str:
    def render(match):
        try:
            return mathml(expression(match), bool(match.group(1) or match.group(4)))
        except Exception:
            return "<code>" + escape(match.group(0)) + "</code>"
    return LATEX.sub(render, text)
