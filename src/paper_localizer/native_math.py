"""Conservative equation transcription from decoded source glyph geometry."""

from __future__ import annotations

from statistics import median
import math
import re
import unicodedata

import pypdfium2 as pdfium

from .math_markup import valid_math


_SYMBOLS = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ϵ": r"\epsilon", "θ": r"\theta", "ϑ": r"\vartheta",
    "κ": r"\kappa", "λ": r"\lambda",
    "μ": r"\mu", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "τ": r"\tau", "φ": r"\varphi", "ϕ": r"\phi", "ψ": r"\psi", "ω": r"\omega",
    "ϖ": r"\varpi", "ϱ": r"\varrho", "ς": r"\varsigma",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq", "≈": r"\approx",
    "±": r"\pm", "×": r"\times", "÷": r"\div", "∞": r"\infty",
    "∈": r"\in", "∉": r"\notin", "∪": r"\cup", "∩": r"\cap",
    "→": r"\to", "∥": r"\Vert", "−": "-", "–": "-",
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ℓ": r"\ell",
    "{": r"\{", "}": r"\}", "_": r"\_",
}
_PUNCTUATION = set("+*/=<>()[].,:;|^_{}-")


def native_expression(page, box) -> str | None:
    """Accept only decoded single-line glyphs with explainable script positions.

    The source image stays authoritative. Fractions, multiline layouts, missing
    Unicode mappings, and unexplained baseline changes defer to the OCR path.
    """
    textpage = page.get_textpage()
    chars = []
    try:
        for index in range(textpage.count_chars()):
            code = pdfium.raw.FPDFText_GetUnicode(textpage, index)
            bounds = textpage.get_charbox(index)
            x, y = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]): continue
            if bounds[0] == bounds[2] or bounds[1] == bounds[3]: continue
            if pdfium.raw.FPDFText_HasUnicodeMapError(textpage, index) != 0: return None
            if code in {9, 10, 13, 32}: continue
            if not 0 < code <= 0x10FFFF: return None
            original = chr(code)
            if original == "^": return None
            unicode_name = unicodedata.name(original, "")
            styled_math = unicode_name.startswith("MATHEMATICAL ") and not unicode_name.startswith("MATHEMATICAL ITALIC ")
            if (styled_math or any(style in unicode_name for style in ("DOUBLE-STRUCK", "FRAKTUR", "SCRIPT"))) and original not in _SYMBOLS:
                return None
            char = original if original in _SYMBOLS else unicodedata.normalize("NFKC", original)
            if original != char and "GREEK" in unicode_name and original not in _SYMBOLS:
                return None
            if any(not (c.isalnum() or c in _SYMBOLS or c in _PUNCTUATION) for c in char): return None
            obj = textpage.get_textobj(index)
            if obj is None: return None
            # Object handles can repeat across characters; do not retain them.
            name = obj.get_font().get_base_name()
            if not name: return None
            matrix = obj.get_matrix()
            size = obj.get_font_size() * math.hypot(matrix.c, matrix.d)
            if not size or size <= 0: return None
            if not bounds[1]-size*.6 <= matrix.f <= bounds[3]+size*.3: return None
            explicit = "sup" if "SUPERSCRIPT" in unicode_name else "sub" if "SUBSCRIPT" in unicode_name else ""
            chars.append((char, size, matrix.f, name, explicit))
            if len(chars) > 128: return None
    finally:
        textpage.close()
    if not chars: return None
    largest = max(size for _, size, _, _, _ in chars)
    base = [baseline for _, size, baseline, _, _ in chars if size >= largest*.9]
    if max(base)-min(base) > max(1, largest*.2): return None
    baseline = median(base)
    groups = []
    for char, size, y, font, explicit in chars:
        delta = y-baseline
        role = explicit or "base"
        if explicit:
            pass
        elif size < largest*.9:
            if largest*.08 < delta < largest*.9: role = "sup"
            elif -largest*.9 < delta < -largest*.08: role = "sub"
            else: return None
        elif abs(delta) > largest*.2:
            return None
        if not groups or groups[-1][0] != role:
            groups.append([role, []])
        groups[-1][1].append((char, font))
    if groups[0][0] != "base": return None
    # A small numerator and denominator beside normal-size text can look like
    # simultaneous scripts. Their fraction bar is not a text glyph, so defer
    # these ambiguous two-dimensional layouts rather than inventing scripts.
    if {"sup", "sub"}.issubset({role for role, _ in groups}): return None
    result = []
    for role, group in groups:
        text = "".join(_SYMBOLS.get(char, char) + (" " if char in _SYMBOLS and _SYMBOLS[char].startswith("\\") else "")
                       for char, _ in group).strip()
        if role != "base":
            if re.fullmatch(r"[A-Za-z]{2,}", text) and all(not re.search(r"Italic|Oblique|It$", font, re.I) for _, font in group):
                text = r"\mathrm{" + text + "}"
            text = ("^{" if role == "sup" else "_{") + text + "}"
        result.append(text)
    expression = "".join(result)
    return expression if valid_math("$$"+expression+"$$") else None
