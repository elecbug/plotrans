"""Shared parsing for generated Markdown images and inline HTML images."""

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
import math
import re


FIGURE_LINK = re.compile(
    r"!\[([^\]\n]*)\]\((figs/[A-Za-z0-9_./-]+)\)"
    r"|<img\b(?=[^>]*\bsrc\s*=\s*[\"']figs/)[^>]*>", re.IGNORECASE
)


class _ImageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.attributes = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            self.attributes = dict(attrs)


@dataclass(frozen=True)
class ImageReference:
    source: str
    alt: str
    inline: bool = False
    height_em: float = 1.0
    align_em: float = -0.15
    aspect_ratio: float = 1.0


def image_reference(match: re.Match) -> ImageReference:
    if match.group(2):
        return ImageReference(match.group(2), match.group(1))
    parser = _ImageParser()
    parser.feed(match.group(0))
    attrs = parser.attributes
    source = attrs.get("src", "")
    if not re.fullmatch(r"figs/[A-Za-z0-9_./-]+", source):
        raise ValueError("Unsupported inline image path.")
    style = attrs.get("style", "") or ""
    height = re.search(r"(?:^|;)\s*height\s*:\s*([0-9.]+)em", style)
    align = re.search(r"(?:^|;)\s*vertical-align\s*:\s*([-+0-9.]+)em", style)
    try:
        aspect = float(attrs.get("width", "16")) / max(float(attrs.get("height", "16")), 1)
        return ImageReference(source, attrs.get("alt", "Equation"), True,
                              float(height.group(1)) if height else 1.0,
                              float(align.group(1)) if align else -0.15, aspect)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid inline image size attributes.") from exc


def inline_image_tag(source: str, alt: str, height_em: float, align_em: float, aspect_ratio: float) -> str:
    values = (height_em, align_em, aspect_ratio)
    if not all(math.isfinite(v) for v in values) or height_em <= 0 or aspect_ratio <= 0:
        raise ValueError("Inline image dimensions must be finite positive values.")
    height_em = min(max(height_em, 0.2), 20)
    align_em = min(max(align_em, -20), 20)
    height_px = max(1, round(height_em * 16))
    width_px = max(1, round(height_px * aspect_ratio))
    # Pixel attributes provide a small inline fallback when a viewer strips CSS.
    return (
        f'<img src="{escape(source, quote=True)}" alt="{escape(alt, quote=True)}" '
        f'class="inline-math" width="{width_px}" height="{height_px}" '
        f'style="display:inline-block;height:{height_em:.3f}em;width:auto;max-width:none;'
        f'vertical-align:{align_em:.3f}em;" />'
    )
