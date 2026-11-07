"""Portable Markdown exports with their referenced figure files."""

from io import BytesIO
import base64
from html import escape
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from .markup import FIGURE_LINK, image_reference, inline_image_tag



def figure_path(markdown_path: Path, reference: str) -> Path:
    base = markdown_path.parent.resolve()
    path = (base / reference).resolve()
    if not path.is_relative_to(base / "figs"):
        raise ValueError("Figure paths must stay inside the figs directory.")
    return path


def build_bundle(markdown_path: Path) -> bytes:
    markdown = markdown_path.read_text(encoding="utf-8")
    html_path = markdown_path.with_suffix(".html")
    html = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
    stream = BytesIO()
    references = {image_reference(match).source for match in FIGURE_LINK.finditer(markdown + "\n" + html)}
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr(markdown_path.name, markdown.encode("utf-8"))
        for suffix in ('.ocr.json', '.translation.json'):
            report = markdown_path.with_suffix(suffix)
            if report.is_file():
                archive.write(report, report.name)
        if html:
            archive.writestr(html_path.name, html.encode("utf-8"))
            for reference in sorted(set(re.findall(r'href="(assets/[A-Za-z0-9_./-]+)"', html))):
                archive.write(_stylesheet_path(markdown_path.parent, reference), reference)
        for reference in sorted(references):
            archive.write(figure_path(markdown_path, reference), reference)
    return stream.getvalue()


def _stylesheet_path(directory: Path, reference: str) -> Path:
    base = directory.resolve()
    path = (base / reference).resolve()
    if not path.is_relative_to(base / "assets"):
        raise ValueError("Stylesheet paths must stay inside the assets directory.")
    return path


def standalone_html(html_path: Path) -> str:
    """Embed local assets for the app's isolated article preview."""
    html = html_path.read_text(encoding="utf-8")
    for match in list(FIGURE_LINK.finditer(html))[::-1]:
        reference = image_reference(match)
        data = base64.b64encode(figure_path(html_path, reference.source).read_bytes()).decode("ascii")
        image = match.group(0).replace(reference.source, "data:image/png;base64," + data, 1)
        html = html[:match.start()] + image + html[match.end():]
    def stylesheet(match):
        css = _stylesheet_path(html_path.parent, match.group(1)).read_text(encoding="utf-8")
        return "<style>" + css + "</style>"
    return re.sub(r'<link rel="stylesheet" href="(assets/[A-Za-z0-9_./-]+)">', stylesheet, html)


def inline_equation_preview(markdown_path: Path, markdown: str) -> str:
    """Render trusted image tags inline and escape all other source HTML."""
    parts: list[str] = []
    cursor = 0
    for match in FIGURE_LINK.finditer(markdown):
        parts.append(escape(markdown[cursor:match.start()], quote=False))
        reference = image_reference(match)
        if reference.inline or "-math-" in reference.source:
            encoded = base64.b64encode(figure_path(markdown_path, reference.source).read_bytes()).decode("ascii")
            source = f"data:image/png;base64,{encoded}"
            if reference.inline:
                parts.append(inline_image_tag(source, reference.alt, reference.height_em,
                                              reference.align_em, reference.aspect_ratio))
            else:
                parts.append(f"![{escape(reference.alt)}]({source})")
        else:
            parts.append(f"![{escape(reference.alt)}]({reference.source})")
        cursor = match.end()
    parts.append(escape(markdown[cursor:], quote=False))
    return "".join(parts)
