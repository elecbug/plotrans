"""Portable Markdown exports with their referenced figure files."""

from io import BytesIO
import base64
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .markup import FIGURE_LINK, image_reference, inline_image_tag



def figure_path(markdown_path: Path, reference: str) -> Path:
    base = markdown_path.parent.resolve()
    path = (base / reference).resolve()
    if not path.is_relative_to(base / "figs"):
        raise ValueError("그림 경로는 figs 폴더 안에 있어야 합니다.")
    return path


def build_bundle(markdown_path: Path) -> bytes:
    markdown = markdown_path.read_text(encoding="utf-8")
    stream = BytesIO()
    references = {image_reference(match).source for match in FIGURE_LINK.finditer(markdown)}
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr(markdown_path.name, markdown.encode("utf-8"))
        for reference in sorted(references):
            archive.write(figure_path(markdown_path, reference), reference)
    return stream.getvalue()


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
