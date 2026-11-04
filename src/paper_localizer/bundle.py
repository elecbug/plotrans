"""Portable Markdown exports with their referenced figure files."""

from io import BytesIO
import base64
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile


FIGURE_LINK = re.compile(r"!\[([^\]\n]*)\]\((figs/[A-Za-z0-9_./-]+)\)")


def figure_path(markdown_path: Path, reference: str) -> Path:
    base = markdown_path.parent.resolve()
    path = (base / reference).resolve()
    if not path.is_relative_to(base / "figs"):
        raise ValueError("그림 경로는 figs 폴더 안에 있어야 합니다.")
    return path


def build_bundle(markdown_path: Path) -> bytes:
    markdown = markdown_path.read_text(encoding="utf-8")
    stream = BytesIO()
    references = {match.group(2) for match in FIGURE_LINK.finditer(markdown)}
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr(markdown_path.name, markdown.encode("utf-8"))
        for reference in sorted(references):
            archive.write(figure_path(markdown_path, reference), reference)
    return stream.getvalue()


def inline_equation_preview(markdown_path: Path, markdown: str) -> str:
    """Keep equation PNGs inline in Streamlit's Markdown preview."""
    def replace(match: re.Match) -> str:
        if "-math-" not in match.group(2):
            return match.group(0)
        encoded = base64.b64encode(figure_path(markdown_path, match.group(2)).read_bytes()).decode("ascii")
        return f"![{match.group(1)}](data:image/png;base64,{encoded})"
    return FIGURE_LINK.sub(replace, markdown)
