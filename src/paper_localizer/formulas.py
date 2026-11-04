"""Preserve equations whose PDF fonts cannot be decoded as ordinary text."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re
import unicodedata

import pypdfium2 as pdfium

from .figures import _PDFIUM_LOCK, _area, _intersection, _nodes, _union


_MATH_FONT = re.compile(r"(?:CMMI|CMSY|CMEX|MSAM|MSBM|Symbol|Math)", re.I)
Box = tuple[float, float, float, float]


@dataclass
class TextSpan:
    index: int
    text: str
    box: Box
    math: bool


def _canonical(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char):
            if not normalized.isspace() and normalized.isprintable():
                characters.append(normalized)
                offsets.append(index)
    return "".join(characters), offsets


def _box(node: dict) -> Box | None:
    value = node.get("bounding box")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(map(float, value))
    return None


def _inside(span: Box, node: Box, padding: float = 2.5) -> bool:
    x, y = (span[0] + span[2]) / 2, (span[1] + span[3]) / 2
    return node[0] - padding <= x <= node[2] + padding and node[1] - padding <= y <= node[3] + padding


def _spans(page) -> list[TextSpan]:
    textpage = page.get_textpage()
    result: list[TextSpan] = []
    try:
        for index, obj in enumerate(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_TEXT], textpage=textpage)):
            text = obj.extract()
            font = obj.get_font().get_base_name()
            box = tuple(obj.get_bounds())
            # Legacy prose fonts can contain soft-hyphen control codes too.
            # They must not turn an entire normal text line into an equation.
            undecodable = not font or bool(_MATH_FONT.search(font))
            if text.strip() and _area(box):
                result.append(TextSpan(index, text, box, undecodable))
    finally:
        textpage.close()
    return result


def _groups(spans: list[TextSpan]) -> list[list[TextSpan]]:
    groups: list[list[TextSpan]] = []
    previous_math = False
    for span in spans:
        if not span.math:
            previous_math = False
            continue
        if groups and previous_math:
            previous = groups[-1][-1]
            gap_x = max(span.box[0] - previous.box[2], previous.box[0] - span.box[2], 0)
            gap_y = max(span.box[1] - previous.box[3], previous.box[1] - span.box[3], 0)
            if gap_x <= 30 and gap_y <= 8:
                groups[-1].append(span)
            else:
                groups.append([span])
        else:
            groups.append([span])
        previous_math = True
    return groups


def _group_box(group: list[TextSpan]) -> Box:
    box = group[0].box
    for span in group[1:]:
        box = _union(box, span.box)
    return box


def _insertions(content: str, spans: list[TextSpan], groups: list[list[TextSpan]]) -> list[tuple[int, int]] | None:
    """Locate each missing formula between matching ordinary-text anchors.

    Source math is excluded before alignment; a gap in the extracted text can
    therefore contain either nothing or a partially decoded equation to replace.
    """
    source = ""
    positions: dict[int, int] = {}
    for span in spans:
        positions[span.index] = len(source)
        if not span.math:
            source += _canonical(span.text)[0]
    target, offsets = _canonical(content)
    if not source or not target:
        return None
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    matches = matcher.get_matching_blocks()
    if sum(match.size for match in matches) / max(len(source), 1) < 0.7:
        return None
    mapping = {match.a + i: match.b + i for match in matches for i in range(match.size)}
    ranges: list[tuple[int, int]] = []
    for group in groups:
        position = positions[group[0].index]
        before = next((mapping[i] + 1 for i in range(position - 1, -1, -1) if i in mapping), 0)
        after = next((mapping[i] for i in range(position, len(source)) if i in mapping), len(target))
        if before > after:
            return None
        start = offsets[before] if before < len(offsets) else len(content)
        end = offsets[after] if after < len(offsets) else len(content)
        removed = content[start:end]
        math_length = sum(len(_canonical(span.text)[0]) for span in group)
        math_words = set(re.findall(r"[A-Za-z]{3,}", "".join(span.text for span in group).lower()))
        removed_words = set(re.findall(r"[A-Za-z]{3,}", removed.lower()))
        if len(removed) > max(32, math_length * 2 + 4) or not removed_words.issubset(math_words):
            return None
        ranges.append((start, end))
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        return None
    return ranges


def recover_formulas(pdf_path: Path, document: dict, output_dir: Path) -> dict:
    """Insert faithful inline PNG equations, keeping ordinary text editable."""
    with pdf_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()[:12]
    asset_dir = output_dir / "figs" / digest
    by_page: dict[int, list[dict]] = defaultdict(list)
    for node in _nodes(document):
        if isinstance(node.get("content"), str) and _box(node):
            by_page[node.get("page number", 1)].append(node)
    figures = document.get("figures", [])
    formulas: list[dict] = []

    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for page_number in range(1, len(pdf) + 1):
            page = pdf[page_number - 1]
            try:
                width, height = page.get_size()
                spans = _spans(page)
                figure_boxes = [_box(f) for f in figures if f["page number"] == page_number]
                spans = [span for span in spans if not any(
                    box and _intersection(span.box, box) / max(_area(span.box), 1) > 0.7 for box in figure_boxes
                )]
                assignments: dict[int, list[TextSpan]] = defaultdict(list)
                unassigned: list[TextSpan] = []
                nodes = by_page.get(page_number, [])
                for span in spans:
                    if not span.math:
                        continue
                    candidates = [(index, _area(_box(node))) for index, node in enumerate(nodes)
                                  if _inside(span.box, _box(node))]
                    if candidates:
                        index = min(candidates, key=lambda item: item[1])[0]
                        assignments[index].append(span)
                    else:
                        unassigned.append(span)
                counter = 0

                def save(box: Box, *, block: bool = False) -> str:
                    nonlocal counter
                    counter += 1
                    left, bottom, right, top = box
                    left, bottom = max(0, left - 0.4), max(0, bottom - 1.0)
                    right, top = min(width, right + 0.4), min(height, top + 1.0)
                    destination = asset_dir / f"page-{page_number:03d}-math-{counter:03d}.png"
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    bitmap = page.render(scale=2, crop=(left, bottom, width - right, height - top), draw_annots=False)
                    image = bitmap.to_pil(); image.save(destination); image.close(); bitmap.close()
                    reference = destination.relative_to(output_dir).as_posix()
                    formulas.append({"source": reference, "page number": page_number,
                                     "bounding box": [left, bottom, right, top], "block": block})
                    return f"![수식-p{page_number}-{counter}]({reference})"

                for index, assigned in assignments.items():
                    node = nodes[index]
                    node_box = _box(node)
                    selected = [s for s in spans if _inside(s.box, node_box)]
                    assigned_ids = {s.index for s in assigned}
                    # Keep normal anchors and only the formulas owned by this node.
                    selected = [s for s in selected if not s.math or s.index in assigned_ids]
                    groups = _groups(selected)
                    ranges = _insertions(node["content"], selected, groups)
                    if ranges is None:
                        # Preserve uncertain alignment as a complete source snapshot
                        # rather than insert an equation into an unrelated sentence.
                        snapshot = save(node_box, block=True)
                        node["formula snapshot"] = snapshot
                        continue
                    content = node["content"]
                    edits = [(start, end, save(_group_box(group)))
                             for (start, end), group in zip(ranges, groups, strict=True)]
                    for start, end, image in reversed(edits):
                        if end < len(content) and content[end].isalnum():
                            image += " "
                        content = content[:start] + image + content[end:]
                    node["content"] = content
                # Display equations that the layout parser dropped entirely.
                unassigned_ids = {span.index for span in unassigned}
                for group in _groups([s for s in spans if not s.math or s.index in unassigned_ids]):
                    box = _group_box(group)
                    image = save(box, block=True)
                    equation = {"type": "formula", "page number": page_number,
                                "bounding box": list(box), "content": image, "formula image": True}
                    insertion = len(document.get("kids", []))
                    for index, node in enumerate(document.get("kids", [])):
                        node_page = node.get("page number", 1)
                        node_box = _box(node)
                        below = node_box and (node_box[3] < box[3] - 2 or (
                            abs(node_box[3] - box[3]) <= 2 and node_box[0] > box[0]
                        ))
                        if node_page > page_number or (node_page == page_number and below):
                            insertion = index
                            break
                    document.setdefault("kids", []).insert(insertion, equation)
            finally:
                page.close()
    document["formulas"] = formulas
    return document
