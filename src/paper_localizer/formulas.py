"""Preserve equations whose PDF fonts cannot be decoded as ordinary text."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import hashlib
import ctypes
import math
from pathlib import Path
import re
import unicodedata

import pypdfium2 as pdfium

from .figures import _PDFIUM_LOCK, _area, _intersection, _nodes, _union
from .markup import inline_image_tag


_MATH_FONT = re.compile(r"(?:CMMI|CMSY|CMEX|MSAM|MSBM|Symbol|Math)", re.I)
Box = tuple[float, float, float, float]


@dataclass
class TextSpan:
    index: int
    text: str
    box: Box
    math: bool
    font_size: float
    baseline: float
    font_name: str = ""


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
    fonts: dict[int, tuple[str, float, float, bool]] = {}
    pending: list[TextSpan] = []
    category = ""

    def flush() -> None:
        nonlocal pending, category
        if pending:
            first = pending[0]
            text = "".join(span.text for span in pending)
            box = first.box
            for span in pending[1:]: box = _union(box, span.box)
            seed = any(span.math for span in pending) or text in {"=", "+", "≤", "≥", "∑", "∫", "√", "±", "×"}
            seed = seed or (len(text) == 1 and text.isalpha() and bool(re.search("Italic|Oblique", first.font_name, re.I)))
            result.append(TextSpan(first.index, text, box, seed, max(s.font_size for s in pending),
                                   first.baseline, first.font_name))
        pending, category = [], ""

    try:
        count = textpage.count_chars()
        for index in range(count):
            code = pdfium.raw.FPDFText_GetUnicode(textpage, index)
            text = chr(code) if 0 < code <= 0x10FFFF else "\ufffd"
            obj = textpage.get_textobj(index)
            if obj is None:
                if text.isspace(): flush()
                continue
            key = ctypes.cast(obj.raw, ctypes.c_void_p).value
            if key not in fonts:
                font = obj.get_font().get_base_name()
                matrix = obj.get_matrix()
                size = obj.get_font_size() * math.hypot(matrix.c, matrix.d)
                fonts[key] = font, size, matrix.f, not font or bool(_MATH_FONT.search(font))
            font, size, baseline, undecodable = fonts[key]
            if text.isspace() and not undecodable:
                flush(); continue
            if not text.isprintable() and not undecodable: continue
            box = tuple(textpage.get_charbox(index))
            if not _area(box): continue
            if text.isspace(): text = "\ufffd"
            kind = "word" if text.isalpha() else "number" if text.isdigit() else text
            decimal = text == "." and category == "number" and index + 1 < count and chr(
                pdfium.raw.FPDFText_GetUnicode(textpage, index + 1)).isdigit()
            if decimal: kind = "number"
            reference_size = max(8, size)
            separated = pending and (abs(baseline - pending[-1].baseline) > reference_size * 0.5
                                      or box[0] - pending[-1].box[2] > reference_size * 0.5)
            if category != kind or kind not in {"word", "number"} or separated: flush()
            category = kind
            pending.append(TextSpan(index, text, box, undecodable, size, baseline, font))
        flush()
    finally:
        textpage.close()
    return _expand_expressions(result)


_FUNCTIONS = {"O", "log", "ln", "exp", "sin", "cos", "tan", "min", "max", "Pr", "lim", "det", "mod", "arg"}


def _expand_expressions(spans: list[TextSpan]) -> list[TextSpan]:
    """Include readable operators, digits, brackets and identifiers in one crop."""
    def candidate(span: TextSpan) -> bool:
        text = span.text
        return (span.math or text in _FUNCTIONS or text.replace(".", "").isdigit()
                or (len(text) == 1 and text.isalpha())
                or (text.isalpha() and bool(re.search("Italic|Oblique", span.font_name, re.I)))
                or all(char in "+-=*/^_()[]{}<>|,:;≤≥≈≠∞·×÷" for char in text))

    def connected(a: TextSpan, b: TextSpan) -> bool:
        size = max(8, min(max(a.font_size, b.font_size), 24))
        gap_x = max(b.box[0] - a.box[2], a.box[0] - b.box[2], 0)
        gap_y = max(b.box[1] - a.box[3], a.box[1] - b.box[3], 0)
        if gap_x > size * 0.75 or gap_y > size * 0.7: return False
        if (a.text.isalpha() and b.text.isalpha() and a.text not in _FUNCTIONS and b.text not in _FUNCTIONS
                and not (a.math and b.math) and gap_x > size * 0.18): return False
        return True

    selected: set[int] = set()
    for seed, span in enumerate(spans):
        if not span.math: continue
        left = right = seed
        while left and candidate(spans[left - 1]) and connected(spans[left - 1], spans[left]): left -= 1
        while right + 1 < len(spans) and candidate(spans[right + 1]) and connected(spans[right], spans[right + 1]): right += 1
        # Commas separate expressions unless they are inside function arguments.
        stack: list[str] = []
        start = left
        for index in range(left, right + 2):
            text = spans[index].text if index <= right else ";"
            if text in {"(", "[", "{"}: stack.append(text)
            elif text in {")", "]", "}"}:
                if stack: stack.pop()
                else:
                    if start <= seed < index: selected.update(range(start, index))
                    start = index + 1
            if text in {",", ";", ":"} and not stack:
                if start <= seed < index: selected.update(range(start, index))
                start = index + 1
        if start <= seed <= right: selected.update(range(start, right + 1))
    return [replace(span, math=True) if index in selected else span for index, span in enumerate(spans)]


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
            if gap_x <= 30 and gap_y <= max(8, min(max(span.font_size, previous.font_size), 18) * 1.4):
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


def _inline_metrics(group: list[TextSpan], spans: list[TextSpan], node: dict) -> tuple[float, float]:
    box = _group_box(group)
    node_font = float(node.get("font size") or 9)
    if node_font < 2:
        node_font = 9
    anchors = [span for span in spans if not span.math and span.font_size >= 2
               and box[1] - node_font * 1.5 <= span.baseline <= box[3] + node_font]
    if anchors:
        def distance(span: TextSpan) -> float:
            horizontal = max(box[0] - span.box[2], span.box[0] - box[2], 0)
            return horizontal + abs(span.baseline - group[0].baseline) * 2
        anchor = min(anchors, key=distance)
        return anchor.font_size, anchor.baseline
    return node_font, group[0].baseline


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
                for group in _groups(spans):
                    group_box = _group_box(group)
                    candidates = [(index, _area(_box(node))) for index, node in enumerate(nodes)
                                  if _intersection(group_box, _box(node)) / max(_area(group_box), 1) >= 0.75]
                    if candidates:
                        index = min(candidates, key=lambda item: item[1])[0]
                        assignments[index].extend(group)
                    else:
                        unassigned.extend(group)
                counter = 0

                def save(box: Box, *, block: bool = False, metrics: tuple[float, float] | None = None) -> str:
                    nonlocal counter
                    counter += 1
                    left, bottom, right, top = box
                    left, bottom = max(0, left - 0.4), max(0, bottom - 1.0)
                    right, top = min(width, right + 0.4), min(height, top + 1.0)
                    destination = asset_dir / f"page-{page_number:03d}-math-{counter:03d}.png"
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    bitmap = page.render(scale=2, crop=(left, bottom, width - right, height - top), draw_annots=False)
                    image = bitmap.to_pil()
                    aspect = image.width / image.height
                    image.save(destination); image.close(); bitmap.close()
                    reference = destination.relative_to(output_dir).as_posix()
                    formulas.append({"source": reference, "page number": page_number,
                                     "bounding box": [left, bottom, right, top], "block": block})
                    alt = f"수식-p{page_number}-{counter}"
                    if block:
                        return f"![{alt}]({reference})"
                    font_size, baseline = metrics or (9, bottom + 1.5)
                    height_em = (top - bottom) / font_size
                    align_em = (bottom - baseline) / font_size
                    return inline_image_tag(reference, alt, height_em, align_em, aspect)

                for index, assigned in assignments.items():
                    node = nodes[index]
                    node_box = _box(node)
                    assigned_ids = {s.index for s in assigned}
                    selected = [s for s in spans if _inside(s.box, node_box) or s.index in assigned_ids]
                    # Keep normal anchors and only the formulas owned by this node.
                    selected = [s for s in selected if not s.math or s.index in assigned_ids]
                    groups = _groups(selected)
                    ranges = _insertions(node["content"], selected, groups)
                    if ranges is None:
                        # Preserve uncertain alignment as a complete source snapshot
                        # rather than insert an equation into an unrelated sentence.
                        whole_box = node_box
                        for group in groups: whole_box = _union(whole_box, _group_box(group))
                        snapshot = save(whole_box, block=True)
                        node["formula snapshot"] = snapshot
                        continue
                    content = node["content"]
                    edits = [(start, end, save(_group_box(group), metrics=_inline_metrics(group, selected, node)))
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
