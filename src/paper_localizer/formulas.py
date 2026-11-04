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
from .markup import FIGURE_LINK, image_reference, inline_image_tag


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
            # Operators in a monospaced program are source code, not formula
            # seeds. Cropping `p =` out of a C declaration destroys the program.
            if re.search(r"Courier|Mono|Consolas", first.font_name, re.I):
                seed = False
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
    identifiers: set[int] = set()
    prose = {"and", "or", "is", "where", "with", "for", "the", "a", "an", "if", "then", "else", "return", "while"}
    for index, span in enumerate(spans):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", span.text) or span.text.lower() in prose: continue
        neighbors = spans[max(0, index - 1):index] + spans[index + 1:index + 2]
        # Named subscripts such as D_lazy are ordinary roman text in many TeX
        # PDFs. Their smaller font and lowered baseline establish membership.
        if any(other.math and span.font_size <= other.font_size * .88
               and 0 < other.baseline - span.baseline < max(12, other.font_size)
               and max(span.box[0]-other.box[2], other.box[0]-span.box[2], 0) < 3
               for other in neighbors):
            identifiers.add(span.index)
        if any(other.text in {"=", "≤", "≥", "+", "*", "/"} for other in neighbors): identifiers.add(span.index)
        if index + 1 < len(spans) and spans[index + 1].text == "(":
            gap = spans[index + 1].box[0] - span.box[2]
            if gap <= max(8, span.font_size) * 0.25: identifiers.add(span.index)
    def candidate(span: TextSpan) -> bool:
        text = span.text
        return (span.math or span.index in identifiers or text in _FUNCTIONS or text.replace(".", "").isdigit()
                or (len(text) == 1 and text.isalpha())
                or (text.isalpha() and bool(re.search("Italic|Oblique", span.font_name, re.I)))
                or all(char in "+-=*/^_()[]{}<>|,:;≤≥≈≠∞·×÷∈∉∪∩…" for char in text)
                or text == "...")

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
    # PDF content streams often emit a summation's lower bound before the
    # fraction and its cases in another text object. Stream adjacency alone
    # splits one displayed equation into overlapping, incomplete snapshots.
    merged: list[list[TextSpan]] = []
    for group in groups:
        box = _group_box(group)
        while True:
            touching = next((i for i, other in enumerate(merged)
                             if _intersection(box, _group_box(other)) >
                             min(_area(box), _area(_group_box(other))) * .08), None)
            if touching is None:
                break
            group = sorted(merged.pop(touching) + group, key=lambda s: s.index)
            box = _group_box(group)
        merged.append(group)
    return sorted(merged, key=lambda group: group[0].index)


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
        math_words = set(re.findall(r"[A-Za-z]{3,}", " ".join(span.text for span in group).lower()))
        removed_words = set(re.findall(r"[A-Za-z]{3,}", removed.lower()))
        if len(removed) > max(32, math_length * 2 + 4) or not removed_words.issubset(math_words):
            return None
        ranges.append((start, end))
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        return None
    return ranges


def _remove_figure_duplicates(nodes: list[dict], spans: list[TextSpan], figure_boxes: list[Box],
                              preserve_captions: bool = True) -> int:
    changed = 0
    for node in nodes:
        content = node.get("content", "")
        box = _box(node)
        if not content or not box: continue
        if preserve_captions and re.match(r"^(?:-\s*)?Fig(?:ure)?\.?\s*\d", content, re.I): continue
        selected = [span for span in spans if _inside(span.box, box)]
        marked = [replace(span, math=any(
            figure and _intersection(span.box, figure) / max(_area(span.box), 1) >= 0.7
            for figure in figure_boxes)) for span in selected]
        covered = [span for span in marked if span.math]
        if not covered: continue
        if len(covered) == len(marked):
            node["content"] = ""; changed += 1; continue
        ranges = _insertions(content, marked, _groups(marked))
        if ranges is None: continue
        for start, end in reversed(ranges): content = content[:start] + content[end:]
        content = re.sub(r" {2,}", " ", content).strip()
        if content != node["content"]:
            node["content"] = content; changed += 1
    return changed


def _split_mixed_heading(node: dict, spans: list[TextSpan]) -> None:
    if node.get("type") != "heading" or "<img" in node.get("content", ""): return
    selected = [span for span in spans if _inside(span.box, _box(node))]
    if not selected or not re.search("Bold|Black", selected[0].font_name, re.I): return
    title = []
    for span in selected:
        if not re.search("Bold|Black", span.font_name, re.I) and not span.math:
            if span.baseline < selected[0].baseline - max(3, selected[0].font_size * 0.4): break
        title.append(span.text)
    else: return
    prefix = _canonical("".join(title))[0]
    content, offsets = _canonical(node["content"])
    if prefix and content.startswith(prefix) and len(content) - len(prefix) > 10:
        node["heading split"] = offsets[len(prefix) - 1] + 1


def _reconcile_snapshots(nodes: list[dict], spans: list[TextSpan], assets: list[dict]) -> int:
    """The largest source snapshot owns every text fragment visible inside it.

    A piecewise equation can occupy three overlapping layout nodes. Clearing
    only the snapshot's originating node leaves the other two on screen.
    """
    regions = {a["source"]: tuple(a["bounding box"]) for a in assets if a.get("block")}
    owners = []
    for node in nodes:
        markup = node.get("formula snapshot") or (node.get("content", "") if node.get("formula image") else "")
        match = next(FIGURE_LINK.finditer(markup), None)
        if match and (source := image_reference(match).source) in regions:
            owners.append((node, regions[source]))
    owners.sort(key=lambda item: _area(item[1]), reverse=True)
    suppressed = set(); changed = 0
    for owner, region in owners:
        if id(owner) in suppressed:
            continue
        for node in nodes:
            if node is owner or id(node) in suppressed or not (box := _box(node)):
                continue
            if not node.get("content") and not node.get("formula snapshot"):
                continue
            if node.get("formula snapshot") or node.get("formula image"):
                markup = node.get("formula snapshot") or node.get("content", "")
                match = next(FIGURE_LINK.finditer(markup), None)
                if match:
                    box = regions.get(image_reference(match).source, box)
            if _intersection(box, region) / max(_area(box), 1) >= .98:
                node["content"] = ""
                node.pop("formula snapshot", None)
                node["suppressed"] = True
                suppressed.add(id(node)); changed += 1
        # Partial paragraph overlap is removed by ordinary-text anchors rather
        # than dropping prose outside the snapshot's visible bounds.
        partial = [n for n in nodes if n is not owner and id(n) not in suppressed
                   and not n.get("formula snapshot") and not FIGURE_LINK.search(n.get("content", ""))]
        changed += _remove_figure_duplicates(partial, spans, [region], preserve_captions=False)
    return changed


def _complete_region(box: Box, nodes: list[dict]) -> Box:
    """Finish overlapping native blocks before suppressing their duplicate text.

    Do not erase a paragraph just because 85% is in a crop. Expand that crop to
    retain its edge too, with a bounded area increase to avoid region chaining.
    """
    initial_area = _area(box)
    while True:
        previous = box
        for node in nodes:
            other = _box(node)
            if not other or not node.get("content") or not _area(other):
                continue
            combined = _union(box, other)
            if (_intersection(other, box) / _area(other) >= .8
                    and _area(combined) <= initial_area * 1.5):
                box = combined
        if box == previous:
            return box


def _recover_code(page, nodes: list[dict]) -> None:
    """Keep monospaced listings as complete fenced blocks with source newlines."""
    candidates = [n for n in nodes if re.search(r"Courier|Mono|Consolas", n.get("font") or "", re.I)
                  and n.get("content") and _box(n)]
    groups: list[list[dict]] = []
    for node in sorted(candidates, key=lambda n: -_box(n)[3]):
        box = _box(node)
        if groups:
            previous = _group_node_box(groups[-1])
            size = float(node.get("font size") or 10)
            if -2 <= previous[1] - box[3] <= size * 1.6 and abs(previous[0] - box[0]) < size * 6:
                groups[-1].append(node)
                continue
        groups.append([node])
    textpage = page.get_textpage()
    try:
        for group in groups:
            box = _group_node_box(group)
            text = textpage.get_text_bounded(box[0] - .5, box[1] - .5,
                                             box[2] + .5, box[3] + .5).replace("\r\n", "\n").strip()
            if not re.search(r"[{};]|\b(?:def|function|return|include)\b", text):
                continue
            if len(text.splitlines()) < 2:
                continue
            owner = group[0]
            owner.update({"type": "code", "content": "```\n" + text + "\n```", "bounding box": list(box)})
            for node in group[1:]:
                node["content"] = ""
                node["suppressed"] = True
    finally:
        textpage.close()


def _group_node_box(nodes: list[dict]) -> Box:
    box = _box(nodes[0])
    for node in nodes[1:]:
        box = _union(box, _box(node))
    return box


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
    formulas: list[dict] = list(document.get("ocr formula assets", []))

    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for page_number in range(1, len(pdf) + 1):
            page = pdf[page_number - 1]
            try:
                # OCR already preserved uncertain/math lines as complete crops.
                # Reusing undecodable native glyphs would overwrite its prose.
                if page_number in document.get("ocr pages", []):
                    continue
                width, height = page.get_size()
                spans = _spans(page)
                undecoded = sum(not s.font_name for s in spans)
                if spans and undecoded / len(spans) > .75:
                    # Some old PDFs encode *all prose* as custom Type3 glyphs.
                    # Preserve one complete page instead of thousands of tiny
                    # pseudo-equations with unusable OCR-free translations.
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    destination = asset_dir / f"page-{page_number:03d}-source.png"
                    bitmap = page.render(scale=1.5, draw_annots=False)
                    image = bitmap.to_pil(); image.save(destination); image.close(); bitmap.close()
                    for node in _nodes(document):
                        if node.get("page number") == page_number:
                            node["suppressed"] = True
                    document["figures"] = [f for f in document.get("figures", []) if f["page number"] != page_number]
                    reference = destination.relative_to(output_dir).as_posix()
                    snapshot = {"type": "formula", "page number": page_number, "bounding box": [0, 0, width, height],
                                "formula image": True,
                                "content": f"![Source page {page_number} — undecodable font]({reference})"}
                    at = next((i for i,n in enumerate(document.get("kids", [])) if n.get("page number",1) > page_number), len(document.get("kids", [])))
                    document.setdefault("kids", []).insert(at, snapshot)
                    document.setdefault("source image pages", []).append(page_number)
                    continue
                figure_boxes = [_box(f) for f in figures if f["page number"] == page_number]
                nodes = by_page.get(page_number, [])
                _recover_code(page, nodes)
                document["deduplicated figure text"] = document.get("deduplicated figure text", 0) + _remove_figure_duplicates(
                    nodes, spans, figure_boxes)
                spans = [span for span in spans if not any(
                    box and _intersection(span.box, box) / max(_area(span.box), 1) > 0.7 for box in figure_boxes
                )]
                assignments: dict[int, list[TextSpan]] = defaultdict(list)
                unassigned: list[TextSpan] = []
                nodes = [node for node in nodes if node.get("content", "").strip()]
                for group in _groups(spans):
                    group_box = _group_box(group)
                    candidates = []
                    for index, node in enumerate(nodes):
                        box = _box(node)
                        padded = (box[0] - 2.5, box[1] - 2.5, box[2] + 2.5, box[3] + 2.5)
                        if (_intersection(group_box, box) / max(_area(group_box), 1) >= 0.75
                                or _intersection(group_box, padded) / max(_area(group_box), 1) >= 0.95):
                            candidates.append((index, _area(box)))
                    if candidates:
                        index = min(candidates, key=lambda item: item[1])[0]
                        assignments[index].extend(group)
                    else:
                        unassigned.extend(group)
                unassigned_ids = {span.index for span in unassigned}
                orphan_groups = _groups([s for s in spans if not s.math or s.index in unassigned_ids])
                document["deduplicated equation text"] = document.get("deduplicated equation text", 0) + _remove_figure_duplicates(
                    nodes, spans, [_group_box(group) for group in orphan_groups], preserve_captions=False)
                counter = 0

                def save(box: Box, *, block: bool = False, metrics: tuple[float, float] | None = None) -> str:
                    nonlocal counter
                    counter += 1
                    if block:
                        box = _complete_region(box, nodes)
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
                    alt = f"Eq.-p{page_number}-{counter}"
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
                    if not node.get("content", "").strip():
                        orphan_groups.extend(_groups([s for s in spans if not s.math or s.index in assigned_ids]))
                        continue
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
                for group in orphan_groups:
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
                for node in nodes:
                    if not node.get("formula snapshot") and re.search(r"[\ue000-\uf8ff\ufffd]", node.get("content", "")):
                        # Preserve undecoded source text instead of exposing a
                        # replacement character after otherwise successful math
                        # alignment (e.g. a symbol in a reference or footnote).
                        if box := _box(node):
                            if _area(box):
                                node["formula snapshot"] = save(box, block=True)
                for node in nodes:
                    _split_mixed_heading(node, spans)
                page_nodes = [n for n in _nodes(document) if n.get("page number") == page_number]
                document["deduplicated equation text"] = document.get("deduplicated equation text", 0) + _reconcile_snapshots(
                    page_nodes, spans, [a for a in formulas if a["page number"] == page_number])
            finally:
                page.close()
    active = {image_reference(m).source for node in _nodes(document) if not node.get("suppressed")
              for text in (node.get("content", ""), node.get("formula snapshot", ""))
              for m in FIGURE_LINK.finditer(text)}
    document["formulas"] = [asset for asset in formulas if asset["source"] in active]
    return document
