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


_MATH_FONT = re.compile(r"(?:CMMI|CMSY|CMEX|MSAM|MSBM|MTSYN|Symbol|Math|AdvP4C4E)", re.I)
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
            seed = seed or (len(text) == 1 and text.isalpha() and bool(re.search("Italic|Oblique|Ital$|It$", first.font_name, re.I)))
            # Operators in a monospaced program are source code, not formula
            # seeds. Cropping `p =` out of a C declaration destroys the program.
            if re.search(r"Courier|Mono|Consolas", first.font_name, re.I):
                seed = False
            if text in {'•','◦','▪','‣'}:
                seed = False
            result.append(TextSpan(first.index, text, box, seed, max(s.font_size for s in pending),
                                   first.baseline, first.font_name))
        pending, category = [], ""

    try:
        count = textpage.count_chars()
        skip_low_surrogate = False
        for index in range(count):
            if skip_low_surrogate:
                skip_low_surrogate = False
                continue
            code = pdfium.raw.FPDFText_GetUnicode(textpage, index)
            if 0xD800 <= code <= 0xDBFF and index+1 < count:
                low = pdfium.raw.FPDFText_GetUnicode(textpage, index+1)
                if 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + ((code-0xD800)<<10) + low-0xDC00
                    skip_low_surrogate = True
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
            # Type3 advance-only glyphs can acquire a tiny positive height
            # from floating-point transforms. They contain no ink and must
            # not become orphan equation images of neighboring descenders.
            if not _area(box) or box[3]-box[1] < .01: continue
            if text.isspace(): text = "\ufffd"
            kind = "word" if text.isalpha() else "number" if text.isdigit() else text
            decimal = text == "." and category == "number" and index + 1 < count and chr(
                pdfium.raw.FPDFText_GetUnicode(textpage, index + 1)).isdigit()
            if decimal: kind = "number"
            reference_size = max(8, size)
            separated = pending and (abs(baseline - pending[-1].baseline) > reference_size * 0.5
                                      or box[0] - pending[-1].box[2] > reference_size * 0.5
                                      or font != pending[-1].font_name
                                      or abs(size - pending[-1].font_size) > max(size, pending[-1].font_size) * .1)
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
    citations: set[int] = set()
    for index in range(1, len(spans) - 2):
        opening, number, closing = spans[index:index + 3]
        # A separated numeric reference after an expression is prose. Keep a
        # tightly attached array index such as A[2] with its mathematical base.
        if (opening.text == "[" and number.text.isdigit() and closing.text == "]"
                and opening.box[0] - spans[index - 1].box[2] > opening.font_size * .3
                and not any(_MATH_FONT.search(s.font_name) for s in (opening, number, closing))):
            citations.update(s.index for s in (opening, number, closing))
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
        if span.index in citations:
            return False
        text = span.text
        return (span.math or span.index in identifiers or text in _FUNCTIONS or text.replace(".", "").isdigit()
                or (len(text) == 1 and text.isalpha())
                or (text.isalpha() and bool(re.search("Italic|Oblique|Ital$|It$", span.font_name, re.I)))
                or all(char in "+-=–*/^_()[]{}<>|,:;≤≥≈≠∞·×÷∈∉∪∩…•◦▪‣" for char in text)
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


def _groups(spans: list[TextSpan], owners: dict[int, Box] | None = None) -> list[list[TextSpan]]:
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
            same_cell = owners is None or owners.get(span.index) == owners.get(previous.index)
            if same_cell and gap_x <= 30 and gap_y <= max(8, min(max(span.font_size, previous.font_size), 18) * 1.4):
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
                             if (owners is None or owners.get(group[0].index) == owners.get(other[0].index))
                             and _intersection(box, _group_box(other)) >
                             min(_area(box), _area(_group_box(other))) * .08), None)
            if touching is None:
                break
            group = sorted(merged.pop(touching) + group, key=lambda s: s.index)
            box = _group_box(group)
        merged.append(group)
    # Italic overhangs can put a following comma/period inside the padded crop.
    # Give that punctuation an explicit owner and remove its duplicate text.
    owned = {span.index for group in merged for span in group}
    for group in merged:
        box = _group_box(group)
        baseline = group[-1].baseline
        punctuation = [span for span in spans if span.index not in owned and span.text in {".", ",", ";", ":"}
                       and (owners is None or owners.get(span.index) == owners.get(group[0].index))
                       and -.5 <= span.box[0]-box[2] <= .4 and abs(span.baseline-baseline) < 1]
        if punctuation:
            span = min(punctuation, key=lambda item: item.box[0])
            group.append(span); owned.add(span.index)
    return sorted(merged, key=lambda group: group[0].index)


def _group_box(group: list[TextSpan]) -> Box:
    box = group[0].box
    for span in group[1:]:
        box = _union(box, span.box)
    return box


def _cell_owners(spans: list[TextSpan], cells: list[Box]) -> dict[int, Box]:
    """Physical cells bound math groups, even when their glyphs are adjacent."""
    return {span.index: min(matches, key=_area) for span in spans
            if (matches := [cell for cell in cells if _inside(span.box, cell, padding=0)])}


def _within_cell(region: Box, cell: Box, padding: float = 2.5) -> bool:
    return (cell[0]-padding <= region[0] and cell[1]-padding <= region[1]
            and region[2] <= cell[2]+padding and region[3] <= cell[3]+padding)


def _plain_text_group(group: list[TextSpan], node: dict) -> bool:
    """Decoded list markers, heading labels, and prose tokens need no crop."""
    text = "".join(span.text for span in group)
    content = node.get("content", "").lstrip()
    if text in {"•", "◦", "▪", "‣"} and content.startswith(text):
        return True
    # Mapped arrows/minus signs often come from CMSY even in simple numeric
    # tables. Keep a complete, flat, source-matching cell editable. A raised
    # exponent or stacked fraction fails the baseline/size checks.
    if (node.get('native table cell') and re.fullmatch(r'[\d.,%()+*/=<>×÷−–→↔ -]+',text)
            and _canonical(content)[0] in {_canonical(text)[0],_canonical(text)[0]+'%'}
            and all(s.font_name for s in group)
            and min(s.font_size for s in group) >= max(s.font_size for s in group)*.95
            and max(s.baseline for s in group)-min(s.baseline for s in group)
                <= max(s.font_size for s in group)*.12):
        return True
    if any(not span.font_name or _MATH_FONT.search(span.font_name) for span in group):
        return False
    if node.get("type") == "heading" and (label := re.match(r"([A-Z])\.\s", content)):
        if text in {label[1], label[1] + "."}:
            return True
    sizes = [span.font_size for span in group]
    baselines = [span.baseline for span in group]
    if min(sizes) < max(sizes) * .95 or max(baselines) - min(baselines) > max(sizes) * .12:
        return False
    if node.get('type')=='heading' and re.fullmatch(r'[A-Za-z]',text):
        return True
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", text):
        return False
    # Italic P2P and version names can be split into letter/number spans by
    # PDFium. A uniform baseline provides no evidence of a sub/superscript.
    if sum(char.isalpha() for char in text) >= 2 and any(char.isdigit() for char in text):
        return True
    return bool(re.match(r"\[\d+\]", content) and re.fullmatch(r"[A-Z]\.", text))


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
    owned_math = {span.index for group in groups for span in group}
    for span in spans:
        positions[span.index] = len(source)
        if span.index not in owned_math:
            source += _canonical(span.text)[0]
    target, offsets = _canonical(content)
    if not source or not target:
        return None
    # When every decoded native character already agrees, use the full glyph
    # positions. Removing math before alignment can make repeated short words
    # such as "when d_in is" match the same letters inside the variable.
    native = ""
    native_positions = {}
    for span in spans:
        start = len(native)
        native += _canonical(span.text)[0]
        native_positions[span.index] = (start,len(native))
    complete_matches = ([(0,0,len(native))] if native==target else
                        [(m.a,m.b,m.size) for m in SequenceMatcher(None,native,target,autojunk=False).get_matching_blocks()])
    if sum(size for _,_,size in complete_matches)>=max(len(native),len(target))*.97:
        direct = []
        for group in groups:
            indices = {s.index for s in group}
            start = min(native_positions[s.index][0] for s in group)
            end = max(native_positions[s.index][1] for s in group)
            if start == end or any(s.index not in indices and native_positions[s.index][0]<end
                                   and native_positions[s.index][1]>start for s in spans): break
            match=next(((a,b) for a,b,size in complete_matches if a<=start and end<=a+size),None)
            if match is None:break
            mapped=match[1]+start-match[0]
            direct.append((offsets[mapped],offsets[mapped+end-start-1]+1))
        else:
            if all(a[1]<=b[0] for a,b in zip(direct,direct[1:])):
                return direct
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
        exact_native = _canonical(removed)[0] == _canonical("".join(span.text for span in group))[0]
        if len(removed) > max(32, math_length * 2 + 4) or not (exact_native or removed_words.issubset(math_words)):
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
        if preserve_captions and (node.get("type") == "caption"
                                  or re.match(r"^(?:-\s*)?(?:Fig(?:ure)?\.?|Table)\s*(?:\d|[IVXLCDM]+\b)", content, re.I)): continue
        selected = [span for span in spans if _inside(span.box, box)]
        cell_box = tuple(node.get('native table cell box', box))
        owned_boxes = [figure for figure in figure_boxes if figure and
                       (not node.get('native table cell') or _within_cell(figure, cell_box))]
        marked = [replace(span, math=any(
            figure and _intersection(span.box, figure) / max(_area(span.box), 1) >= 0.7
            for figure in owned_boxes)) for span in selected]
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
    inline_regions = {a['source']:tuple(a['bounding box']) for a in assets if not a.get('block')}
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
            if node.get('native table cell') and not _within_cell(region, tuple(node.get('native table cell box', box))):
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
        for node in nodes:
            if (node is owner or node.get('suppressed') or node.get('formula snapshot')
                    or node.get('native table cell') or node.get('type') not in {
                        'paragraph','list item','abstract','footnote','frontmatter'}):
                continue
            def retire_covered_inline(match):
                source = image_reference(match).source
                box = inline_regions.get(source)
                if box and _intersection(box,region)/max(_area(box),1)>=.98:
                    node.setdefault('source inline regions covered by snapshot',[]).append({
                        'source':source,'box':list(box),'snapshot':owner.get('formula snapshot') or owner.get('content','')})
                    return ' '
                return match[0]
            node['content'] = FIGURE_LINK.sub(retire_covered_inline,node.get('content',''))
        partial = [n for n in nodes if n is not owner and id(n) not in suppressed
                   and not n.get("formula snapshot") and not FIGURE_LINK.search(n.get("content", ""))]
        changed += _remove_figure_duplicates(partial, spans, [region], preserve_captions=False)
    return changed


def _merge_overlapping_prose_snapshots(nodes, spans, assets, save, textpage):
    """Keep an overlapping prose continuation whole in one source snapshot.

    An inline crop inside another paragraph's snapshot proves duplicate source
    ownership. Merely deleting that continuation would lose its outside prose.
    Limit expansion to the same column and twice the original snapshot area.
    """
    regions = {a['source']:tuple(a['bounding box']) for a in assets}
    prose = {'paragraph','list item','abstract','footnote','frontmatter'}
    barriers = [n for n in nodes if n.get('type') in {'heading','caption','table','code','image','picture'}
                and not n.get('suppressed') and _box(n)
                and (n.get('type')!='table' or any(c.get('content') or c.get('formula snapshot') for c in _nodes(n)))]
    owners = []
    for node in nodes:
        if (node.get('suppressed') or node.get('type') not in prose or node.get('native table cell')
                or node.get('source display equation') or node.get('source display definition')):
            continue
        match = next(FIGURE_LINK.finditer(node.get('formula snapshot','')),None)
        if match and (source:=image_reference(match).source) in regions:
            owners.append((node,regions[source]))
    changed = 0
    for owner,initial in sorted(owners,key=lambda item:_area(item[1]),reverse=True):
        if owner.get('suppressed'):
            continue
        region = initial
        members = []
        size = float(owner.get('font size') or 9)
        while True:
            added = False
            for node in nodes:
                if (node is owner or node in members or node.get('suppressed') or node.get('type') not in prose
                        or node.get('native table cell') or node.get('formula snapshot') or not (box:=_box(node))):
                    continue
                refs = [regions.get(image_reference(m).source) for m in FIGURE_LINK.finditer(node.get('content',''))]
                if not any(b and _intersection(b,region)/max(_area(b),1)>=.98 for b in refs):
                    continue
                if abs(box[0]-initial[0])>max(12,size*3) or abs(box[2]-initial[2])>max(12,size*3):
                    continue
                combined = _union(region,box)
                for span in spans:
                    if _inside(span.box,box,padding=.5):
                        combined = _union(combined,span.box)
                if _area(combined)>_area(initial)*2:
                    continue
                if any(_intersection(combined,_box(n))>_intersection(initial,_box(n))+.5 for n in barriers):
                    continue
                members.append(node);region=combined;added=True
            if not added:
                break
        if not members:
            continue
        owner['source merged paragraph fragments'] = [owner.get('content','')]+[n.get('content','') for n in members]
        owner['content'] = ' '.join(textpage.get_text_bounded(*region).split())
        owner['bounding box'] = list(region)
        owner['formula snapshot'] = save(region,block=True,complete=False)
        for node in members:
            node['content']='';node['suppressed']=True
            node['covered by source snapshot']=owner['formula snapshot']
            changed += 1
    return changed


def _merge_legacy_equation_snapshots(nodes: list[dict], spans: list[TextSpan], assets: list[dict], save) -> int:
    """Unify overlapping pieces of a display equation in legacy math fonts."""
    if not any(re.search(r"AdvP4C4E", s.font_name, re.I) for s in spans): return 0
    regions={a["source"]:tuple(a["bounding box"]) for a in assets if a.get("block")}
    candidates=[]
    for node in nodes:
        if node.get("suppressed") or node.get("type") in {"heading", "caption"}: continue
        markup=node.get("formula snapshot") or (node.get("content", "") if node.get("formula image") else "")
        match=next(FIGURE_LINK.finditer(markup),None)
        if not match or (source:=image_reference(match).source) not in regions: continue
        box=regions[source];selected=[s for s in spans if _inside(s.box,box,padding=0)]
        if not selected or not any(re.search(r"AdvP4C4E",s.font_name,re.I) for s in selected): continue
        if any(not s.math and re.search(r"[A-Za-z]{3,}",s.text) for s in selected): continue
        candidates.append((node,box))
    groups=[]
    for node,box in candidates:
        members=[node]
        while True:
            hit=next((i for i,(others,other) in enumerate(groups)
                      if min(box[3],other[3])-max(box[1],other[1])>0
                      and max(box[0]-other[2],other[0]-box[2],0)<=14
                      and _area(_union(box,other))<=max(_area(box),_area(other))*3),None)
            if hit is None:break
            others,other=groups.pop(hit);members.extend(others);box=_union(box,other)
        groups.append((members,box))
    changed=0
    for members,box in groups:
        if len(members)<2:continue
        owner=members[0];image=save(box,block=True)
        owner["bounding box"]=list(box)
        if owner.get("formula image"):owner["content"]=image
        else:owner["formula snapshot"]=image
        for node in members[1:]:
            node["content"]="";node.pop("formula snapshot",None);node["suppressed"]=True;changed+=1
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


def _complete_prose_words(box: Box, spans: list[TextSpan]) -> Box:
    """Keep an ordinary word whole before its covered text is deduplicated."""
    original = box
    for span in spans:
        if (not span.math and span.font_name and not _MATH_FONT.search(span.font_name)
                and not re.search(r'Courier|Mono|Consolas',span.font_name,re.I)
                and len(span.text)>=2 and all(c.isalpha() and ('LATIN' in unicodedata.name(c,'')
                                                                   or ord(c)<128) for c in span.text)
                and _inside(span.box,original,padding=0)):
            box = _union(box,span.box)
    return box if _area(box)<=max(_area(original),1)*1.25 else original


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


def _complete_display_equations(nodes, spans, textpage, width, tables=()):
    """Join nearby pieces of isolated, indented display equations."""
    prose=[n for n in nodes if n.get('type')=='paragraph' and not n.get('suppressed')
           and len(n.get('content','').split())>=15 and _box(n) and _box(n)[2]-_box(n)[0]>width*.5]
    if not prose:return
    left=min(_box(n)[0] for n in prose);right=max(_box(n)[2] for n in prose)
    groups=[]
    for node in nodes:
        box=_box(node);text=node.get('content','');size=float(node.get('font size') or 10)
        definition=(len(re.findall(r'[A-Za-z]{3,}',text))>=4
                    and re.match(r'^(?:[A-Za-z][A-Za-z0-9_]*|[A-Za-z]\([^)]{1,12}\))\s*=',text))
        if (not box or not text or node.get('suppressed') or node.get('type') not in {'paragraph','formula'}
                or re.search(r'Courier|Mono|Consolas',node.get('font',''),re.I)
                or box[0]<=left+size*1.5 or box[2]>right+1 or box[2]-box[0]>(right-left)*.8
                or box[3]-box[1]>size*5 or len(re.findall(r'[A-Za-z]{3,}',text))>2 and not definition
                or any(_intersection(box,t)/max(_area(box),1)>.8 for t in tables)):
            continue
        if definition and any(s.math and _inside(s.box,box,padding=.5) for s in spans):
            node.update({'source display equation':True,'source display definition':True,'display fragments':1})
            continue
        group=[node]
        while True:
            hit=next((i for i,(other,b) in enumerate(groups)
                      if max(box[1]-b[3],b[1]-box[3],0)<=2
                      and max(box[0]-b[2],b[0]-box[2],0)<=size*2.5
                      and _union(box,b)[3]-_union(box,b)[1]<=size*4),None)
            if hit is None:break
            other,b=groups.pop(hit);group+=other;box=_union(box,b)
        groups.append((group,box))
    for group,box in groups:
        if len(group)<2 or not any('=' in n['content'] for n in group):continue
        selected=[s for s in spans if _inside(s.box,box,padding=.5)]
        if sum(s.math for s in selected)<3:continue
        ordinary=[s for s in selected if not s.math and re.search(r'[A-Za-z]{3,}',s.text)]
        if len(ordinary)>3:continue
        owner=min(group,key=lambda n:_box(n)[0])
        owner.update({'type':'formula','content':' '.join(textpage.get_text_bounded(*box).split()),
                      'bounding box':list(box),'source display equation':True,'display fragments':len(group)})
        for node in group:
            if node is not owner:node['suppressed']=True


def recover_formulas(pdf_path: Path, document: dict, output_dir: Path) -> dict:
    """Insert faithful inline PNG equations, keeping ordinary text editable."""
    with pdf_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()[:12]
    asset_dir = output_dir / "figs" / digest
    by_page: dict[int, list[dict]] = defaultdict(list)
    table_boxes: dict[int, list[Box]] = defaultdict(list)
    cell_boxes: dict[int, list[Box]] = defaultdict(list)
    for cell in list(_nodes(document)):
        if cell.get('type') != 'table cell' or not (box := _box(cell)):
            continue
        cell_boxes[cell.get('page number', 1)].append(box)
        for child in _nodes(cell):
            if isinstance(child.get('content'), str):
                child['native table cell'] = True
                child['native table cell box'] = list(box)
    for node in _nodes(document):
        if node.get('type')=='table' and _box(node):
            table_boxes[node.get('page number',1)].append(_box(node))
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
                from .review_layout import filter_counters
                from .source_text import normalized_spans
                spans = filter_counters(_spans(page),document,page_number)
                spans = normalized_spans(spans,document,page_number)
                owners = _cell_owners(spans, cell_boxes[page_number])
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
                textpage=page.get_textpage()
                try:_complete_display_equations(nodes,spans,textpage,width,table_boxes[page_number])
                finally:textpage.close()
                _recover_code(page, nodes)
                document["deduplicated figure text"] = document.get("deduplicated figure text", 0) + _remove_figure_duplicates(
                    nodes, spans, figure_boxes)
                spans = [span for span in spans if not any(
                    box and _intersection(span.box, box) / max(_area(span.box), 1) > 0.7 for box in figure_boxes
                )]
                displays=[n for n in nodes if n.get('source display equation') and not n.get('suppressed')]
                spans=[s for s in spans if not any(_inside(s.box,_box(n),padding=.5) for n in displays)]
                assignments: dict[int, list[TextSpan]] = defaultdict(list)
                unassigned: list[TextSpan] = []
                nodes = [node for node in nodes if node.get("content", "").strip() and not node.get('suppressed')]
                plain_ids = set()
                for group in _groups(spans, owners):
                    if any(_inside(_group_box(group), _box(node), padding=.5)
                           and _plain_text_group(group, node) for node in nodes):
                        plain_ids.update(span.index for span in group)
                spans = [replace(span, math=False) if span.index in plain_ids else span for span in spans]
                for group in _groups(spans, owners):
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
                orphan_groups = _groups([s for s in spans if not s.math or s.index in unassigned_ids], owners)
                document["deduplicated equation text"] = document.get("deduplicated equation text", 0) + _remove_figure_duplicates(
                    nodes, spans, [_group_box(group) for group in orphan_groups], preserve_captions=False)
                counter = 0

                def save(box: Box, *, block: bool = False, metrics: tuple[float, float] | None = None,
                         complete: bool = True, prose: bool = False) -> str:
                    nonlocal counter
                    counter += 1
                    if block and complete:
                        box = _complete_region(box, nodes)
                    if block and prose:
                        box = _complete_prose_words(box,spans)
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

                for node in displays:
                    if node.get('source display definition'):
                        node['formula snapshot']=save(_box(node),block=True)
                    else:
                        node['content']=save(_box(node),block=True)
                        node['formula image']=True
                    document.setdefault('completed display equations',[]).append({'page':page_number,'box':node['bounding box'],
                                                                                'fragments':node['display fragments'],
                                                                                'definition':bool(node.get('source display definition'))})
                for index, assigned in assignments.items():
                    node = nodes[index]
                    node_box = _box(node)
                    assigned_ids = {s.index for s in assigned}
                    if not node.get("content", "").strip():
                        orphan_groups.extend(_groups([s for s in spans if not s.math or s.index in assigned_ids], owners))
                        continue
                    selected = [s for s in spans if _inside(s.box, node_box) or s.index in assigned_ids]
                    # Keep normal anchors and only the formulas owned by this node.
                    selected = [s for s in selected if not s.math or s.index in assigned_ids]
                    groups = _groups(selected, owners)
                    ranges = _insertions(node["content"], selected, groups)
                    if ranges is None:
                        # Preserve uncertain alignment as a complete source snapshot
                        # rather than insert an equation into an unrelated sentence.
                        whole_box = node_box
                        for group in groups: whole_box = _union(whole_box, _group_box(group))
                        snapshot = save(whole_box, block=True,prose=(
                            not node.get('native table cell') and node.get('type') in {
                                'paragraph','list item','abstract','footnote','frontmatter'}
                            and len(re.findall(r'\b[A-Za-z]{3,}\b',node.get('content','')))>=4))
                        node["formula snapshot"] = snapshot
                        continue
                    content = node["content"]
                    edits = [(start, end, save(_group_box(group), metrics=_inline_metrics(group, selected, node)))
                             for (start, end), group in zip(ranges, groups, strict=True)]
                    for start, end, image in reversed(edits):
                        if end < len(content) and (content[end].isalnum()
                                or (end > start and content[end - 1].isspace())):
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
                textpage = page.get_textpage()
                try:
                    document['merged overlapping prose fragments'] = document.get('merged overlapping prose fragments',0) + _merge_overlapping_prose_snapshots(
                        page_nodes,spans,[a for a in formulas if a['page number']==page_number],save,textpage)
                finally:
                    textpage.close()
                document["deduplicated equation text"] = document.get("deduplicated equation text", 0) + _reconcile_snapshots(
                    page_nodes, spans, [a for a in formulas if a["page number"] == page_number])
                document["merged legacy equation fragments"] = document.get("merged legacy equation fragments", 0) + _merge_legacy_equation_snapshots(
                    page_nodes, spans, [a for a in formulas if a["page number"] == page_number], save)
            finally:
                page.close()
    active = {image_reference(m).source for node in _nodes(document) if not node.get("suppressed")
              for text in (node.get("content", ""), node.get("formula snapshot", ""))
              for m in FIGURE_LINK.finditer(text)}
    document["formulas"] = [asset for asset in formulas if asset["source"] in active]
    return document
