"""Recover document structure from source geometry without generating text."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from statistics import median

import pypdfium2 as pdfium

from .figures import _PDFIUM_LOCK, _union
from .formulas import _canonical, _inside, _spans
from .markup import FIGURE_LINK
from .source_links import restore_printed_targets, source_printed_target


_NUMBERED_TITLE = re.compile(r"^(?:[IVXLCDM]+\.\s+|[A-Z]\.\s+|\d+\)\s+)\S")
_LIST_MARKER = re.compile(r"^(?:[-•▪‣◦]\s+|\d+[.)]\s+|[a-z]\)\s+|\[\d+\]\s+)")
_ABSTRACT = re.compile(r"^abstract\b", re.I)
_KEYWORDS = re.compile(r"^(?:index terms|keywords)\b", re.I)


def _flatten(document: dict) -> list[dict]:
    result = []

    def visit(node: dict, page: int = 1, suppressed: bool = False) -> None:
        page = node.get("page number", page)
        suppressed = suppressed or node.get("suppressed", False)
        children = [child for key in ("kids", "list items") for child in node.get(key, [])]
        if node.get("type") == "table":
            node["page number"] = page
            if suppressed: node["suppressed"] = True
            result.append(node)
            return
        if node.get("content") or node.get("formula snapshot") or node.get("type") in {"image", "picture", "formula"}:
            item = {key: value for key, value in node.items() if key not in {"kids", "list items"}}
            item["page number"] = page
            if suppressed: item["suppressed"] = True
            result.append(item)
        for child in children:
            if isinstance(child, dict): visit(child, page, suppressed)

    for node in document.get("kids", []):
        visit(node)
    return result


def _source_lines(node: dict, spans) -> list[list]:
    box = node.get("bounding box")
    if not box: return []
    lines = []
    for span in sorted((s for s in spans if _inside(s.box, box, padding=.8)),
                       key=lambda s: (-s.baseline, s.box[0])):
        line = next((line for line in lines if abs(line[0].baseline - span.baseline) < 1.5), None)
        if line is None: lines.append([span])
        else: line.append(span)
    return [sorted(line, key=lambda s: s.box[0]) for line in lines]


def _line_text(line) -> str:
    text = " ".join(span.text for span in line)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    return re.sub(r"\s*-\s*", "-", text)


def _is_title(text: str) -> bool:
    letters = "".join(char for char in text if char.isalpha())
    return bool(letters and letters.isupper() and (
        _NUMBERED_TITLE.match(text) or text in {"REFERENCES", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"}))


def _heading_level(text: str, font_name: str = "") -> int:
    if text == "REFERENCES": return 2
    if re.match(r"^[IVXLCDM]+\.\s", text) and not re.search(r"Italic|Oblique|It$", font_name, re.I): return 2
    if re.match(r"^[A-Z]\.\s", text): return 3
    return 4


def _line_box(lines) -> list[float]:
    boxes = [span.box for line in lines for span in line]
    box = boxes[0]
    for other in boxes[1:]: box = _union(box, other)
    return list(box)


def _split_titles(node: dict, spans) -> list[dict]:
    text = node.get("content", "").strip()
    if not text or node.get("type") in {"table", "caption", "image", "picture", "header", "footer"}:
        return [node]
    lines = _source_lines(node, spans)
    if lines:
        node["source first line indent"] = lines[0][0].box[0] - node["bounding box"][0]
    if not lines or not _is_title(_line_text(lines[0])):
        return [node]
    source, offsets = _canonical(text)
    cursor, line_at, result = 0, 0, []
    while line_at < len(lines) and _is_title(_line_text(lines[line_at])):
        first, end = line_at, line_at + 1
        while end < len(lines):
            candidate = _line_text(lines[end])
            letters = "".join(c for c in candidate if c.isalpha())
            if not letters.isupper() or _NUMBERED_TITLE.match(candidate): break
            if lines[end][0].font_name != lines[first][0].font_name: break
            end += 1
        title = " ".join(_line_text(line) for line in lines[first:end])
        key = _canonical(title)[0]
        if not source[cursor:].startswith(key):
            return [node]  # A title is promoted only when the source text agrees.
        start_at = offsets[cursor]
        cursor += len(key)
        end_at = offsets[cursor - 1] + 1
        item = dict(node, type="heading", content=text[start_at:end_at].strip(),
                    **{"heading level": _heading_level(title, lines[first][0].font_name),
                       "bounding box": _line_box(lines[first:end])})
        item.pop("heading split", None)
        result.append(item)
        line_at = end
    if cursor < len(source):
        body = dict(node, type="paragraph", content=text[offsets[cursor]:].strip())
        body.pop("heading level", None)
        body.pop("heading split", None)
        if line_at < len(lines):
            body["bounding box"] = _line_box(lines[line_at:])
            body["source first line indent"] = lines[line_at][0].box[0] - body["bounding box"][0]
            body["font"] = lines[line_at][0].font_name
            body["font size"] = lines[line_at][0].font_size
        result.append(body)
    return result or [node]


def _front_matter(document: dict) -> None:
    nodes = [n for n in document["kids"] if n.get("page number") == 1 and n.get("content")]
    abstract = next((n for n in nodes if _ABSTRACT.match(n["content"])), None)
    title = next((n for n in nodes if n.get("type") == "heading" and n.get("heading level") == 1), None)
    if not abstract or not title: return
    abstract_top = abstract["bounding box"][3]
    title_bottom = title["bounding box"][1]
    authors = []
    for node in nodes:
        box = node.get("bounding box")
        if not box or node is title: continue
        if box[1] > abstract_top:
            if (node.get("type") == "heading" and box[3] < title_bottom
                    and not _NUMBERED_TITLE.match(node["content"])):
                authors.append(node["content"])
            node["type"] = "frontmatter"
            node.pop("heading level", None)
        elif _KEYWORDS.match(node["content"]):
            node["type"] = "keywords"
    if not document.get("author") and authors:
        document["author"] = " ".join(authors)
    document["title bounds"] = title["bounding box"]
    height = document["page sizes"]["1"][1]
    sizes = [n["font size"] for n in nodes if n.get("font size") and n.get("type") == "paragraph"
             and n.get("bounding box", [0, 0, 0, 0])[3] <= abstract_top]
    body_size = median(sizes) if sizes else 10
    for node in nodes:
        box = node.get("bounding box")
        if not box or box[3] > height * .12: continue
        text = node["content"]
        if re.fullmatch(r"\d{1,8}", text.strip()) or re.fullmatch(r"VOLUME\s+\d+,?\s+\d{4}", text.strip(), re.I):
            node["type"] = "footer"
        elif re.search(r"copyright|licensed under|creative commons", text, re.I):
            node["type"] = "frontmatter"
        elif float(node.get("font size") or body_size) < body_size * .86 and len(text.split()) > 6:
            node["type"] = "footnote"


def _restore_source_spacing(node: dict, textpage) -> bool:
    """Use native spacing only when every printable source character agrees."""
    text, box = node.get("content", ""), node.get("bounding box")
    if not text or not box or node.get("type") in {"table", "image", "picture"}: return False
    native = textpage.get_text_bounded(*box).strip()
    if native.endswith(("\x02", "\ufffe")) and text.rstrip().endswith("-"):
        native = native[:-1] + "-"
    native = native.replace("\ufffe", "").replace("\x02", "")
    if any(not char.isprintable() and not char.isspace() for char in native): return False
    if not native or _canonical(native)[0] != _canonical(text)[0]: return False
    native = " ".join(native.split())
    if native == text: return False
    node["content"] = native
    return True


def repair_structure(pdf_path: Path, document: dict) -> dict:
    """Split source-confirmed run-in headings and preserve front matter roles."""
    nodes = _flatten(document)
    by_page = defaultdict(list)
    for node in nodes:
        by_page[node.get("page number", 1)].append(node)
    repaired, sizes, spacing_repairs, link_repairs = [], {}, [], []
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for page_number in range(1, len(pdf) + 1):
            page = pdf[page_number - 1]
            try:
                sizes[str(page_number)] = list(page.get_size())
                spans = [] if page_number in document.get("ocr pages", []) else _spans(page)
                textpage = page.get_textpage()
                try:
                    links = [link for link in document.get("source links", []) if page_number in link.get("pages", [])]
                    if spans and links:
                        native_page = textpage.get_text_bounded()
                        for link in links:
                            if printed := source_printed_target(native_page, link.get("url")):
                                link.setdefault("printed targets", {})[str(page_number)] = printed
                    for node in by_page.get(page_number, []):
                        if spans and not node.get("suppressed") and _restore_source_spacing(node, textpage):
                            spacing_repairs.append({"page": page_number, "bounding box": node["bounding box"]})
                        if spans and not node.get("suppressed") and node.get("content") and node.get("bounding box"):
                            if links:
                                native = textpage.get_text_bounded(*node["bounding box"])
                                node["content"], repairs = restore_printed_targets(node["content"], native, links)
                                link_repairs.extend(dict(repair, page=page_number) for repair in repairs)
                        text = node.get("content", "").strip()
                        if node.get("type") == "list item" and not _LIST_MARKER.match(text):
                            node["type"] = "paragraph"
                        if node.get("suppressed"):
                            repaired.append(node)
                        else:
                            repaired.extend(_split_titles(node, spans))
                finally:
                    textpage.close()
            finally:
                page.close()
    document["kids"] = repaired
    document["page sizes"] = sizes
    document["source spacing repairs"] = spacing_repairs
    document["source link text repairs"] = link_repairs
    for node in repaired:
        if node.get("type") in {"paragraph", "list item"}:
            node["source paragraph text"] = node.get("content", "")
    _front_matter(document)
    return document


def _page_order(nodes: list[dict], width: float) -> list[dict]:
    def top_order():
        return sorted(nodes, key=lambda n: (-n.get("bounding box", [0, 0, 0, 0])[3],
                                            n.get("bounding box", [0, 0, 0, 0])[0]))

    textual = [n for n in nodes if n.get("content") and not n.get("suppressed")
               and n.get("type") not in {"caption", "header", "footer"}
               and (box := n.get("bounding box")) and width * .18 < box[2] - box[0] < width * .6]
    lefts = sorted({round(n["bounding box"][0], 1) for n in textual})
    gaps = [(b - a, a, b) for a, b in zip(lefts, lefts[1:])]
    if not gaps or max(gaps)[0] < width * .18:
        return top_order()
    _, left, right = max(gaps)
    left_edges = [n["bounding box"][2] for n in textual if n["bounding box"][0] <= left + .5]
    gutter = (min(max(left_edges), right) + right) / 2
    evidence = textual + [n for n in nodes if n.get("type") in {"image", "picture"}
                          and (box := n.get("bounding box")) and width*.18 < box[2]-box[0] < width*.6]
    left_column = [n["bounding box"] for n in evidence if (n["bounding box"][0]+n["bounding box"][2])/2 < gutter]
    right_column = [n["bounding box"] for n in evidence if (n["bounding box"][0]+n["bounding box"][2])/2 >= gutter]
    if (not left_column or not right_column
            or min(max(b[3] for b in left_column), max(b[3] for b in right_column))
            <= max(min(b[1] for b in left_column), min(b[1] for b in right_column))):
        return top_order()
    wide = [n for n in nodes if (box := n.get("bounding box"))
            and box[0] < gutter - width * .04 and box[2] > gutter + width * .04]
    remaining = [n for n in nodes if n not in wide]
    result = []

    def columns(items):
        return sorted(items, key=lambda n: (
            (sum(n.get("bounding box", [0, 0, 0, 0])[::2]) / 2) >= gutter,
            -n.get("bounding box", [0, 0, 0, 0])[3], n.get("bounding box", [0, 0, 0, 0])[0]))

    for anchor in sorted(wide, key=lambda n: -n["bounding box"][3]):
        above = [n for n in remaining if n.get("bounding box", [0, 0, 0, 0])[3] > anchor["bounding box"][3] + 1]
        result.extend(columns(above))
        remaining = [n for n in remaining if n not in above]
        result.append(anchor)
    result.extend(columns(remaining))
    return result


def order_document(document: dict) -> dict:
    """Read each column completely between full-width figures and text bands."""
    by_page = defaultdict(list)
    for node in _flatten(document):
        by_page[node.get("page number", 1)].append(node)
    ordered = []
    for number, nodes in sorted(by_page.items()):
        width = document.get("page sizes", {}).get(str(number), [612, 792])[0]
        ordered.extend(_page_order(nodes, width))
    in_references = False
    for index, node in enumerate(ordered):
        if node.get("type") == "heading":
            in_references = bool(re.fullmatch(r"references", node.get("content", ""), re.I))
        if in_references and node.get("asset kind") == "portrait":
            box = node["bounding box"]
            ordered.insert(index, {"type": "heading", "heading level": 2, "content": "Author biographies",
                                   "page number": node["page number"], "generated heading": True,
                                   "bounding box": [box[0], box[3]+1, box[2], box[3]+3]})
            break
    document["kids"] = ordered
    previous = None
    paragraph_number = 0
    for node in ordered:
        kind = node.get("type")
        if node.get("suppressed") or kind in {"image", "picture", "caption", "frontmatter", "keywords", "footnote", "header", "footer"}:
            continue
        if kind != "paragraph":
            previous = None
            continue
        text = node.get("source paragraph text", FIGURE_LINK.sub(" X ", node.get("content", ""))).strip()
        continuation = False
        if previous and text:
            prior_text = previous.get("source paragraph text", FIGURE_LINK.sub(" X ", previous.get("content", ""))).strip()
            lower_start = (text[0].islower() or bool(re.match(r"^[A-Za-z]\s+[a-z]", text))
                           or bool(re.match(r"^\[\d+\](?:[,.;]|\s)", text)))
            same_page = node["page number"] == previous["page number"]
            next_page = node["page number"] == previous["page number"] + 1
            first_box, second_box = previous.get("bounding box"), node.get("bounding box")
            geometry = next_page
            boundary = next_page
            if same_page and first_box and second_box:
                column_change = second_box[0] - first_box[0] > 70
                boundary = column_change
                adjacent = abs(second_box[0] - first_box[0]) < 15 and abs(first_box[1] - second_box[3]) < 16
                around_float = (first_box[0] - second_box[0] > 35 and abs(first_box[2] - second_box[2]) < 8
                                and abs(first_box[1] - second_box[3]) < 3)
                lower_start = lower_start or around_float
                geometry = column_change or adjacent or around_float
            indent = node.get("source first line indent")
            sizes = [float(n.get("font size") or 0) for n in (previous, node)]
            if (boundary and text[0].isupper() and indent is not None and -1 <= indent <= 2
                    and min(sizes) > 0 and min(sizes) >= max(sizes) * .95):
                lower_start = True
            continuation = bool(prior_text and re.search(r"[\w,–—-]$", prior_text) and lower_start and geometry)
        if not continuation:
            paragraph_number += 1
        node["paragraph id"] = f"flow-{paragraph_number}"
        previous = node
    _reference_flows(ordered)
    return document


def _reference_flows(nodes: list[dict]) -> None:
    """Keep hanging-indent continuations inside their numbered reference."""
    in_references, previous, reference_id = False, None, ""
    for index, node in enumerate(nodes):
        kind = node.get("type")
        if kind == "heading":
            in_references = bool(re.fullmatch(r"references", node.get("content", ""), re.I))
            previous, reference_id = None, ""
            continue
        if node.get("suppressed") or kind in {"header", "footer"}:
            continue
        if not in_references or kind not in {"paragraph", "list item"}:
            previous, reference_id = None, ""
            continue
        text = node.get("source paragraph text", node.get("content", "")).strip()
        if re.match(r"^\[\d+\]\s+", text):
            reference_id = f"reference-{index}"
        elif previous and reference_id:
            first, second = previous.get("bounding box"), node.get("bounding box")
            next_page = node["page number"] == previous["page number"] + 1
            adjacent = (node["page number"] == previous["page number"] and first and second
                        and -2 <= first[1] - second[3] <= 16
                        and -2 <= second[0] - first[0] <= 35)
            if not (next_page or adjacent):
                reference_id = ""
        else:
            reference_id = ""
        if reference_id:
            node["paragraph id"] = reference_id
        previous = node
