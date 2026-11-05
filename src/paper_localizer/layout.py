"""Conservative reading-order repairs and explicit figure/caption ownership."""

from __future__ import annotations

from collections import defaultdict
import re

from .figures import _nodes, _union


CAPTION = re.compile(r"^(?:[-•■▪‣]\s*)?(Fig(?:ure)?\.?|Table)\s*((?:[A-Z]\.)?\d+(?:\.\d+)*|[IVXLCDM]+)([.:]\s*|\s+)", re.I)
_REFERENCE_VERBS = re.compile(r"^(?:shows?|illustrates?|depicts?|presents?|plots?|evaluates?|is|are|provides?|compares?|summarizes?)\b", re.I)


def _font_family(name: str) -> str:
    name = re.sub(r"^[A-Z]{6}\+", "", name)
    return re.sub(r"(?:[-_, ]?(?:Bold|Italic|Oblique|Regular|Roman|Book|Medium|Semibold|Md|It|Reg|MT))+$",
                  "", name, flags=re.I).casefold()


def caption_label(text: str) -> str:
    match = CAPTION.match(text.strip())
    if not match or _REFERENCE_VERBS.match(text[match.end():].lstrip()):
        return ""
    remainder = text[match.end():].lstrip()
    if match.group(3).isspace() and remainder and remainder[0].islower():
        return ""
    return ("Table " if match.group(1).lower() == "table" else "Fig. ") + match.group(2)


def prepare_layout(document: dict) -> dict:
    """Keep captions atomic even when the layout engine calls them headings.

    Continuations must share the same column, size and tight line spacing. Body
    paragraphs and another caption terminate the region instead of being eaten.
    """
    by_page = defaultdict(list)
    for node in _nodes(document):
        if node.get("content") and node.get("bounding box"):
            by_page[node.get("page number", 1)].append(node)
    heading_fonts = {n.get("font") for nodes in by_page.values() for n in nodes
                     if re.match(r"^\d+\s+[A-Z][A-Z ]+$", n.get("content", ""))
                     and n.get("font") and n["font"] != "OCR" and not re.search(r"Courier|Mono", n["font"], re.I)}
    captions = []
    for nodes in by_page.values():
        consumed = set()
        for node in nodes:
            text = node.get("content", "").strip()
            label = caption_label(text)
            if not label or id(node) in consumed:
                continue
            node["type"] = "caption"
            node["caption label"] = label
            node["content"] = re.sub(r"^(?:[-•■▪‣]\s*)?Fig(?:ure)?\.?\s*", "Fig. ", text, flags=re.I)
            box = tuple(node["bounding box"])
            size = float(node.get("font size") or 9)
            parts = [node["content"]]
            while True:
                candidates = []
                for other in nodes:
                    if other is node or id(other) in consumed or not other.get("content"):
                        continue
                    if other.get("type") in {"header", "footer"} or caption_label(other["content"]):
                        continue
                    if re.match(r"^(?:\d+(?:\.\d+)*[. ]|[IVX]+\. )", other["content"]):
                        continue
                    if (node.get("font") and other.get("font")
                            and _font_family(node["font"]) != _font_family(other["font"])):
                        continue
                    b = tuple(other["bounding box"])
                    gap = box[1] - b[3]
                    # Bounding boxes slightly overlap on superscripts in PDF text.
                    continuation_overlap = (other.get("type") == "caption" and not parts[-1].rstrip().endswith((".", "!", "?"))
                                            and b[3] < box[3] and gap >= -size*3)
                    if (not (-size * 1.3 <= gap <= size * .55 + 1) and not continuation_overlap) or b[1] >= box[1]:
                        continue
                    tolerance = max(.35,size*.3) if node.get("font") == "OCR" else .35
                    if abs(float(other.get("font size") or size) - size) > tolerance:
                        continue
                    if abs(b[0] - box[0]) > 12:
                        continue
                    if len(other["content"].split()) < 2:
                        continue
                    candidates.append((gap, other))
                if not candidates:
                    break
                _, other = min(candidates, key=lambda pair: pair[0])
                parts.append(other["content"].strip())
                if node.get("ocr snapshot") or other.get("ocr snapshot"):
                    node["ocr snapshot"] = True
                consumed.add(id(other)); other["content"] = ""
                box = _union(box, tuple(other["bounding box"]))
            node["content"] = " ".join(parts)
            node["bounding box"] = list(box)
            captions.append(node)
        # IEEE small-cap section titles are sometimes ordinary paragraphs.
        for node in nodes:
            text = node.get("content", "").strip()
            if node.get("type") not in {"paragraph", "list item"} or len(text) > 115:
                continue
            roman = re.match(r"^(?:[IVX]{1,7}\.\s+|[A-Z]\.\s+)[A-Z]", text)
            numeric = re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text) and node.get("font") in heading_fonts
            if (roman or numeric) and len(text.split()) <= 13 and not text.endswith((".", ";")):
                node["type"] = "heading"; node["heading level"] = 2
    document["captions"] = captions
    return document


def caption_for_region(region, captions, height, nodes):
    """Choose a figure caption below, or table caption above, without intervening body text."""
    candidates = []
    for caption in captions:
        label = caption.get("caption label", "")
        if not label.startswith(("Fig.", "Table ")):
            continue
        left, bottom, right, top = caption["bounding box"]
        is_table = label.startswith("Table ")
        gap = region[1] - (height - bottom) if is_table else height - top - region[3]
        table_below = is_table and -4 <= height-top-region[3] <= 24
        if table_below: gap = height-top-region[3]
        overlap = min(right, region[2]) - max(left, region[0])
        if not (-4 <= gap <= (120 if is_table else 420)) or overlap < min(right-left, region[2]-region[0]) * .55:
            continue
        between_bottom, between_top = ((height-region[1], bottom) if is_table and not table_below else (top, height-region[3]))
        blocked = False
        for node in nodes:
            if node is caption or node.get("type") == "caption" or not node.get("content"):
                continue
            b = node.get("bounding box")
            if not b or b[1] <= between_bottom or b[3] >= between_top-2:
                continue
            if min(b[2], right)-max(b[0], left) < (right-left)*.5:
                continue
            if len(node["content"].split()) > 30 or node.get("type") == "heading":
                blocked = True; break
        if not blocked:
            candidates.append((gap, caption))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def bind_captions(document: dict) -> dict:
    """Attach the complete recovered caption regardless of parser reading order."""
    nodes = list(_nodes(document))
    used = set()
    for figure in document.get("figures", []):
        label = figure.get("caption label")
        if not label:
            continue
        caption = next((n for n in nodes if n.get("type") == "caption" and n.get("caption label") == label
                        and n.get("page number") == figure["page number"] and id(n) not in used), None)
        if caption:
            figure["caption node"] = caption
            figure["alt"] = label
            caption["attached caption"] = True; used.add(id(caption))
    return document
