"""Recover raster and vector figures without OCR or a remote layout service."""

from __future__ import annotations

import hashlib
import ctypes
import math
import re
import threading
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from pypdf import PdfWriter
import pypdfium2 as pdfium


# PDFium must not be used concurrently, including by separate Streamlit sessions.
_PDFIUM_LOCK = threading.Lock()
_FIGURE_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|그림)\s*\d", re.IGNORECASE)
Box = tuple[float, float, float, float]


def _area(box: Box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _nodes(node: dict):
    yield node
    for key in ("kids", "list items", "rows", "cells"):
        for child in node.get(key, []):
            if isinstance(child, dict):
                yield from _nodes(child)


def _node_box(node: dict, height: float) -> Box | None:
    box = node.get("bounding box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    left, bottom, right, top = map(float, box)
    return left, height - top, right, height - bottom


def _connected_regions(mask: Image.Image, gap: int = 7, *, min_width: int = 40,
                       min_height: int = 18, min_ink: int = 30) -> list[Box]:
    """Find components in a graphics mask, joining nearby drawing fragments."""
    expanded = mask.copy() if gap == 0 else mask.filter(ImageFilter.MaxFilter(gap * 2 + 1))
    data = expanded.tobytes()
    width, height = expanded.size
    parents: list[int] = []
    bounds: list[Box] = []

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = data[y * width:(y + 1) * width]
        current: list[tuple[int, int, int]] = []
        x = 0
        cursor = 0
        while (start := row.find(b"\xff", x)) >= 0:
            end = row.find(b"\x00", start)
            if end < 0:
                end = width
            index = len(parents)
            parents.append(index)
            bounds.append((start, y, end, y + 1))
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            other = cursor
            while other < len(previous) and previous[other][0] <= end:
                parent = root(previous[other][2])
                own = root(index)
                if parent != own:
                    parents[own] = parent
                other += 1
            current.append((start, end, index))
            x = end
        previous = current
    grouped: dict[int, Box] = {}
    for index, box in enumerate(bounds):
        key = root(index)
        grouped[key] = _union(grouped[key], box) if key in grouped else box
    regions: list[Box] = []
    for box in grouped.values():
        # Undo dilation using the actual visible pixels, including clipping/masks.
        tile = mask.crop(tuple(map(int, box)))
        tight = tile.getbbox()
        if tight:
            candidate = (box[0] + tight[0], box[1] + tight[1],
                         box[0] + tight[2], box[1] + tight[3])
            w, h = candidate[2] - candidate[0], candidate[3] - candidate[1]
            if w >= min_width and h >= min_height and tile.histogram()[255] >= min_ink:
                regions.append(candidate)
        tile.close()
    expanded.close()
    return regions


def _horizontal_rules(mask: Image.Image, box: Box) -> bool:
    tile = mask.crop(box)
    width, height = tile.size
    pixels = tile.tobytes(); tile.close()
    counts = [pixels[y*width:(y+1)*width].count(255) for y in range(height)]
    total = sum(counts)
    dense = sum(n for n in counts if n > width*.4)
    return bool(total and dense/total > .9 and sum(n > 0 for n in counts) < max(30, height*.3))


def _repeated_margin_graphics(graphics) -> dict[int, list[Box]]:
    """Find repeated margin artwork before it can be assigned to a caption.

    Repetition and a short region confined to the page margin are both required;
    an ordinary uncaptioned drawing in the body remains a real image.
    """
    candidates = defaultdict(list)
    for index in range(len(graphics)):
        page = graphics[index]
        try:
            width, height = page.get_size()
            bitmap = page.render(scale=1, grayscale=True, draw_annots=False)
            image = bitmap.to_pil()
            mask = image.point(lambda value: 255 if value < 235 else 0).convert("L")
            # A running rule may touch a nearby figure after dilation. Detect
            # its thin pixel rows before connected components join them.
            for y in list(range(int(height*.05))) + list(range(int(height*.95), int(height))):
                row = mask.crop((0, y, int(width), y+1))
                dense = row.histogram()[255] >= width*.8
                bounds = row.getbbox();row.close()
                if dense and bounds:
                    box = (bounds[0], y, bounds[2], y+1)
                    candidates["rule", round(y/height*300)].append((index+1, box))
            for box in _connected_regions(mask):
                if box[3] - box[1] > height * .08: continue
                side = "top" if box[3] < height * .09 else "bottom" if box[1] > height * .91 else ""
                if not side: continue
                key = side, round(box[0] / width * 20), round(box[2] / width * 20), round(box[1] / height * 20)
                candidates[key].append((index + 1, box))
            mask.close(); image.close(); bitmap.close()
        finally:
            page.close()
    repeated = defaultdict(list)
    minimum = max(3, math.ceil(len(graphics) * .3))
    for items in candidates.values():
        if len({number for number, _ in items}) >= minimum:
            for number, box in items: repeated[number].append(box)
    return repeated


def _has_figure_caption(region: Box, nodes: list[dict], height: float) -> bool:
    for node in nodes:
        text = str(node.get("content", "")).strip()
        box = _node_box(node, height)
        if not box or not _FIGURE_CAPTION.match(text):
            continue
        horizontal = min(region[2], box[2]) - max(region[0], box[0])
        if horizontal > 0 and -8 <= box[1] - region[3] <= 65:
            return True
    return False


def _is_table(region: Box, nodes: list[dict], height: float) -> bool:
    if _has_figure_caption(region, nodes, height):
        return False
    for node in nodes:
        if node.get("type") != "table":
            continue
        box = _node_box(node, height)
        rows = node.get("rows", [])
        if not box or len(rows) < 3:
            continue
        cells = [cell for row in rows for cell in row.get("cells", [])]
        columns = max((len(row.get("cells", [])) for row in rows), default=0)
        filled = sum(any(str(n.get("content", "")).strip() for n in _nodes(cell)) for cell in cells)
        if (columns >= 2 and cells and filled / len(cells) >= 0.6
                and _intersection(region, box) / max(_area(region), 1) >= 0.85
                and _intersection(region, box) / max(_area(box), 1) >= 0.7):
            return True
    return False


def _include_labels(region: Box, nodes: list[dict], width: float, height: float) -> Box:
    result = region
    caption_boxes = [box for node in nodes
                     if (node.get("type") == "caption" or _FIGURE_CAPTION.match(str(node.get("content", "")).strip()))
                     and (box := _node_box(node, height))]
    body_boxes = [box for node in nodes
                  if len(str(node.get("content", "")).split()) > 25
                  and node.get("type") != "caption"
                  and not _FIGURE_CAPTION.match(str(node.get("content", "")).strip())
                  and (box := _node_box(node, height))]
    for node in nodes:
        text = str(node.get("content", "")).strip()
        box = _node_box(node, height)
        if not text or not box:
            continue
        # A PDF text object may span several columns even if its decoded text is
        # short. Such bounds must never pull the neighboring plot into this crop.
        if box[2] - box[0] > max(160, (region[2] - region[0]) * 1.6):
            continue
        vertical = box[3] - box[1] > (box[2] - box[0]) * 2.5 and box[2] - box[0] <= 20
        if (len(text) > 120 or _FIGURE_CAPTION.match(text) or node.get("type") == "caption"
                or node.get("type") == "heading" or (box[3] - box[1] > 40 and not vertical)):
            continue
        if any(_intersection(box, caption) / max(_area(box), 1) > 0.5 for caption in caption_boxes):
            continue
        if any(_intersection(box, body) / max(_area(box), 1) > 0.8 for body in body_boxes):
            continue
        if len(text.split()) >= 9 and text.endswith((".", ";", ":")):
            continue
        gap_x = max(region[0] - box[2], box[0] - region[2], 0)
        gap_y = max(region[1] - box[3], box[1] - region[3], 0)
        above_ocr_label = (node.get("font") == "OCR" and box[3] <= region[1]
                           and len(text.split()) <= 8 and not text.endswith((".", ";", ":")))
        if (gap_x <= 8 and gap_y <= (28 if above_ocr_label else 13)) or (vertical and gap_x <= 35 and gap_y == 0):
            result = _union(result, box)
    bottom = min(height, result[3] + 4)
    for caption in caption_boxes:
        if region[3] <= caption[1] and caption[1] < bottom:
            bottom = max(region[3], caption[1] - 2)
    return (max(0, result[0] - 4), max(0, result[1] - 4), min(width, result[2] + 4), bottom)


def _portrait_label(region: Box, nodes: list[dict], height: float) -> str:
    width, tall = region[2] - region[0], region[3] - region[1]
    if not tall or not .45 <= width / tall <= 1.2 or tall > height * .3:
        return ""
    for node in nodes:
        text = node.get("content", "")
        match = re.match(r"^([A-Z][A-Z.'’\-]+(?:\s+[A-Z][A-Z.'’\-]+){1,5})"
                         r"(?:\s+\([^)]{1,50}\))?\s+(?:received|earned)\s+(?:the|a)\s+"
                         r"(?:B\.?S\.?|M\.?S\.?|Ph\.?D\.?|bachelor|master|doctoral|degree)(?=\W|$)", text)
        box = _node_box(node, height)
        if not match or not box: continue
        overlap = min(box[3], region[3]) - max(box[1], region[1])
        if -5 <= box[0] - region[2] <= 25 and overlap > min(tall, box[3]-box[1]) * .5:
            return match.group(1)
    return ""


def _text_labels(page) -> list[dict]:
    # Object bounds inside an XObject are in *local* coordinates. Character
    # bounds from the text page include every enclosing form transform, which
    # is essential for plot legends and labels imported from another PDF.
    textpage = page.get_textpage()
    labels: list[dict] = []
    objects = {}
    try:
        for index in range(textpage.count_chars()):
            code = pdfium.raw.FPDFText_GetUnicode(textpage, index)
            if not code: continue
            obj = textpage.get_textobj(index)
            if obj is None: continue
            key = ctypes.cast(obj.raw, ctypes.c_void_p).value
            if chr(code).isspace():
                if key in objects: objects[key][0] += " "
                continue
            box = tuple(textpage.get_charbox(index))
            if not _area(box): continue
            if key in objects:
                objects[key][1] = _union(objects[key][1], box)
                objects[key][0] += chr(code)
            else:
                objects[key] = [chr(code), box]
        for text, box in objects.values():
            if text and len(text) <= 120:
                labels.append({"content": text, "bounding box": list(box)})
    finally:
        textpage.close()
    return labels


def _refine_caption_tops(page, nodes: list[dict]) -> None:
    """Correct caption bounds using the printed figure or table label.

    Find the printed label using page character coordinates. Keep uncertain
    tall mathematics intact rather than inferring a smaller caption rectangle.
    """
    captions = [n for n in nodes if n.get("type") == "caption"
                and n.get("caption label", "").startswith(("Fig.", "Table "))]
    if not captions: return
    textpage = page.get_textpage()
    try:
        codes = [pdfium.raw.FPDFText_GetUnicode(textpage,i) for i in range(textpage.count_chars())]
        text = "".join(chr(code) if code else "\ufffd" for code in codes)
        for node in captions:
            left,bottom,right,top = node["bounding box"]
            size = float(node.get("font size") or 9)
            is_table = node["caption label"].startswith("Table ")
            number = re.escape(node["caption label"][6:] if is_table else node["caption label"][5:]).replace(r"\.", r"\.\s*")
            prefix = r"\bTable\s*" if is_table else r"\bFig(?:ure)?\.?\s*"
            matches = []
            for match in re.finditer(prefix+number+r"(?!\d|\.\d)",text,re.I):
                boxes = [tuple(textpage.get_charbox(i)) for i in range(match.start(),match.end()) if not text[i].isspace()]
                if not boxes: continue
                box = boxes[0]
                for other in boxes[1:]: box = _union(box,other)
                if left-4 <= box[0] <= left+15 and bottom-3 <= box[1] < top:
                    matches.append(box)
            if not matches: continue
            label = max(matches,key=lambda b:b[3])
            # Table labels can sit only a few points below the final border.
            # A full line-box ascent would clip that border during separation.
            padding = .7 if is_table else max(1.2, size*.18)
            if top-label[3] < (padding+.5 if is_table else max(4,size*.8)): continue
            tall_math = False
            for index,code in enumerate(codes):
                if not code or chr(code).isspace(): continue
                box = tuple(textpage.get_charbox(index))
                if box[0] < left or box[2] > right or box[3] <= label[3]+2 or box[1] >= top:
                    continue
                obj = textpage.get_textobj(index)
                font = obj.get_font().get_base_name() if obj else ""
                if (not font or re.search(r"CMMI|CMSY|CMEX|MSAM|MSBM|Symbol|Math",font,re.I)
                        or chr(code) in "=+≤≥∑∫√" or 0x1D400 <= code <= 0x1D7FF):
                    tall_math = True; break
            if not tall_math:
                node["bounding box"] = [left,bottom,right,label[3]+padding]
    finally:
        textpage.close()


def _merge_regions(regions: list[Box]) -> list[Box]:
    merged: list[Box] = []
    for region in regions:
        while True:
            hit = next((index for index, other in enumerate(merged)
                        if _intersection(region, other) > 0), None)
            if hit is None:
                break
            region = _union(region, merged.pop(hit))
        merged.append(region)
    return merged


def _caption_boundaries(regions: list[Box], captions: list[dict], height: float) -> list[Box]:
    """Separate artwork joined across a caption, and remove caption slivers.

    The graphics mask omits text, so its dilation can bridge two tables on
    either side of a caption. Keep the caption outside both source crops.
    """
    result = list(regions)
    for caption in captions:
        boundary = _node_box(caption, height)
        if not boundary:
            continue
        left, top, right, bottom = boundary
        margin = .5 if caption.get("caption label", "").startswith("Table ") else 2
        separated = []
        for region in result:
            overlap = min(region[2], right) - max(region[0], left)
            if (overlap < min(region[2]-region[0], right-left)*.55
                    or region[3] < top or region[1] > bottom):
                separated.append(region)
                continue
            pieces = []
            if top - region[1] > 18:
                pieces.append((region[0], region[1], region[2], top-margin))
            if region[3] - bottom > 18:
                pieces.append((region[0], bottom+margin, region[2], region[3]))
            # An uncertain caption covering an entire small graphic is not
            # enough evidence to discard that graphic.
            separated.extend(pieces or [region])
        result = separated
    return result


def _region_captions(regions: list[Box], captions: list[dict], nodes: list[dict],
                     height: float, source_tables: list[tuple[Box, str]]) -> list[dict | None]:
    """Assign adjacent tables jointly before crop expansion changes their gaps."""
    from .layout import caption_for_region
    owners: list[dict | None] = [None] * len(regions)
    used = set()
    pending = []
    for index, region in enumerate(regions):
        label = next((label for box, label in source_tables
                      if _intersection(region,box)/max(_area(region),1) > .85
                      and _intersection(region,box)/max(_area(box),1) > .85), None)
        if label:
            owners[index] = next((n for n in captions if n.get("caption label") == label), None)
            if owners[index] is not None:
                used.add(id(owners[index]))
            continue
        choices = []
        remaining = [n for n in captions if n.get("caption label") not in {label for _, label in source_tables}]
        while (caption := caption_for_region(region, remaining, height, nodes)) is not None:
            choices.append(caption)
            remaining = [n for n in remaining if n is not caption]
        if not choices:
            continue
        if choices[0]["caption label"].startswith("Table "):
            tables = [n for n in choices if n["caption label"].startswith("Table ")]
            pending.append((index, tables))
        else:
            owners[index] = choices[0]
    # A table with one eligible caption establishes the boundary for its
    # neighbor, which can otherwise match both the caption above and below.
    for index, choices in sorted(pending, key=lambda item: len(item[1])):
        owners[index] = next((n for n in choices if id(n) not in used), choices[0])
        used.add(id(owners[index]))
    return owners


def _merge_captioned_regions(regions: list[tuple[Box, dict | None]]) -> list[tuple[Box, dict | None]]:
    """Keep detached panels with their owner without joining different captions."""
    grouped = {}
    for region, caption in regions:
        if caption is None:
            continue
        key = id(caption)
        grouped[key] = (_union(grouped[key][0], region) if key in grouped else region, caption)
    unbound = _merge_regions([region for region, caption in regions if caption is None])
    remaining = []
    for region in unbound:
        matches = [key for key, (box, _) in grouped.items() if _intersection(region, box) > 0]
        if len(matches) == 1:
            key = matches[0]
            box, caption = grouped[key]
            grouped[key] = _union(box, region), caption
        else:
            remaining.append((region, None))
    return list(grouped.values()) + remaining


def _pseudocode_region(caption: dict, nodes: list[dict], width: float, height: float) -> Box | None:
    """Text-only algorithm figures have no ink after suppressing text drawing."""
    label = caption.get("content", "")
    if not re.search(r"\b(?:pseudo\s*code|(?:source )?code|(?:property|function|procedure|algorithm)\s+\d+\s+definition)\b", label, re.I): return None
    cl, cb, cr, ct = caption["bounding box"]
    if cr-cl > width*.55: return None
    centered = cl < width*.42 and cr > width*.58
    left, right = (0, width) if centered else ((0, width/2) if (cl+cr)/2 < width/2 else (width/2, width))
    above = [n for n in nodes if n is not caption and (b := n.get("bounding box"))
             and left <= (b[0]+b[2])/2 <= right and b[1] > ct
             and n.get("content") and not n.get("suppressed")]
    ceiling = min((n["bounding box"][1] for n in nodes if n is not caption and n.get("caption label")
                   and (b := n.get("bounding box")) and b[1] > ct and b[0] < right and b[2] > left),
                  default=height*.96)
    group = [];top = ct
    for node in sorted(above, key=lambda n: n["bounding box"][1]):
        box = node["bounding box"];text = node["content"]
        if box[3] > ceiling or node.get("caption label") or node.get("type") in {"header", "footer", "frontmatter"}: break
        if box[1]-top > 45: break
        if any(re.match(r"^\(\d+\)", n["content"].strip()) for n in group) and not re.match(r"^\(\d+\)", text.strip()): break
        if len(text.split()) > 25 and not re.search(r"//|;|\b(?:for|if|while|return)\b", text): break
        if node.get("type") == "heading" and re.match(r"^(?:\d+(?:\.\d+)*\.?\s+|[IVXLCDM]+\.\s+)[A-Z]", text): break
        group.append(node);top=max(top,box[3])
    control = sum(bool(re.match(r"^(?:for|while|if|else|return|//)\b|^//", n["content"].strip())) for n in group)
    numbered = [int(m[1]) for n in sorted(group,key=lambda n:-n["bounding box"][3])
                if (m := re.match(r"^\(?(\d+)\)?(?:\s+|$)", n["content"].strip()))]
    numbered_code = (len(numbered)>=4 and numbered==list(range(1,len(numbered)+1))
                     and sum(bool(re.search(r"[(){};=]",n["content"])) for n in group)>=len(group)*.6)
    if len(group) < 4 or control < 2 and not numbered_code: return None
    for node in group:
        if node.get("type") == "caption" and not node.get("caption label"):
            node["type"] = "paragraph"
    box = tuple(group[0]["bounding box"])
    for node in group[1:]: box = _union(box, tuple(node["bounding box"]))
    if centered:
        left, right = max(0, min(cl, box[0])-2), min(width, max(cr, box[2])+2)
    return left, max(0,height-box[3]-2), right, height-box[1]+2


def _include_graphic_fragments(regions: list[Box], fragments: list[Box],
                               captions: list[dict], nodes: list[dict], height: float) -> list[Box]:
    """Keep small detached drawing parts beside a captioned diagram.

    Require the fragment's vertical extent to sit within the original drawing,
    a nearby horizontal edge, and a bounded area change. Assign against the
    original regions once so fragments cannot chain into neighboring drawings.
    """
    from .layout import caption_for_region
    owners = [index for index, region in enumerate(regions)
              if caption_for_region(region, captions, height, nodes)]
    result = list(regions)
    for fragment in fragments:
        candidates = []
        for index in owners:
            region = regions[index]
            gap = max(region[0] - fragment[2], fragment[0] - region[2], 0)
            if (region[1] <= fragment[1] and fragment[3] <= region[3] and gap <= 20
                    and _area(_union(region, fragment)) <= _area(region) * 1.25):
                candidates.append((gap, abs((fragment[0]+fragment[2]) - (region[0]+region[2])), index))
        if candidates:
            index = min(candidates)[2]
            result[index] = _union(result[index], fragment)
    return result


def _replace_figure_text(document: dict, figures: list[dict], heights: dict[int, float]) -> None:
    emitted: set[str] = set()
    by_page: dict[int, list[dict]] = defaultdict(list)
    for figure in figures:
        by_page[figure["page number"]].append(figure)

    def rewrite(node: dict, inherited_page: int = 1) -> list[dict]:
        page = node.get("page number", inherited_page)
        box = _node_box(node, heights.get(page, 0))
        text = str(node.get("content", "")).strip()
        if node.get("type") == "caption":
            # The layout engine can put the *next* figure underneath a caption
            # node. Emit recovered figures at document level, avoiding cycles
            # and making caption ownership independent of that nesting.
            return [node]
        has_caption = any(n is not node and n.get("type") == "caption" for n in _nodes(node))
        if box and _area(box) and not has_caption and node.get("type") != "caption" and not _FIGURE_CAPTION.match(text):
            for figure in by_page.get(page, []):
                figure_box = _node_box(figure, heights[page])
                if _intersection(box, figure_box) / _area(box) >= 0.7:
                    key = figure["source"]
                    if key not in emitted:
                        emitted.add(key)
                        return [figure]
                    return []
        for key in ("kids", "list items"):
            if key in node:
                node[key] = [item for child in node[key] for item in rewrite(child, page)]
        return [node]

    document["kids"] = [item for node in document.get("kids", []) for item in rewrite(node)]
    # Raster images without a text label may have no corresponding text node.
    for figure in figures:
        if figure["source"] in emitted:
            continue
        page = figure["page number"]
        figure_box = _node_box(figure, heights[page])
        insertion = len(document["kids"])
        for index, node in enumerate(document["kids"]):
            node_page = node.get("page number", 1)
            node_box = _node_box(node, heights.get(node_page, 0))
            if node_page > page or (node_page == page and node_box and node_box[1] >= figure_box[1]):
                insertion = index
                break
        document["kids"].insert(insertion, figure)
    document["figures"] = figures


def attach_figures(pdf_path: Path, document: dict, output_dir: Path) -> dict:
    """Write cropped PNGs to output_dir/figs and remove duplicate diagram labels.

    A graphics-only in-memory copy suppresses text drawing operations. Its actual
    rendered pixels locate vector diagrams, rasters and clipped plots. The final
    crops always come from the unchanged original, retaining figure labels.
    """
    with pdf_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()[:12]
    asset_dir = output_dir / "figs" / digest
    page_nodes: dict[int, list[dict]] = defaultdict(list)
    for node in _nodes(document):
        if "page number" in node:
            page_nodes[node["page number"]].append(node)
    figures: list[dict] = []
    heights: dict[int, float] = {}
    writer = PdfWriter(clone_from=pdf_path)
    buffer = BytesIO()
    try:
        writer.remove_text()
        writer.write(buffer)
    finally:
        writer.close()
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as original, pdfium.PdfDocument(buffer.getvalue()) as graphics:
        repeated = _repeated_margin_graphics(graphics)
        excluded = document.setdefault("excluded artwork", [])
        for index in range(len(original)):
            page_number = index + 1
            page, graphics_page = original[index], graphics[index]
            try:
                width, height = page.get_size()
                heights[page_number] = height
                nodes = page_nodes.get(page_number, [])

                def furniture(region):
                    repeated_margin = any(_intersection(region, box) / max(_area(region), 1) > .8
                                          for box in repeated.get(page_number, []))
                    title = document.get("title bounds")
                    before_title = (page_number == 1 and title and region[3] < height - title[3] - 2
                                    and region[3] - region[1] < height * .08)
                    if repeated_margin or before_title:
                        reason = "repeated margin artwork" if repeated_margin else "artwork above the article title"
                        entry = {"page number": page_number,
                                 "bounding box": [region[0], height-region[3], region[2], height-region[1]],
                                 "reason": reason}
                        if entry not in excluded: excluded.append(entry)
                        return True
                    return False

                for node in nodes:
                    if node.get("type") in {"image", "picture"} and (box := _node_box(node, height)) and furniture(box):
                        node["suppressed"] = True
                _refine_caption_tops(page,nodes)
                bitmap = graphics_page.render(scale=1, grayscale=True, draw_annots=False)
                image = bitmap.to_pil()
                mask = image.point(lambda value: 255 if value < 235 else 0).convert("L")
                draw = ImageDraw.Draw(mask)
                for l,t,r,b in repeated.get(page_number, []):
                    if b-t <= 3 and r-l >= width*.75:
                        draw.rectangle((l-1,t-1,r+1,b+1), fill=0)
                if page_number in document.get("ocr pages", []):
                    # Scans retain prose in their raster even after remove_text.
                    # Suppress detected text only in the search mask; final
                    # crops always use the unchanged original page.
                    draw = ImageDraw.Draw(mask)
                    for l,b,r,t in document.get("ocr line boxes",{}).get(str(page_number),[]):
                        draw.rectangle((max(0,l-1),max(0,height-t-1),min(width,r+1),min(height,height-b+1)),fill=0)
                components = [region for region in _connected_regions(
                    mask, min_width=3, min_height=3) if not furniture(region)]
                regions = [region for region in components
                           if region[2]-region[0] >= 40 and region[3]-region[1] >= 18]
                fragments = [region for region in components if region not in regions]
                rules = {r for r in regions if _horizontal_rules(mask, r)}
                # A diagram's detached ID-space axis can look like a page
                # separator. Retain it when it sits directly above a captioned
                # graphic in the same column (e.g. a tree and its scale).
                diagram_axes = {r for r in rules if any(
                    other not in rules and other[3] - other[1] > 35
                    and 0 <= other[1] - r[3] <= 40
                    and min(r[2], other[2]) - max(r[0], other[0]) > min(r[2]-r[0], other[2]-other[0])*.65
                    and _has_figure_caption(other, nodes, height)
                    for other in regions)}
                mask.close(); image.close(); bitmap.close()
                regions = [r for r in regions if r not in rules or r in diagram_axes or any(
                    (n.get("type") in {"image", "picture", "table"}
                     and (b := _node_box(n, height)) and _intersection(r,b)/max(_area(r),1) > .5)
                    or (n.get("type") == "caption" and (b := _node_box(n,height))
                        and min(r[2],b[2])-max(r[0],b[0]) > 0
                        and min(abs(b[1]-r[3]), abs(r[1]-b[3])) < 75)
                    for n in nodes)]
                # Small raster images can fall below the graphics-component threshold.
                for node in nodes:
                    if node.get("type") in {"image", "picture"} and not node.get("suppressed"):
                        box = _node_box(node, height)
                        if page_number in document.get("ocr pages", []) and box and _area(box) > width*height*.65:
                            continue
                        if box and _area(box) > 100:
                            clipped = (max(0, box[0]), max(0, box[1]),
                                       min(width, box[2]), min(height, box[3]))
                            if _area(clipped) and not any(
                                _intersection(clipped, r) / _area(clipped) >= 0.8 for r in regions
                            ):
                                regions.append(clipped)
                regions = [r for r in regions if not _is_table(r, nodes, height)]
                for table in document.get("source table regions", []):
                    if table["page number"] != page_number: continue
                    box = _node_box(table, height)
                    regions = [r for r in regions if _intersection(r,box)/max(_area(r),1)<.85]
                    regions.append(box)
                labels = nodes + ([] if page_number in document.get("ocr pages", []) else _text_labels(page))
                # Multi-panel plots share one caption. Group by the caption's
                # column instead of joining everything that has nearby graphics.
                captions = [n for n in nodes if n.get("type") == "caption"
                            and n.get("content", "").strip() and n.get("caption label")]
                source_tables = [(_node_box(t,height),t["caption label"])
                                 for t in document.get("source table regions",[]) if t["page number"]==page_number]
                table_labels = {label for _,label in source_tables} | {
                    n["caption label"] for n in nodes
                    if n.get("source reconstructed table") and n.get("caption label")}
                regions = _include_graphic_fragments(regions, fragments, captions, nodes, height)
                regions = _caption_boundaries(regions, captions, height)
                available = [n for n in captions if n["caption label"] not in table_labels
                             or n["caption label"] in {label for _, label in source_tables}]
                owners = _region_captions(regions, available, nodes, height, source_tables)
                expanded = []
                for region, caption in zip(regions, owners):
                    if any(_intersection(region,box)/max(_area(region),1)>.98
                           and _intersection(region,box)/max(_area(box),1)>.98 for box,_ in source_tables):
                        expanded.append((region, caption))
                        continue
                    original_region = region
                    if caption:
                        left, _, right, _ = caption["bounding box"]
                        # A centered caption may be shorter than its figure.
                        # Use the page column/gutter as a boundary; never trim
                        # the detected graphics to the caption's text width.
                        center = (left + right) / 2
                        low, high = 0, width
                        if right-left < width*.48 and region[2]-region[0] < width*.48:
                            if center < width/2 and (region[0]+region[2])/2 < width/2:
                                high = width/2
                            elif center >= width/2 and (region[0]+region[2])/2 >= width/2:
                                low = width/2
                        for neighbor in captions:
                            if neighbor is caption: continue
                            nl, nb, nr, nt = neighbor["bounding box"]
                            if abs(nt-caption["bounding box"][3]) > 22: continue
                            nc = (nl+nr)/2
                            if nc < center: low = max(low, (nc+center)/2)
                            elif nc > center: high = min(high, (nc+center)/2)
                        low, high = min(low, region[0]), max(high, region[2])
                        local_labels = [n for n in labels if (b := _node_box(n, height))
                                        and b[0] >= low and b[2] <= high]
                        region = _include_labels(region, local_labels, width, height)
                    else:
                        region = _include_labels(region, [] if _portrait_label(region, nodes, height) else labels, width, height)
                    # Padding or a detached text label must not bridge back
                    # across the caption that separated the original artwork.
                    region = max(_caption_boundaries([region], captions, height),
                                 key=lambda box: _intersection(box, original_region))
                    expanded.append((region, caption))
                # Text labels can bring an initially distant fragment within
                # caption range. Keep established owners and resolve only the
                # remaining parts, then reconnect overlapping unowned panels.
                claimed = {id(caption) for _, caption in expanded if caption is not None
                           and caption["caption label"].startswith("Table ")}
                from .layout import caption_for_region
                expanded = [(region, caption if caption is not None else caption_for_region(
                    region, [n for n in available if id(n) not in claimed], height, nodes))
                    for region, caption in expanded]
                region_captions = sorted(_merge_captioned_regions(expanded),
                                         key=lambda item: (item[0][1], item[0][0]))
                listing_captions = set()
                for caption in captions:
                    if any(owner is caption for _,owner in region_captions): continue
                    if region := _pseudocode_region(caption, nodes, width, height):
                        region_captions.append((region,caption))
                        listing_captions.add(id(caption))
                region_captions.sort(key=lambda item: (item[0][1], item[0][0]))
                for figure_index, (region, caption) in enumerate(region_captions, 1):
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"page-{page_number:03d}-fig-{figure_index:02d}.png"
                    destination = asset_dir / filename
                    left, top, right, bottom = region
                    crop = (left, max(0, height - bottom), max(0, width - right), top)
                    bitmap = page.render(scale=2.5, crop=crop, draw_annots=False)
                    image = bitmap.to_pil()
                    pixel_size = list(image.size)
                    image.save(destination)
                    image.close(); bitmap.close()
                    portrait = _portrait_label(region, nodes, height) if not caption else ""
                    figures.append({
                        "type": "image", "page number": page_number,
                        "bounding box": [left, height - bottom, right, height - top],
                        "source": destination.relative_to(output_dir).as_posix(),
                        "pixel size": pixel_size,
                        "alt": caption["caption label"] if caption else portrait or f"Image p.{page_number}-{figure_index}",
                        "asset kind": "table" if caption and caption["caption label"].startswith("Table ") else "portrait" if portrait else "figure",
                        **({"caption label": caption["caption label"]} if caption else {}),
                        **({"source listing": True} if caption is not None and id(caption) in listing_captions else {}),
                    })
            finally:
                page.close(); graphics_page.close()
    _replace_figure_text(document, figures, heights)
    return document
