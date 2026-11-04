"""Recover raster and vector figures without OCR or a remote layout service."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter
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


def _connected_regions(mask: Image.Image, gap: int = 7) -> list[Box]:
    """Find components in a graphics mask, joining nearby drawing fragments."""
    expanded = mask.filter(ImageFilter.MaxFilter(gap * 2 + 1))
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
            if w >= 40 and h >= 18 and tile.histogram()[255] >= 30:
                regions.append(candidate)
        tile.close()
    expanded.close()
    return regions


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
                     if _FIGURE_CAPTION.match(str(node.get("content", "")).strip())
                     and (box := _node_box(node, height))]
    body_boxes = [box for node in nodes
                  if len(str(node.get("content", "")).split()) > 25
                  and not _FIGURE_CAPTION.match(str(node.get("content", "")).strip())
                  and (box := _node_box(node, height))]
    for node in nodes:
        text = str(node.get("content", "")).strip()
        box = _node_box(node, height)
        if not text or not box:
            continue
        vertical = box[3] - box[1] > (box[2] - box[0]) * 2.5 and box[2] - box[0] <= 20
        if (len(text) > 120 or _FIGURE_CAPTION.match(text)
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
        if (gap_x <= 8 and gap_y <= 13) or (vertical and gap_x <= 35 and gap_y == 0):
            result = _union(result, box)
    bottom = min(height, result[3] + 4)
    for caption in caption_boxes:
        if region[3] <= caption[1] and caption[1] < bottom:
            bottom = max(region[3], caption[1] - 2)
    return (max(0, result[0] - 4), max(0, result[1] - 4), min(width, result[2] + 4), bottom)


def _text_labels(page) -> list[dict]:
    # Native text-object bounds also retain rotated y-axis labels that layout
    # extraction may split or assign an incorrect bounding box.
    textpage = page.get_textpage()
    labels: list[dict] = []
    try:
        for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_TEXT], textpage=textpage):
            text = obj.extract().strip()
            if text and len(text) <= 120:
                labels.append({"content": text, "bounding box": list(obj.get_bounds())})
    finally:
        textpage.close()
    return labels


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


def _replace_figure_text(document: dict, figures: list[dict], heights: dict[int, float]) -> None:
    emitted: set[str] = set()
    by_page: dict[int, list[dict]] = defaultdict(list)
    for figure in figures:
        by_page[figure["page number"]].append(figure)

    def rewrite(node: dict, inherited_page: int = 1) -> list[dict]:
        page = node.get("page number", inherited_page)
        box = _node_box(node, heights.get(page, 0))
        text = str(node.get("content", "")).strip()
        if box and _area(box) and not _FIGURE_CAPTION.match(text):
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
        for index in range(len(original)):
            page_number = index + 1
            page, graphics_page = original[index], graphics[index]
            try:
                width, height = page.get_size()
                heights[page_number] = height
                bitmap = graphics_page.render(scale=1, grayscale=True, draw_annots=False)
                image = bitmap.to_pil()
                mask = image.point(lambda value: 255 if value < 235 else 0).convert("L")
                regions = _connected_regions(mask)
                mask.close(); image.close(); bitmap.close()
                nodes = page_nodes.get(page_number, [])
                # Small raster images can fall below the graphics-component threshold.
                for node in nodes:
                    if node.get("type") in {"image", "picture"}:
                        box = _node_box(node, height)
                        if box and _area(box) > 100:
                            clipped = (max(0, box[0]), max(0, box[1]),
                                       min(width, box[2]), min(height, box[3]))
                            if _area(clipped) and not any(
                                _intersection(clipped, r) / _area(clipped) >= 0.8 for r in regions
                            ):
                                regions.append(clipped)
                regions = [r for r in regions if not _is_table(r, nodes, height)]
                labels = nodes + _text_labels(page)
                regions = [_include_labels(r, labels, width, height) for r in regions]
                regions = _merge_regions(regions)
                for figure_index, region in enumerate(sorted(regions, key=lambda b: (b[1], b[0])), 1):
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"page-{page_number:03d}-fig-{figure_index:02d}.png"
                    destination = asset_dir / filename
                    left, top, right, bottom = region
                    crop = (left, max(0, height - bottom), max(0, width - right), top)
                    bitmap = page.render(scale=2.5, crop=crop, draw_annots=False)
                    image = bitmap.to_pil()
                    image.save(destination)
                    image.close(); bitmap.close()
                    figures.append({
                        "type": "image", "page number": page_number,
                        "bounding box": [left, height - bottom, right, height - top],
                        "source": destination.relative_to(output_dir).as_posix(),
                        "alt": f"그림 p.{page_number}-{figure_index}",
                    })
            finally:
                page.close(); graphics_page.close()
    _replace_figure_text(document, figures, heights)
    return document
