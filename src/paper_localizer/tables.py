"""Validate table transcriptions as data; never trust model-generated HTML."""

from __future__ import annotations

from collections import Counter
from html import escape
from html.parser import HTMLParser
import re
import unicodedata


class TableError(ValueError):
    pass


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.opened = False
        self.finished = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.opened or self.finished: raise TableError("Nested or multiple tables are unsupported.")
            self.opened = True
        elif tag in {"thead", "tbody", "tfoot"}:
            if not self.opened: raise TableError("Table section outside a table.")
        elif tag == "tr":
            if not self.opened or self.row is not None: raise TableError("Invalid table row.")
            self.row = []
        elif tag in {"td", "th"}:
            if self.row is None or self.cell is not None: raise TableError("Invalid table cell.")
            values = dict(attrs)
            try:
                rowspan, colspan = int(values.get("rowspan", "1")), int(values.get("colspan", "1"))
            except (TypeError, ValueError) as exc:
                raise TableError("Invalid table span.") from exc
            if not 1 <= rowspan <= 256 or not 1 <= colspan <= 64: raise TableError("Table span is out of range.")
            self.cell = {"text": "", "rowspan": rowspan, "colspan": colspan, "header": tag == "th"}
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"
        else:
            raise TableError("Unsupported markup in table transcription.")

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            if self.cell is None or self.row is None: raise TableError("Unmatched table cell.")
            self.cell["text"] = self.cell["text"].strip()
            self.row.append(self.cell); self.cell = None
        elif tag == "tr":
            if self.row is None or self.cell is not None: raise TableError("Unclosed table cell.")
            self.rows.append(self.row); self.row = None
        elif tag == "table":
            if not self.opened or self.row is not None: raise TableError("Unclosed table row.")
            self.opened = False; self.finished = True
        elif tag not in {"thead", "tbody", "tfoot", "br"}:
            raise TableError("Unsupported markup in table transcription.")

    def handle_data(self, text):
        if self.cell is not None:
            self.cell["text"] += text
        elif text.strip():
            raise TableError("Unexpected text outside table cells.")

    def handle_comment(self, text):
        raise TableError("Unexpected comment in table transcription.")


def parse_table(text: str) -> list[list[dict]]:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:html)?\s*\n(.*)\n```", text, re.S)
        if not match: raise TableError("Invalid table code fence.")
        text = match.group(1)
    if len(text) > 200_000: raise TableError("Table transcription is too large.")
    parser = _TableParser(); parser.feed(text); parser.close()
    if not parser.finished or parser.opened or parser.row is not None or parser.cell is not None:
        raise TableError("Incomplete table transcription.")
    grid = table_grid(parser.rows)
    if len(grid) < 2 or len(grid[0]) < 2: raise TableError("A table needs at least two rows and columns.")
    return parser.rows


def table_grid(rows: list[list[dict]]) -> tuple[tuple[str, ...], ...]:
    if not rows or len(rows) > 256: raise TableError("Invalid table row count.")
    grid = {}
    for row_number, row in enumerate(rows):
        column = 0
        for cell in row:
            while (row_number, column) in grid: column += 1
            for y in range(row_number, row_number + cell["rowspan"]):
                for x in range(column, column + cell["colspan"]):
                    if y >= len(rows) or x >= 64 or (y, x) in grid:
                        raise TableError("Overlapping or out-of-range table spans.")
                    grid[y, x] = cell["text"]
            column += cell["colspan"]
    columns = max((x for _, x in grid), default=-1) + 1
    if any((y, x) not in grid for y in range(len(rows)) for x in range(columns)):
        raise TableError("Table rows have inconsistent widths.")
    return tuple(tuple(grid[y, x] for x in range(columns)) for y in range(len(rows)))


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    if "$" not in text and re.match(r"^[A-Za-z]{3,}\b", text):
        text = text[0].lower() + text[1:]
    return re.sub(r"\s+", "", text)


def tables_agree(first: list[list[dict]], second: list[list[dict]]) -> bool:
    a, b = table_grid(first), table_grid(second)
    return len(a) == len(b) and len(a[0]) == len(b[0]) and all(
        _key(x) == _key(y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def _numbers(text: str) -> Counter:
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    return Counter(re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", text))


def verify_table_text(rows: list[list[dict]], ocr_lines: list[dict]) -> dict:
    """Require independent word support and every plain-text numeric value."""
    expected = " ".join(cell["text"] for row in rows for cell in row)
    expected = re.sub(r"\\[A-Za-z]+", "", expected)
    lines = [line for line in ocr_lines if line.get("score", 0) >= .85]
    observed = " ".join(line["text"] for line in lines)
    joined_words = []
    for first, second in zip(lines, lines[1:]):
        if first.get("column") is None or first.get("column") != second.get("column"): continue
        prefix = re.search(r"([A-Za-z]+)[-–]\s*$", first["text"])
        suffix = re.match(r"\s*([A-Za-z]+)", second["text"])
        if prefix and suffix:
            joined_words.append(prefix.group(1) + suffix.group(1))
    # Long words are robust against line wrapping and small punctuation changes.
    observed_words = re.findall(r"[a-z]+", observed.casefold()) + [word.casefold() for word in joined_words]
    words = set(re.findall(r"[A-Za-z]{5,}", expected.casefold()))
    supported = {word for word in words if any(word in observed for observed in observed_words)}
    support = len(supported) / max(len(words), 1)
    # Formula cells retain their source images; OCR math needs separate checks.
    plain_text = " ".join(cell["text"] for row in rows for cell in row if "$" not in cell["text"])
    missing_numbers = _numbers(plain_text) - _numbers(observed)
    return {"accepted": support >= .98 and not missing_numbers,
            "word_support": round(support, 4), "unsupported_words": sorted(words-supported),
            "missing_numbers": dict(missing_numbers), "numeric_values_checked": sum(_numbers(plain_text).values())}


def table_column_bounds(rows: list[list[dict]], lines: list[dict], image) -> list[tuple[int, int]]:
    """Find column cuts from independently recognized headers and whitespace."""
    import numpy as np

    grid = table_grid(rows)
    columns = len(grid[0])
    header = next((row for row in grid[:3] if len(set(row)) == columns), None)
    if header is None: return []
    width, height = image.size
    boxes = []
    letters = lambda value: re.sub(r"[^a-z0-9]", "", value.casefold())
    for title in header:
        key = letters(title)
        matches = []
        for line in lines:
            text = letters(line["text"])
            if len(text) < min(4, len(key)) or text not in key or line.get("score", 0) < .85: continue
            box = line["box"]
            if max(point[1] for point in box) > height * .3: continue
            matches.append(((min(p[0] for p in box), min(p[1] for p in box),
                             max(p[0] for p in box), max(p[1] for p in box)), len(text)))
        if not matches:
            boxes = []
            break
        primary = max(matches, key=lambda pair: pair[1])[0]
        local = [box for box, _ in matches if abs((box[0]+box[2]-primary[0]-primary[2])/2)
                 < max(primary[2]-primary[0], width/columns*.5)]
        boxes.append((min(b[0] for b in local), max(b[2] for b in local)))
    if any(left[1] >= right[0] for left, right in zip(boxes, boxes[1:])):
        boxes = []
    gray = image.convert("L")
    ink = np.asarray(gray) < 180
    gray.close()
    dense = np.flatnonzero(ink.sum(axis=1) > width * .5)
    for row in dense:
        ink[max(0, row-2):min(height, row+3), :] = False
    projection = ink.sum(axis=0)
    cuts = [0]
    for index in range(1, columns):
        if boxes:
            left, right = boxes[index-1:index+1]
            low, high = max(cuts[-1]+1, int(left[1])-3), min(width-1, int(right[0])+3)
            middle = (left[1]+right[0])/2
        else:
            middle = width * index / columns
            low, high = max(cuts[-1]+int(width*.04), int(middle-width*.1)), min(width-1, int(middle+width*.1))
        if low >= high: return []
        minimum = projection[low:high+1].min()
        choices = np.flatnonzero(projection[low:high+1] == minimum) + low
        cut = min(choices, key=lambda value: abs(value-middle))
        cuts.append(int(cut))
    cuts.append(width)
    return list(zip(cuts, cuts[1:]))


def table_html(rows: list[list[dict]], render_cell=escape) -> str:
    table_grid(rows)
    parts = ['<table class="transcribed-table"><tbody>']
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            tag = "th" if cell.get("header") else "td"
            spans = "".join(f' {key}="{cell[key]}"' for key in ("rowspan", "colspan") if cell[key] > 1)
            parts.append(f'<{tag}{spans}>' + render_cell(cell["text"]) + f'</{tag}>')
        parts.append("</tr>")
    return "".join(parts) + "</tbody></table>"
