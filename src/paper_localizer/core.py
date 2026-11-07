"""OpenDataLoader extraction and local Ollama processing."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from collections import Counter
from pathlib import Path
from typing import Callable

from .figures import attach_figures
from .formulas import recover_formulas
from .bundle import FIGURE_LINK
from .article import build_article, article_markdown, write_article
from .layout import bind_captions, prepare_layout
from .ocr import OcrOptions, apply_ocr, preserve_ocr_regions
from .structure import order_document, repair_structure
from .source_links import read_source_links


TRANSLATION_MODEL = "kaelri/hy-mt2:7b-q4_K_M"
OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


class ProcessingError(RuntimeError):
    """A user-facing failure with a concrete next step."""


class TranslationValidationError(ProcessingError):
    """A candidate cannot safely replace the original text."""


class GenerationLimitError(ProcessingError):
    """The model stopped before completing its response."""


@dataclass(frozen=True)
class Block:
    page: int
    kind: str
    text: str
    caption_target: str = ""
    source_pages: tuple[int, ...] = ()
    paragraph_id: str = ""
    table_data: dict | None = None


@dataclass(frozen=True)
class Chunk:
    index: int
    pages: tuple[int, ...]
    source: str
    kind: str = "text"
    caption_target: str = ""
    paragraph_ids: tuple[str, ...] = ()
    source_parts: tuple[str, ...] = ()
    table_data: dict | None = None
    reference_section: bool = False


def _descendant_text(node: dict) -> str:
    if node.get('suppressed'):
        return ''
    if snapshot := node.get('formula snapshot'):
        return snapshot
    pieces: list[str] = []
    content = node.get("content")
    if isinstance(content, str) and content.strip():
        pieces.append(content.strip())
    for key in ("kids", "list items"):
        for child in node.get(key, []):
            if isinstance(child, dict):
                child_text = _descendant_text(child)
                if child_text:
                    pieces.append(child_text)
    return " ".join(pieces)


def _table_markdown(node: dict) -> str:
    from .tables import TableError, table_grid
    native = _native_table_rows(node)
    try:
        grid = table_grid(native)
    except TableError:
        grid = ()
    if grid:
        header_rows = max((i + cell["rowspan"] for i, row in enumerate(native[:3])
                           for cell in row if cell["header"]), default=1)
        header_rows = min(header_rows, len(grid) - 1) if len(grid) > 1 else 1
        headers = []
        for column in range(len(grid[0])):
            labels = list(dict.fromkeys(grid[row][column] for row in range(header_rows) if grid[row][column]))
            headers.append(" / ".join(labels))
        rows = [headers] + [list(row) for row in grid[header_rows:]]
        escaped = [[re.sub(r"\s+", " ", cell).replace("|", "\\|") for cell in row] for row in rows]
        lines = ["| " + " | ".join(row) + " |" for row in escaped]
        lines.insert(1, "| " + " | ".join(["---"] * len(headers)) + " |")
        return "\n".join(lines)
    rows: list[list[str]] = []
    for row in node.get("rows", []):
        cells = sorted(row.get("cells", []), key=lambda c: c.get("column number", 0))
        values = [re.sub(r"\s+", " ", _descendant_text(cell)).replace("|", "\\|") for cell in cells]
        if values:
            rows.append(values)
    if not rows:
        return ""
    width = max(map(len, rows))
    lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _native_table_rows(node: dict) -> list[list[dict]]:
    rows = []
    for row in sorted(node.get("rows", []), key=lambda value: value.get("row number", 0)):
        rows.append([{"text": _descendant_text(cell),
                      "rowspan": max(1, int(cell.get("row span", 1))),
                      "colspan": max(1, int(cell.get("column span", 1))),
                      "header": bool(cell.get("is_header") or cell.get("pdfua_tag") == "TH")}
                     for cell in sorted(row.get("cells", []), key=lambda value: value.get("column number", 0))])
    return rows


def extract_blocks(document: dict) -> list[Block]:
    """Flatten reading-order JSON while retaining page references and tables."""
    blocks: list[Block] = []
    footnotes: list[Block] = []
    front_matter: list[Block] = []

    def visit(node: dict, inherited_page: int = 1) -> None:
        kind = str(node.get("type", "")).lower()
        page = node.get("page number", inherited_page)
        page = page if isinstance(page, int) and page > 0 else inherited_page
        if kind in {"header", "footer"} or node.get("suppressed"):
            return
        if node.get("attached caption"):
            return
        if kind == "footnote":
            if content := node.get('formula snapshot') or node.get("content"):
                footnotes.append(Block(page, "formula" if node.get('formula snapshot') else "paragraph", content))
            return
        if kind == "frontmatter":
            if content := node.get('formula snapshot') or node.get("content"):
                front_matter.append(Block(page, kind, content))
            return
        if kind == "keywords":
            if content := node.get("content"):
                blocks.append(Block(page, kind, content))
            return
        if kind in {"image", "picture"}:
            source = node.get("source")
            if isinstance(source, str) and source.startswith("figs/"):
                alt = str(node.get("alt", f"Fig. p.{page}")).replace("[", "").replace("]", "")
                kind = {"table": "table-image", "portrait": "portrait"}.get(node.get("asset kind"), "image")
                blocks.append(Block(page, kind, f"![{alt}]({source})"))
                if caption := node.get("caption node"):
                    value = caption.get("formula snapshot") or caption.get("content", "")
                    if value:
                        blocks.append(Block(page, "caption-image" if caption.get("formula snapshot") else "caption", value, source))
            return
        if kind == "table":
            table = _table_markdown(node)
            if table:
                from .tables import TableError, table_grid
                rows = _native_table_rows(node)
                try:
                    table_grid(rows)
                    data = {"rows": rows}
                except TableError:
                    data = None
                blocks.append(Block(page, kind, table, table_data=data))
            return
        if snapshot := node.get("formula snapshot"):
            snapshot_kind = {"caption": "caption-image", "abstract": "abstract-image"}.get(kind, "formula")
            blocks.append(Block(page, snapshot_kind, snapshot))
            return
        content = node.get("content")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if node.get("formula image"):
                blocks.append(Block(page, "formula", text))
                return
            if kind == "heading":
                level = node.get("heading level", 2)
                level = min(max(level, 1), 6) if isinstance(level, int) else 2
                if split := node.get("heading split"):
                    blocks.append(Block(page, kind, f"{'#' * level} {text[:split].strip()}"))
                    blocks.append(Block(page, "paragraph", text[split:].strip()))
                    return
                text = f"{'#' * level} {text}"
            elif kind == "list item" or re.match(r"^[•■▪‣◦]\s+", text):
                if not re.match(r"^(?:\d+[.)]\s+|\[\d+\]\s+)", text):
                    text = "- " + re.sub(r"^[-•■▪‣◦]\s*", "", text)
            blocks.append(Block(page, kind, text, paragraph_id=node.get("paragraph id", "")))
        for key in ("kids", "list items"):
            for child in node.get(key, []):
                if isinstance(child, dict):
                    visit(child, page)

    for element in document.get("kids", []):
        if isinstance(element, dict):
            visit(element)
    continued: list[Block] = []
    for block in blocks:
        anchor = len(continued) - 1
        while anchor >= 0 and continued[anchor].kind in {'image', 'caption'}:
            anchor -= 1
        previous = continued[anchor] if anchor >= 0 else None
        previous_pages = (previous.source_pages or (previous.page,)) if previous else ()
        floats = continued[anchor + 1:]
        # A figure-only page may intervene in the middle of a sentence. Finish
        # that sentence before displaying its floats instead of asking the
        # translation model to invent the missing continuation.
        gap = block.page - previous_pages[-1] if previous else 0
        intervening_pages = {item.page for item in floats}
        adjacent = gap == 1 or (gap == 2 and previous_pages[-1] + 1 in intervening_pages)
        if (previous and previous.kind == block.kind == "paragraph"
                and adjacent and all(previous_pages[-1] <= item.page <= block.page for item in floats)
                and len(previous.text) > 50 and re.search(r"[A-Za-z,]$", previous.text)
                and re.match(r"^[a-z]", block.text)
                and not FIGURE_LINK.search(previous.text + block.text)):
            continued[anchor] = Block(previous.page, "paragraph", previous.text + " " + block.text,
                                      source_pages=previous_pages + (block.page,), paragraph_id=previous.paragraph_id)
        else:
            continued.append(block)
    blocks = front_matter + continued
    if footnotes:
        blocks.append(Block(footnotes[0].page, "heading", "## Notes"))
        blocks.extend(footnotes)
    return blocks


def _split_text(text: str, limit: int) -> list[str]:
    from .math_markup import LATEX
    opaque: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        token = f'[[CHUNK_{len(opaque) + 1:04d}]]'
        opaque[token] = match[0]
        return token

    protected = LATEX.sub(protect, FIGURE_LINK.sub(protect, text))
    if len(protected) <= limit:
        return [text]
    parts: list[str] = []
    for piece, separator in _translation_segments(protected, limit):
        for token, original in opaque.items():
            piece = piece.replace(token, original)
        if piece.strip():
            parts.append(piece.strip())
    return parts


def make_chunks(blocks: list[Block], limit: int = 1800) -> list[Chunk]:
    from .math_markup import LATEX
    if limit < 200:
        raise ValueError("limit must be at least 200 characters")
    chunks: list[Chunk] = []
    lines: list[str] = []
    paragraph_ids: list[str] = []
    pages: set[int] = set()
    size = 0

    def flush() -> None:
        nonlocal lines, paragraph_ids, pages, size
        if lines:
            chunks.append(Chunk(len(chunks) + 1, tuple(sorted(pages)), "\n\n".join(lines),
                                paragraph_ids=tuple(paragraph_ids), source_parts=tuple(lines)))
        lines, paragraph_ids, pages, size = [], [], set(), 0

    for block_number, block in enumerate(blocks, 1):
        paragraph_id = block.paragraph_id or f"block-{block_number}"
        if block.kind == "heading" or re.match(r"^(?:Abstract|초록)\s*[.:：]", block.text, re.I):
            flush()
        if block.kind in {"image", "table", "table-image", "portrait", "formula", "caption", "caption-image", "code", "frontmatter", "keywords", "abstract", "abstract-image"}:
            flush()
            kind = "figure" if block.kind == "image" else block.kind
            chunks.append(Chunk(len(chunks) + 1, (block.page,), block.text, kind, block.caption_target,
                                table_data=block.table_data))
            continue
        if len(block.source_pages) > 1:
            flush()
            for part in _split_text(block.text, limit):
                chunks.append(Chunk(len(chunks)+1, block.source_pages, part,
                                    paragraph_ids=(paragraph_id,), source_parts=(part,)))
            continue
        for part in _split_text(block.text, limit):
            part_size = len(LATEX.sub("[[MATH_0001]]", FIGURE_LINK.sub("[[MATH_0001]]", part)))
            # A paragraph's displayed provenance must not inherit the pages of
            # unrelated paragraphs that happened to share its model chunk.
            if pages and block.page not in pages:
                flush()
            if size + part_size + 2 > limit:
                flush()
            lines.append(part)
            paragraph_ids.append(paragraph_id)
            pages.add(block.page)
            size += part_size + 2
    flush()
    reference_level: int | None = None
    for index, chunk in enumerate(chunks):
        if heading := re.match(r'^(#{1,6})\s+([^\n]+)', chunk.source):
            level = len(heading[1])
            label = re.sub(r'^(?:\d+(?:\.\d+)*[.)]?|[IVXLCDM]+[.)])\s+', '', heading[2], flags=re.I)
            if re.fullmatch(r'(?:references?|bibliography|works cited|literature cited|참고문헌|参考文献|參考文獻)\s*[:.]?', label, re.I):
                reference_level = level
            elif reference_level is not None and level <= reference_level:
                reference_level = None
        if reference_level is not None:
            chunks[index] = replace(chunk, reference_section=True)
    return chunks


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._translation_formats: dict[str, str] = {}

    def _translation_format(self, model: str) -> str:
        if model not in self._translation_formats:
            mode = 'chat'
            if 'hy-mt2' in model.casefold():
                info = self._request('/api/show', {'model': model, 'verbose': True}).get('model_info', {})
                if info.get('general.architecture') == 'hunyuan-dense':
                    tokens = set(info.get('tokenizer.ggml.tokens', []))
                    if not {'<|startoftext|>', '<|extra_0|>', '<|eos|>'} <= tokens:
                        raise ProcessingError('HY-MT2 native prompt tokens could not be verified from the installed model.')
                    mode = 'hy-mt2-dense'
            self._translation_formats[model] = mode
        return self._translation_formats[model]

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProcessingError(f"Cannot connect to Ollama ({self.base_url}): {exc}") from exc

    def check_models(self, *models: str) -> None:
        available = self._request("/api/tags").get("models", [])
        installed = {item.get("name") for item in available}
        missing = [model for model in models if model not in installed]
        if missing:
            commands = "\n".join(f"ollama pull {model}" for model in missing)
            raise ProcessingError(f"Required models are not installed. Run the following commands:\n{commands}")
        self.model_digests = {item['name']: item.get('digest', '') for item in available
                              if item.get('name') in models}

    def chat(self, model: str, prompt: str, *, translation: bool = False) -> str:
        options = (
            {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05, "num_predict": 4096}
            if translation else {"temperature": 0.2, "num_predict": 1200}
        )
        options['num_ctx'] = 8192
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": options,
        }
        native = translation and self._translation_format(model) == 'hy-mt2-dense'
        if native:
            # Some distributed MT2 tags carry the MT1.x template. Bypass it
            # only after verifying the actual dense MT2 tokenizer vocabulary.
            payload.pop('messages'); payload.pop('think')
            payload.update(raw=True, prompt='<|startoftext|>' + prompt + '<|extra_0|>')
        response = self._request('/api/generate' if native else '/api/chat', payload)
        if response.get('done_reason') == 'length':
            raise GenerationLimitError(f"{model} reached its generation limit; incomplete output was not accepted. Reduce the chunk size and retry.")
        answer = (response.get('response', '') if native else response.get("message", {}).get("content", ""))
        if not isinstance(answer, str):
            raise ProcessingError(f"{model} returned an invalid response.")
        answer = answer.strip()
        if not answer:
            raise ProcessingError(f"{model} returned an empty response.")
        return answer


def prepare_document(pdf_path: Path, document: dict, output_dir: Path, *, ocr: OcrOptions | None = None) -> dict:
    document["source links"] = read_source_links(pdf_path)
    document = apply_ocr(pdf_path, document, output_dir, ocr)
    document = repair_structure(pdf_path, document)
    document = prepare_layout(document)
    document = preserve_ocr_regions(pdf_path, document, output_dir)
    document = attach_figures(pdf_path, document, output_dir)
    document = recover_formulas(pdf_path, document, output_dir)
    from .table_evidence import supplement_source_tables
    document = supplement_source_tables(pdf_path, document)
    if ocr and ocr.vision_model:
        from .vision import refine_document
        document = refine_document(pdf_path, document, output_dir,
                                  model=ocr.vision_model, url=ocr.vision_url, expected_digest=ocr.vision_digest)
    return order_document(bind_captions(document))


def parse_pdf(pdf_path: Path, output_dir: Path | None = None, *, ocr: OcrOptions | None = None) -> dict:
    try:
        import opendataloader_pdf
    except ImportError as exc:
        raise ProcessingError("opendataloader-pdf is not installed. Run `pip install -e .`.") from exc
    with tempfile.TemporaryDirectory(prefix="paper-localizer-") as directory:
        root = Path(directory)
        source = root / "paper.pdf"
        shutil.copyfile(pdf_path, source)
        output = root / "parsed"
        output.mkdir()
        try:
            opendataloader_pdf.convert(
                input_path=str(source), output_dir=str(output), format="json", quiet=True,
                image_output="external", image_format="png",
            )
        except Exception as exc:
            raise ProcessingError(f"OpenDataLoader conversion failed: {exc}") from exc
        result = output / "paper.json"
        if not result.exists():
            raise ProcessingError("OpenDataLoader did not produce a JSON file. Check the input PDF.")
        with result.open(encoding="utf-8") as file:
            document = json.load(file)
        try:
            destination = output_dir or Path("outputs")
            return prepare_document(pdf_path, document, destination, ocr=ocr)
        except Exception as exc:
            raise ProcessingError(f"PDF/OCR, figure, or equation processing failed: {exc}") from exc


def _translation_prompt(source: str, target_language: str, context: str = '', requirements: str = '',
                        terms: tuple[tuple[str, str], ...] = ()) -> str:
    background = ("Research context for terminology only; do not translate this context:\n" + context + "\n\n") if context else ''
    glossary = ('Reference the following translations:\n' +
                '\n'.join(f'{source_term} translates to {target_term}' for source_term, target_term in terms) + '\n\n') if terms else ''
    return background + glossary + (
        f"Translate the following text into {target_language}. Note that you should only output the "
        "translated result without any additional explanation. Preserve Markdown headings, lists, "
        "table structure, citations, numbers, equations, and reference identifiers exactly. "
        "If equation placeholders are present, keep each one exactly once and unchanged in its original position. "
        + (requirements + " " if requirements else "")
        + "Do not add or omit information:\n\n" + source
    )


def _translation_math_regions(source: str) -> list[tuple[int, int, str | None]]:
    """Keep split tuple delimiters and their variable labels as one opaque unit."""
    from .math_markup import LATEX, expression
    matches = list(LATEX.finditer(source))
    paired = {r'\langle': r'\rangle', r'\{': r'\}', r'\lvert': r'\rvert'}
    regions = []; index = 0
    while index < len(matches):
        match = matches[index]
        meaning = expression(match)
        if index+1 < len(matches) and meaning.strip() in paired:
            end = matches[index+1]
            inside = source[match.end():end.start()]
            if (expression(end).strip() == paired[meaning.strip()] and '\n' not in inside
                    and re.fullmatch(r'[\w ,;:+-]{1,64}', inside)):
                regions.append((match.start(), end.end(), None))
                index += 2
                continue
        regions.append((match.start(), match.end(), meaning))
        index += 1
    return regions


def _literal_text(text: str) -> str:
    from .math_markup import LATEX
    return re.sub(r'\[\[MATH_\d{4}\]\]', ' ', LATEX.sub(' ', FIGURE_LINK.sub(' ', text)))


def _numeric_literals(text: str) -> Counter:
    text = re.sub(r'(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)', lambda m: m[0].replace(',', ''), _literal_text(text))
    return Counter(re.findall(r'\d+(?:\.\d+)*', text))


def _citation_identifiers(text: str) -> Counter:
    return Counter(re.sub(r'\s+', '', m[0]).replace('–', '-').replace('−', '-')
                   for m in re.finditer(r'\[\d+(?:\s*[,;–−-]\s*\d+)*\]', _literal_text(text)))


def _sentence_boundaries(text: str) -> list[tuple[int, int]]:
    boundaries = []
    for match in re.finditer(r'[.!?]["”’\)\]]*\s+(?=[A-Z“"\(\[])', text):
        whitespace = re.search(r'\s+$', match[0])
        start = match.start() + whitespace.start()
        prefix = text[:start]
        if (prefix.endswith('..') or prefix.count('“') > prefix.count('”')
                or prefix.count('"') % 2 or prefix.count('(') > prefix.count(')')
                or re.search(r'(?:^|\n)\s*\d+\.$', prefix)):
            continue
        if re.search(r'\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Dr|Prof|Mr|Ms|vs|cf|approx|No|pp)\.$', prefix, re.I):
            continue
        if re.search(r'(?:^|\s)[A-Z]\.$', prefix):
            continue
        boundaries.append((start, match.end()))
    return boundaries


def _translation_segments(text: str, limit: int = 700) -> list[tuple[str, str]]:
    """Split protected text at paragraphs/sentences, never inside a clause."""
    if len(text) <= limit:
        return [(text, '')]
    result: list[tuple[str, str]] = []
    start = 0
    paragraphs = list(re.finditer(r'\n[ \t]*\n(?:[ \t]*\n)*', text))
    for paragraph_end, next_start in [(m.start(), m.end()) for m in paragraphs] + [(len(text), len(text))]:
        paragraph = text[start:paragraph_end]
        boundaries = _sentence_boundaries(paragraph)
        position = 0
        while len(paragraph) - position > limit:
            before = [b for b in boundaries if position < b[0] <= position + limit]
            after = [b for b in boundaries if b[0] > position + limit]
            boundary = before[-1] if before else after[0] if after else None
            if boundary is None:
                break
            result.append((paragraph[position:boundary[0]], paragraph[boundary[0]:boundary[1]]))
            position = boundary[1]
        if paragraph[position:]:
            result.append((paragraph[position:], text[paragraph_end:next_start]))
        elif result:
            piece, separator = result[-1]
            result[-1] = (piece, separator + text[paragraph_end:next_start])
        elif next_start:
            result.append(('', text[paragraph_end:next_start]))
        start = next_start
    return result


def _translate_chunk(client: OllamaClient, model: str, source: str, language: str,
                     equation_text: dict[str, str] | None = None, *, context: str = '') -> str:
    equations: dict[str, str] = {}
    equation_context: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        token = f"[[MATH_{len(equations) + 1:04d}]]"
        equations[token] = match.group(0)
        from .markup import image_reference
        if latex := (equation_text or {}).get(image_reference(match).source):
            equation_context[token] = latex
        return token

    protected = FIGURE_LINK.sub(protect, source)
    replacements = []
    for start, end, meaning in _translation_math_regions(protected):
        token = f"[[MATH_{len(equations) + 1:04d}]]"
        equations[token] = protected[start:end]
        if meaning is not None:
            equation_context[token] = meaning
        replacements.append((start, end, token))
    for start, end, token in reversed(replacements):
        protected = protected[:start] + token + protected[end:]
    dedicated = 'hy-mt2' in model.casefold()
    from .terminology import translation_terms

    def translate_piece(piece: str, *, allow_split: bool = True) -> str:
        expected = {token for token in equations if token in piece}
        hints = [context] if context and not dedicated else []
        if expected and not dedicated:
            hints.append("Required unchanged placeholders: " + ", ".join(sorted(expected)))
        meanings = {token: value for token, value in equation_context.items() if token in expected}
        if meanings and not dedicated:
            hints.append("Local OCR equation context (keep placeholders unchanged):\n" + json.dumps(meanings, ensure_ascii=False))
        terms = translation_terms(piece, context, language) if dedicated else ()
        for attempt in range(2):
            requirements = ('Do not infer missing text or equation values. Keep incomplete sentences and terminal punctuation incomplete. '
                            'Preserve every equation placeholder exactly once.' if dedicated and attempt else '')
            prompt = _translation_prompt(piece, language, '' if dedicated else '\n\n'.join(hints), requirements, terms)
            translated = client.chat(model, prompt, translation=True)
            found = re.findall(r"\[\[MATH_\d{4}\]\]", translated)
            if ('\ufffd' not in translated and all(translated.count(token) == 1 for token in expected)
                    and set(found) == expected
                    and not _numeric_literals(piece) - _numeric_literals(translated)
                    and _citation_identifiers(piece) == _citation_identifiers(translated)
                    and not (terms and re.search(r'Reference the following translations:|\btranslates to\b', translated)
                             and not re.search(r'Reference the following translations:|\btranslates to\b', piece))):
                return translated
            hints.append("Required unchanged placeholders: " + ", ".join(sorted(expected)))
            if '\ufffd' in translated:
                hints.append('Return valid Unicode text without replacement characters.')
        if allow_split:
            sentences = _translation_segments(piece, limit=1)
            if len(sentences) > 1:
                return ''.join((translate_piece(sentence, allow_split=False) if sentence.strip() else sentence) + separator
                               for sentence, separator in sentences)
        raise TranslationValidationError("The translation model returned damaged characters, omitted numeric literals, or altered citation/equation markers. Reduce the chunk size and retry.")

    translated = ''.join((translate_piece(piece) if piece.strip() else piece) + separator
                         for piece, separator in _translation_segments(protected))
    for token, image in equations.items():
        translated = translated.replace(token, image)
    return translated


def _translate_native_table(client: OllamaClient, model: str, rows: list[list[dict]], language: str,
                            equation_text: dict[str, str] | None = None, *, context: str = '',
                            cache: dict[str, str] | None = None, failures: list[dict] | None = None) -> list[list[dict]]:
    """Translate cell values; source geometry and mathematical data stay fixed."""
    from copy import deepcopy
    from .math_markup import LATEX
    from .terminology import TABLE_HEADERS, table_header_translation, translation_terms
    result = deepcopy(rows)
    cache = cache if cache is not None else {}
    header_depth = max((index + cell['rowspan'] for index, row in enumerate(rows[:3])
                        for cell in row if cell['header']), default=1)
    for row_number, row in enumerate(result, 1):
        for cell_number, cell in enumerate(row, 1):
            source = cell['text']
            literal = LATEX.sub('', FIGURE_LINK.sub('', source)).strip()
            if ((row_number <= header_depth or cell['header']) and source.strip() == literal
                    and (header := table_header_translation(literal, language))):
                cell['text'] = header
                continue
            if (not re.search(r'[^\W\d_]{2,}', literal)
                    or ((row_number <= header_depth or cell['header'])
                        and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', literal)
                        and literal.casefold() not in TABLE_HEADERS
                        and not translation_terms(literal, context, language))
                    or re.fullmatch(r'(?:[A-Za-z][A-Za-z0-9]*\([^()]*\)|\([A-Za-z0-9 +*/^-]+\))(?:\^?\d+)?', literal)
                    or re.fullmatch(r'[a-z]{1,4}[A-Z][A-Za-z0-9]*', literal)):
                continue
            if source not in cache:
                try:
                    cache[source] = _translate_chunk(client, model, source, language, equation_text, context=context)
                except (TranslationValidationError, GenerationLimitError) as exc:
                    if failures is None:
                        raise
                    failures.append({'kind': 'table-cell', 'row': row_number, 'cell': cell_number,
                                     'source': source, 'reason': str(exc)})
                    continue
            cell['text'] = cache[source]
    return result


def _native_rows_markdown(rows: list[list[dict]]) -> str:
    node = {'rows': [{'cells': [{'kids': [{'content': cell['text']}],
             'row span': cell['rowspan'], 'column span': cell['colspan'], 'is_header': cell['header']}
            for cell in row]} for row in rows]}
    return _table_markdown(node)


def _document_title(document: dict, fallback: str) -> str:
    title = str(document.get("title") or "").strip()
    if not title or Path(title).suffix.lower() in {".dvi", ".pdf", ".ps"}:
        for block in extract_blocks(document):
            if block.kind == "heading":
                return FIGURE_LINK.sub("", block.text).lstrip("# ").replace("\n", " ").strip()
        title = Path(fallback).stem
    return title.replace("\n", " ").strip()


def _translation_context(document: dict, filename: str) -> str:
    from .math_markup import LATEX
    title = _document_title(document, filename)
    abstract = next((block.text for block in extract_blocks(document) if block.kind == 'abstract'), '')
    abstract = LATEX.sub(' ', FIGURE_LINK.sub(' ', abstract))
    context = 'Paper title: ' + title[:250]
    if re.search(r'peer[ -]to[ -]peer|\bP2P\b|gossipsub|blockchain', title+' '+abstract, re.I):
        context += ('\nDomain: computer networking. Churn refers to network nodes joining and leaving. '
                    'Keep protocol names unchanged.')
    return context


def render_markdown(document: dict, pdf_name: str, chunks: list[Chunk],
                    translations: list[str], language: str) -> str:
    article = build_article(document, _document_title(document, pdf_name), pdf_name,
                            chunks, translations, language)
    return article_markdown(article)


def extract_pdf(pdf_path: Path, output_path: Path, *, chunk_size: int = 1800, ocr: OcrOptions | None = None) -> Path:
    """Write a source-only preview without requiring Ollama."""
    pdf_path, output_path = Path(pdf_path), Path(output_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ProcessingError("Specify an existing PDF file.")
    if output_path.suffix.lower() != ".md":
        raise ProcessingError("The output file must have a .md extension.")
    document = parse_pdf(pdf_path, output_path.parent, ocr=ocr)
    chunks = make_chunks(extract_blocks(document), chunk_size)
    if not chunks:
        raise ProcessingError("No body text was extracted. Scanned PDFs may require OCR.")
    return write_article(document, _document_title(document, pdf_path.name), pdf_path.name,
                         chunks, None, "English", output_path)


def process_pdf(pdf_path: Path, output_path: Path, *,
                translation_model: str = TRANSLATION_MODEL,
                target_language: str = "English",
                chunk_size: int = 1800,
                ollama_url: str = OLLAMA_URL,
                progress: Callable[[str, int, int], None] | None = None,
                parser: Callable[[Path], dict] | None = None,
                client: OllamaClient | None = None,
                ocr: OcrOptions | None = None) -> Path:
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ProcessingError("Specify an existing PDF file.")
    if output_path.suffix.lower() != ".md":
        raise ProcessingError("The output file must have a .md extension.")
    if pdf_path.resolve() == output_path.resolve():
        raise ProcessingError("The output file must differ from the input PDF.")
    with pdf_path.open("rb") as file:
        header = file.read(5)
    if header != b"%PDF-":
        raise ProcessingError("The input file has an invalid PDF header.")
    client = client or OllamaClient(ollama_url)
    client.check_models(translation_model)
    if progress:
        progress("Analyzing PDF", 0, 1)
    document = parser(pdf_path) if parser else parse_pdf(pdf_path, output_path.parent, ocr=ocr)
    blocks = extract_blocks(document)
    if not blocks:
        raise ProcessingError("No body text was extracted. Scanned PDFs may require OCR.")
    chunks = make_chunks(blocks, chunk_size)
    with pdf_path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    identity = {
        "inference_profile": "native-mt2-scoped-terms-sentences-v15",
        "terminology_profile": "scientific-korean-v4",
        "translation_segment_chars": 700,
        "retry_policy": "two-attempts-then-complete-sentences",
        "table_header_policy": "korean-label-lexicon-v1-translate-generic-preserve-identifiers",
        "numeric_policy": "preserve-source-literals",
        "citation_policy": "preserve-identifiers",
        "reference_policy": "preserve-original",
        "reference_chunks": [chunk.index for chunk in chunks if chunk.reference_section],
        "pdf_sha256": fingerprint,
        "translation_model": translation_model,
        "model_digests": getattr(client, 'model_digests', {}),
        "target_language": target_language,
        "chunk_size": chunk_size,
        "sources": [chunk.source for chunk in chunks],
        "kinds": [chunk.kind for chunk in chunks],
        "native_tables": {str(chunk.index): chunk.table_data for chunk in chunks if chunk.table_data},
        "math_transcriptions": document.get("math transcriptions", {}),
        "vision_ocr_identity": {key: value for key, value in document.get("vision ocr", {}).items() if key != "regions"},
    }
    checkpoint = output_path.with_suffix(".progress.json")
    translations: list[str] = []
    translated_tables: dict[str, dict] = {}
    cell_cache: dict[str, str] = {}
    translation_issues: list[dict] = []
    translation_context = _translation_context(document, pdf_path.name)
    if checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("identity") == identity:
                translations = saved.get("translations", [])
                translated_tables = saved.get('translated_tables', {})
                cell_cache = saved.get('table_cell_translations', {})
                translation_issues = saved.get('translation_issues', [])
                if len(translations) > len(chunks):
                    translations = []
                    translated_tables, cell_cache = {}, {}
                    translation_issues = []
        except (OSError, ValueError):
            translations = []
            translated_tables, cell_cache, translation_issues = {}, {}, []
    for index, chunk in enumerate(chunks, start=1):
        if index <= len(translations):
            continue
        if progress:
            progress("Translating", index, len(chunks))
        if chunk.reference_section or chunk.kind in {"figure", "table-image", "portrait", "formula", "caption-image", "abstract-image", "code", "frontmatter"}:
            translations.append(chunk.source)
        else:
            if chunk.kind == 'table' and chunk.table_data:
                failures: list[dict] = []
                rows = _translate_native_table(client, translation_model, chunk.table_data['rows'], target_language,
                        document.get('math transcriptions'), context=translation_context, cache=cell_cache,
                        failures=failures)
                translation_issues.extend(dict(failure, chunk=chunk.index, pages=list(chunk.pages))
                                          for failure in failures)
                translated_tables[str(chunk.index)] = {'rows': rows}
                translations.append(_native_rows_markdown(rows))
            else:
                try:
                    translations.append(_translate_chunk(client, translation_model, chunk.source, target_language,
                                                           document.get("math transcriptions"), context=translation_context))
                except (TranslationValidationError, GenerationLimitError) as exc:
                    translations.append("")
                    translation_issues.append({'kind': 'excerpt', 'chunk': chunk.index,
                                               'pages': list(chunk.pages), 'reason': str(exc)})
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"identity": identity, "translations": translations,
                                          'translated_tables': translated_tables,
                                          'table_cell_translations': cell_cache,
                                          'translation_issues': translation_issues}, ensure_ascii=False), encoding="utf-8")
    if progress:
        progress("Writing HTML and Markdown", len(chunks), len(chunks))
    document['translated native tables'] = translated_tables
    document['translation issues'] = translation_issues
    audit = {key: identity[key] for key in ('inference_profile', 'pdf_sha256', 'translation_model',
                                          'model_digests', 'target_language', 'reference_policy', 'reference_chunks',
                                          'numeric_policy', 'citation_policy', 'terminology_profile', 'translation_segment_chars', 'retry_policy', 'table_header_policy')}
    audit.update({'chunks': len(chunks), 'native_tables': len(translated_tables),
                  'issues': translation_issues,
                  'validation_scope': 'Checks generation completion, Unicode, equation markers, source numeric literals and citation identifiers; does not establish semantic accuracy.'})
    output_path.with_suffix('.translation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    write_article(document, _document_title(document, pdf_path.name), pdf_path.name,
                  chunks, translations, target_language, output_path)
    checkpoint.unlink(missing_ok=True)
    return output_path
