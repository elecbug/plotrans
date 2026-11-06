"""Supplement source table images with independently checked native cells."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import version
import re
import unicodedata

import pdfplumber
import pypdfium2 as pdfium
from pdfminer.psparser import PSException

from .figures import _PDFIUM_LOCK, _area, _intersection
from .formulas import _canonical, _inside, _spans
from .native_math import native_expression
from .tables import TableError, table_grid


def _readable(text):
    return all((char.isprintable() or char.isspace())
               and char != '\ufffd' and unicodedata.category(char) != 'Co' for char in text)


def _checked_rows(table, page, source_box, spans):
    """A layout proposal cannot supply unverified or repeated source text.

    pdfplumber proposes the grid and word spacing. PDFium independently owns
    complete positioned spans and validates text order within every cell.
    Ambiguous merged cells and incomplete glyph encodings retain the image.
    """
    proposed = table.extract(x_tolerance=2, y_tolerance=2)
    if not proposed or not 2 <= len(proposed[0]) <= 12:
        raise TableError('No regular table grid.')
    columns = len(proposed[0])
    retained = [i for i, row in enumerate(proposed) if any(row)]
    if not 2 <= len(retained) <= 100:
        raise TableError('Insufficient nonempty rows.')
    geometry = [table.rows[i].cells for i in retained]
    if (any(len(proposed[index]) != columns for index in retained)
            or any(len(cells) != columns or any(cell is None for cell in cells) for cells in geometry)):
        raise TableError('Merged or inconsistent cells require source review.')
    bounds = [(min(c[1] for c in cells), max(c[3] for c in cells)) for cells in geometry]
    height = page.get_size()[1]
    cuts = [height-source_box[3]]
    cuts += [(upper[1]+lower[0])/2 for upper, lower in zip(bounds, bounds[1:])]
    cuts.append(height-source_box[1])
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise TableError('Overlapping row boundaries.')
    selected = [span for span in spans if _inside(span.box, source_box, padding=.2)]
    if not selected:
        raise TableError('No positioned native text.')
    ownership = Counter()
    rows, evidence = [], []
    for position, index in enumerate(retained):
        row = []
        for column, (cell, text) in enumerate(zip(geometry[position], proposed[index])):
            if not text or not _readable(text):
                raise TableError('Empty or undecodable cells require source review.')
            box = (source_box[0] if column == 0 else cell[0], height-cuts[position+1],
                   source_box[2] if column == columns-1 else cell[2], height-cuts[position])
            if not _inside(box, source_box, padding=.2) or _area(box) <= 0:
                raise TableError('Cell boundaries extend outside the source table.')
            local = [span for span in selected if _inside(span.box, box, padding=.2)]
            native = ''.join(span.text for span in local)
            if (not _readable(native) or not local
                    or _canonical(native)[0] != _canonical(text)[0]):
                raise TableError('Cell text disagrees with complete positioned source spans.')
            if any(re.search(r'AdvP4C4E', span.font_name, re.I) for span in local):
                raise TableError('Ambiguous legacy font mapping.')
            ownership.update(span.index for span in local)
            value = ' '.join(text.split())
            scripted = (len({round(span.baseline, 1) for span in local}) > 1
                        and max(span.font_size for span in local) > 1.15*min(span.font_size for span in local))
            expression = None
            if any(span.math for span in local) or scripted:
                expression = native_expression(page, box)
                if expression is None:
                    raise TableError('Cell notation cannot be resolved from source geometry.')
                value = '$' + expression + '$'
            row.append({'text': value, 'rowspan': 1, 'colspan': 1, 'header': position == 0})
            evidence.append({'row': position+1, 'column': column+1, 'box': list(box),
                             'native_text': native, 'proposed_text': text,
                             'native_expression': expression})
        rows.append(row)
    if ownership != Counter({span.index: 1 for span in selected}):
        raise TableError('Source spans are omitted or owned by multiple cells.')
    table_grid(rows)
    return rows, {'method': 'native-geometry', 'accepted': True,
                  'cell_count': len(evidence), 'source_span_count': len(selected),
                  'complete_span_ownership': True, 'cells': evidence}


def supplement_source_tables(pdf_path, document):
    """Try only existing ruled-table regions; the original image always stays."""
    existing = document.setdefault('table transcriptions', {})
    targets = []
    for region in document.get('source table regions', []):
        box = tuple(region['bounding box'])
        assets = [asset for asset in document.get('figures', [])
                  if asset.get('asset kind') == 'table'
                  and asset.get('page number') == region['page number']
                  and asset.get('caption label') == region['caption label']
                  and _intersection(tuple(asset['bounding box']), box)/max(_area(box), 1) >= .95]
        if len(assets) == 1 and assets[0]['source'] not in existing:
            targets.append((region, assets[0]))
    if not targets:
        return document
    identity = {'layout_engine': 'pdfplumber', 'layout_version': version('pdfplumber'),
                'text_engine': 'pypdfium2', 'text_version': version('pypdfium2'),
                'profile': 'native-table-cells-v1'}
    try:
        _supplement_targets(pdf_path, document, targets, existing, identity)
    except (PSException, ValueError) as exc:
        recorded = {row['source'] for row in document.get('native table transcriptions', [])}
        for region, asset in targets:
            if asset['source'] not in recorded:
                document.setdefault('native table transcriptions', []).append({
                    **identity, 'page': region['page number'], 'label': region['caption label'],
                    'source': asset['source'], 'accepted': False,
                    'reason': 'Native table parser unavailable for this PDF: ' + str(exc)})
    return document


def _supplement_targets(pdf_path, document, targets, existing, identity):
    with _PDFIUM_LOCK, pdfplumber.open(pdf_path) as layout_pdf, pdfium.PdfDocument(pdf_path) as native_pdf:
        for number in sorted({region['page number'] for region, _ in targets}):
            page = native_pdf[number-1]
            layout_page = layout_pdf.pages[number-1]
            try:
                spans = _spans(page)
                height = page.get_size()[1]
                for region, asset in targets:
                    if region['page number'] != number:
                        continue
                    detail = {**identity, 'page': number, 'label': region['caption label'],
                              'source': asset['source'], 'accepted': False}
                    try:
                        if abs(layout_page.height-height) > .1 or abs(layout_page.width-page.get_size()[0]) > .1:
                            raise TableError('PDF engines disagree on page coordinates.')
                        left, bottom, right, top = region['bounding box']
                        crop = layout_page.crop((left, height-top, right, height-bottom))
                        tables = crop.find_tables({'vertical_strategy': 'text', 'horizontal_strategy': 'text',
                                                  'min_words_vertical': 2, 'min_words_horizontal': 1,
                                                  'intersection_tolerance': 5,
                                                  'text_x_tolerance': 2, 'text_y_tolerance': 2})
                        if len(tables) != 1:
                            raise TableError('No unambiguous table grid.')
                        rows, verification = _checked_rows(tables[0], page, tuple(region['bounding box']), spans)
                        verification.update(identity)
                        existing[asset['source']] = {'rows': rows, 'verification': verification}
                        detail.update(accepted=True, cell_count=verification['cell_count'])
                    except (TableError, ValueError, PSException) as exc:
                        detail['reason'] = str(exc)
                    document.setdefault('native table transcriptions', []).append(detail)
            finally:
                page.close()
                layout_page.close()
