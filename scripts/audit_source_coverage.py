"""Locate native PDF words that need source review after an extraction audit.

This is a diagnostic, not an accuracy score. It distinguishes editable text,
visible original crops, intentionally removed furniture, and uncertain words.
Scanned/undecodable pages require a separate OCR/source review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pypdfium2 as pdfium

from paper_localizer.formulas import _spans, _MATH_FONT
from paper_localizer.markup import FIGURE_LINK, image_reference
from paper_localizer.source_text import normalized_spans, _span_indices
from paper_localizer.core import extract_blocks


def words(text):
    text = unicodedata.normalize('NFKD', text.casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.findall(r'[a-z]{3,}', text)


def owns(outer, inner):
    x, y = (inner[0]+inner[2])/2, (inner[1]+inner[3])/2
    return outer[0]-.5 <= x <= outer[2]+.5 and outer[1]-.5 <= y <= outer[3]+.5


def nodes(document):
    """Honor inherited suppression while retaining removed furniture evidence."""
    visible, hidden = [], []

    def visit(node, inherited=False, page=1):
        page = node.get('page number', page)
        suppressed = inherited or node.get('suppressed') or node.get('type') in {'header','footer'}
        if node.get('bounding box') and node.get('content'):
            record = {'page':page, 'box':node['bounding box'], 'type':node.get('type'),
                      'text':node['content'], 'words':Counter(words(FIGURE_LINK.sub('',node['content']))),
                      'snapshot':bool(node.get('formula snapshot') or node.get('formula image'))}
            (hidden if suppressed else visible).append(record)
        for key in ('kids','list items','rows','cells'):
            for child in node.get(key,[]):
                if isinstance(child,dict):
                    visit(child,suppressed,page)
        if isinstance(node.get('caption node'),dict):
            visit(node['caption node'],suppressed,page)

    visit(document)
    return visible, hidden


def visible_crops(document, markdown):
    refs = {image_reference(m).source for m in FIGURE_LINK.finditer(markdown)}
    main_refs = {image_reference(m).source for block in extract_blocks(document)
                 for m in FIGURE_LINK.finditer(block.text)}
    crops = defaultdict(list)
    seen = set()

    def visit(value):
        if isinstance(value,dict):
            source, box, page = value.get('source'),value.get('bounding box'),value.get('page number')
            if isinstance(source,str) and source in refs and box and page:
                key = (source,page,tuple(box))
                if key not in seen:
                    crops[page].append({'source':source,'box':box,'main':source in main_refs})
                    seen.add(key)
            for child in value.values():
                if isinstance(child,(list,dict)):
                    visit(child)
        elif isinstance(value,list):
            for child in value:
                visit(child)

    visit(document)
    for page in document.get('source image pages',[]):
        size = document.get('page sizes',{}).get(str(page))
        if size:
            crops[page].append({'source':'complete source page','box':[0,0,*size],'main':True})
    return crops


def native_words(page, raw_spans, document, page_number):
    """Rejoin physical word fragments, retaining every original source index."""
    groups = []
    textpage = page.get_textpage()
    try:
        indices = _span_indices(textpage,raw_spans,{s.index for s in raw_spans})
        spans = normalized_spans(raw_spans,document,page_number)
        overrides = document.get('source glyph overrides',{}).get(str(page_number),{})
        previous_end = None
        for span in spans:
            if not re.fullmatch(r'[A-Za-zÀ-ž]+',span.text) or _MATH_FONT.search(span.font_name or ''):
                previous_end = None
                continue
            previous = groups[-1] if groups else None
            gap = range(previous_end+1,span.index) if previous_end is not None else None
            adjacent = gap is not None and all(overrides.get(str(i)) == '' for i in gap)
            if (previous and abs(previous.baseline-span.baseline) < span.font_size*.12
                    and -.2 <= span.box[0]-previous.box[2] < span.font_size*.18
                    and adjacent):
                from dataclasses import replace
                from paper_localizer.figures import _union
                groups[-1] = replace(previous,text=previous.text+span.text,box=_union(previous.box,span.box))
            else:
                groups.append(span)
            positions = indices.get(str(span.index),[])
            previous_end = positions[-1] if positions else None
        aliases = {}
        for left, right in zip(spans,spans[1:]):
            positions = indices.get(str(left.index),[])
            if not positions or not re.fullmatch('[A-Za-z]+',left.text+right.text):
                continue
            index = positions[-1]+1
            if (index < textpage.count_chars() and right.index == index+1
                    and pdfium.raw.FPDFText_GetUnicode(textpage,index) == 2
                    and pdfium.raw.FPDFText_IsHyphen(textpage,index) == 1):
                aliases[left.index] = aliases[right.index] = (left.text+right.text).casefold()
    finally:
        textpage.close()
    return groups, aliases


def repeated_margins(spans_by_page, sizes):
    groups = defaultdict(list)
    for page,spans in spans_by_page.items():
        height = sizes[page][1]
        lines = defaultdict(list)
        for span in spans:
            y = (span.box[1]+span.box[3])/2
            side = 'bottom' if y < height*.06 else 'top' if y > height*.90 else ''
            if side:
                lines[side,round(span.baseline)].append(span)
        for (side,_),line in lines.items():
            line.sort(key=lambda s:s.box[0])
            key = side,tuple(words(' '.join(s.text for s in line)))
            if key[1]:
                groups[key].append((page,{s.index for s in line}))
    result = defaultdict(set)
    for instances in groups.values():
        if len({p for p,_ in instances}) >= max(3,len(spans_by_page)*.3):
            for page,indices in instances:
                result[page].update(indices)
    return result


def inspect(source, document, markdown):
    visible, hidden = nodes(document)
    by_page, hidden_page = defaultdict(list), defaultdict(list)
    for node in visible:
        by_page[node['page']].append(node)
    for node in hidden:
        hidden_page[node['page']].append(node)
    crops = visible_crops(document,markdown)
    repairs = defaultdict(list)
    for repair in document.get('source typography repairs',[]):
        repairs[repair['page']].append(repair)
    totals, pages = Counter(), []
    with pdfium.PdfDocument(source) as pdf:
        spans_by_page,sizes = {},{}
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                sizes[index+1] = page.get_size()
                spans_by_page[index+1] = _spans(page)
            finally:
                page.close()
        margins = repeated_margins(spans_by_page,sizes)
        for number in range(1,len(pdf)+1):
            if number in document.get('ocr pages',[]):
                pages.append({'page':number,'status':'OCR requires visual source review'})
                continue
            page = pdf[number-1]
            try:
                counts, uncertain = Counter(), []
                spans, aliases = native_words(page,spans_by_page.pop(number),document,number)
                for span in spans:
                    if not re.fullmatch(r'[A-Za-zÀ-ž]{3,}',span.text) or _MATH_FONT.search(span.font_name or ''):
                        continue
                    token = ''.join(words(span.text))
                    if not token:
                        continue
                    image = next((c for c in crops[number] if c['main'] and owns(c['box'],span.box)),None)
                    comparison = next((c for c in crops[number] if not c['main'] and owns(c['box'],span.box)),None)
                    containing = [n for n in by_page[number] if owns(n['box'],span.box) and not n['snapshot']]
                    matching = [n for n in containing if n['words'][token]]
                    native_wrap = any(n['words'][aliases.get(span.index,'')] for n in containing)
                    reader_label = any((span.text=='Abstract' and n['type']=='abstract')
                                       or (span.text in {'Figure','FIGURE'} and n['type']=='caption'
                                           and n['text'].startswith('Fig.')) for n in containing)
                    typography = any(r['kind']=='line-wrap' and r.get('removed hyphen')
                                     and token in words(r['original'])
                                     and any(owns(b,span.box) for b in r['native boxes'])
                                     for r in repairs[number])
                    if matching:
                        category = 'editable-word-match'
                    elif native_wrap:
                        category = 'native-wrap-word-match'
                    elif reader_label:
                        category = 'reader-label-normalization'
                    elif image:
                        category = 'visible-source-crop'
                    elif typography:
                        category = 'source-confirmed-line-wrap'
                    elif any(owns(n['box'],span.box) for n in hidden_page[number]):
                        category = 'removed-furniture-or-covered-node'
                    elif span.index in margins[number]:
                        category = 'repeated-outer-margin-text'
                    elif comparison:
                        category = 'comparison-appendix-only'
                    else:
                        category = 'needs-source-review'
                        uncertain.append({'word':span.text,'box':list(span.box),'font':span.font_name,
                                          'math_flag':span.math,
                                          'containing_text':[n['text'][:240] for n in containing[:2]]})
                    counts[category] += 1
                totals.update(counts)
                pages.append({'page':number,'counts':dict(counts),'uncertain':uncertain})
            finally:
                page.close()
    return {'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'counts':dict(totals),'pages':pages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--audits',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',default='')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    reports = []
    for source in sorted(args.input.glob('*.pdf')):
        if args.only.casefold() not in source.name.casefold():
            continue
        stem = re.sub(r'[^a-z0-9]+','-',source.stem.lower()).strip('-')[:100]
        audit = args.audits/(stem+'.audit.json')
        if not audit.exists():
            continue
        report = inspect(source,json.loads(audit.read_text()),audit.with_suffix('').with_suffix('.md').read_text())
        reports.append(report)
        args.output.write_text(json.dumps({'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                                          'scope':__doc__,'documents':reports},indent=2,ensure_ascii=False))
        print(json.dumps({'file':source.name,**report['counts']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
