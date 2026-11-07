"""Compare an extracted article with its source PDF and create offline review pages."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

from PIL import Image
import pypdfium2 as pdfium

from paper_localizer import core
from paper_localizer.article import build_article, write_article
from paper_localizer.markup import FIGURE_LINK, image_reference


def nodes(document, page=1):
    page = document.get("page number", page)
    yield document, page
    for key in ("kids", "list items", "rows", "cells"):
        for child in document.get(key, []):
            if isinstance(child, dict): yield from nodes(child, page)


def canonical(text):
    text = FIGURE_LINK.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Native TeX marks can be separate glyphs while the reading edition uses
    # composed Unicode. Count their base letters consistently; typography is
    # verified separately from this character-preservation diagnostic.
    return "".join(c for c in unicodedata.normalize("NFKD", text.replace('ı','i')).casefold() if c.isalnum())


def inside(glyph, box, padding=.5):
    return box and box[0]-padding <= glyph["x"] <= box[2]+padding and box[1]-padding <= glyph["y"] <= box[3]+padding


def source_glyphs(page):
    text = page.get_textpage()
    glyphs = []
    try:
        count = text.count_chars()
        skip_low = False
        for index in range(count):
            if skip_low:
                skip_low = False
                continue
            code = pdfium.raw.FPDFText_GetUnicode(text, index)
            if 0xd800<=code<=0xdbff and index+1<count:
                low = pdfium.raw.FPDFText_GetUnicode(text,index+1)
                if 0xdc00<=low<=0xdfff:
                    code = 0x10000+((code-0xd800)<<10)+low-0xdc00
                    skip_low = True
            box = text.get_charbox(index)
            if box[0] == box[2] or box[3]-box[1] < .01: continue
            char = chr(code) if 0 < code <= 0x10FFFF else "\ufffd"
            obj = text.get_textobj(index)
            baseline = obj.get_matrix().f if obj is not None else (box[1]+box[3])/2
            glyphs.append({"index": index, "char": char, "box": list(box), "x": (box[0]+box[2])/2,
                           "y": (box[1]+box[3])/2, "baseline": baseline})
    finally:
        text.close()
    return glyphs


def repeated_margin_text(pages, sizes):
    groups = defaultdict(list)
    for number, glyphs in pages.items():
        height = sizes[number][1]
        lines = defaultdict(list)
        for glyph in glyphs:
            side = "bottom" if glyph["y"] < height*.06 else "top" if glyph["y"] > height*.90 else ""
            if side: lines[side, round(glyph["baseline"], 0)].append(glyph)
        for (side, _), line in lines.items():
            text = "".join(g["char"] for g in sorted(line, key=lambda g: g["x"]))
            key = side, re.sub(r"\d+", "#", canonical(text))
            if key[1]: groups[key].append((number, [g["index"] for g in line]))
    result = defaultdict(set)
    for instances in groups.values():
        if len({number for number, _ in instances}) >= max(3, len(pages)*.3):
            for number, indexes in instances: result[number].update(indexes)
    return result


def content_coverage(document, article_path, pages, sizes):
    contents = article_path.read_text(encoding="utf-8") + article_path.with_suffix(".html").read_text(encoding="utf-8")
    active = {image_reference(match).source for match in FIGURE_LINK.finditer(contents)}
    assets = document.get("figures", []) + document.get("formulas", [])
    # Full original OCR pages are retained in the comparison appendix. Count
    # that image preservation separately from editable/semantic OCR accuracy.
    assets += [{'source':a['source'],'page number':a['page'],'bounding box':[0,0,*sizes[a['page']]]}
               for a in document.get('ocr sources',[]) if a.get('page') in sizes]
    assets_by_page,excluded_by_page,counters_by_page = defaultdict(list),defaultdict(list),defaultdict(list)
    for asset in assets:
        if asset['source'] in active:
            assets_by_page[asset['page number']].append(asset)
    for asset in document.get('excluded artwork',[]):
        excluded_by_page[asset['page number']].append(asset)
    for counter in document.get('review line numbers',[]):
        counters_by_page[counter['page']].append(counter)
    by_page = defaultdict(list)
    for node, number in nodes(document):
        if node.get("content") and not node.get("suppressed"): by_page[number].append(node)
    furniture = repeated_margin_text(pages, sizes)
    result = []
    invalid_assets = []
    for asset in assets:
        if asset["source"] not in active: continue
        path = article_path.parent / asset["source"]
        try:
            with Image.open(path) as image:
                gray = image.convert("L"); ink = gray.getextrema()[0] < 245; gray.close()
        except OSError:
            ink = False
        if not ink: invalid_assets.append(asset["source"])
    for number, glyphs in pages.items():
        ignored = set(furniture[number])
        visual = set()
        local_assets = [a for a in assets_by_page[number] if a['source'] not in invalid_assets]
        local_excluded = excluded_by_page[number]
        review_counters = set()
        margin_nodes = [n for n in by_page[number] if n.get('type') in {'header','footer'}]
        for glyph in glyphs:
            if any(inside(glyph,n.get('bounding box')) for n in margin_nodes):
                ignored.add(glyph["index"])
            if any(inside(glyph,a['bounding box']) for a in local_excluded):
                ignored.add(glyph["index"])
            if any(inside(glyph,c['box'],padding=0) for c in counters_by_page[number]):
                ignored.add(glyph['index']);review_counters.add(glyph['index'])
            if any(inside(glyph,a['bounding box'],padding=0) for a in local_assets):
                visual.add(glyph["index"])
        units = {}
        for glyph in glyphs:
            if glyph["index"] in ignored: continue
            for offset, char in enumerate(canonical(glyph["char"])):
                units[glyph["index"], offset] = char
        covered = {key for key in units if key[0] in visual}
        normalized = set()
        for node in by_page[number]:
            if node.get("type") in {"header", "footer"} or not node.get("bounding box"): continue
            selected = [g for g in glyphs if g["index"] not in visual and g["index"] not in ignored
                        and inside(g, node["bounding box"])]
            source, keys = [], []
            for glyph in selected:
                for offset, char in enumerate(canonical(glyph["char"])):
                    source.append(char); keys.append((glyph["index"], offset))
            expected, observed = "".join(source), canonical(node["content"])
            if expected.startswith("figure") and observed.startswith("fig") and not observed.startswith("figure"):
                normalized.update(keys[3:6])
                expected = expected[:3] + expected[6:]
                keys = keys[:3] + keys[6:]
            for match in SequenceMatcher(None, expected, observed, autojunk=False).get_matching_blocks():
                covered.update(keys[match.a:match.a+match.size])
        covered.update(normalized)
        missing = sorted(set(units)-covered)
        absent_indexes = {key[0] for key in missing}
        snippets, pending = [], []
        for glyph in glyphs:
            if glyph["index"] in absent_indexes: pending.append(glyph["char"])
            elif pending:
                snippets.append("".join(pending)); pending = []
        if pending: snippets.append("".join(pending))
        opaque = [g for g in glyphs if (g["char"] == "\ufffd" or not g["char"].isprintable()) and g["index"] not in ignored]
        result.append({"page": number, "decodable_character_units": len(units), "covered_units": len(covered),
                       "coverage": round(len(covered)/max(len(units), 1), 6), "unmatched_units": len(missing),
                       "caption_abbreviation_units": len(normalized),
                       "unmatched_snippets": snippets, "image_preserved_glyphs": len(visual),
                       "furniture_glyphs": len(ignored), "undecodable_glyphs": len(opaque),
                       "review_counter_glyphs":len(review_counters),
                       "undecodable_glyphs_preserved_as_images": sum(g["index"] in visual for g in opaque)})
    return result, invalid_assets


def write_review(pdf_path, document, article_path, output):
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source-pages"; source_dir.mkdir(exist_ok=True)
    parsed_dir = output / "parsed-pages"; parsed_dir.mkdir(exist_ok=True)
    if (article_path.parent/"figs").is_dir():
        shutil.copytree(article_path.parent/"figs", parsed_dir/"figs", dirs_exist_ok=True)
    pages, sizes = {}, {}
    with pdfium.PdfDocument(pdf_path) as pdf:
        for number in range(1, len(pdf)+1):
            page = pdf[number-1]
            try:
                sizes[number] = list(page.get_size()); pages[number] = source_glyphs(page)
                bitmap = page.render(scale=1.5, draw_annots=False); image = bitmap.to_pil()
                image.save(source_dir/f"page-{number:03d}.png"); image.close(); bitmap.close()
            finally: page.close()
    coverage, invalid_assets = content_coverage(document, article_path, pages, sizes)
    blocks = core.extract_blocks(document)
    for number in pages:
        page_blocks = [b for b in blocks if number in (b.source_pages or (b.page,))]
        if number>1:
            first_body = next((b for b in page_blocks if b.kind not in {
                'frontmatter','abstract','abstract-image','keywords'}),None)
            if first_body is not None and first_body.kind!='heading':
                # A standalone pane lacks the heading from the preceding
                # source page. Keep its continued body visible in the pane.
                page_blocks.insert(page_blocks.index(first_body),core.Block(
                    number,'heading',f'## Source page {number}'))
        page_document = {"number of pages": 1, "figures": document.get("figures", []),
                         "table transcriptions": document.get("table transcriptions", {}),
                         "source links": [link for link in document.get("source links", [])
                                          if number in link.get("pages", [])]}
        write_article(page_document, f"Parsed source page {number}", pdf_path.name,
                      core.make_chunks(page_blocks), None, "English", parsed_dir/f"page-{number:03d}.md")
    article = build_article(document, core._document_title(document, pdf_path.name), pdf_path.name, core.make_chunks(blocks))
    graphics = document.get("figures", [])
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_pdf": str(pdf_path.resolve()),
              "review_engine_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "article": str(article_path.resolve()),
              "pages": len(pages), "figures": sum(a.get("asset kind", "figure") == "figure" for a in graphics),
              "tables": sum(a.get("asset kind") == "table" for a in graphics),
              "portraits": sum(a.get("asset kind") == "portrait" for a in graphics),
              "equation_images": len(document.get("formulas", [])), "sections": [s.title for s in article.sections],
              "abstract_paragraphs": len(article.abstract), "source_spacing_repairs": len(document.get("source spacing repairs", [])),
              "table_transcriptions": len(document.get("table transcriptions", {})), "invalid_assets": invalid_assets,
              "coverage": coverage,
              "interpretation": "Character coverage concerns decodable source letters and digits represented by text or retained source images. It does not measure overall semantic accuracy, reading order, OCR accuracy, or text stored as vector outlines. Review every source page and the retained table/equation images."}
    (output/"review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = []
    for page in coverage:
        number = page["page"]
        sections.append(f'<section id="page-{number}"><h2>Source page {number}</h2>'
                        f'<p>Decodable character coverage: {page["coverage"]:.2%}; unmatched units: {page["unmatched_units"]}.</p>'
                        '<div class="comparison"><figure><img loading="lazy" '
                        f'src="source-pages/page-{number:03d}.png" alt="Original PDF page {number}"></figure>'
                        f'<iframe loading="lazy" src="parsed-pages/page-{number:03d}.html" title="Parsed source page {number}"></iframe></div></section>')
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PLoTrans source comparison</title><style>body{font:15px/1.6 system-ui;margin:24px;background:#f4f6f8;color:#172e40}'
            'header,section{background:white;padding:20px;margin-bottom:24px}nav{display:flex;gap:12px;flex-wrap:wrap}'
            '.comparison{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img{width:100%;height:auto}'
            'iframe{width:100%;height:1200px;border:1px solid #ccd5dd}a{color:#00629b}@media(max-width:850px){.comparison{grid-template-columns:1fr}}'
            '</style></head><body><header><h1>PLoTrans source comparison</h1>'
            f'<p>{escape(pdf_path.name)} · {len(pages)} pages · {report["figures"]} figures · {report["tables"]} tables</p>'
            f'<p>{escape(report["interpretation"])}</p>'
            '<p>Reading panes keep continued paragraphs whole. Paragraphs spanning source pages may appear in more than one pane; page references record their provenance.</p><nav>'
            + "".join(f'<a href="#page-{number}">{number}</a>' for number in pages) + '</nav></header>'
            + "".join(sections) + '</body></html>')
    (output/"review.html").write_text(html, encoding="utf-8")
    print(json.dumps({"report": str(output/"review.html"), "pages": len(pages),
                      "unmatched_units": sum(p["unmatched_units"] for p in coverage), "invalid_assets": invalid_assets}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("document", type=Path, help="Processed .audit.json from audit_corpus.py")
    parser.add_argument("article", type=Path, help="Generated Markdown article with matching HTML")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.document.read_text(encoding="utf-8"))
    write_review(args.pdf, document, args.article, args.output)


if __name__ == "__main__":
    main()
