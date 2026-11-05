"""Reproducible local parsing audit against a folder of real papers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from html.parser import HTMLParser
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import tempfile
import time

from PIL import Image, ImageDraw
import pypdfium2 as pdfium

from paper_localizer import core
from paper_localizer.article import build_article, write_article
from paper_localizer.figures import _area, _intersection, _nodes, attach_figures
from paper_localizer.formulas import recover_formulas
from paper_localizer.layout import prepare_layout
from paper_localizer.markup import FIGURE_LINK, image_reference
from paper_localizer.ocr import OcrOptions


CAPTION = re.compile(r"^(?:-\s*)?(?:Fig(?:ure)?\.?|Table)\s*\d+", re.I)


class LinkAudit(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.ids = set(); self.links = []; self.external = set(); self.feed(html)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs: self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"): self.links.append(attrs["href"][1:])
        if tag == "a" and attrs.get("href", "").lower().startswith(("http://", "https://")): self.external.add(attrs["href"])


def slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:100]


def native_cache_path(pdf: Path, directory: Path, engine_version: str) -> Path:
    identity = {"source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "engine": engine_version, "profile": "json-without-images-v1"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return directory / f"{pdf.stem}-{digest}.json"


def metrics(document: dict, raw: dict, output: Path) -> dict:
    blocks = core.extract_blocks(document)
    visible = FIGURE_LINK.sub("", "\n".join(block.text for block in blocks))
    overlap = []
    for node in _nodes(document):
        content, box = node.get("content", ""), node.get("bounding box")
        if not content or not box or node.get("formula snapshot") or node.get("formula image") or FIGURE_LINK.search(content): continue
        for asset in document.get("formulas", []):
            if asset.get("block") and node.get("page number") == asset["page number"]:
                if _intersection(tuple(box), tuple(asset["bounding box"])) / max(_area(tuple(box)), 1) > .85:
                    overlap.append({"page": node.get("page number"), "id": node.get("id"), "text": content[:160]}); break
    chunks = core.make_chunks(blocks)
    article = build_article(document, core._document_title(document, output.name), output.name, chunks)
    raw_captions = prepare_layout(deepcopy(raw)).get("captions", [])
    all_units = article.front + article.abstract + [u for s in article.sections for u in s.units]
    attached = [u for u in all_units if u.kind == "figure" and u.caption]
    attached_tables = [u for u in all_units if u.kind == "table-image" and u.caption]
    detached = [{"page":n.get("page number"), "label":n["caption label"]}
                for n in _nodes(document) if n.get("caption label", "").startswith("Fig.")
                and n.get("type") == "caption" and not n.get("attached caption") and not n.get("suppressed")]
    retained = {(n.get("page number"),n.get("caption label")) for n in _nodes(document)
                if n.get("type") == "caption"}
    lost = [{"page":n.get("page number"), "label":n["caption label"]} for n in raw_captions
            if (n.get("page number"),n.get("caption label")) not in retained
            and n.get("page number") not in document.get("source image pages", [])]
    markdown = output.read_text(encoding="utf-8")
    html = output.with_suffix(".html").read_text(encoding="utf-8")
    refs = {image_reference(m).source for m in FIGURE_LINK.finditer(markdown + "\n" + html)}
    audit = LinkAudit(html)
    graphics = document.get("figures", [])
    return {"pages": document.get("number of pages"),
            "figures": sum(f.get("asset kind", "figure") == "figure" for f in graphics),
            "tables": sum(f.get("asset kind") == "table" for f in graphics),
            "portraits": sum(f.get("asset kind") == "portrait" for f in graphics),
            "graphic_regions": len(graphics),
            "equations": len(document.get("formulas", [])), "source_caption_starts": len(raw_captions),
            "attached_figure_captions": len(attached), "overlapping_text": overlap,
            "attached_table_captions": len(attached_tables),
            "table_transcriptions": len(document.get("table transcriptions", {})),
            "detached_captions": detached, "source_image_pages": document.get("source image pages", []),
            "missing_source_captions": lost,
            "sections": len(article.sections),
            "ocr_pages": document.get("ocr pages", []), "ocr_details": document.get("ocr details", []),
            "unresolved_glyphs": len(re.findall(r"[\ue000-\uf8ff\ufffd]", visible)),
            "missing_assets": [ref for ref in refs if not (output.parent / ref).is_file()],
            "broken_anchors": [link for link in audit.links if link not in audit.ids],
            "source_links": len(document.get("source links", [])),
            "missing_source_links": [link["url"] for link in document.get("source links", [])
                                     if link["url"] not in audit.external]}


def review_sheet(pdf: Path, document: dict, destination: Path) -> None:
    """Original page next to selected source crops; no browser is required."""
    assets = document.get("figures", [])
    if document.get("source image pages"):
        assets = [{"source": f"figs/{hashlib.sha256(pdf.read_bytes()).hexdigest()[:12]}/page-{p:03d}-source.png",
                   "page number": p, "bounding box": [0,0,612,792]} for p in document["source image pages"]]
    if not assets: assets = [f for f in document.get("formulas", []) if f.get("block")]
    if not assets: return
    selected = [assets[0], assets[len(assets) // 2], assets[-1]]
    canvas = Image.new("RGB", (1400, len(selected) * 580), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    with pdfium.PdfDocument(pdf) as source:
        for row, asset in enumerate(selected):
            page = source[asset["page number"] - 1]
            bitmap = page.render(scale=1); original = bitmap.to_pil().convert("RGB")
            height = page.get_size()[1]; left, bottom, right, top = asset["bounding box"]
            ImageDraw.Draw(original).rectangle((left, height - top, right, height - bottom), outline="red", width=2)
            original.thumbnail((480, 540)); canvas.paste(original, (15, row * 580 + 30))
            with Image.open(destination.parent / asset["source"]) as crop:
                crop.thumbnail((880, 400 if asset.get("caption node") else 510)); canvas.paste(crop, (510, row * 580 + 45))
            if caption := asset.get("caption node"):
                l,b,r,t = caption["bounding box"]; w,h = page.get_size()
                cap_bitmap = page.render(scale=2, crop=(max(0,l-1),max(0,b-1),max(0,w-r-1),max(0,h-t-1)))
                cap_image = cap_bitmap.to_pil(); cap_image.thumbnail((880,100))
                canvas.paste(cap_image, (510,row*580+465)); cap_image.close(); cap_bitmap.close()
            draw.text((15, row * 580 + 8), f"Source p.{asset['page number']} / extracted crop", fill="black")
            original.close(); bitmap.close(); page.close()
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/review"))
    parser.add_argument("--cache", type=Path, default=Path(".scratch/corpus/native"))
    parser.add_argument("--only", default="")
    parser.add_argument("--sheets", action="store_true")
    parser.add_argument("--ocr-mode",choices=("auto","off","force"),default="auto")
    parser.add_argument("--vision-ocr-model", default="")
    parser.add_argument("--vision-ocr-url", default=core.OLLAMA_URL)
    args = parser.parse_args()
    filters = args.only.lower().split(",")
    pdfs = [p for p in sorted(args.input.glob("*.pdf")) if any(part.strip() in p.name.lower() for part in filters)]
    args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    native_files = {pdf: native_cache_path(pdf, args.cache, version("opendataloader-pdf")) for pdf in pdfs}
    missing = [p for p in pdfs if not native_files[p].is_file()]
    if missing:
        import opendataloader_pdf
        print(f"Extracting {len(missing)} uncached PDFs with OpenDataLoader", flush=True)
        with tempfile.TemporaryDirectory(prefix="paper-localizer-native-audit-", dir=args.cache.resolve()) as temporary:
            staging = Path(temporary)
            opendataloader_pdf.convert(input_path=[str(p.resolve()) for p in missing],
                                      output_dir=str(staging), format="json", image_output="off", quiet=True)
            for pdf in missing:
                (staging / (pdf.stem + ".json")).replace(native_files[pdf])
    results = []
    for index, pdf in enumerate(pdfs, 1):
        start = time.monotonic()
        try:
            raw = json.loads(native_files[pdf].read_text(encoding="utf-8"))
            document = deepcopy(raw)
            if hasattr(core, "prepare_document"):
                document = core.prepare_document(pdf, document, args.output,ocr=OcrOptions(
                    mode=args.ocr_mode, vision_model=args.vision_ocr_model, vision_url=args.vision_ocr_url))
            else:
                document = recover_formulas(pdf, attach_figures(pdf, document, args.output), args.output)
            output = args.output / (slug(pdf) + ".md")
            write_article(document, core._document_title(document, pdf.name), pdf.name,
                          core.make_chunks(core.extract_blocks(document)), None, "", "English", output)
            result = {"file": pdf.name, "output": output.name,
                      "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                      **metrics(document, raw, output)}
            (args.output / (slug(pdf) + ".audit.json")).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            if args.sheets: review_sheet(pdf, document, args.output / (slug(pdf) + ".review.png"))
        except Exception as exc:
            result = {"file": pdf.name, "error": str(exc)}
        result["seconds"] = round(time.monotonic() - start, 2); results.append(result)
        print(f"[{index}/{len(pdfs)}] {pdf.stem[:48]}: " + json.dumps({k:v for k,v in result.items() if k not in {"file", "output"}},ensure_ascii=True), flush=True)
        (args.output / "audit-report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Parsing corpus audit", "", "Generated locally from the supplied PDFs. These are extraction previews with local OCR where needed; Ollama translation and summary inference were not run.", "",
             "Checks cover asset paths, internal links, duplicate text fully inside equation snapshots and caption ownership. A zero does not prove complete semantic extraction. Review sheets compare selected original pages, figure crops and full source captions.", "",
             "| Paper | Pages | Fig. | Eq./source crops | Captions attached | Text overlaps | Undecoded glyphs |", "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        if "error" in result: lines.append(f"| {result['file']} | ERROR: {result['error']} | | | | | |")
        else: lines.append(f"| [{result['file']}]({Path(result['output']).with_suffix('.html').name}) | {result['pages']} | {result['figures']} | {result['equations']} | {result['attached_figure_captions']} | {len(result['overlapping_text'])} | {result['unresolved_glyphs']} |")
    lines.extend(["", "## Items requiring source review", ""])
    for result in results:
        if "error" in result: continue
        notes = []
        if result["ocr_pages"]: notes.append("local OCR pages: " + ", ".join(map(str,result["ocr_pages"])))
        if result["source_image_pages"]: notes.append("undecodable fonts; source page images retained, OCR required for translation")
        if result["detached_captions"]: notes.append("captions retained in reading order without a matched graphic: " + ", ".join(f"{n['label']} (p.{n['page']})" for n in result["detached_captions"]))
        if result["missing_source_captions"]: notes.append("missing source captions: " + str(result["missing_source_captions"]))
        sheet = Path(result["output"]).with_suffix(".review.png")
        if (args.output / sheet).is_file():
            lines.append(f"- [{result['file']}]({sheet.as_posix()}): " + ("; ".join(notes) if notes else "selected source crops available for visual review"))
    report = "\n".join(lines)+"\n"
    (args.output / "audit-report.md").write_text(report, encoding="utf-8")
    from markdown_it import MarkdownIt
    body = MarkdownIt("commonmark").enable("table").render(report)
    (args.output / "audit-report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PLoTrans · Corpus audit</title><link rel="stylesheet" href="assets/article.css"><body><header class="masthead">PLoTrans · Corpus audit</header><main style="max-width:1400px;margin:auto;padding:30px">'+body+'</main></body></html>',encoding="utf-8")


if __name__ == "__main__": main()
