"""Offline CPU OCR for scans and PDFs whose prose uses undecodable fonts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from statistics import median
import threading

import pypdfium2 as pdfium

from .figures import _PDFIUM_LOCK, _nodes, _union
from .formulas import _spans


OCR_VERSION = "rapidocr-3.10.0/onnxruntime-1.23.2/ppocr-v6-small/layout-1"
_OCR_LOCK = threading.Lock()


@dataclass(frozen=True)
class OcrOptions:
    mode: str = "auto"
    dpi: int = 216
    min_confidence: float = .85
    vision_model: str = ""
    vision_url: str = "http://127.0.0.1:11434"
    vision_digest: str = ""

    def __post_init__(self):
        if self.mode not in {"auto", "off", "force"}:
            raise ValueError("OCR mode must be auto, off or force")
        if not 96 <= self.dpi <= 400:
            raise ValueError("OCR DPI must be between 96 and 400")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")


class OcrError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def model_paths() -> dict[str, Path]:
    """Use only wheel-bundled models, verified against the pinned manifest."""
    try:
        import rapidocr
        import onnxruntime
    except ImportError as exc:
        raise OcrError('Local OCR is unavailable. Install it with: pip install -e ".[ocr]"') from exc
    from importlib.metadata import version
    if version("rapidocr") != "3.10.0" or version("onnxruntime") != "1.23.2":
        raise OcrError('OCR versions differ from the pinned setup. Run: pip install -e ".[ocr]"')
    onnxruntime.disable_telemetry_events()
    root = Path(rapidocr.__file__).parent / "models"
    manifest = json.loads(files("paper_localizer").joinpath("assets/ocr-models.json").read_text(encoding="utf-8"))
    result = {}
    for stage, entry in manifest.items():
        path = root / entry["filename"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise OcrError(f"Bundled OCR model is missing or has an invalid checksum: {path.name}. Reinstall rapidocr==3.10.0.")
        result[stage] = path
    return result


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    paths = model_paths()
    return RapidOCR(params={
        "Global.log_level": "warning", "Global.max_side_len": 3600,
        "Det.limit_side_len": 2048, "Det.limit_type": "max",
        "Det.model_path": str(paths["det"]), "Cls.model_path": str(paths["cls"]),
        "Rec.model_path": str(paths["rec"]), "Global.text_score": 0.0,
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
    })


def _recognize(image) -> list[dict]:
    # Fail before RapidOCR's import if the optional dependency is absent.
    model_paths()
    with _OCR_LOCK:
        result = _engine()(image)
    if result.boxes is None or result.txts is None:
        return []
    return [{"text": str(text), "score": float(score), "box": box.tolist()}
            for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)]


def _math_line(text: str) -> bool:
    if "⊕" in text:
        return True
    # Superscripts are often flattened with high OCR confidence. Preserve the
    # source when a likely exponent or logarithm base has lost its typography.
    if re.search(r"\b\d+[ijkb](?:[+\-−]\d+)?\b|\b\d+[−-][a-z]\b|\blog\d+[a-z]\b", text):
        return True
    if re.search(r"https?://|\S+@\S+",text): return False
    return bool(re.search(r"[=<>≤≥∑∫√∈∉≈≠²³⁰¹⁴⁵⁶⁷⁸⁹]|\b(?:log|ln|sin|cos|Pr)\s*\(|\bO\s*\(", text)
                or (len(text.split()) < 5 and re.search(r"[{}^_]|\d\s*[+*/]\s*\d", text)))


def _reading_order(lines: list[dict], width: float) -> list[dict]:
    """Read columns completely, splitting at full-width titles/captions."""
    middle = width/2
    left = [n for n in lines if n["bounding box"][2] < middle+5]
    right = [n for n in lines if n["bounding box"][0] > middle-5]
    wide = [n for n in lines if n not in left and n not in right]
    if min(len(left), len(right)) < 5 or len(wide) > len(lines)*.25:
        return sorted(lines, key=lambda n: (-n["bounding box"][3], n["bounding box"][0]))
    anchors = sorted(wide, key=lambda n: -n["bounding box"][3])
    ordered = []; remaining = left+right
    for anchor in anchors:
        top = anchor["bounding box"][3]
        band = [n for n in remaining if n["bounding box"][3] > top]
        ordered.extend(sorted(band, key=lambda n: (n["bounding box"][0] >= middle, -n["bounding box"][3])))
        remaining = [n for n in remaining if n not in band]
        ordered.append(anchor)
    ordered.extend(sorted(remaining, key=lambda n: (n["bounding box"][0] >= middle, -n["bounding box"][3])))
    return ordered


def _layout(lines: list[dict], page: int, width: float, height: float, options: OcrOptions,
            *, in_references: bool = False) -> list[dict]:
    if not lines: return []
    typical = median(n["bounding box"][3]-n["bounding box"][1] for n in lines)
    consumed = set(); replacements = {}
    for i,line in enumerate(lines):
        if i in consumed: continue
        if re.fullmatch(r"\d+(?:\.\d+)*\.?",line["text"].strip()):
            l,b,r,t = line["bounding box"]
            for j,other in enumerate(lines):
                if j==i or j in consumed or not re.match(r"[A-Z][a-zA-Z]{3}",other["text"]):continue
                a,c,d,e = other["bounding box"]
                if 0 <= a-r <= typical*2 and abs((b+t)-(c+e)) < typical*.8:
                    line = {"text":line["text"]+" "+other["text"],"score":min(line["score"],other["score"]),
                            "bounding box":list(_union(tuple(line["bounding box"]),tuple(other["bounding box"])))}
                    consumed.add(j);break
        replacements[i] = line
    # A prefix can occur after its title in OCR's initial row ordering.
    lines = [replacements.get(i,line) for i,line in enumerate(lines) if i not in consumed]
    result = []
    for line in _reading_order(lines, width):
        box = line["bounding box"]; text = line["text"].strip()
        if not text: continue
        text = re.sub(r"^(\d+(?:\.\d+)*)([A-Z][a-z])", r"\1 \2", text)
        size = box[3]-box[1]
        if text == "References": in_references = True
        heading = not in_references and re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", text) and len(text) < 100 and not text.endswith(".")
        title = page == 1 and box[1] > height*.6 and size > typical*1.3
        previous = result[-1] if result else None
        if previous and previous.get("ocr title"):
            b = previous["bounding box"]
            if (abs((b[0]+b[2])-(box[0]+box[2])) < typical*2 and 0 <= b[1]-box[3] <= typical*.5
                    and size >= typical*.95):title = True
        kind = "heading" if heading or title or text in {"Abstract", "References"} else "paragraph"
        if (box[1] < height*.04 and re.fullmatch(r"\d+",text)) or box[3] > height*.98:
            kind = "footer" if box[1] < height*.04 else "header"
        uncertain = line["score"] < options.min_confidence or _math_line(text)
        node = {"type": kind, "page number": page, "content": text,
                "bounding box": list(box), "font size": size, "font": "OCR",
                "ocr score": line["score"], "ocr snapshot": uncertain, "ocr title": title}
        if kind == "heading": node["heading level"] = 1 if title else 2
        if previous:
            b = previous["bounding box"]; gap = b[1]-box[3]
            same_column = abs(b[0]-box[0]) < max(8, typical*2) and box[0]-b[0] < typical*.8 and min(b[2],box[2])-max(b[0],box[0]) > 0
            if title and previous.get("ocr title"):
                same_column = abs((b[0]+b[2])-(box[0]+box[2])) < typical*2
            compatible = (kind == previous["type"] == "paragraph" and not uncertain and not previous["ocr snapshot"])
            compatible = compatible or (title and previous.get("ocr title"))
            if compatible and same_column and -typical*.1 <= gap <= typical*.8:
                separator = " "
                if previous["content"].endswith("-") and text[0].islower():
                    previous["content"] = previous["content"][:-1]; separator = ""
                previous["content"] += separator+text
                previous["bounding box"] = list(_union(tuple(b),tuple(box)))
                previous["ocr score"] = min(previous["ocr score"],line["score"])
                continue
        result.append(node)
    for node in result:
        if (page == 1 and node["type"] == "paragraph" and node["bounding box"][1] < height*.3
                and re.match(r"^(?:[*†]\s*)?This (?:research|work) (?:was|is) (?:partially )?(?:supported|funded)\b", node["content"])):
            node["type"] = "footnote"
    return result


def apply_ocr(pdf_path: Path, document: dict, output_dir: Path, options: OcrOptions | None = None,
              *, recognizer=None) -> dict:
    """Replace only affected pages; never OCR already-decodable prose in auto mode."""
    options = options or OcrOptions()
    if options.mode == "off": return document
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]
    asset_dir = output_dir/"figs"/digest
    identity = {"version": OCR_VERSION, "dpi": options.dpi}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for number in range(1,len(pdf)+1):
            page = pdf[number-1]
            try:
                spans = _spans(page)
                unknown = bool(spans) and sum(not s.font_name for s in spans)/len(spans) > .75
                if options.mode != "force" and not unknown and sum(len(s.text) for s in spans if s.font_name) >= 30:
                    continue
                width,height = page.get_size()
                if not spans:
                    probe = page.render(scale=.5, grayscale=True)
                    image = probe.to_pil(); ink = image.point(lambda x:255 if x < 235 else 0)
                    blank = ink.getbbox() is None; ink.close(); image.close(); probe.close()
                    if blank: continue
                asset_dir.mkdir(parents=True,exist_ok=True)
                source = asset_dir/f"page-{number:03d}-ocr-source.png"
                bitmap = page.render(scale=options.dpi/72,draw_annots=False)
                image = bitmap.to_pil().convert("RGB"); bitmap.close()
                try:
                    image.save(source)
                    cache = output_dir/".ocr-cache"/digest/f"page-{number:03d}-{key}.json"
                    cached = None
                    if cache.is_file():
                        try:
                            candidate = json.loads(cache.read_text(encoding="utf-8"))
                            if candidate.get("identity") == identity: cached = candidate.get("lines")
                        except (ValueError,OSError): pass
                    raw = cached if cached is not None else (recognizer or _recognize)(image)
                    if cached is None:
                        cache.parent.mkdir(parents=True,exist_ok=True)
                        temporary = cache.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"identity":identity,"lines":raw},ensure_ascii=False),encoding="utf-8")
                        temporary.replace(cache)
                    lines = []
                    for line in raw:
                        points = line["box"]
                        left = max(0,min(p[0] for p in points)*width/image.width)
                        right = min(width,max(p[0] for p in points)*width/image.width)
                        bottom = max(0,height-max(p[1] for p in points)*height/image.height)
                        top = min(height,height-min(p[1] for p in points)*height/image.height)
                        if right > left and top > bottom:
                            lines.append({"text":line["text"],"score":float(line["score"]),"bounding box":[left,bottom,right,top]})
                finally:
                    image.close()
                if not lines:
                    # Keep the original page when the OCR engine finds no text.
                    document.setdefault("ocr failures",[]).append(number)
                    continue
                headings = [n for n in _nodes(document) if n.get("type")=="heading" and n.get("page number",number)<number]
                nodes = _layout(lines,number,width,height,options,
                                in_references=bool(headings and headings[-1].get("content")=="References"))
                def keep(node,inherited=1):
                    current = node.get("page number",inherited)
                    for field in ("kids","list items","rows","cells"):
                        if field in node:
                            node[field] = [n for child in node[field] if (n:=keep(child,current)) is not None]
                    if current != number: return node
                    children = [n for field in ("kids","list items","rows","cells") for n in node.get(field,[])]
                    if not children: return None
                    # A list/table can span OCR and native pages. Retain the
                    # other pages' descendants instead of deleting its parent.
                    node.pop("content",None);node.pop("bounding box",None)
                    node["page number"] = min(n.get("page number",number+1) for n in children)
                    if node.get("type") not in {"table","row","cell","list"}:node["type"]="group"
                    return node
                document["kids"] = [n for child in document.get("kids",[]) if (n:=keep(child)) is not None]
                at = next((i for i,n in enumerate(document["kids"]) if n.get("page number",1) > number),len(document["kids"]))
                document["kids"][at:at] = nodes
                document.setdefault("ocr pages",[]).append(number)
                document.setdefault("ocr line boxes",{})[str(number)] = [n["bounding box"] for n in lines]
                document.setdefault("ocr sources",[]).append({"page":number,"source":source.relative_to(output_dir).as_posix()})
                document.setdefault("ocr details",[]).append({"page":number,"reason":"undecodable fonts" if unknown else "scan or forced OCR",
                    "lines":len(lines),"mean confidence":sum(n["score"] for n in lines)/len(lines),"cache hit":cached is not None})
            finally:
                page.close()
    if document.get("ocr pages"):
        document["ocr engine"] = OCR_VERSION
    return document


def preserve_ocr_regions(pdf_path: Path, document: dict, output_dir: Path) -> dict:
    """Retain whole low-confidence/math lines, instead of trusting damaged OCR."""
    nodes = list(_nodes(document)); pending = []
    page_fonts: dict[int, list[float]] = {}
    for node in nodes:
        if (node.get('type') == 'paragraph' and node.get('font') == 'OCR'
                and not node.get('ocr snapshot') and float(node.get('font size') or 0) > 0):
            page_fonts.setdefault(node['page number'], []).append(float(node['font size']))
    for node in nodes:
        if not node.get("ocr snapshot") or not node.get("content"):continue
        if pending:
            previous = pending[-1]; l,b,r,t = node["bounding box"]; a,c,d,e = previous["bounding box"]
            size = float(node.get("font size") or 9)
            # Join adjacent uncertain equation rows; a readable prose block
            # between them prevents joining through unrelated sentences.
            between = nodes[nodes.index(previous)+1:nodes.index(node)]
            if (node.get("type")==previous.get("type")=="paragraph" and node["page number"]==previous["page number"]
                    and min(r,d)>max(l,a) and -size*.15 <= c-t <= size*.9
                    and not any(n.get("content") for n in between)):
                previous["bounding box"]=list(_union((a,c,d,e),(l,b,r,t)))
                previous["content"] += "\n"+node["content"]
                node["content"]="";node["suppressed"]=True
                continue
        pending.append(node)
    if not pending: return document
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]
    asset_dir = output_dir/"figs"/digest; asset_dir.mkdir(parents=True,exist_ok=True)
    source_fonts: dict[int, float] = {}
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for node in pending:
            page_number = node["page number"]; page = pdf[page_number-1]
            try:
                width,height = page.get_size(); l,b,r,t = node["bounding box"]
                l,b,r,t = max(0,l-1),max(0,b-2),min(width,r+1),min(height,t+2)
                region_key = hashlib.sha256(json.dumps([node["bounding box"],node["content"]],ensure_ascii=False).encode()).hexdigest()[:10]
                destination = asset_dir/f"page-{page_number:03d}-ocr-region-{region_key}.png"
                bitmap = page.render(scale=3,crop=(l,b,width-r,height-t),draw_annots=False)
                image = bitmap.to_pil(); image.save(destination); image.close(); bitmap.close()
                ref = destination.relative_to(output_dir).as_posix()
                if page_number not in source_fonts:
                    sizes = [span.font_size for span in _spans(page) if 4 <= span.font_size <= 48]
                    source_fonts[page_number] = median(sizes) if sizes else 0
                size = (source_fonts[page_number] or
                        (median(page_fonts[page_number]) if page_fonts.get(page_number) else float(node.get('font size') or 9)))
                width_em = (r-l)/size
                height_em = (t-b)/size
                # PNGs are rendered at 216 DPI. Use source font geometry so
                # raster resolution does not enlarge prose in the reading view.
                node["formula snapshot"] = (
                    f'<img src="{escape(ref, quote=True)}" alt="Original notation or uncertain OCR" class="ocr-region" '
                    f'width="{max(1, round(width_em*16))}" height="{max(1, round(height_em*16))}" '
                    f'style="width:{width_em:.3f}em;max-width:100%;height:auto;" />')
                document.setdefault("ocr formula assets",[]).append({"source":ref,"page number":page_number,
                    "bounding box":[l,b,r,t],"block":True,"ocr":True})
            finally:
                page.close()
    return document
