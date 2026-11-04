"""Selective local document OCR with source crops and reproducible model identity."""

from __future__ import annotations

import base64
from collections import Counter
from difflib import SequenceMatcher
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

import pypdfium2 as pdfium
from PIL import Image, ImageChops, ImageOps

from .figures import _PDFIUM_LOCK, _nodes, _area
from .markup import FIGURE_LINK, image_reference


VISION_VERSION = "glm-crops-4x-6x-v2"
DEFAULT_VISION_MODEL = "glm-ocr:latest"


class VisionOcrError(RuntimeError):
    pass


def _request(url: str, path: str, body=None, timeout: int = 300):
    request = urllib.request.Request(url.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise VisionOcrError(f"Local vision OCR failed ({url}): {exc}") from exc


def _math_key(text: str) -> str:
    """Ignore TeX layout choices, but retain symbols, operands and exponents."""
    text = re.sub(r"\s+", "", text).strip("$")
    text = re.sub(r"\\(?:left|right|displaystyle|,|;|!)", "", text)
    text = re.sub(r"\\begin\{array\}\{[clr|]+\}", r"\\begin{array}", text)
    text = re.sub(r"\\(?:mathrm|text|mathit)\{([^{}]*)\}", r"\1", text)
    return text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def _strip_math_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("$$") and text.endswith("$$"):
        return text[2:-2].strip()
    if text.startswith("\\[") and text.endswith("\\]"):
        return text[2:-2].strip()
    return text.strip("$").strip()


def _latin_operands(text: str, latex: bool = False) -> Counter:
    if latex:
        text = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[clr|\s]+\})?", "", text)
        text = re.sub(r"\\(log|ln|sin|cos|exp|Pr)\b", r"\1", text)
        text = re.sub(r"\\[A-Za-z]+", "", text)
    return Counter(re.findall(r"[A-Za-z0-9]", text))


def _semantic_latex(text: str) -> str:
    # TeX ignores whitespace between digits, but a text-only language model may
    # read `1 6` as two values. Preserve its mathematical value as `16`.
    text = re.sub(r"(?<=[0-9.])\s+(?=[0-9.])", "", text)
    def word(match):
        body=match.group(2).strip()
        if re.fullmatch(r"(?:[A-Za-z]\s+)+[A-Za-z]",body):body=re.sub(r"\s+","",body)
        return "\\"+match.group(1)+"{"+body+"}"
    return re.sub(r"\\(mathrm|text|mathit)\s*\{([^{}]*)\}",word,text)


def _prose_agrees(original: str, candidate: str) -> bool:
    """Reject changed data and large prose omissions; this is not an accuracy score."""
    words = lambda s: Counter(re.findall(r"\b[A-Za-z]{3,}\b", s.lower()))
    before, after = words(original), words(candidate)
    if sum(before.values()) < 4:
        return False
    overlap = sum((before & after).values()) / sum(before.values())
    numbers = lambda s: Counter(re.findall(r"\d+(?:\.\d+)?", s))
    return overlap >= .72 and numbers(original) == numbers(candidate) and .4 <= len(candidate)/max(len(original),1) <= 3


def _prose_consistent(first: str, second: str) -> bool:
    from .math_markup import LATEX, expression
    first_math = [_math_key(expression(match)) for match in LATEX.finditer(first)]
    second_math = [_math_key(expression(match)) for match in LATEX.finditer(second)]
    if first_math != second_math:
        return False
    key = lambda text: _math_key(text).replace("$", "").lower()
    return (SequenceMatcher(None, key(first), key(second), autojunk=False).ratio() >= .985
            and Counter(re.findall(r"\d+(?:\.\d+)?", first)) == Counter(re.findall(r"\d+(?:\.\d+)?", second)))


def _formula_signature(path: Path):
    with Image.open(path) as source:
        ink = source.convert("L").point(lambda value: 255 if value < 180 else 0)
    bounds = ink.getbbox()
    if not bounds:
        ink.close()
        return None
    crop = ink.crop(bounds); ink.close()
    ratio = crop.width/crop.height
    signature = crop.resize((max(1,round(ratio*64)),64)).point(lambda value:255 if value>=128 else 0)
    crop.close()
    return signature,ratio


def _ink_overlap(first, second):
    other = second.resize(first.size).point(lambda value:255 if value>=128 else 0)
    intersection = ImageChops.darker(first,other); union = ImageChops.lighter(first,other)
    score = intersection.histogram()[255]/max(union.histogram()[255],1)
    other.close();intersection.close();union.close()
    return score


def _same_short_formula(first, second) -> bool:
    """Compare repeated short expressions per glyph, including vertical position."""
    a,ar=first;b,br=second
    if abs(ar/br-1)>.06 or not .5<ar<8 or _ink_overlap(a,b)<.88:
        return False
    def glyphs(image):
        occupied=[image.crop((x,0,x+1,image.height)).getbbox() is not None for x in range(image.width)]
        runs=[];start=None
        for x,value in enumerate(occupied+[False]):
            if value and start is None:start=x
            elif not value and start is not None:
                runs.append((start,0,x,image.height));start=None
        return runs
    ag,bg=glyphs(a),glyphs(b)
    if len(ag)!=len(bg):return False
    for ac,bc in zip(ag,bg):
        ca,cb=a.crop(ac),b.crop(bc)
        score=_ink_overlap(ca,cb);ca.close();cb.close()
        if score<.80:return False
    return True


def refine_document(pdf_path: Path, document: dict, output_dir: Path, *, model: str,
                    url: str, expected_digest: str = "", timeout: int = 300, requester=None) -> dict:
    """Refine OCR prose and supply semantic equation text to text-only models.

    Original equation images stay in the reading edition. Formula transcriptions
    require agreement at two *native rendering* resolutions, plus valid MathML.
    Agreement is a consistency check, not proof that a transcription is correct.
    """
    request = requester or _request
    tags = request(url, "/api/tags", timeout=timeout).get("models", [])
    name = model if ":" in model else model + ":latest"
    entry = next((m for m in tags if m.get("name") in {model, name}), None)
    if not entry or not entry.get("digest"):
        raise VisionOcrError(f"Vision OCR model is not installed. Run: ollama pull {model}")
    if expected_digest and entry["digest"].removeprefix("sha256:") != expected_digest.removeprefix("sha256:"):
        raise VisionOcrError(f"Vision OCR model digest differs from the requested version: {entry['digest']}")
    identity = {"profile": VISION_VERSION, "model": name, "digest": entry["digest"]}
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]
    cache_dir = output_dir / ".vision-cache" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    details, math_text, sources = [], {}, []

    def recognize(page, box, mode, scale):
        key_data = {**identity, "box": box, "page": page_number, "mode": mode, "scale": scale}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        cache = cache_dir / (key + ".json")
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("identity") == key_data and isinstance(cached.get("text"),str) and cached["text"]:
                    return cached["text"]
            except (ValueError,OSError):
                pass
        width, height = page.get_size(); l,b,r,t = box
        padding = .7 if mode == "Text" else 0
        bitmap = page.render(scale=scale, crop=(max(0,l-padding),max(0,b-padding),max(0,width-r-padding),max(0,height-t-padding)), draw_annots=False)
        image = bitmap.to_pil(); bordered = ImageOps.expand(image, border=12, fill="white"); buffer = BytesIO()
        bordered.save(buffer, format="PNG"); bordered.close(); image.close(); bitmap.close()
        result = request(url, "/api/generate", {
            "model": name, "prompt": mode + " Recognition:", "images": [base64.b64encode(buffer.getvalue()).decode()],
            "stream": False, "keep_alive": "5m",
            "options": {"temperature": 0, "seed": 0, "num_predict": 4096, "num_ctx": 8192},
        }, timeout=timeout)
        text = result.get("response", "").strip()
        if not text or result.get("done_reason") != "stop":
            raise VisionOcrError("Vision OCR returned an empty or truncated transcription; source retained. Increase the token limit or use a smaller region.")
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": key_data, "text": text}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
        return text

    from .math_markup import valid_math
    nodes = list(_nodes(document))
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for page_number in range(1, len(pdf)+1):
            page = pdf[page_number-1]
            try:
                from .formulas import _spans, _inside
                page_spans = _spans(page)
                for node in nodes:
                    if node.get("page number") != page_number or node.get("suppressed"):
                        continue
                    original = node.get("content", "")
                    box = node.get("bounding box")
                    if not original or not box or node.get("type") in {"heading", "header", "footer", "image", "picture", "code"}:
                        continue
                    # OCR pages also contain confident-looking flattened
                    # exponents. Refine prose even if RapidOCR's score is high.
                    if ((node.get("font") == "OCR" and (len(original.split()) >= 7 or box[2]-box[0] > page.get_size()[0]*.35))
                            or (node.get("formula snapshot") and len(re.findall(r"\b[A-Za-z]{3,}\b", original)) >= 4)):
                        candidate = recognize(page, box, "Text", 4)
                        accepted = _prose_agrees(original, candidate) and valid_math(candidate)
                        reason = "prose and numeric checks passed" if accepted else "source disagreement; original retained"
                        second = ""
                        protected_numbers = (node.get("font") == "OCR" and node.get("ocr score", 0) >= .85
                                             and len(re.findall(r"\b[A-Za-z]{3,}\b", original)) >= 4
                                             and Counter(re.findall(r"\d+(?:\.\d+)?", original)) != Counter(re.findall(r"\d+(?:\.\d+)?", candidate)))
                        if not accepted and valid_math(candidate) and not protected_numbers:
                            second = recognize(page, box, "Text", 6)
                            accepted = valid_math(second) and _prose_consistent(candidate, second)
                            if accepted:
                                reason = "two resolutions agree; original OCR disagreed"
                        details.append({"page": page_number, "kind": "prose", "accepted": accepted,
                                        "box": box, "reason": reason, "original": original,
                                        "candidate": candidate, "second": second})
                        if accepted:
                            snapshot = node.pop("formula snapshot", "")
                            if snapshot:
                                ref = image_reference(next(FIGURE_LINK.finditer(snapshot)))
                                sources.append({"page": page_number, "source": ref.source})
                            node["ocr original"] = original
                            node["content"] = candidate
                            node["vision refined"] = True
                            node["ocr snapshot"] = False
                for asset in document.get("formulas", []):
                    if asset["page number"] != page_number:
                        continue
                    # Whole prose snapshots are handled above, not treated as a
                    # single gigantic mathematical expression.
                    if any(n.get("formula snapshot") and asset["source"] in n["formula snapshot"]
                           and len(re.findall(r"\b[A-Za-z]{3,}\b", n.get("content", ""))) >= 4 for n in nodes):
                        continue
                    # Do not transcribe retired snapshots of refined prose.
                    active = any(asset["source"] in n.get("content", "") or asset["source"] in n.get("formula snapshot", "")
                                 for n in nodes if not n.get("suppressed"))
                    if not active:
                        continue
                    box = asset["bounding box"]
                    local = [span for span in page_spans if _inside(span.box, box, padding=0)]
                    textpage = page.get_textpage()
                    try:
                        native = textpage.get_text_bounded(*box).strip()
                        # Word spans may contain both a base and its subscript.
                        # Inspect the original character sizes before trusting
                        # native plain text such as `qz` as a flat expression.
                        sizes = [pdfium.raw.FPDFText_GetFontSize(textpage, index)
                                 for span in local for index in range(span.index, min(textpage.count_chars(), span.index+len(span.text)))
                                 if pdfium.raw.FPDFText_GetUnicode(textpage,index)>32]
                        uniform = bool(sizes) and min(sizes) >= max(sizes)*.9
                    finally: textpage.close()
                    # A decoded, single-line expression with a uniform baseline
                    # is stronger evidence than a model guess (e.g. `p > q`).
                    if (local and uniform and all(span.font_name for span in local) and "\n" not in native
                            and not re.search(r"[\ue000-\uf8ff\ufffd\x00-\x08]", native)
                            and min(span.font_size for span in local) >= max(span.font_size for span in local)*.9
                            and max(span.baseline for span in local)-min(span.baseline for span in local) < 1
                            and re.fullmatch(r"[A-Za-z0-9\s+*/=<>.,−-]+", native)):
                        math_text[asset["source"]] = native
                        details.append({"page":page_number,"kind":"formula","source":asset["source"],
                                        "accepted":True,"reason":"uniform-baseline native Unicode","first":native})
                        continue
                    first = _strip_math_fence(recognize(page, box, "Formula", 4))
                    second = _strip_math_fence(recognize(page, box, "Formula", 6))
                    escalation = ""
                    if _math_key(first) != _math_key(second):
                        third = _strip_math_fence(recognize(page, box, "Formula", 8))
                        if _math_key(second) == _math_key(third):
                            first, second = second, third
                            escalation = " (6x/8x retry)"
                    accepted = _math_key(first) == _math_key(second) and valid_math("$$"+first+"$$")
                    native_agrees = not local or not all(span.font_name for span in local) or (
                        _latin_operands(native) == _latin_operands(first, latex=True))
                    accepted = accepted and native_agrees
                    details.append({"page": page_number, "kind": "formula", "source": asset["source"],
                                    "accepted": accepted, "first": first, "second": second,
                                    "native_operands_agree": native_agrees,
                                    "reason": "two resolutions and native operands agree" + escalation if accepted else "transcriptions or native operands disagree; image retained"})
                    if accepted:
                        math_text[asset["source"]] = first
            finally:
                page.close()
    # Repeated short notation provides another independent observation. Only
    # compare an ambiguous candidate with an already accepted identical formula,
    # and require both whole-image and per-glyph similarity. No chaining.
    templates = [(source,text,_formula_signature(output_dir/source)) for source,text in math_text.items()
                 if len(_math_key(text))<=24 and (output_dir/source).is_file()]
    try:
        for detail in details:
            if (detail["kind"]!="formula" or detail["accepted"] or not detail.get("second")
                    or not detail.get("native_operands_agree",True)):continue
            candidates={_math_key(detail["first"]),_math_key(detail["second"])}
            signature=_formula_signature(output_dir/detail["source"])
            if not signature:continue
            try:
                match=next(((source,text) for source,text,template in templates if template
                            and _math_key(text) in candidates and _same_short_formula(signature,template)),None)
                if match:
                    source,text=match;math_text[detail["source"]]=text
                    detail.update(accepted=True,reason="repeated formula matches an accepted source image per glyph",template=source)
            finally:signature[0].close()
    finally:
        for _,_,template in templates:
            if template:template[0].close()
    document["vision ocr"] = {**identity, "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "regions": details}
    document["math transcriptions"] = {source:_semantic_latex(text) for source,text in math_text.items()}
    document["vision sources"] = sources
    active = {image_reference(match).source for node in nodes if not node.get("suppressed")
              for text in (node.get("content", ""), node.get("formula snapshot", ""))
              for match in FIGURE_LINK.finditer(text)}
    document["formulas"] = [asset for asset in document.get("formulas", []) if asset["source"] in active]
    return document
