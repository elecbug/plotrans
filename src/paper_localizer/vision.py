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
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request

import pypdfium2 as pdfium
from PIL import Image, ImageChops, ImageOps

from .figures import _PDFIUM_LOCK, _nodes, _area
from .markup import FIGURE_LINK, image_reference
from .tables import (TableError, parse_table, table_column_bounds, table_grid,
                     tables_agree, verify_table_text)
from .native_math import native_expression
from .math_fidelity import rendered_formula, source_symbols_agree, source_delimiters_agree


VISION_VERSION = "glm-crops-4x-6x-v2"
DEFAULT_VISION_MODEL = "glm-ocr:latest"
_LIMIT_CACHE_SECONDS = 24 * 60 * 60


class VisionOcrError(RuntimeError):
    pass


class _RegionTranscriptionError(VisionOcrError):
    """A model response failed for one crop; keep that source representation."""


def _cached_generation_limit(path, identity):
    try:
        cached = json.loads(path.read_text(encoding='utf-8'))
        age = time.time() - cached['created_at']
        return (cached.get('identity') == identity and cached.get('done_reason') == 'length'
                and 0 <= age < _LIMIT_CACHE_SECONDS)
    except (OSError,ValueError,TypeError,KeyError):
        return False


def _cache_generation_limit(path, identity, result):
    payload = {'identity':identity,'created_at':time.time(),'done_reason':'length',
               'eval_count':result.get('eval_count'),
               'reason':'The model reached its generation limit. Source representation retained.',
               'maximum_age_seconds':_LIMIT_CACHE_SECONDS}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix=path.stem+'-',suffix='.tmp',delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload,handle,ensure_ascii=False)
        temporary.replace(path)
    except OSError:
        # Cache availability must not change source-preserving OCR fallback.
        return
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
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
        try:
            text = rendered_formula(text)[0]
            return Counter(re.findall(r'[A-Za-z0-9]',unicodedata.normalize('NFKC',text)))
        except Exception:
            pass
        text = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\{[clr|\s]+\})?", "", text)
        text = re.sub(r"\\(log|ln|sin|cos|exp|Pr)\b", r"\1", text)
        text = re.sub(r"\\[A-Za-z]+", "", text)
    return Counter(re.findall(r"[A-Za-z0-9]", unicodedata.normalize('NFKC',text)))


def _semantic_latex(text: str) -> str:
    # TeX ignores whitespace between digits, but a text-only language model may
    # read `1 6` as two values. Preserve its mathematical value as `16`.
    text = re.sub(r"(?<=[0-9.])\s+(?=[0-9.])", "", text)
    def word(match):
        body=match.group(2).strip()
        if re.fullmatch(r"(?:[A-Za-z]\s+)+[A-Za-z]",body):body=re.sub(r"\s+","",body)
        return "\\"+match.group(1)+"{"+body+"}"
    return re.sub(r"\\(mathrm|text|mathit|operatorname)\s*\{([^{}]*)\}",word,text)


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


def _prose_words(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=[^\W\d_])-\s+(?=[^\W\d_])", "", text)
    text = re.sub(r"\\[a-z]+", " ", text)
    return [word for word in re.findall(r'[^\W\d_]+',text) if len(word)>=3]


def _native_prose_preserved(original: str, candidate: str) -> bool:
    """Two agreeing OCR responses cannot override readable native prose."""
    before, after = _prose_words(original), _prose_words(candidate)
    numbers = lambda text: Counter(re.findall(r"\d+(?:\.\d+)?", text))
    return bool(before) and before == after and numbers(original) == numbers(candidate)


def _native_crop_prose_preserved(spans, box, candidate):
    """Also protect prose covered by a snapshot beyond its layout node's text.

    Layout boxes can include the end of the preceding paragraph. Snapshot
    reconciliation removes that duplicate prose, so dropping the snapshot must
    account for these source words too. This check can only veto acceptance.
    """
    from .formulas import _inside, _MATH_FONT
    expected = Counter(word for span in spans
                       if span.font_name and not _MATH_FONT.search(span.font_name)
                       and _inside(span.box, box, padding=0)
                       for word in _prose_words(span.text))
    return not (expected - Counter(_prose_words(candidate)))


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


def _verify_table_image(page, box, rows, cache_dir: Path, identity: dict) -> dict:
    from .ocr import OCR_VERSION, _recognize

    width, height = page.get_size(); left, bottom, right, top = box
    attempts = []
    for scale in (3, 4.5):
        profile = {**identity, "ocr": OCR_VERSION, "box": box, "scale": scale,
                   "columns": len(table_grid(rows)[0]), "profile": "table-columns-v1"}
        key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        cache = cache_dir / f"table-cpu-{key}.json"
        cached = None
        if cache.is_file():
            try:
                candidate = json.loads(cache.read_text(encoding="utf-8"))
                if candidate.get("identity") == profile: cached = candidate
            except (ValueError, OSError): pass
        bitmap = page.render(scale=scale, crop=(left, bottom, width-right, height-top), draw_annots=False)
        image = bitmap.to_pil()
        try:
            lines = cached["lines"] if cached else _recognize(image)
            verification = verify_table_text(rows, lines)
            method = "whole table"
            if not verification["accepted"]:
                bounds = table_column_bounds(rows, lines, image)
                column_lines = cached.get("column_lines", []) if cached else []
                if not column_lines:
                    for column, (start, end) in enumerate(bounds):
                        crop = image.crop((start, 0, end, image.height))
                        try:
                            column_lines.extend({**line, "column": column} for line in _recognize(crop))
                        finally: crop.close()
                if column_lines:
                    candidate = verify_table_text(rows, column_lines)
                    if candidate["accepted"] or candidate["word_support"] > verification["word_support"]:
                        verification, method = candidate, "separate columns"
                cached = {"identity": profile, "lines": lines, "column_lines": column_lines, "bounds": bounds}
            else:
                cached = {"identity": profile, "lines": lines}
            cache.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            verification.update(scale=scale, method=method, independent_ocr_profile=OCR_VERSION)
            attempts.append(verification)
            if verification["accepted"]: return verification
        finally:
            image.close(); bitmap.close()
    result = max(attempts, key=lambda attempt: (not attempt["missing_numbers"], attempt["word_support"]))
    result["attempts"] = [dict(attempt) for attempt in attempts]
    return result


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
    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    digest = source_sha256[:12]
    cache_dir = output_dir / ".vision-cache" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    details, math_text, sources, table_text = [], {}, [], {}
    backend_unavailable = False

    def recognize(page, box, mode, scale):
        nonlocal backend_unavailable
        key_data = {**identity, "box": box, "page": page_number, "mode": mode, "scale": scale}
        if mode == "Table": key_data["table_profile"] = "borderless-crop-v1"
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        cache = cache_dir / (key + ".json")
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("identity") == key_data and isinstance(cached.get("text"),str) and cached["text"]:
                    return cached["text"]
            except (ValueError,OSError):
                pass
        if backend_unavailable:
            raise _RegionTranscriptionError("Vision OCR became unavailable; cached results and source representations retained.")
        width, height = page.get_size(); l,b,r,t = box
        padding = .7 if mode == "Text" else 0
        bitmap = page.render(scale=scale, crop=(max(0,l-padding),max(0,b-padding),max(0,width-r-padding),max(0,height-t-padding)), draw_annots=False)
        image = bitmap.to_pil(); bordered = ImageOps.expand(image, border=0 if mode == "Table" else 12, fill="white"); buffer = BytesIO()
        bordered.save(buffer, format="PNG"); bordered.close(); image.close(); bitmap.close()
        body = {
            "model": name, "prompt": mode + " Recognition:", "images": [base64.b64encode(buffer.getvalue()).decode()],
            "stream": False, "keep_alive": "5m",
            "options": {"temperature": 0, "seed": 0, "num_predict": 4096, "num_ctx": 8192},
        }
        failure_identity = {**key_data,'profile_failure':'generation-limit-v1',
                            'source_sha256':source_sha256,'image_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
                            'url':url.rstrip('/'),'prompt':body['prompt'],'options':body['options']}
        failure_cache = cache.with_suffix('.failed.json')
        if _cached_generation_limit(failure_cache,failure_identity):
            raise _RegionTranscriptionError(
                'Vision OCR previously reached its token limit for this unchanged crop; '
                'source retained. This failure is cached for up to 24 hours.')
        for attempt in range(2):
            try:
                result = request(url, "/api/generate", body, timeout=timeout)
                break
            except VisionOcrError as exc:
                if attempt:
                    backend_unavailable = True
                    raise _RegionTranscriptionError(f"{exc} after one retry.") from exc
        if not isinstance(result, dict) or not isinstance(result.get("response"), str):
            raise _RegionTranscriptionError("Vision OCR returned an invalid transcription.")
        text = result["response"].strip()
        if not text or result.get("done_reason") != "stop":
            if text and result.get('done_reason') == 'length':
                _cache_generation_limit(failure_cache,failure_identity,result)
            raise _RegionTranscriptionError("Vision OCR returned an empty or truncated transcription.")
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": key_data, "text": text}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
        return text

    from .math_markup import valid_math
    from .ocr import OcrError
    nodes = list(_nodes(document))
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        for page_number in range(1, len(pdf)+1):
            page = pdf[page_number-1]
            try:
                from .formulas import _spans, _inside
                from .source_text import normalized_spans
                page_spans = normalized_spans(_spans(page), document, page_number)
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
                        try:
                            candidate = recognize(page, box, "Text", 4)
                        except _RegionTranscriptionError as exc:
                            details.append({"page": page_number, "kind": "prose", "accepted": False,
                                            "box": box, "reason": f"{exc} Source retained.", "original": original})
                            continue
                        accepted = _prose_agrees(original, candidate) and valid_math(candidate)
                        reason = "prose and numeric checks passed" if accepted else "source disagreement; original retained"
                        second = ""
                        protected_numbers = (node.get("font") == "OCR" and node.get("ocr score", 0) >= .85
                                             and len(re.findall(r"\b[A-Za-z]{3,}\b", original)) >= 4
                                             and Counter(re.findall(r"\d+(?:\.\d+)?", original)) != Counter(re.findall(r"\d+(?:\.\d+)?", candidate)))
                        if not accepted and valid_math(candidate) and not protected_numbers:
                            try:
                                second = recognize(page, box, "Text", 6)
                            except _RegionTranscriptionError:
                                second = ""
                            accepted = valid_math(second) and _prose_consistent(candidate, second)
                            if accepted:
                                reason = "two resolutions agree; original OCR disagreed"
                        crop_prose_preserved = None
                        if accepted and node.get("font") != "OCR":
                            if not _native_prose_preserved(original, candidate):
                                accepted = False
                                reason = "native prose was omitted or adjacent prose was added; source retained"
                            else:
                                crop_prose_preserved = _native_crop_prose_preserved(page_spans, box, candidate)
                                if not crop_prose_preserved:
                                    accepted = False
                                    reason = "readable native words in the complete source crop were omitted; source retained"
                        details.append({"page": page_number, "kind": "prose", "accepted": accepted,
                                        "box": box, "reason": reason, "original": original,
                                        "candidate": candidate, "second": second,
                                        "native_crop_prose_preserved": crop_prose_preserved})
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
                    if expression := native_expression(page, box):
                        math_text[asset["source"]] = expression
                        details.append({"page": page_number, "kind": "formula", "source": asset["source"],
                                        "accepted": True, "reason": "source Unicode and character geometry", "first": expression})
                        continue
                    textpage = page.get_textpage()
                    try:
                        native = textpage.get_text_bounded(*box).strip()
                        native_mapping_valid = True
                        for index in range(textpage.count_chars()):
                            bounds = textpage.get_charbox(index)
                            if _area(bounds) and _inside(bounds, box, padding=0):
                                if (pdfium.raw.FPDFText_HasUnicodeMapError(textpage, index) != 0
                                        or pdfium.raw.FPDFText_GetUnicode(textpage, index) == 0):
                                    native_mapping_valid = False
                                    break
                    finally: textpage.close()
                    try:
                        first = _strip_math_fence(recognize(page, box, "Formula", 4))
                        second = _strip_math_fence(recognize(page, box, "Formula", 6))
                        escalation = ""
                        if _math_key(first) != _math_key(second):
                            third = _strip_math_fence(recognize(page, box, "Formula", 8))
                            if _math_key(second) == _math_key(third):
                                first, second = second, third
                                escalation = " (6x/8x retry)"
                    except _RegionTranscriptionError as exc:
                        details.append({"page": page_number, "kind": "formula", "source": asset["source"],
                                        "accepted": False, "reason": f"{exc} Source image retained."})
                        continue
                    accepted = _math_key(first) == _math_key(second) and valid_math("$$"+first+"$$")
                    native_available = bool(local) and all(span.font_name for span in local) and native_mapping_valid
                    native_agrees = not native_available or _latin_operands(native) == _latin_operands(first, latex=True)
                    symbols_agree = source_symbols_agree(native,first) if native_available else None
                    accepted = accepted and native_agrees and symbols_agree is not False
                    delimiters_agree = (source_delimiters_agree(output_dir/asset['source'],first)
                                        if accepted and symbols_agree is None else None)
                    accepted = accepted and delimiters_agree is not False
                    details.append({"page": page_number, "kind": "formula", "source": asset["source"],
                                    "accepted": accepted, "first": first, "second": second,
                                    "native_operands_agree": native_agrees,
                                    "native_operands_available": native_available,
                                    'native_symbols_available':symbols_agree is not None,
                                    'native_symbols_agree':symbols_agree,
                                    'source_delimiter_strokes_agree':delimiters_agree,
                                    "reason": (("two resolutions and native operands agree" if native_available else
                                                "two resolutions agree; native operands unavailable") + escalation
                                               if accepted else ('explicit native symbols or decimal values disagree; image retained'
                                                                 if symbols_agree is False else
                                                                 'OCR delimiters lack supporting source strokes; image retained'
                                                                 if delimiters_agree is False else
                                                                 "transcriptions or native operands disagree; image retained"))})
                    if accepted:
                        math_text[asset["source"]] = first
                for asset in document.get("figures", []):
                    if asset["page number"] != page_number or asset.get("asset kind") != "table": continue
                    detail = {"page": page_number, "kind": "table", "source": asset["source"], "accepted": False}
                    try:
                        candidates, failures, first = [], [], None
                        for scale in (3, 4.5, 6):
                            try:
                                candidate = parse_table(recognize(page, asset["bounding box"], "Table", scale))
                            except TableError as exc:
                                failures.append(str(exc)); continue
                            first = next((prior for prior in candidates if tables_agree(prior, candidate)), None)
                            if first is not None: break
                            candidates.append(candidate)
                        if first is None:
                            raise TableError("Table cells or structure were not confirmed at two resolutions. " + "; ".join(failures))
                        if not all(valid_math(cell["text"]) for row in first for cell in row):
                            raise TableError("Table transcription contains invalid equation markup.")
                        verification = _verify_table_image(page, asset["bounding box"], first, cache_dir, identity)
                        detail.update(accepted=verification["accepted"], verification=verification,
                                      shape=[len(table_grid(first)), len(table_grid(first)[0])],
                                      reason="rendering resolutions and independent OCR agree" if verification["accepted"]
                                      else "independent OCR did not confirm table text or numbers; source image retained")
                        if verification["accepted"]:
                            table_text[asset["source"]] = {"rows": first, "verification": verification}
                    except (TableError, VisionOcrError) as exc:
                        detail["reason"] = str(exc) + "; source image retained"
                    except ImportError:
                        detail["reason"] = 'Independent CPU OCR is unavailable. Install ".[ocr]" to verify table transcriptions; source image retained.'
                    except OcrError as exc:
                        detail["reason"] = f"Independent CPU OCR is unavailable: {exc}; source image retained"
                    details.append(detail)
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
                    or not detail.get("native_operands_agree",True)
                    or detail.get('native_symbols_agree') is False
                    or detail.get('source_delimiter_strokes_agree') is False):continue
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
    document["vision ocr"] = {**identity, "acceptance_profile": "native-prose-and-symbols-v3",
                              "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "regions": details}
    document["math transcriptions"] = {source:_semantic_latex(text) for source,text in math_text.items()}
    document["table transcriptions"] = table_text
    document["vision sources"] = sources
    active = {image_reference(match).source for node in nodes if not node.get("suppressed")
              for text in (node.get("content", ""), node.get("formula snapshot", ""))
              for match in FIGURE_LINK.finditer(text)}
    document["formulas"] = [asset for asset in document.get("formulas", []) if asset["source"] in active]
    return document
