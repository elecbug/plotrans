"""Preserve web links explicitly stored in PDF annotations."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from urllib.parse import quote, unquote, urlsplit

from pypdf import PdfReader


def web_target(value) -> str:
    if not isinstance(value, str) or any(ord(char) < 32 for char in value) or "\\" in value:
        return ""
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        _ = parsed.port
        return quote(value, safe=":/?#[]@!$&'*,;+%=~.-_")
    except ValueError:
        return ""


def read_source_links(pdf_path: Path) -> list[dict]:
    links = {}
    for number, page in enumerate(PdfReader(pdf_path).pages, 1):
        annotations = page.get("/Annots", [])
        annotations = annotations.get_object() if hasattr(annotations, "get_object") else annotations
        if not isinstance(annotations, (list, tuple)):
            continue
        for reference in annotations:
            annotation = reference.get_object()
            if not isinstance(annotation, dict) or annotation.get("/Subtype") != "/Link":
                continue
            action = annotation.get("/A")
            action = action.get_object() if action is not None else {}
            if not isinstance(action, dict) or action.get("/S") != "/URI":
                continue
            target = web_target(action.get("/URI"))
            if not target:
                continue
            link = links.setdefault(target, {"url": target, "pages": [], "rectangles": []})
            if number not in link["pages"]:
                link["pages"].append(number)
            box = annotation.get("/Rect")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    link["rectangles"].append({"page": number, "box": [float(value) for value in box]})
                except (TypeError, ValueError):
                    pass
    return list(links.values())


def link_kind(target: str) -> str:
    host = urlsplit(target).hostname
    if host in {"orcid.org", "www.orcid.org"}:
        return "ORCID"
    if host in {"doi.org", "dx.doi.org"}:
        return "DOI"
    return "Source link"


def source_printed_target(native: str, target: str) -> str:
    """Return an annotation's printed spelling only if source glyphs confirm it."""
    target = web_target(target)
    if not target:
        return ""
    # PDFium can expose a visible line-end hyphen as a discretionary marker.
    # The annotation must still confirm the complete target including that dash.
    native = re.sub(r"\s+", "", native.replace("\x02", "-").replace("\ufffe", "-"))
    printed = urlsplit(target).path.lstrip("/") if link_kind(target) == "DOI" else target
    for candidate in dict.fromkeys((printed, unquote(printed))):
        boundary = r"[A-Za-z0-9/_-]"
        if len(candidate) >= 8 and re.search(r"(?<!" + boundary + ")" + re.escape(candidate) + r"(?!" + boundary + ")", native):
            return candidate
    return ""


def restore_printed_targets(text: str, native: str, links: list[dict]) -> tuple[str, list[dict]]:
    """Repair only whitespace and omitted hyphens confirmed by source glyphs."""
    repairs = []
    for link in links:
        target = web_target(link.get("url"))
        candidate = source_printed_target(native, target)
        if not candidate:
            continue
        boundary = r"[A-Za-z0-9/_-]"
        pattern = r"\s*".join("-?" if char == "-" else re.escape(char) for char in candidate)
        matches = list(re.finditer(r"(?<!" + boundary + ")" + pattern + r"(?!" + boundary + ")", text))
        if len(matches) != 1 or matches[0].group() == candidate:
            continue
        match = matches[0]
        repairs.append({"before": match.group(), "after": candidate, "url": target})
        text = text[:match.start()] + candidate + text[match.end():]
    return text, repairs


def _key(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def assign_source_links(links: list[dict], units: list) -> tuple[list[dict], list[dict]]:
    """Attach only a uniquely matching printed target on its source page.

    An icon-only identifier remains source metadata. Ambiguous links stay in a
    separate list instead of being attributed to the wrong reference or author.
    """
    identifiers, unmatched = [], []
    for original in links:
        target = web_target(original.get("url"))
        if not target:
            continue
        link = dict(original, url=target)
        kind = link_kind(target)
        parsed = urlsplit(target)
        if kind == "ORCID" and re.fullmatch(r"/\d{4}-\d{4}-\d{4}-\d{3}[\dXx]/?", parsed.path):
            link["label"] = "ORCID " + parsed.path.strip("/")
            identifiers.append(link)
            continue
        needle = _key(unquote(parsed.path if kind == "DOI" else target))
        owners = [unit for unit in units if len(needle) >= 8
                  and unit.kind in {"text", "caption"}
                  and set(unit.pages).intersection(link.get("pages", []))
                  and needle in _key(unit.source)]
        link["label"] = kind
        if len(owners) == 1:
            for page in owners[0].pages:
                if printed := link.get("printed targets", {}).get(str(page)):
                    owners[0].source, repairs = restore_printed_targets(owners[0].source, printed, [link])
                    if repairs:
                        link["text repairs"] = repairs
                    break
            owners[0].links.append(link)
        else:
            unmatched.append(link)
    return identifiers, unmatched
