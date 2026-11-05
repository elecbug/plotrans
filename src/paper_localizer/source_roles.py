"""Recover front matter and caption roles from native source glyphs."""

from __future__ import annotations

import re
import unicodedata

from .figures import _union
from .formulas import _canonical, _inside
from .layout import caption_label


def _key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalpha())


def _person(text: str) -> bool:
    words = text.strip().split()
    if not 2 <= len(words) <= 6 or any(char.isdigit() or char in "@/&:;()" for char in text):
        return False
    if re.search(r"\b(?:university|institute|laboratory|department|college|school|computer|society|research|systems|network|networks|laboratories|eth|mit|usa|germany|switzerland|france|abstract)\b", text, re.I):
        return False
    particles = {"von", "van", "de", "del", "da", "di", "du", "der", "den", "la", "le"}
    return all(word in particles or word[0].isupper() and all(c.isalpha() or c in ".-'’" for c in word) for word in words)


def recover_authors(document: dict, spans) -> None:
    """Augment incomplete metadata only with names printed in the author band."""
    bounds = document.get("title bounds")
    if not bounds or not spans: return
    first_page = [n for n in document.get("kids", []) if n.get("page number") == 1]
    abstract_top = max((n["bounding box"][3] for n in first_page
                        if n.get("type") == "abstract" and n.get("bounding box")), default=0)
    if not abstract_top: return
    selected = [s for s in spans if abstract_top < (s.box[1]+s.box[3])/2 < bounds[1]
                and s.box[2]-s.box[0] > 0 and s.box[3]-s.box[1] < s.font_size*2]
    lines = []
    for span in sorted(selected, key=lambda s: (-s.baseline, s.box[0])):
        line = next((line for line in lines if abs(line[0].baseline-span.baseline) < 1.2), None)
        if line is None: lines.append([span])
        else: line.append(span)
    original = str(document.get("author") or "").strip()
    known = [part.strip(" []\"'") for part in re.split(r";|,|\s+and\s+", original) if part.strip(" []\"'")]
    known_keys = [_key(part) for part in known if _person(part.strip("*∗"))]
    names, evidence = [], []
    for line in lines:
        size = max(s.font_size for s in line)
        line = sorted((s for s in line if s.font_size >= size*.9), key=lambda s: s.box[0])
        groups = []
        for span in line:
            if not groups or span.box[0]-groups[-1][-1].box[2] > size*1.3:
                groups.append([span])
            else: groups[-1].append(span)
        found = []
        for group in groups:
            text = " ".join(s.text for s in group)
            text = re.sub(r"\s+([,.;])", r"\1", text)
            text = unicodedata.normalize("NFKC", text)
            found.extend(part.strip(" ,.*∗†‡") for part in re.split(r",|\s+and\s+", text))
        valid = [name for name in found if _person(name)]
        if not valid: continue
        # An existing author anchors the source row. With no usable metadata,
        # accept only standalone name groups in the title/abstract interval.
        if known_keys and not any(key in _key(" ".join(valid)) for key in known_keys): continue
        if not known_keys and len(valid) != len(found): continue
        names.extend(valid)
        evidence.append({"baseline": line[0].baseline, "names": valid,
                         "boxes": [list(s.box) for s in line]})
    names = list(dict.fromkeys(names))
    joined = _key(" ".join(names))
    if not names or known_keys and not all(key in joined for key in known_keys): return
    if known_keys and len(names) <= len(known_keys): return
    document["author"] = "; ".join(names)
    document["authors"] = names
    document["author recovery"] = {"original": original, "source": "native glyphs between title and abstract",
                                   "rows": evidence}


def split_captions(node: dict, spans, width: float) -> list[dict]:
    """Separate two captions combined across columns by the layout extractor."""
    text = node.get("content", "").strip()
    if not node.get("bounding box") or not caption_label(text): return [node]
    markers = list(re.finditer(r"(?<!\w)(?:Fig(?:ure)?\.?|Table)\s*(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.:]\s+", text, re.I))
    if len(markers) < 2 or markers[0].start() != 0: return [node]
    selected = sorted((s for s in spans if _inside(s.box, node["bounding box"])), key=lambda s: s.index)
    characters, owners = [], []
    for span in selected:
        key = _canonical(span.text)[0]
        characters.extend(key); owners.extend([span]*len(key))
    native = "".join(characters)
    result = []
    for i, marker in enumerate(markers):
        part = text[marker.start():markers[i+1].start() if i+1 < len(markers) else len(text)].strip()
        key = _canonical(part)[0];at = native.find(key)
        if not key or at < 0 or native.find(key, at+1) >= 0: return [node]
        glyphs = owners[at:at+len(key)];box = glyphs[0].box
        for glyph in glyphs[1:]: box = _union(box, glyph.box)
        item = dict(node, content=part, type="caption", **{"bounding box": list(box),
                    "source caption split": node.get("id"), "font": glyphs[0].font_name,
                    "font size": glyphs[0].font_size})
        item.pop("linked content id", None);item.pop("heading level", None)
        if i:item.pop("id", None)
        result.append(item)
    centers = [(n["bounding box"][0]+n["bounding box"][2])/2 for n in result]
    if not min(centers)<width/2<max(centers): return [node]
    if any(n["bounding box"][2]-n["bounding box"][0]>width*.6 for n in result): return [node]
    if any(abs(a["bounding box"][0]-b["bounding box"][0]) < width*.18
           or min(a["bounding box"][2],b["bounding box"][2])-max(a["bounding box"][0],b["bounding box"][0]) > 0
           for a,b in zip(result,result[1:])): return [node]
    return result
