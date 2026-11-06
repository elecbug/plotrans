"""Recover document structure from source geometry without generating text."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata
from statistics import median

import pypdfium2 as pdfium

from .figures import _PDFIUM_LOCK, _union, _nodes
from .formulas import _canonical, _inside, _spans, _MATH_FONT
from .markup import FIGURE_LINK
from .source_links import restore_printed_targets, source_printed_target
from .source_roles import recover_authors, split_captions
from .native_tables import recover_tables
from .review_layout import remove_line_counters
from .source_text import (positioned_accents, normalize_accents, native_vocabulary,
                          line_wraps, normalize_line_wraps, restore_local_spacing)
from .source_text import restore_prose_dashes
from .source_text import geometric_word_spaces, normalize_geometric_spaces
from .source_text import restore_prose_omissions


_NUMBERED_TITLE = re.compile(r"^(?:[IVXLCDM]+\.\s+|[A-Z]\.\s+|\d+(?:\.\d+)*[.)]?\s+)\S")
_LIST_MARKER = re.compile(r"^(?:[-•▪‣◦]\s+|\d+[.)]\s+|[a-z]\)\s+|\[\d+\]\s+)")
_ABSTRACT = re.compile(r"^(?:abstract\b|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t\b)", re.I)
_KEYWORDS = re.compile(r"^(?:index terms|keywords)\b", re.I)


def _flatten(document: dict) -> list[dict]:
    result = []

    def visit(node: dict, page: int = 1, suppressed: bool = False) -> None:
        page = node.get("page number", page)
        suppressed = suppressed or node.get("suppressed", False)
        children = [child for key in ("kids", "list items") for child in node.get(key, [])]
        if node.get("type") == "table":
            node["page number"] = page
            if suppressed: node["suppressed"] = True
            result.append(node)
            return
        if node.get("content") or node.get("formula snapshot") or node.get("type") in {"image", "picture", "formula"}:
            item = {key: value for key, value in node.items() if key not in {"kids", "list items"}}
            item["page number"] = page
            if suppressed: item["suppressed"] = True
            result.append(item)
        for child in children:
            if isinstance(child, dict): visit(child, page, suppressed)

    for node in document.get("kids", []):
        visit(node)
    return result


def _source_lines(node: dict, spans) -> list[list]:
    box = node.get("bounding box")
    if not box: return []
    lines = []
    for span in sorted((s for s in spans if _inside(s.box, box, padding=.8)),
                       key=lambda s: (-s.baseline, s.box[0])):
        line = next((line for line in lines if abs(line[0].baseline - span.baseline) < 1.5), None)
        if line is None: lines.append([span])
        else: line.append(span)
    return [sorted(line, key=lambda s: s.box[0]) for line in lines]


def _line_text(line) -> str:
    text = " ".join(span.text for span in line)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    return re.sub(r"\s*-\s*", "-", text)


def _is_title(text: str, font_name: str = "", outline: bool = False) -> bool:
    letters = "".join(char for char in text if char.isalpha())
    capitals = bool(len(letters) >= 3 and letters.isupper() and (
        _NUMBERED_TITLE.match(text) or re.sub(r"\s+", "", text) in {"REFERENCES", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"}))
    mixed = bool(re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", text)
                 and (re.search(r"Bold|Medi|Semibold|TB$", font_name, re.I)
                      or (outline or re.search(r'Italic|Oblique|Ital|It$',font_name))
                      and re.match(r"^\d+(?:\.\d+)+\.?\s",text))
                 and len(text.split()) <= 15 and not text.endswith((".", ";")))
    return capitals or mixed


def _heading_level(text: str, font_name: str = "") -> int:
    if re.sub(r"\s+", "", text).upper() in {"REFERENCES", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"}: return 2
    if re.match(r"^[IVXLCDM]+\.\s", text) and not re.search(r"Italic|Oblique|It$", font_name, re.I): return 2
    if re.match(r"^[A-Z]\.\s", text): return 3
    if match := re.match(r"^(\d+(?:\.\d+)+)\.?\s", text): return min(6,len(match[1].split('.'))+1)
    if re.match(r"^\d+[.)]?\s", text): return 2
    return 4


def _line_box(lines) -> list[float]:
    boxes = [span.box for line in lines for span in line]
    box = boxes[0]
    for other in boxes[1:]: box = _union(box, other)
    return list(box)


def _title_characters(text):
    key,offsets=_canonical(text)
    characters,positions=[],[]
    for char,offset in zip(key,offsets):
        for part in unicodedata.normalize('NFD',char):
            if unicodedata.category(part) not in {'Mn','Mc','Me','Sk'}:
                characters.append(part);positions.append(offset)
    return ''.join(characters),positions


def _split_titles(node: dict, spans) -> list[dict]:
    text = node.get("content", "").strip()
    if not text or node.get("type") in {"table", "caption", "image", "picture", "header", "footer"}:
        return [node]
    lines = _source_lines(node, spans)
    outline = node.get("pdfua_tag") == "LI"
    if lines:
        node["source first line indent"] = lines[0][0].box[0] - node["bounding box"][0]
    if not lines: return [node]
    if not _is_title(_line_text(lines[0]), lines[0][0].font_name, outline):
        source_key = _title_characters(text)[0]
        start = next((i for i,line in enumerate(lines)
                      if _is_title(_line_text(line), line[0].font_name, outline)
                      and source_key.startswith(_title_characters(_line_text(line))[0])), None)
        if start is None: return [node]
        # Layout bounds can include the tail of the preceding paragraph. That
        # neighboring line is not part of a source-matched heading's content.
        lines = lines[start:]
    source, offsets = _title_characters(text)
    cursor, line_at, result = 0, 0, []
    while line_at < len(lines) and _is_title(_line_text(lines[line_at]), lines[line_at][0].font_name, outline):
        first, end = line_at, line_at + 1
        while end < len(lines):
            candidate = _line_text(lines[end])
            letters = "".join(c for c in candidate if c.isalpha())
            if _NUMBERED_TITLE.match(candidate): break
            if not letters.isupper() and not re.search(r"Bold|Medi|Semibold|TB$", lines[first][0].font_name, re.I): break
            if lines[end][0].font_name != lines[first][0].font_name: break
            end += 1
        title = " ".join(_line_text(line) for line in lines[first:end])
        key = _title_characters(title)[0]
        if not source[cursor:].startswith(key):
            return [node]  # A title is promoted only when the source text agrees.
        start_at = offsets[cursor]
        cursor += len(key)
        end_at = offsets[cursor - 1] + 1
        heading_text = re.sub(r"^(\d+(?:\.\d+)*)(?=[A-Z])", r"\1 ", text[start_at:end_at].strip())
        item = dict(node, type="heading", content=heading_text,
                    **{"heading level": _heading_level(title, lines[first][0].font_name),
                       "bounding box": _line_box(lines[first:end])})
        item.pop("heading split", None)
        result.append(item)
        line_at = end
    if cursor < len(source):
        body = dict(node, type="paragraph", content=text[offsets[cursor]:].strip())
        body.pop("heading level", None)
        body.pop("heading split", None)
        if line_at < len(lines):
            body["bounding box"] = _line_box(lines[line_at:])
            body["source first line indent"] = lines[line_at][0].box[0] - body["bounding box"][0]
            body["font"] = lines[line_at][0].font_name
            body["font size"] = lines[line_at][0].font_size
        result.append(body)
    return result or [node]


def _split_abstract_keywords(node: dict, spans) -> list[dict]:
    text=node.get('content','')
    if node.get('page number')!=1 or not _ABSTRACT.match(text): return [node]
    match=re.search(r'\b(?:keywords|index terms)\s*[:—-]',text,re.I)
    if not match: return [node]
    lines=_source_lines(node,spans)
    split=next((i for i,line in enumerate(lines) if _KEYWORDS.match(_line_text(line))),None)
    if split is None or split==0: return [node]
    tail=' '.join(_line_text(line) for line in lines[split:])
    if _canonical(tail)[0]!=_canonical(text[match.start():])[0]: return [node]
    abstract=dict(node,content=text[:match.start()].strip(),**{'bounding box':_line_box(lines[:split])})
    keywords=dict(node,type='keywords',content=text[match.start():].strip(),
                  **{'bounding box':_line_box(lines[split:])})
    keywords.pop('id',None)
    return [abstract,keywords]


def _front_matter(document: dict) -> None:
    nodes = [n for n in document["kids"] if n.get("page number") == 1 and n.get("content")]
    width, height = document["page sizes"]["1"]
    for node in nodes:
        box = node.get("bounding box")
        if box and box[2]-box[0] < width*.08 and box[3]-box[1] > height*.25:
            node["type"] = "frontmatter"
            node.pop("heading level", None)
    abstract = next((n for n in nodes if _ABSTRACT.match(n["content"])), None)
    introduction = next((n for n in nodes if re.match(r"^(?:1\s*\.?\s*|I\.\s*)Introduction\b", n["content"], re.I)), None)
    if not abstract and not introduction: return
    title_key = re.sub(r"\W", "", str(document.get("title") or "")).casefold()
    candidates = [n for n in nodes if n.get("bounding box") and len(n["content"]) > 20
                  and n["bounding box"][2]-n["bounding box"][0] > width*.4
                  and not _ABSTRACT.match(n["content"])
                  and (not abstract or n["bounding box"][1] > abstract["bounding box"][3])]
    title = next((n for n in candidates if title_key and re.sub(r"\W", "", n["content"]).casefold() == title_key), None)
    if title is None:
        title = max(candidates, key=lambda n: float(n.get("font size") or 0), default=None)
    if title is None: return
    title["type"], title["heading level"] = "heading", 1
    if not title_key or str(document.get("title", "")).lower().endswith((".dvi", ".ps", ".pdf")):
        document["title"] = title["content"]
    if not abstract and introduction:
        intro_top = introduction["bounding box"][3]
        prose = [n for n in nodes if n.get("bounding box") and n is not title
                 and n["bounding box"][1] > intro_top and n["bounding box"][3] < title["bounding box"][1]
                 and len(n["content"].split()) >= 45
                 and not re.match(r"CCS Concepts|Additional Key|ACM Reference|Authors.? addresses", n["content"], re.I)]
        abstract = max(prose, key=lambda n: n["bounding box"][3], default=None)
    if not abstract: return
    abstract_top = abstract["bounding box"][3]
    abstract_box = abstract["bounding box"]
    if _ABSTRACT.fullmatch(abstract['content'].strip()):
        # A centered standalone label can be much narrower than its prose
        # column. Classify the following prose using that column's left edge.
        center = (abstract_box[0]+abstract_box[2])/2
        following = [n for n in nodes if n is not abstract and n.get('bounding box')
                     and n.get('type') in {'paragraph','list item','abstract'}
                     and len(n.get('content','').split())>=45
                     and 0<=abstract_box[1]-n['bounding box'][3]<=max(36,float(abstract.get('font size') or 10)*3)
                     and n['bounding box'][0]<=center<=n['bounding box'][2]]
        if following:
            abstract_box = max(following,key=lambda n:n['bounding box'][3])['bounding box']
    boundaries = [n["bounding box"][3] for n in nodes if n.get("bounding box") and n is not abstract
                  and (n.get("type") == "heading" or _KEYWORDS.match(n["content"])
                       or re.match(r"CCS Concepts|Additional Key|ACM Reference", n["content"], re.I))
                  and n["bounding box"][3] < abstract_top
                  and n["bounding box"][0] < abstract_box[2]+10 and n["bounding box"][2] > abstract_box[0]-10]
    abstract_bottom = max(boundaries, default=height*.2)
    for node in nodes:
        box = node.get("bounding box")
        if not box or node is title or node.get('type') in {'header','footer'}: continue
        aligned = abs(box[0]-abstract_box[0]) < 15
        if (node is abstract or aligned and box[3] <= abstract_top+2 and box[1] > abstract_bottom
                and node.get("type") not in {"frontmatter", "footer", "header"}
                and not _KEYWORDS.match(node["content"])
                and not re.match(r"CCS Concepts|Additional Key|ACM Reference", node["content"], re.I)):
            node["type"] = "abstract"
            node["content"] = _ABSTRACT.sub("", node["content"], count=1).lstrip(" :.—-\n")
            node.pop("heading level", None)
    title_bottom = title["bounding box"][1]
    authors = []
    for node in nodes:
        box = node.get("bounding box")
        if not box or node is title: continue
        if box[1] > abstract_top:
            if (node.get("type") == "heading" and box[3] < title_bottom
                    and not _NUMBERED_TITLE.match(node["content"])):
                authors.append(node["content"])
            node["type"] = "frontmatter"
            node.pop("heading level", None)
        elif _KEYWORDS.match(node["content"]):
            node["type"] = "keywords"
    if not document.get("author") and authors:
        document["author"] = " ".join(authors)
    document["title bounds"] = title["bounding box"]
    sizes = [n["font size"] for n in nodes if n.get("font size") and n.get("type") == "paragraph"
             and n.get("bounding box", [0, 0, 0, 0])[3] <= abstract_top]
    body_size = median(sizes) if sizes else 10
    for node in nodes:
        box = node.get("bounding box")
        if not box or node.get("type") in {"abstract", "heading", "caption", "table", "header", "footer"}: continue
        if (box[2]<abstract_box[0]-4 and re.search(r'Received:|Revised:|Academic Editor:|Copyright:',node['content'],re.I)):
            node['type']='frontmatter'
            continue
        if re.match(r"^(?:[-•■▪‣]\s*)?(?:Fig(?:ure)?\.?|Table)\s*(?:\d|[IVXLCDM]+\b)|^\[\d+\]\s", node["content"], re.I): continue
        if (box[3] < height*.30 and float(node.get("font size") or body_size) < body_size*.94
                and len(node["content"].split()) > 4):
            node["type"] = "footnote"
        if box[3] > height * .12: continue
        text = node["content"]
        if re.fullmatch(r"\d{1,8}", text.strip()) or re.fullmatch(r"VOLUME\s+\d+,?\s+\d{4}", text.strip(), re.I):
            node["type"] = "footer"
        elif re.search(r"copyright|licensed under|creative commons", text, re.I):
            node["type"] = "frontmatter"
        elif float(node.get("font size") or body_size) < body_size * .86 and len(text.split()) > 6:
            node["type"] = "footnote"


def _restore_source_spacing(node: dict, textpage) -> bool:
    """Use native spacing only when every printable source character agrees."""
    text, box = node.get("content", ""), node.get("bounding box")
    if not text or not box or node.get("type") in {"table", "image", "picture"}: return False
    native = textpage.get_text_bounded(*box).strip()
    if native.endswith(("\x02", "\ufffe")) and text.rstrip().endswith("-"):
        native = native[:-1] + "-"
    native = native.replace("\ufffe", "").replace("\x02", "")
    if any(not char.isprintable() and not char.isspace() for char in native): return False
    if not native: return False
    source_key, text_key = _canonical(native)[0], _canonical(text)[0]
    if source_key != text_key:
        pattern = "".join("[–—−-]?" if char in "–—−-" and 0 < i < len(source_key)-1
                          and source_key[i-1].isdigit() and source_key[i+1].isdigit()
                          else re.escape(char) for i,char in enumerate(source_key))
        if not re.fullmatch(pattern, text_key): return False
    native = re.sub(r"(?<=\d)([–—−-])\s+(?=\d)", r"\1", native)
    native = " ".join(native.split())
    if native == text: return False
    node["content"] = native
    return True


def _repair_table_cells(table: dict, textpage) -> None:
    """Redistribute source-confirmed header text into its actual cell boxes."""
    cells = [cell for row in table.get("rows", []) for cell in row.get("cells", [])]
    if not cells or any(not cell.get("bounding box") for cell in cells): return
    old = [" ".join(n.get("content", "") for n in _nodes(cell) if n.get("content")) for cell in cells]
    native = [" ".join(textpage.get_text_bounded(*cell["bounding box"]).split()) for cell in cells]
    if any("\ufffd" in text or any(not char.isprintable() for char in text) for text in native): return
    if Counter(_canonical("".join(old))[0]) != Counter(_canonical("".join(native))[0]): return
    if old == native: return
    for cell, text in zip(cells, native):
        template = next((n for n in _nodes(cell) if n.get("content")), {})
        cell["kids"] = [dict(template, type="paragraph", content=text,
                             **{"bounding box": cell["bounding box"], "page number": table["page number"]})] if text else []
    table["source cell text repaired"] = True


def _mark_page_furniture(document: dict) -> None:
    groups = defaultdict(list)
    total_pages = len(document.get("page sizes", {}))
    for node in document["kids"]:
        text = node.get("content", "").strip()
        box = node.get("bounding box");page = node.get("page number", 1)
        if not box or not re.fullmatch(r"\d{1,12}", text) or node.get("type") in {"caption", "table"}: continue
        width, height = document["page sizes"].get(str(page), [612, 792])
        side = "bottom" if box[3] < height*.08 else "top" if box[1] > height*.92 else ""
        if not side: continue
        key = side, round((box[0]+box[2])/2/width*40), round(box[1]/height*50)
        groups[key].append((page, text, node))
    minimum = max(3, (total_pages+3)//4)
    furniture = []
    for group in groups.values():
        if len({page for page, _, _ in group}) < minimum: continue
        patterns = defaultdict(list)
        for page, text, node in group:
            patterns["plain", int(text)-page].append(node)
            marker = str(page)
            if text.endswith(marker) and len(text) > len(marker):
                patterns["suffix", int(text[:-len(marker)])-page].append(node)
            if text.startswith(marker) and len(text) > len(marker):
                patterns["prefix", int(text[len(marker):])-page].append(node)
        for pattern, candidates in patterns.items():
            if len({n["page number"] for n in candidates}) < minimum: continue
            for node in candidates:
                if node.get("type") == "footer" or node.get("type") == "header": continue
                node["type"] = "footer" if group[0][2]["bounding box"][3] < document["page sizes"][str(group[0][0])][1]*.08 else "header"
                node["page furniture reason"] = "sequential page label in a repeated margin position"
                furniture.append({"page": node["page number"], "text": node["content"], "box": node["bounding box"]})
    document["page furniture"] = furniture
    recovered_margins = defaultdict(list)
    for node in document["kids"]:
        if not node.get("source text recovered"): continue
        box = node["bounding box"];page = node["page number"]
        width, height = document["page sizes"][str(page)]
        side = "footer" if box[3] < height*.09 else "header" if box[1] > height*.91 else ""
        if not side or box[3]-box[1] > height*.08: continue
        words = re.findall(r"[a-z]{3,}", node["content"].lower())
        key = side, round(box[1]/height*50), tuple(words[-12:])
        recovered_margins[key].append(node)
    for (side, _, words), candidates in recovered_margins.items():
        if (len(words) < 3 or len(''.join(words)) < 18
                or len({n["page number"] for n in candidates}) < minimum): continue
        for node in candidates:
            node["type"] = side
            node["page furniture reason"] = "repeated native text in the same outer margin"
            furniture.append({"page": node["page number"], "text": node["content"], "box": node["bounding box"]})
    repeated_text=defaultdict(list)
    for node in document['kids']:
        box=node.get('bounding box');page=node.get('page number',1)
        if not box or not node.get('content'): continue
        height=document['page sizes'][str(page)][1]
        if not (box[3]<height*.09 or box[1]>height*.91): continue
        text=re.sub(r'\b\d+\s+of\s+\d+\b','',node['content'],flags=re.I)
        words=tuple(re.findall(r'[a-z]{2,}',text.casefold()))
        if len(words)>=6:repeated_text[words].append(node)
    for candidates in repeated_text.values():
        if len({n['page number'] for n in candidates})<minimum:continue
        for node in candidates:
            height=document['page sizes'][str(node['page number'])][1]
            node['type']='footer' if node['bounding box'][3]<height*.09 else 'header'
            node['page furniture reason']='repeated source text in an outer page margin'


def _recover_uncovered_prose(nodes: list[dict], spans, textpage, width, height, page_number,
                             known_labels=()) -> list[dict]:
    """Recover complete native prose lines omitted by the layout extractor."""
    boxes = [n["bounding box"] for n in nodes if n.get("bounding box")]
    sizes = Counter(round(s.font_size,1) for s in spans if len(s.text)>2 and not s.math)
    body_size = sizes.most_common(1)[0][0] if sizes else 0
    lines = []
    for span in sorted(spans, key=lambda s: (-s.baseline, s.box[0])):
        line = next((line for line in reversed(lines[-4:])
                     if abs(line[0].baseline-span.baseline) < 1.5
                     and -2 <= span.box[0]-line[-1].box[2] < max(25, span.font_size*3)), None)
        if line is None: lines.append([span])
        else: line.append(span)
    missing, labels, role_headings = [], [], []
    for line in lines:
        box = _line_box([line])
        if box[1] < height*.025 or box[3] > height*.98: continue
        if any(s.math for s in line): continue
        if any(any(_inside(s.box, b, padding=.8) for b in boxes) for s in line): continue
        text = " ".join(textpage.get_text_bounded(*[box[0]-.2,box[1]-.2,box[2]+.2,box[3]+.2]).split())
        if not text or "\ufffd" in text: continue
        if re.fullmatch(r"[\d\W]+", text): continue
        role = _canonical(text)[0].casefold()
        if (page_number==1 and role in {'abstract','introduction'}
                and max(s.font_size for s in line)>=body_size*1.05
                and all(s.font_name and not s.math and not _MATH_FONT.search(s.font_name) for s in line)):
            center = (box[0]+box[2])/2
            following = [n for n in nodes if n.get('bounding box') and not n.get('suppressed')
                         and n.get('type') in {'paragraph','list item','abstract'}
                         and len(n.get('content','').split())>=(45 if role=='abstract' else 12)
                         and 0<=box[1]-n['bounding box'][3]<=max(36,max(s.font_size for s in line)*3)
                         and n['bounding box'][0]<=center<=n['bounding box'][2]]
            if following:
                role_headings.append({'type':'heading','heading level':2,'content':text,
                                      'bounding box':list(box),'page number':page_number,
                                      'font':line[0].font_name,'font size':max(s.font_size for s in line),
                                      'source role heading recovered':True})
                continue
        key = (_canonical(text)[0],line[0].font_name,round(line[0].font_size,1))
        if key in known_labels and all(s.font_name==line[0].font_name for s in line):
            labels.append({'type':'paragraph','content':text,'bounding box':list(box),
                           'page number':page_number,'font':line[0].font_name,
                           'font size':line[0].font_size,'source label recovered':True})
        missing.append((box, text, line))
    recovered, group = [], []
    def flush():
        if not group: return
        text = " ".join(item[1] for item in group)
        box = _line_box([s for _,_,s in group])
        # Layout extraction can omit the short final line of a numbered
        # footnote. Recovering only long standalone paragraphs misses it.
        # Require a directly adjacent, small-font numbered note whose whole
        # existing text agrees with the native source before appending it.
        for node in nodes:
            old_box, old = node.get('bounding box'), node.get('content','')
            size = float(node.get('font size') or 0)
            if (not old_box or node.get('suppressed') or node.get('type') not in {'paragraph','footnote'}
                    or old_box[3] > height*.3 or not re.match(r'^\d{1,3}\s*[A-Za-z]',old)
                    or not size or size >= body_size*.94 or abs(size-group[0][2][0].font_size) > size*.12
                    or not 0 <= old_box[1]-box[3] < size*1.2
                    or abs(old_box[0]-box[0]) > size
                    or box[2] > old_box[2]+size or re.search(r'[.!?]["”\u2019]?$',old.rstrip())):
                continue
            native = textpage.get_text_bounded(*old_box).replace('\x02','').replace('\ufffe','')
            if _canonical(native)[0] != _canonical(old)[0]:
                continue
            node.setdefault('source note continuations',[]).append({'text':text,'box':list(box),
                                                                   'previous box':list(old_box)})
            node['content'] = old.rstrip()+' '+text
            node['bounding box'] = list(_union(old_box,box))
            return
        minimum=4 if box[3]<height*.09 or box[1]>height*.91 else 7
        if any(_inside(label['bounding box'],box,padding=.5) for label in labels):
            minimum=2
        if len(re.findall(r"\b[A-Za-z]{2,}\b", text)) < minimum: return
        if not re.search(r"[.!?,;:]", text): return
        recovered.append({"type": "paragraph", "content": text, "bounding box": box,
                          "page number": page_number, "font": group[0][2][0].font_name,
                          "font size": group[0][2][0].font_size, "source text recovered": True})
    for item in missing:
        if group and (abs(item[0][0]-group[-1][0][0]) > 20
                      or group[-1][0][1]-item[0][3] > max(22, item[2][0].font_size*2.3)):
            flush();group=[]
        group.append(item)
    flush()
    # Labels supplement normal prose recovery. Removing a label from its
    # native line group first would strand short following values such as
    # "Unresolved." and cause a new omission at a page boundary.
    recovered.extend(label for label in labels
                     if not any(_inside(label['bounding box'],node.get('bounding box'),padding=.5)
                                for node in recovered+nodes if node.get('bounding box')))
    return recovered+role_headings


def _join_drop_caps(nodes: list[dict]) -> None:
    for cap in nodes:
        text, box = cap.get("content", ""), cap.get("bounding box")
        if cap.get("type") not in {'heading','paragraph'} or not re.fullmatch(r"[A-Z]", text) or not box: continue
        following = next((n for n in nodes if n is not cap and n.get("bounding box")
                          and cap.get('type')=='heading'
                          and 0 <= n["bounding box"][0]-box[2] <= 4
                          and abs(n["bounding box"][3]-box[3]) < 4
                          and re.match(r"^[A-Z]{2,}[- ]", n.get("content", ""))), None)
        if following is None and cap.get('page number')==1:
            size = float(cap.get('font size') or 0)
            from .source_text import _known_inflection
            following = next((n for n in nodes if n is not cap and not n.get('suppressed')
                              and n.get('type')=='paragraph' and n.get('bounding box')
                              and (body_size:=float(n.get('font size') or 0))>0 and size>=body_size*2.5
                              and not re.search(r'Math|Symbol|Italic|Oblique|Courier|Mono',cap.get('font') or '',re.I)
                              and 0<=n['bounding box'][0]-box[2]<=max(4,body_size*.6)
                              and 0<=box[3]-n['bounding box'][3]<=body_size*2.5
                              and box[1]<=n['bounding box'][1]
                              and (word:=re.match(r'^[a-z]{2,}',n.get('content','')))
                              and _known_inflection(text+word[0])),None)
        if following:
            following['source drop cap'] = {'text':text,'box':list(box)}
            following["content"] = text+following["content"]
            following["bounding box"] = list(_union(box, following["bounding box"]))
            cap["suppressed"] = True


def repair_structure(pdf_path: Path, document: dict) -> dict:
    """Split source-confirmed run-in headings and preserve front matter roles."""
    nodes = _flatten(document)
    by_page = defaultdict(list)
    for node in nodes:
        by_page[node.get("page number", 1)].append(node)
    repaired, sizes, spacing_repairs, link_repairs, front_spans = [], {}, [], [], []
    with _PDFIUM_LOCK, pdfium.PdfDocument(pdf_path) as pdf:
        page_spans = {}
        for index in range(len(pdf)):
            page = pdf[index]
            sizes[str(index+1)] = list(page.get_size())
            if index + 1 in document.get('ocr pages', []):
                page_spans[index + 1] = []
                page.close()
                continue
            try:
                page_spans[index + 1] = _spans(page)
            finally:
                page.close()
        vocabulary = native_vocabulary(page_spans.values())
        label_counts = Counter()
        for node in nodes:
            text,box = node.get('content',''),node.get('bounding box')
            font,size = node.get('font') or '',float(node.get('font size') or 0)
            height = sizes.get(str(node.get('page number',1)),[612,792])[1]
            if (box and box[3] < height*.88 and not node.get('suppressed')
                    and node.get('type') in {'paragraph','heading'}
                    and re.search(r'Bold|Demi|Heavy',font,re.I)
                    and re.fullmatch(r'[A-Za-z][A-Za-z -]{1,48}:?',text.strip())):
                label_counts[(_canonical(text.strip())[0],font,round(size,1))] += 1
        known_labels = {key for key,count in label_counts.items() if count>=2}
        for page_number in range(1, len(pdf) + 1):
            page = pdf[page_number - 1]
            try:
                sizes[str(page_number)] = list(page.get_size())
                spans = page_spans.pop(page_number)
                if page_number == 1: front_spans = spans
                textpage = page.get_textpage()
                try:
                    links = [link for link in document.get("source links", []) if page_number in link.get("pages", [])]
                    if spans and links:
                        native_page = textpage.get_text_bounded()
                        for link in links:
                            if printed := source_printed_target(native_page, link.get("url")):
                                link.setdefault("printed targets", {})[str(page_number)] = printed
                    page_nodes = by_page.get(page_number, [])
                    if spans:
                        spans = remove_line_counters(page_nodes,spans,page.get_size()[0],page_number,document)
                    accents = positioned_accents(page, spans) if spans else []
                    wraps = line_wraps(page, spans, vocabulary) if spans else []
                    word_spaces = geometric_word_spaces(spans) if spans else []
                    _join_drop_caps(page_nodes)
                    if spans:
                        page_nodes.extend(_recover_uncovered_prose(page_nodes, spans, textpage,
                                                                   *page.get_size(),page_number,known_labels))
                        page_nodes = [part for node in page_nodes for part in split_captions(node,spans,page.get_size()[0])]
                        page_nodes.extend(recover_tables(page,page_nodes,spans,textpage,page_number,document))
                    for node in page_nodes:
                        if node.get("type") == "table":
                            _repair_table_cells(node, textpage)
                        if spans and not node.get("suppressed") and _restore_source_spacing(node, textpage):
                            spacing_repairs.append({"page": page_number, "bounding box": node["bounding box"]})
                        if spans and not node.get("suppressed") and node.get("content") and node.get("bounding box"):
                            if links:
                                native = textpage.get_text_bounded(*node["bounding box"])
                                node["content"], repairs = restore_printed_targets(node["content"], native, links)
                                link_repairs.extend(dict(repair, page=page_number) for repair in repairs)
                        if accents:
                            for text_node in _nodes(node):
                                normalize_accents(text_node,accents,page_number,document,textpage)
                        if wraps:
                            for text_node in _nodes(node):
                                normalize_line_wraps(text_node,wraps,page_number,document)
                        if spans:
                            for text_node in _nodes(node):
                                restore_local_spacing(text_node,textpage,page_number,document)
                                normalize_geometric_spaces(text_node,word_spaces,page_number,document)
                                restore_prose_dashes(text_node,textpage,page_number,document)
                                restore_prose_omissions(text_node,textpage,spans,page_number,document)
                        text = node.get("content", "").strip()
                        if re.match(r"^(?:Disclaimer/)?Publisher['’]s Note\s*:",text,re.I):
                            node['type']='frontmatter'
                            node['source information role']='publisher note'
                        if node.get("type") == "list item" and not _LIST_MARKER.match(text):
                            node["type"] = "paragraph"
                        if node.get('type')=='heading' and text.casefold()=='references':
                            node['heading level']=2
                        if node.get("suppressed"):
                            repaired.append(node)
                        else:
                            for part in split_captions(node, spans, page.get_size()[0]):
                                for section in _split_abstract_keywords(part,spans):
                                    repaired.extend(_split_titles(section, spans))
                finally:
                    textpage.close()
            finally:
                page.close()
    document["kids"] = repaired
    document["page sizes"] = sizes
    document["source spacing repairs"] = spacing_repairs
    document["source link text repairs"] = link_repairs
    document["source text recoveries"] = [{"page": n["page number"], "text": n["content"], "box": n["bounding box"]}
                                          for n in repaired if n.get("source text recovered")]
    _mark_page_furniture(document)
    for node in repaired:
        if node.get("type") in {"paragraph", "list item"}:
            node["source paragraph text"] = node.get("content", "")
    _front_matter(document)
    recover_authors(document, front_spans)
    return document


def _page_order(nodes: list[dict], width: float) -> list[dict]:
    def top_order():
        return sorted(nodes, key=lambda n: (-n.get("bounding box", [0, 0, 0, 0])[3],
                                            n.get("bounding box", [0, 0, 0, 0])[0]))

    textual = [n for n in nodes if n.get("content") and not n.get("suppressed")
               and n.get("type") not in {"caption", "header", "footer", "frontmatter", "abstract", "keywords", "footnote"}
               and (box := n.get("bounding box")) and width * .18 < box[2] - box[0] < width * .6]
    lefts = sorted({round(n["bounding box"][0], 1) for n in textual})
    gaps = [(b - a, a, b) for a, b in zip(lefts, lefts[1:])]
    if not gaps or max(gaps)[0] < width * .18:
        return top_order()
    _, left, right = max(gaps)
    left_edges = [n["bounding box"][2] for n in textual if n["bounding box"][0] <= left + .5]
    gutter = (min(max(left_edges), right) + right) / 2
    evidence = textual + [n for n in nodes if n.get("type") in {"image", "picture"}
                          and (box := n.get("bounding box")) and width*.18 < box[2]-box[0] < width*.6]
    left_column = [n["bounding box"] for n in evidence if (n["bounding box"][0]+n["bounding box"][2])/2 < gutter]
    right_column = [n["bounding box"] for n in evidence if (n["bounding box"][0]+n["bounding box"][2])/2 >= gutter]
    if (not left_column or not right_column
            or min(max(b[3] for b in left_column), max(b[3] for b in right_column))
            <= max(min(b[1] for b in left_column), min(b[1] for b in right_column))):
        return top_order()
    wide = [n for n in nodes if (box := n.get("bounding box"))
            and box[0] < gutter - width * .04 and box[2] > gutter + width * .04]
    remaining = [n for n in nodes if n not in wide]
    result = []

    def columns(items):
        return sorted(items, key=lambda n: (
            (sum(n.get("bounding box", [0, 0, 0, 0])[::2]) / 2) >= gutter,
            -n.get("bounding box", [0, 0, 0, 0])[3], n.get("bounding box", [0, 0, 0, 0])[0]))

    for anchor in sorted(wide, key=lambda n: -n["bounding box"][3]):
        above = [n for n in remaining if n.get("bounding box", [0, 0, 0, 0])[3] > anchor["bounding box"][3] + 1]
        result.extend(columns(above))
        remaining = [n for n in remaining if n not in above]
        result.append(anchor)
    result.extend(columns(remaining))
    return result


def order_document(document: dict) -> dict:
    """Read each column completely between full-width figures and text bands."""
    by_page = defaultdict(list)
    for node in _flatten(document):
        by_page[node.get("page number", 1)].append(node)
    ordered = []
    for number, nodes in sorted(by_page.items()):
        width = document.get("page sizes", {}).get(str(number), [612, 792])[0]
        ordered.extend(_page_order(nodes, width))
    in_references = False
    for index, node in enumerate(ordered):
        if node.get("type") == "heading":
            in_references = bool(re.fullmatch(r"references", node.get("content", ""), re.I))
        if in_references and node.get("asset kind") == "portrait":
            box = node["bounding box"]
            ordered.insert(index, {"type": "heading", "heading level": 2, "content": "Author biographies",
                                   "page number": node["page number"], "generated heading": True,
                                   "bounding box": [box[0], box[3]+1, box[2], box[3]+3]})
            break
    document["kids"] = ordered
    previous = None
    paragraph_number = 0
    for node in ordered:
        kind = node.get("type")
        if node.get("suppressed") or kind in {"image", "picture", "caption", "frontmatter", "keywords", "footnote", "header", "footer"}:
            continue
        if kind == "table" and node.get("caption label"):
            continue
        if kind != "paragraph":
            previous = None
            continue
        text = node.get("source paragraph text", FIGURE_LINK.sub(" X ", node.get("content", ""))).strip()
        continuation = False
        if previous and text:
            prior_text = previous.get("source paragraph text", FIGURE_LINK.sub(" X ", previous.get("content", ""))).strip()
            lower_start = (text[0].islower() or bool(re.match(r"^[A-Za-z]\s+[a-z]", text))
                           or bool(re.match(r"^\[\d+\](?:[,.;]|\s)", text)))
            same_page = node["page number"] == previous["page number"]
            next_page = node["page number"] == previous["page number"] + 1
            first_box, second_box = previous.get("bounding box"), node.get("bounding box")
            geometry = next_page
            boundary = next_page
            if same_page and first_box and second_box:
                column_change = second_box[0] - first_box[0] > 70
                boundary = column_change
                adjacent = abs(second_box[0] - first_box[0]) < 15 and abs(first_box[1] - second_box[3]) < 16
                around_float = (first_box[0] - second_box[0] > 35 and abs(first_box[2] - second_box[2]) < 8
                                and abs(first_box[1] - second_box[3]) < 3)
                lower_start = lower_start or around_float
                geometry = column_change or adjacent or around_float
            indent = node.get("source first line indent")
            sizes = [float(n.get("font size") or 0) for n in (previous, node)]
            if (boundary and text[0].isupper() and indent is not None and -1 <= indent <= 2
                    and min(sizes) > 0 and min(sizes) >= max(sizes) * .95):
                lower_start = True
            continuation = bool(prior_text and re.search(r"[\w,–—-]$", prior_text) and lower_start and geometry)
        if not continuation:
            paragraph_number += 1
        node["paragraph id"] = f"flow-{paragraph_number}"
        previous = node
    _reference_flows(ordered)
    return document


def _reference_flows(nodes: list[dict]) -> None:
    """Keep hanging-indent continuations inside their numbered reference."""
    in_references, previous, reference_id = False, None, ""
    for index, node in enumerate(nodes):
        kind = node.get("type")
        if kind == "heading":
            in_references = bool(re.fullmatch(r"references", node.get("content", ""), re.I))
            previous, reference_id = None, ""
            continue
        if node.get("suppressed") or kind in {"header", "footer"}:
            continue
        if not in_references or kind not in {"paragraph", "list item"}:
            previous, reference_id = None, ""
            continue
        text = node.get("source paragraph text", node.get("content", "")).strip()
        if re.match(r"^(?:\[\d+\]|\d{1,3}[.)])\s+", text):
            reference_id = f"reference-{index}"
        elif previous and reference_id:
            first, second = previous.get("bounding box"), node.get("bounding box")
            next_page = node["page number"] == previous["page number"] + 1
            adjacent = (node["page number"] == previous["page number"] and first and second
                        and -2 <= first[1] - second[3] <= 16
                        and -2 <= second[0] - first[0] <= 35)
            if not (next_page or adjacent):
                reference_id = ""
        else:
            reference_id = ""
        if reference_id:
            node["paragraph id"] = reference_id
        previous = node
