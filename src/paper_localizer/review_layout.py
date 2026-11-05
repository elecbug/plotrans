"""Separate review line counters from the source text by native geometry."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re
from statistics import median

from .formulas import _canonical, _inside
from .figures import _union


def line_counters(spans, width):
    ordinary = [s.font_size for s in spans if len(re.sub(r"\W", "", s.text)) >= 3
                and not s.math and re.search(r"[A-Za-z]", s.text)]
    if not ordinary: return []
    size = median(ordinary)
    rails = defaultdict(list)
    for span in spans:
        if not re.fullmatch(r"\d{1,5}", span.text.strip()) or span.font_size > size*.72: continue
        left, _, right, _ = span.box
        if right-left > width*.035: continue
        if left > width*.88:
            rails['right',round(right/width*100)].append(span)
        elif right < width*.12:
            rails['left',round(left/width*100)].append(span)
    result = []
    for (side,_), rail in rails.items():
        rail.sort(key=lambda s:-s.baseline)
        if len(rail)<8: continue
        numbers = [int(s.text) for s in rail]
        if any(b<=a for a,b in zip(numbers,numbers[1:])):continue
        if sum(b==a+1 for a,b in zip(numbers,numbers[1:])) < (len(rail)-1)*.85: continue
        # A small numerical axis or table column does not qualify: counters
        # must sit beside larger text on the same source baselines.
        beside = sum(any(t.font_size >= s.font_size*1.35
                         and not re.search(r'Courier|Mono|NimbusMon|Consolas',t.font_name,re.I)
                         and abs(t.baseline-s.baseline)<max(2,s.font_size*.6)
                         and (t.box[2]<s.box[0] if side=='right' else t.box[0]>s.box[2])
                         for t in spans) for s in rail)
        if beside >= len(rail)*.8: result.extend(rail)
    return result


def remove_line_counters(nodes, spans, width, page_number, document):
    counters = line_counters(spans,width)
    if not counters: return spans
    ids = {s.index for s in counters}
    records = document.setdefault('review line numbers',[])
    records.extend({'page':page_number,'value':s.text,'index':s.index,'box':list(s.box)} for s in counters)
    for node in nodes:
        text,box = node.get('content',''),node.get('bounding box')
        if not text or not box or node.get('type') in {'table','image','picture'}: continue
        selected = sorted((s for s in spans if _inside(s.box,box,padding=.8)),key=lambda s:s.index)
        owned = [s for s in selected if s.index in ids]
        if not owned: continue
        kept = [s for s in selected if s.index not in ids]
        if not kept:
            if re.fullmatch(r'[\d\s]+',text): node['suppressed']=True
            continue
        native, positions = '',{}
        for span in selected:
            start=len(native);native+=_canonical(span.text)[0];positions[span.index]=(start,len(native))
        target,offsets = _canonical(text)
        # Combining accents can appear in a different order in the layout
        # extractor. Permit that only when the complete character inventory
        # agrees, and every removed counter lies inside an exact match.
        if Counter(native)!=Counter(target): continue
        matches=SequenceMatcher(None,native,target,autojunk=False).get_matching_blocks()
        if sum(m.size for m in matches)<len(native)*.9: continue
        edits=[]
        for span in owned:
            start,end=positions[span.index]
            match=next((m for m in matches if m.a<=start and end<=m.a+m.size),None)
            if match is None: break
            at=match.b+start-match.a
            edits.append((offsets[at],offsets[at+end-start-1]+1))
        else:
            node['review original text']=text
            for start,end in sorted(edits,reverse=True):text=text[:start]+text[end:]
            node['content']=' '.join(text.split())
            bounds=kept[0].box
            for span in kept[1:]:bounds=_union(bounds,span.box)
            node['bounding box']=list(bounds)
            node['review counters removed']=[s.text for s in owned]
    return [s for s in spans if s.index not in ids]


def filter_counters(spans, document, page_number):
    ids={r['index'] for r in document.get('review line numbers',[]) if r['page']==page_number}
    return [s for s in spans if s.index not in ids]
