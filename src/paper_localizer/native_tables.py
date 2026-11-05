"""Recover simple ruled tables omitted by the native layout extractor."""

from __future__ import annotations

from collections import Counter
import re

import pypdfium2 as pdfium

from .figures import _area, _intersection, _union
from .formulas import _canonical, _inside
from .layout import caption_label


def _table_grid(page, box, spans, textpage):
    selected = [s for s in spans if _inside(s.box, box, padding=0)]
    if not selected: return None
    size = max(s.font_size for s in selected)
    lines = []
    for span in sorted(selected, key=lambda s: (-s.baseline, s.box[0])):
        if span.font_size < size*.85: continue
        line = next((line for line in lines if abs(line[0].baseline-span.baseline) < 1.2), None)
        if line is None: lines.append([span])
        else: line.append(span)
    if not 3 <= len(lines) <= 50: return None
    for line in lines: line.sort(key=lambda s:s.box[0])
    header_count = 0
    for line in lines:
        if sum(bool(re.search(r"Bold|Medi|Semibold|TB$", s.font_name, re.I)) for s in line) < len(line)*.6: break
        header_count += 1
    if not 1 <= header_count <= 3 or len(lines)-header_count < 2: return None
    groups = []
    for span in lines[0]:
        if not groups or span.box[0]-groups[-1][-1].box[2] > max(5, size*.75): groups.append([span])
        else: groups[-1].append(span)
    if not 2 <= len(groups) <= 12: return None
    occupied = []
    for span in sorted(selected,key=lambda s:s.box[0]):
        if occupied and span.box[0] <= occupied[-1][1]: occupied[-1][1] = max(occupied[-1][1],span.box[2])
        else: occupied.append([span.box[0],span.box[2]])
    gaps = [(a[1],b[0]) for a,b in zip(occupied,occupied[1:]) if b[0]-a[1] >= 1.5]
    cuts = [box[0]]
    for left,right in zip(groups,groups[1:]):
        centers = ((left[0].box[0]+left[-1].box[2])/2,(right[0].box[0]+right[-1].box[2])/2)
        possible = [(b-a,(a+b)/2) for a,b in gaps if centers[0] < (a+b)/2 < centers[1]]
        if not possible: return None
        cuts.append(max(possible)[1])
    cuts.append(box[2])
    line_boxes = []
    for line in lines:
        bounds = line[0].box
        for span in line[1:]: bounds = _union(bounds,span.box)
        line_boxes.append(bounds)
    y_cuts = [box[3]]
    for upper,lower in zip(line_boxes,line_boxes[1:]):
        if upper[1] <= lower[3]: return None
        y_cuts.append((upper[1]+lower[3])/2)
    y_cuts.append(box[1])
    rows = []
    for r in range(len(lines)):
        cells = []
        for c in range(len(cuts)-1):
            cell_box = [cuts[c],y_cuts[r+1],cuts[c+1],y_cuts[r]]
            value = " ".join(textpage.get_text_bounded(*cell_box).split())
            if any(not ch.isprintable() or ch == "\ufffd" for ch in value): return None
            cells.append({"type":"table cell", "row number":r+1, "column number":c+1,
                          "row span":1,"column span":1,"is_header":r<header_count,
                          "bounding box":cell_box,"kids":[{"type":"paragraph","content":value,
                              "bounding box":cell_box,"font":lines[r][0].font_name,"font size":lines[r][0].font_size,
                              "native table cell":True}]})
        if r>=header_count and (not cells[0]['kids'][0]['content'] or sum(bool(c['kids'][0]['content']) for c in cells)<2): return None
        rows.append({"type":"table row","row number":r+1,"cells":cells})
    # A label such as Method occupies the height of a wrapped neighboring
    # header, while Successful Victim / Connections spans two printed lines.
    for c in range(len(cuts)-1):
        if header_count>1 and rows[0]['cells'][c]['kids'][0]['content'] and all(not rows[r]['cells'][c]['kids'][0]['content'] for r in range(1,header_count)):
            rows[0]['cells'][c]['row span']=header_count
            for r in range(1,header_count): rows[r]['cells'][c]['occupied by header']=True
    for row in rows: row['cells']=[c for c in row['cells'] if not c.pop('occupied by header',False)]
    return rows


def recover_tables(page, nodes: list[dict], spans, textpage, page_number: int, document: dict) -> list[dict]:
    """Require rules, a nearby table caption, clear columns, and exact text."""
    width,height=page.get_size()
    groups=[]
    # Root paths use page coordinates. Bounds inside Form XObjects are local
    # and must not be mistaken for table rules on the page.
    for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH],max_depth=1):
        box=tuple(obj.get_bounds())
        if box[2]-box[0]<60 or not 0<box[3]-box[1]<=2.5: continue
        if box[0]<0 or box[2]>width: continue
        group=next((g for g in groups if abs(g[0][0]-box[0])<2 and abs(g[0][2]-box[2])<2),None)
        if group is None:groups.append([box])
        else:group.append(box)
    recovered=[];candidates=[]
    for rules in groups:
        if len(rules)<3:continue
        box=rules[0]
        for rule in rules[1:]:box=_union(box,rule)
        if not 20<box[3]-box[1]<height*.4:continue
        if any(n.get('type')=='table' and n.get('bounding box') and _intersection(box,n['bounding box'])/max(_area(box),1)>.6 for n in nodes):continue
        captions=[]
        for node in nodes:
            if not (label:=caption_label(node.get('content',''))).startswith('Table '):continue
            b=node.get('bounding box')
            if not b:continue
            overlap=min(box[2],b[2])-max(box[0],b[0])
            gap=min(abs(b[1]-box[3]),abs(box[1]-b[3]))
            if overlap>=min(box[2]-box[0],b[2]-b[0])*.65 and gap<40:captions.append((gap,label,id(node)))
        if not captions:continue
        candidates.append((box,captions))
    used=set()
    # Assign the constrained matches first. Two nearby tables must not share
    # the caption between them simply because one gap is slightly smaller.
    for box,captions in sorted(candidates,key=lambda item:len(item[1])):
        choices=[c for c in captions if c[2] not in used]
        if not choices:continue
        _,label,owner_id=min(choices)
        owners=[n for n in nodes if n.get('content') and not n.get('suppressed') and not caption_label(n['content'])
                and n.get('bounding box') and _intersection(box,n['bounding box'])/max(_area(n['bounding box']),1)>.85]
        if not owners:continue
        used.add(owner_id)
        rows=_table_grid(page,box,spans,textpage)
        before=Counter(_canonical(''.join(n['content'] for n in owners))[0])
        after=Counter(_canonical(''.join(c['kids'][0]['content'] for r in rows for c in r['cells']))[0]) if rows else None
        if rows and before==after:
            table={"type":"table","page number":page_number,"bounding box":list(box),"rows":rows,
                   "number of rows":len(rows),"number of columns":len(rows[0]['cells']),
                   "source reconstructed table":True,"caption label":label}
            for row in rows:
                for cell in row['cells']:
                    cell['page number']=page_number
                    for child in cell['kids']:child['page number']=page_number
            for node in owners:node['suppressed']=True
            recovered.append(table)
            document.setdefault('native table recoveries',[]).append({"page":page_number,"label":label,"box":list(box)})
        else:
            document.setdefault('source table regions',[]).append({"page number":page_number,"bounding box":list(box),"caption label":label})
    return recovered
