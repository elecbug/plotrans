"""Compare explicit native math symbols with rendered OCR notation."""

from collections import Counter
from functools import lru_cache
import re
import unicodedata
import xml.etree.ElementTree as ET
from statistics import median

from PIL import Image

from .math_markup import mathml
from .native_math import normalized_math_character


_OPERATORS = set('+-=<>≤≥≠±∓×÷⋅*∑∏∫√∞∈∉⊂⊆⊃⊇∀∃∂∇→↔↦|∥′…✓≈≃≡∝∧∨¬⇒⇔⇐⊢⊨⊥⊤∅∪∩⊎∘⊗⊕∖')
_VARIANTS = str.maketrans({'−':'-','–':'-','∣':'|','·':'⋅','•':'⋅','∗':'*','″':'′′',"'":'′'})
_DECIMAL = re.compile(r'(?<![\d.])\d+\.\d+(?![\d.])')


def _without_variation_selectors(text):
    return ''.join(c for c in text if not (0xfe00<=ord(c)<=0xfe0f or 0xe0100<=ord(c)<=0xe01ef))


def _normalize(text):
    # TeX can emit a negation stroke before the equals glyph in PDF order.
    text = re.sub('\u0338\\s*=|=\\s*\u0338','≠',_without_variation_selectors(text))
    text = ''.join(normalized_math_character(c) for c in text).translate(_VARIANTS)
    return re.sub(r'[.⋅](?:\s*[.⋅]){2}','…',text)


def _symbols(text):
    return Counter(c for c in _normalize(text)
                   if c in _OPERATORS or c.isalpha() and 'GREEK' in unicodedata.name(c,''))


def _complete_unicode(text):
    text = re.sub('\u0338\\s*=|=\\s*\u0338','≠',_without_variation_selectors(text))
    for char in text:
        if char.isspace() or 32 <= ord(char) < 127 or char in _OPERATORS or ord(char) in _VARIANTS:
            continue
        name = unicodedata.name(char,'')
        if (char.isalpha() and 'GREEK' in name or name.startswith('MATHEMATICAL ')
                or name.startswith(('SUPERSCRIPT ','SUBSCRIPT '))
                or any(style in name for style in ('DOUBLE-STRUCK','SCRIPT','FRAKTUR'))):
            continue
        return False
    return True


@lru_cache(maxsize=2048)
def rendered_formula(latex):
    """Return rendered characters and decimal leaves, excluding TeX syntax."""
    latex = re.sub(r'(?<=[0-9.])\s+(?=[0-9.])','',latex)
    root = ET.fromstring(mathml(latex))
    text = ''.join(root.itertext())
    text += '√'*sum(n.tag.rsplit('}',1)[-1] in {'msqrt','mroot'} for n in root.iter())
    decimals = Counter()
    for node in root.iter():
        if node.text:
            decimals.update(_DECIMAL.findall(_normalize(node.text)))
    return text, tuple(sorted(decimals.items()))


def source_symbols_agree(native, latex):
    """Return None when legacy/private glyphs prevent a complete comparison.

    Images stay authoritative. Agreement only checks explicit symbols and
    single-line decimal literals; it does not validate mathematical meaning.
    A missing known symbol can fail even an incomplete comparison.
    """
    try:
        rendered, decimals = rendered_formula(latex)
    except Exception:
        return False
    expected,actual = _symbols(native),_symbols(rendered)
    if expected-actual:
        return False
    if not _complete_unicode(native):
        return None
    if expected != actual:
        return False
    if '\n' not in native and '\r' not in native:
        normalized = re.sub(r'(?<=[0-9.])\s+(?=[0-9.])','',_normalize(native))
        if Counter(_DECIMAL.findall(normalized)) != Counter(dict(decimals)):
            return False
    return True


def source_delimiters_agree(path, latex):
    """Veto unsupported absolute/norm bars when native Unicode is unavailable.

    Narrow solid components can support bars, but are not proof of their
    identity. This check can only reject; it never authorizes a transcription.
    """
    try:
        text = _normalize(rendered_formula(latex)[0])
        required = text.count('|')+2*text.count('∥')
        if not required:
            return None
        with Image.open(path) as original:
            image = original.convert('L')
        width,height = image.size
        pixels = image.tobytes()
        ink = {index for index,value in enumerate(pixels) if value<180}
        image.close()
    except (OSError,ValueError):
        return None
    components = []
    while ink:
        start = ink.pop()
        pending = [start]
        left=right=start%width
        top=bottom=start//width
        area = 0
        while pending:
            value = pending.pop()
            x,y = value%width,value//width
            left,right,top,bottom = min(left,x),max(right,x),min(top,y),max(bottom,y)
            area += 1
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    nx,ny = x+dx,y+dy
                    if 0<=nx<width and 0<=ny<height:
                        neighbor = ny*width+nx
                        if neighbor in ink:
                            ink.remove(neighbor)
                            pending.append(neighbor)
        components.append((right-left+1,bottom-top+1,area))
    heights = [h for _,h,_ in components if h>=3]
    minimum = max(4,median(heights)*.6) if heights else 4
    # A delimiter can touch an adjacent parenthesis at one antialiased pixel.
    # Inspect solid vertical runs rather than requiring a disconnected box.
    strokes = []
    for x in range(width):
        start = None
        for y in range(height+1):
            dark = y<height and pixels[y*width+x]<180
            if dark and start is None:
                start = y
            elif not dark and start is not None:
                if y-start>=minimum:
                    group = next((g for g in reversed(strokes) if g[1]==x-1
                                  and max(0,min(g[3],y)-max(g[2],start))>=max(g[3]-g[2],y-start)*.8),None)
                    if group is None:
                        strokes.append([x,x,start,y])
                    else:
                        group[1]=x;group[2]=min(group[2],start);group[3]=max(group[3],y)
                start = None
    supported = 0
    for left,right,top,bottom in strokes:
        h,w = bottom-top,right-left+1
        if w/h>.28:
            continue
        isolated = sum((left==0 or pixels[y*width+left-1]>=180)
                       and (right==width-1 or pixels[y*width+right+1]>=180) for y in range(top,bottom))
        if isolated/h>=.8:
            supported += 1
    return supported>=required
