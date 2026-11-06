"""Normalize source typography only when native glyph geometry confirms it."""

from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from functools import lru_cache
import gzip
from importlib.resources import files
import json
import re
import unicodedata

import pypdfium2 as pdfium
from rapidfuzz.distance import LCSseq

from .figures import _union
from .formulas import _MATH_FONT, _inside, _canonical, TextSpan


_ACCENTS = {
    "´": "\u0301", "`": "\u0300", "^": "\u0302", "˜": "\u0303", "~": "\u0303",
    "¨": "\u0308", "˝": "\u030b", "ˇ": "\u030c", "˘": "\u0306", "¯": "\u0304", "˙": "\u0307",
}
_BROKEN_ACCENT_WORD = re.compile(r"[A-Za-z\u00c0-\u024f]+(?:[´`^˜~¨˝ˇ˘¯˙][A-Za-z\u00c0-\u024f]*)+")
_CODE_FONT = re.compile(r"Courier|Mono|NimbusMon|Consolas", re.I)
_LINK = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.I)
_WORD = re.compile(r'[A-Za-z]{3,}')
_LATIN_WORD = re.compile(r'[A-Za-z\u00c0-\u024f]+')
_COMPOUND_PREFIX = {'anti', 'co', 'cross', 'ex', 'in', 'inter', 'intra', 'multi', 'non', 'post',
                    'pre', 're', 'self', 'semi', 'sub', 'super', 'trans', 'ultra', 'un'}
_INFLECTION = {'ed', 'ing', 'ly', 'ness', 'ment', 'ments'}


def native_vocabulary(page_spans):
    """Count unbroken ordinary words without learning from OCR or formulas."""
    vocabulary = Counter()
    for spans in page_spans:
        vocabulary.update(s.text.casefold() for s in spans
                          if _WORD.fullmatch(s.text) and s.font_name and not s.math
                          and not _MATH_FONT.search(s.font_name) and not _CODE_FONT.search(s.font_name))
        for left, hyphen, right in zip(spans,spans[1:],spans[2:]):
            if (hyphen.text in {'-','‐'} and re.fullmatch('[A-Za-z]{2,}',left.text)
                    and re.fullmatch('[A-Za-z]{2,}',right.text)
                    and left.font_name == right.font_name and not left.math and not right.math
                    and abs(left.baseline-right.baseline) < left.font_size*.2
                    and 0 <= right.box[0]-left.box[2] < left.font_size*.7):
                vocabulary[(left.text+'-'+right.text).casefold()] += 1
    return vocabulary


@lru_cache(maxsize=1)
def _english_words():
    data = files('paper_localizer').joinpath('assets/english-words.json.gz').read_bytes()
    return frozenset(json.loads(gzip.decompress(data)))


@lru_cache(maxsize=1)
def _hyphenators():
    try:
        import pyphen
    except ImportError:
        # Older environments still preserve the source hyphen safely.
        return ()
    return tuple(pyphen.Pyphen(lang=lang) for lang in ('en_US', 'en_GB'))


def _can_join(left, right, vocabulary):
    if len(left) < 2 or len(right) < 2:
        return False
    a, b = left.casefold(), right.casefold()
    joined, compound = a+b, a+'-'+b
    lexical = _english_words()
    observed = vocabulary[joined]
    if (joined not in lexical and observed < 2) or compound in lexical or vocabulary[compound]:
        return False
    if a in _COMPOUND_PREFIX and b in lexical:
        return False
    if not any(len(left) in dictionary.positions(left+right) for dictionary in _hyphenators()):
        return False
    if observed and len(left) >= 3 and len(right) >= 3:
        return True
    # Preserve two independently valid words, e.g. long-term, even when a
    # dictionary also lists their concatenation. Inflectional endings such
    # as report-ing are a separate, explicitly bounded case.
    return joined in lexical and (a not in lexical or b not in lexical or b in _INFLECTION)


def _known_inflection(word):
    """Protect ordinary inflections omitted from the spelling inventory."""
    lexical = _english_words()
    word = word.casefold()
    if word in lexical:
        return True
    stems = set()
    if word.endswith('s'):
        stems.add(word[:-1])
    if word.endswith('ies'):
        stems.add(word[:-3]+'y')
    if word.endswith('ly'):
        stems.add(word[:-2])
    for suffix in ('ed','ing'):
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            stems.update((stem,stem+'e'))
            if len(stem)>2 and stem[-1]==stem[-2]:
                stems.add(stem[:-1])
    return bool(stems & lexical)


def _separate_words(left, right):
    lexical = _english_words()
    a,b = left.casefold(),right.casefold()
    return (min(len(a),len(b))>=3 and a in lexical and b in lexical
            and not _known_inflection(a+b))


def geometric_word_spaces(spans):
    """Find a real word gap at a prose font change without a space glyph."""
    records = []
    for left,right in zip(spans,spans[1:]):
        size = min(left.font_size,right.font_size)
        gap = right.box[0]-left.box[2]
        if (size <= 0 or not re.fullmatch('[A-Za-z]{2,}',left.text)
                or not re.fullmatch('[A-Za-z]{2,}',right.text)
                or not left.font_name or not right.font_name or left.font_name == right.font_name
                or left.math or right.math
                or any(_MATH_FONT.search(s.font_name) or _CODE_FONT.search(s.font_name) for s in (left,right))
                or abs(left.font_size-right.font_size) > size*.1
                or abs(left.baseline-right.baseline) > size*.12
                or not size*.28 <= gap <= size*.7
                or not _separate_words(left.text,right.text)):
            continue
        records.append({'left':left.text,'right':right.text,'gap':gap,
                        'boxes':[list(left.box),list(right.box)]})
    return records


def normalize_geometric_spaces(node, records, page_number, document):
    text,box = node.get('content',''),node.get('bounding box')
    if (not text or not box or node.get('suppressed') or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') in {'formula','code','header','footer','image','picture','table'}):
        return
    for record in records:
        if not all(_inside(b,box,padding=0) for b in record['boxes']):
            continue
        pattern = re.compile(r'(?<!\w)'+re.escape(record['left']+record['right'])+r'(?!\w)')
        matches = list(pattern.finditer(text))
        if len(matches)!=1:
            continue
        match = matches[0]
        if any(m.start()<match.end() and match.start()<m.end() for m in _LINK.finditer(text)):
            continue
        node.setdefault('source text before typography',text)
        node['content'] = text = text[:match.start()]+record['left']+' '+record['right']+text[match.end():]
        document.setdefault('source typography repairs',[]).append({
            'kind':'geometric-spacing','page':page_number,'box':list(box),
            'original':match[0],'normalized':record['left']+' '+record['right'],
            'native gap':record['gap'],'native boxes':record['boxes'],
        })


def line_wraps(page, spans, vocabulary):
    """Identify same-column native line endings before changing any text."""
    lines = []
    for span in sorted(spans, key=lambda s: (-s.baseline, s.box[0])):
        if not span.text.strip():
            continue
        line = next((line for line in reversed(lines[-6:])
                     if abs(line[0].baseline - span.baseline) < 1.2
                     and -2 <= span.box[0] - line[-1].box[2] < max(18, span.font_size * 2)), None)
        if line is None:
            lines.append([span])
        else:
            line.append(span)
    # PDFium may expose a printed line-ending hyphen as a non-printable
    # discretionary marker, which ordinary text spans intentionally omit.
    # Its flag proves the glyph exists, but does not prove it can be removed:
    # compound words still use the same conservative vocabulary checks.
    textpage = page.get_textpage()
    discretionary = set()
    try:
        indices = _span_indices(textpage, spans, {line[-1].index for line in lines})
        count = textpage.count_chars()
        for line in lines:
            last = line[-1]
            positions = indices.get(str(last.index), [])
            if not positions or not re.fullmatch('[A-Za-z]{2,}', last.text):
                continue
            index = positions[-1] + 1
            if (index < count and pdfium.raw.FPDFText_GetUnicode(textpage,index) == 2
                    and pdfium.raw.FPDFText_IsHyphen(textpage,index) == 1):
                line.append(replace(last,index=index,text='-',box=tuple(textpage.get_charbox(index))))
                discretionary.add(index)
    finally:
        textpage.close()
    records = []
    for line in lines:
        if len(line) < 2 or line[-1].text not in {'-', '‐', '\u00ad'}:
            continue
        prefix, hyphen = line[-2:]
        if (not re.fullmatch('[A-Za-z]{2,}', prefix.text) or not prefix.font_name or prefix.math
                or _MATH_FONT.search(prefix.font_name) or _CODE_FONT.search(prefix.font_name)):
            continue
        following = [other for other in lines
                     if .8 * prefix.font_size < line[0].baseline - other[0].baseline < 1.85 * prefix.font_size
                     and abs(other[0].box[0] - line[0].box[0]) < prefix.font_size * 2.6
                     and re.fullmatch('[a-z][A-Za-z]+', other[0].text)
                     and other[0].font_name == prefix.font_name and not other[0].math
                     and abs(other[0].font_size - prefix.font_size) < prefix.font_size * .1]
        if len(following) != 1:
            continue
        suffix = following[0][0]
        joined = _can_join(prefix.text, suffix.text, vocabulary)
        records.append({'left': prefix.text, 'right': suffix.text, 'join': joined,
                        'restore': (prefix.text+'-'+suffix.text).casefold() in _english_words()
                                   or bool(vocabulary[(prefix.text+'-'+suffix.text).casefold()])
                                   or (_separate_words(prefix.text,suffix.text)
                                       and not vocabulary[(prefix.text+suffix.text).casefold()]),
                        'source occurrences': vocabulary[(prefix.text + suffix.text).casefold()],
                        'boxes': [list(prefix.box), list(hyphen.box), list(suffix.box)],
                        'hyphen_index': hyphen.index,
                        'discretionary': hyphen.index in discretionary,
                        'hyphen span': {'index':hyphen.index,'text':'-','box':list(hyphen.box),'math':False,
                                        'font_size':prefix.font_size,'baseline':prefix.baseline,'font_name':prefix.font_name}})
    return records


def normalize_line_wraps(node, records, page_number, document):
    text, box = node.get('content', ''), node.get('bounding box')
    if (not text or not box or node.get('suppressed') or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') in {'formula', 'code', 'header', 'footer'}):
        return
    for record in records:
        if not all(_inside(b, box, padding=.5) for b in record['boxes']):
            continue
        protected = [m.span() for m in _LINK.finditer(text)]
        separator = r'[-‐\u00ad]?\s*' if record.get('restore') and not record['join'] else r'[-‐\u00ad]\s*'
        pattern = re.compile(r'(?<!\w)' + re.escape(record['left']) + separator
                             + re.escape(record['right']) + r'(?!\w)')

        def repair(match):
            if any(a < match.end() and match.start() < b for a, b in protected):
                return match[0]
            fixed = record['left'] + ('' if record['join'] else '-') + record['right']
            if record.get('restore') and not record['join'] and record.get('discretionary'):
                document.setdefault('source discretionary hyphens',{}).setdefault(str(page_number),{})[
                    str(record['hyphen_index'])] = record['hyphen span']
            if fixed != match[0]:
                document.setdefault('source typography repairs', []).append({
                    'kind': 'line-wrap', 'page': page_number, 'box': list(box),
                    'original': match[0], 'normalized': fixed,
                    'joined source occurrences': record['source occurrences'],
                    'removed hyphen': record['join'], 'native boxes': record['boxes'],
                    'inserted hyphen': not record['join'] and not any(c in match[0] for c in '-‐\u00ad'),
                })
                if record['join']:
                    index = str(record['hyphen_index'])
                    document.setdefault('source glyph overrides', {}).setdefault(str(page_number), {})[index] = ''
                    document.setdefault('source span glyph indices', {}).setdefault(str(page_number), {})[index] = [int(index)]
            return fixed

        fixed = pattern.sub(repair, text, count=1)
        if fixed != text:
            node.setdefault('source text before typography', text)
            node['content'] = text = fixed


def restore_local_spacing(node, textpage, page_number, document):
    """Restore spaces in long matching prose around undecodable inline math.

    Only whitespace is inserted. Every affected letter and the surrounding
    source sequence must agree; URLs and code retain their existing spelling.
    """
    text, box = node.get('content',''), node.get('bounding box')
    if (not text or not box or node.get('suppressed') or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') in {'formula','code','header','footer','image','picture','table'}):
        return False
    native = textpage.get_text_bounded(*box)
    source_key, source_offsets = _canonical(native)
    target_key, target_offsets = _canonical(text)
    if source_key == target_key or min(len(source_key),len(target_key)) < 24:
        return False
    matcher = SequenceMatcher(None,source_key,target_key,autojunk=False)
    if matcher.ratio() < .9:
        return False
    protected = [m.span() for m in _LINK.finditer(text)]
    insertions = set()
    for a,b,size in matcher.get_matching_blocks():
        if size < 24:
            continue
        source = native[source_offsets[a]:source_offsets[a+size-1]+1]
        if len(re.findall(r'\b[A-Za-z]{2,}\b',source)) < 4:
            continue
        for offset in range(size-1):
            left, right = target_offsets[b+offset], target_offsets[b+offset+1]
            if right != left+1 or not text[left].isalpha() or not text[right].isalpha():
                continue
            gap = native[source_offsets[a+offset]+1:source_offsets[a+offset+1]]
            if not gap or not gap.isspace() or any(start < right < end for start,end in protected):
                continue
            insertions.add(right)
    if not insertions:
        return False
    fixed = ''.join((' ' if i in insertions else '')+char for i,char in enumerate(text))
    node.setdefault('source text before typography',text)
    node['content'] = fixed
    document.setdefault('source typography repairs',[]).append({
        'kind':'spacing','page':page_number,'box':list(box),'inserted spaces':len(insertions),
        'source agreement':round(matcher.ratio(),6),
    })
    return True


def _source_alignment(source, target):
    """Align repeated source prose without greedy longest-block selection."""
    matches = LCSseq.opcodes(source, target).as_matching_blocks()
    agreement = 2 * sum(match.size for match in matches) / max(len(source) + len(target), 1)
    return matches, agreement


def restore_prose_omissions(node, textpage, spans, page_number, document):
    """Restore small native prose omissions between matching text anchors.

    The paragraph must have at least 90% agreement; existing characters in
    each repaired gap remain a source subsequence. Only native letters and
    their spacing can be added. Mathematical and undecodable additions are
    excluded, including inside otherwise mixed prose/math paragraphs.
    """
    text,box = node.get('content',''),node.get('bounding box')
    if (not text or not box or node.get('suppressed') or node.get('formula snapshot')
            or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') not in {'paragraph','abstract','list item','footnote','frontmatter'}):
        return False
    selected = [s for s in spans if _inside(s.box,box,padding=0)]
    if not selected:
        return False
    native = textpage.get_text_bounded(*box).replace('\x02','').replace('\ufffe','')
    if any(not c.isprintable() and not c.isspace() for c in native):
        return False
    source,source_offsets = _canonical(native)
    target,target_offsets = _canonical(text)
    if source==target or min(len(source),len(target))<48:
        return False
    matches, complete_agreement = _source_alignment(source, target)
    matches = [m for m in matches if m.size]
    strong = [m for m in matches if m.size>=8]
    agreement = complete_agreement
    if strong:
        first,last = strong[0],strong[-1]
        matched = sum(m.size for m in matches if first.a<=m.a<=last.a and first.b<=m.b<=last.b)
        bounded = 2*matched/max(last.a+last.size-first.a+last.b+last.size-first.b,1)
        if matched/max(len(target),1)>=.9:
            # Layout bounds can include clipped lines from neighboring
            # paragraphs. Their outside text cannot become an addition.
            agreement = max(agreement,bounded)
    if agreement<.9:
        return False
    # Project native font ownership into the same character sequence before
    # restoring prose within a paragraph that also contains mathematics.
    # If projection is incomplete, only wholly readable prose can proceed.
    projected,ownership = [],[]
    for span in selected:
        chars = _canonical(span.text)[0]
        projected.append(chars)
        ownership.extend([span]*len(chars))
    aligned = ''.join(projected)==source
    projection_matches = []
    if not aligned:
        projection, projection_agreement = _source_alignment(''.join(projected), source)
        if projection_agreement>=.9:
            projection_matches = [m for m in projection if m.size>=24]
    def ordinary(span):
        # A list bullet contributes no letters and cannot be part of a prose
        # addition. Its Symbol/Math font must not veto the ordinary list text.
        return (span.text in {'•','◦','▪','‣'} or
                not span.math and bool(span.font_name) and not _MATH_FONT.search(span.font_name)
                and not _CODE_FONT.search(span.font_name))
    wholly_ordinary = all(ordinary(span) for span in selected)
    if not aligned and not projection_matches and not wholly_ordinary:
        return False
    anchors = [m for m in matches if m.size>=8]
    protected = [m.span() for m in _LINK.finditer(text)]
    edits = []
    for left,right in zip(anchors,anchors[1:]):
        if right.size<16:
            continue
        a,b = left.a+left.size,right.a
        c,d = left.b+left.size,right.b
        addition = source[a:b]
        reference_initial = (bool(re.fullmatch(r'[A-Z][.]',addition))
                             and bool(re.match(r'^\[\d+\]\s',text))
                             and bool(re.search(r'\b(?:19|20)\d{2}\b',text)))
        if (not 1<=len(addition)<=80 or not any(char.isalpha() for char in addition)
                or not reference_initial and any(not char.isalpha() and char not in "-‐'’,;:" for char in addition)):
            continue
        if len(addition)<=d-c or LCSseq.similarity(addition,target[c:d])!=d-c:
            continue
        start,end = target_offsets[c-1]+1,target_offsets[d]
        if any(x<end and start<y for x,y in protected):
            continue
        original = text[start:end]
        replacement = re.sub(r'\s+',' ',native[source_offsets[a-1]+1:source_offsets[b]])
        owners = ownership[a:b] if aligned else None
        if owners is None:
            match = next((m for m in projection_matches if m.b+4<=a and b+4<=m.b+m.size),None)
            if match:
                owners = ownership[match.a+a-match.b:match.a+b-match.b]
            elif not wholly_ordinary:
                continue
        if len(addition)<3 and not reference_initial:
            whole_short_word = bool(len(addition)==2 and re.fullmatch(r'\s+[A-Za-z]{2}\s+',replacement))
            known_source_word = bool(owners and all(s is owners[0] for s in owners)
                                     and len(owners[0].text)>=3 and owners[0].text.isalpha()
                                     and _known_inflection(owners[0].text))
            if not whole_short_word and not known_source_word:
                continue
        if (len(re.findall(r'[^\W\d_]+',replacement))>8
                or any(char.isalpha() and not ('LATIN' in unicodedata.name(char,'') or ord(char)<128)
                       for char in replacement)
                or owners is not None and not all(ordinary(span) for span in owners)):
            continue
        edits.append((start,end,replacement,original))
    if not edits:
        return False
    node.setdefault('source text before prose recovery',text)
    for start,end,replacement,original in reversed(edits):
        text = text[:start]+replacement+text[end:]
        document.setdefault('source prose insertions',[]).append({
            'page':page_number,'box':list(box),'original':original,'restored':replacement,
            'source agreement':round(agreement,6),'complete native box agreement':round(complete_agreement,6),
            'alignment':'rapidfuzz-lcs-v1',
        })
    node['content'] = text
    return True


def restore_prose_dashes(node, textpage, page_number, document):
    """Restore a dropped prose dash between long, exact native anchors."""
    text,box = node.get('content',''),node.get('bounding box')
    if (not text or not box or node.get('suppressed') or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') in {'formula','code','header','footer','image','picture','table'}):
        return False
    native = textpage.get_text_bounded(*box)
    if not any(c in native for c in '–—'):
        return False
    source,_ = _canonical(native)
    target,offsets = _canonical(text)
    matcher = SequenceMatcher(None,source,target,autojunk=False)
    if matcher.ratio() < .9:
        return False
    opcodes = matcher.get_opcodes()
    protected = [m.span() for m in _LINK.finditer(text)]
    edits = []
    for index,(kind,a,b,c,d) in enumerate(opcodes):
        if kind!='delete' or source[a:b] not in {'–','—'} or not 0<c<len(target):
            continue
        if not 0<index<len(opcodes)-1:
            continue
        before,after = opcodes[index-1],opcodes[index+1]
        if before[0]!='equal' or after[0]!='equal' or before[2]-before[1]<16 or after[2]-after[1]<16:
            continue
        left,right = offsets[c-1],offsets[c]
        if not text[left].isalpha() or not text[right].isalpha():
            continue
        gap = text[left+1:right]
        if gap and not gap.isspace() or any(start<right<end for start,end in protected):
            continue
        edits.append((left+1,right,source[a:b]))
    if not edits:
        return False
    fixed = text
    for start,end,dash in reversed(edits):
        fixed = fixed[:start]+dash+fixed[end:]
    node.setdefault('source text before typography',text)
    node['content'] = fixed
    document.setdefault('source typography repairs',[]).extend({
        'kind':'prose-dash','page':page_number,'box':list(box),'dash':dash,
        'source agreement':round(matcher.ratio(),6),
    } for _,_,dash in edits)
    return True


def _letters(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.replace('ı','i'))
                   if c.isalpha() and c not in _ACCENTS)


def _span_indices(textpage, spans, affected):
    """Verify a span's actual native indices, including UTF-16 pairs/gaps."""
    result = {}
    ordered = sorted(spans, key=lambda s: s.index)
    for at, span in enumerate(ordered):
        end = ordered[at+1].index if at+1 < len(ordered) else textpage.count_chars()
        if not any(span.index <= index < end for index in affected): continue
        text, indices, index = '', [], span.index
        while index < end and len(text) < len(span.text):
            start = index
            code = pdfium.raw.FPDFText_GetUnicode(textpage, index)
            index += 1
            if 0xd800 <= code <= 0xdbff and index < end:
                low = pdfium.raw.FPDFText_GetUnicode(textpage, index)
                if 0xdc00 <= low <= 0xdfff:
                    code = 0x10000+((code-0xd800)<<10)+low-0xdc00
                    index += 1
            char = chr(code) if 0 < code <= 0x10ffff else '\ufffd'
            box = textpage.get_charbox(start)
            if (char.isspace() or not char.isprintable() or box[2] <= box[0] or box[3] <= box[1]): continue
            text += char
            indices.append(start)
        if text == span.text: result[str(span.index)] = indices
    return result


def positioned_accents(page, spans):
    """Compose separate marks using physical words, independent of PDF order."""
    records = []
    textpage = page.get_textpage()
    try:
        ordinary = [s for s in spans if s.font_name
                    and not _MATH_FONT.search(s.font_name)
                    and not _CODE_FONT.search(s.font_name)]
        words = [s for s in ordinary if _LATIN_WORD.fullmatch(s.text) and s.text.isalpha() and not s.math]
        marks = [s for s in ordinary if s.text in _ACCENTS
                 and s.text not in {'^', '~', '`'}]
        if not marks:
            return records
        # A mark can be stored after an entire line, away from its base in
        # native character order. Verify individual glyph indices first.
        positions = _span_indices(textpage, words + marks, {s.index for s in words + marks})
        glyphs = []
        for word in words:
            indices = positions.get(str(word.index), [])
            if len(indices) != len(word.text):
                continue
            for char, index in zip(word.text, indices):
                glyphs.append((word, char, index, tuple(textpage.get_charbox(index))))
        located = []
        for mark in marks:
            center = (mark.box[0] + mark.box[2]) / 2
            candidates = []
            for word, char, index, box in glyphs:
                # TeX commonly uses dotless i underneath an acute accent.
                composed = unicodedata.normalize('NFC', ('i' if char=='ı' else char) + _ACCENTS[mark.text])
                if len(composed) != 1:
                    continue
                if (box[0] - .2 <= center <= box[2] + .2
                        and abs(word.baseline - mark.baseline) < word.font_size * .3
                        and mark.box[1] >= box[1] + word.font_size * .45
                        and mark.box[2] - mark.box[0] <= (box[2] - box[0]) * 1.6):
                    candidates.append((abs(center - (box[0] + box[2]) / 2), index, composed, word, box))
            if candidates:
                _, base, replacement, word, box = min(candidates, key=lambda c: c[:2])
                located.append((mark, base, replacement, word, box))
        completed = set()
        for mark, base, replacement, word, base_box in located:
            # Join adjacent native word fragments only across a sub-letter
            # gap. A real inter-word space is wider and ends the word.
            line = sorted([s for s in words if abs(s.baseline - word.baseline) < word.font_size * .2
                           and abs(s.font_size - word.font_size) < word.font_size * .1],
                          key=lambda s: s.box[0])
            at = line.index(word)
            start, end = at, at + 1
            while start > 0 and -.2 <= line[start].box[0] - line[start-1].box[2] <= word.font_size * .18:
                start -= 1
            while end < len(line) and -.2 <= line[end].box[0] - line[end-1].box[2] <= word.font_size * .18:
                end += 1
            group = line[start:end]
            key = tuple(s.index for s in group)
            if key in completed:
                continue
            completed.add(key)
            letters = [(c, i, b) for s, c, i, b in glyphs if s in group]
            letters.sort(key=lambda v: v[2][0])
            initial = False
            if len(letters)==1 and letters[0][0].isupper():
                index = letters[0][1]+1
                while index < textpage.count_chars():
                    code = pdfium.raw.FPDFText_GetUnicode(textpage,index)
                    if not 0 < code <= 0x10ffff:
                        break
                    if chr(code) not in _ACCENTS:
                        initial = code==ord('.')
                        break
                    index += 1
            if len(letters) < 3 and not initial:
                continue
            accents = [v for v in located if v[3] in group]
            changes = {v[1]: v[2] for v in accents}
            # Multiple different marks over one base are ambiguous here.
            if len(changes) != len(accents):
                continue
            box = group[0].box
            for span in group[1:] + [v[0] for v in accents]:
                box = _union(box, span.box)
            original = ''.join(c for c, _, _ in letters) + ''.join(v[0].text for v in accents)
            normalized = ''.join(changes.get(i, c) for c, i, _ in letters)
            affected = set(changes) | {v[0].index for v in accents}
            record = {'kind': 'accent', 'original': original, 'normalized': normalized,
                      'initial':initial,'marks':''.join(v[0].text for v in accents),
                      'box': list(box), 'glyph_overrides': {str(i): c for i,c in changes.items()},
                      'span_indices': {k: v for k,v in positions.items() if any(i in affected for i in v)}}
            record['glyph_overrides'].update({str(v[0].index): '' for v in accents})
            records.append(record)
    finally:
        textpage.close()
    return records


def normalize_accents(node, records, page_number, document, textpage=None):
    text, box = node.get('content', ''), node.get('bounding box')
    if (not records or not text or not box or node.get('suppressed') or _CODE_FONT.search(node.get('font') or '')
            or node.get('type') in {'formula', 'code', 'header', 'footer'}): return
    reference = (re.match(r'^(?:\[\d+\]|\d{1,3}[.)])\s*',text)
                 and re.search(r'\b(?:18|19|20)\d{2}\b',text))
    local = [r for r in records if (not r.get('initial') or reference) and _inside(r['box'], box, padding=.5)
             and r['box'][0] >= box[0]-1 and r['box'][2] <= box[2]+1]
    variants = defaultdict(set)
    for record in local:
        if not record.get('initial'):
            variants[_letters(record['original'])].add(record['normalized'])
    links = [m.span() for m in _LINK.finditer(text)]
    changes = []

    def repair(match):
        if any(a < match.end() and match.start() < b for a,b in links): return match[0]
        options = variants.get(_letters(match[0]), set())
        if len(options) != 1: return match[0]
        normalized = next(iter(options))
        changes.append({'original': match[0], 'normalized': normalized})
        return normalized

    fixed = _BROKEN_ACCENT_WORD.sub(repair, text)
    for record in local:
        if not record.get('initial'):
            continue
        letter = _letters(record['original'])
        marks = '['+re.escape(record['marks'])+']'
        pattern = re.compile(r'(?<!\w)'+re.escape(letter)+r'(?:\s*\.\s*'+marks+'|'+marks+r'\s*\.)')
        protected = [m.span() for m in _LINK.finditer(fixed)]
        def repair_initial(match):
            if any(a < match.end() and match.start() < b for a,b in protected):
                return match[0]
            normalized = record['normalized']+'.'
            changes.append({'original':match[0],'normalized':normalized,'glyph normalized':record['normalized']})
            return normalized
        fixed = pattern.sub(repair_initial,fixed)
    if textpage is not None:
        # Some extractors omit the separate mark entirely. Restore it only
        # for one unique source word and one unique extracted occurrence.
        native = textpage.get_text_bounded(*box)
        token = re.compile(r'[A-Za-z\u00c0-\u024f´`^˜~¨˝ˇ˘¯˙]+')
        native_counts = defaultdict(int)
        for match in token.finditer(native): native_counts[_letters(match[0])] += 1
        target_counts = defaultdict(int)
        for match in token.finditer(fixed): target_counts[_letters(match[0])] += 1
        protected = [m.span() for m in _LINK.finditer(fixed)]

        def restore_missing(match):
            key = _letters(match[0])
            options = variants.get(key, set())
            if len(options) != 1 or native_counts[key] != 1 or target_counts[key] != 1: return match[0]
            normalized = next(iter(options))
            if normalized == match[0] or any(a < match.end() and match.start() < b for a,b in protected): return match[0]
            changes.append({'original': match[0], 'normalized': normalized})
            return normalized

        fixed = token.sub(restore_missing, fixed)
    if fixed == text: return
    node.setdefault('source text before typography', text)
    node['content'] = fixed
    applied = document.setdefault('source typography repairs', [])
    applied.extend(dict(change, kind='accent', page=page_number, box=list(box)) for change in changes)
    overrides = document.setdefault('source glyph overrides', {})
    page = overrides.setdefault(str(page_number), {})
    indices = document.setdefault('source span glyph indices', {}).setdefault(str(page_number), {})
    for record in local:
        if any(c.get('glyph normalized',c['normalized']) == record['normalized'] for c in changes):
            page.update(record['glyph_overrides'])
            indices.update(record['span_indices'])


def normalized_spans(spans, document, page_number):
    """Keep source indices/bounds while matching normalized ordinary text."""
    changes = document.get('source glyph overrides', {}).get(str(page_number), {})
    indices = document.get('source span glyph indices', {}).get(str(page_number), {})
    hyphens = document.get('source discretionary hyphens',{}).get(str(page_number),{})
    if not changes and not hyphens: return spans
    result = []
    for span in spans:
        positions = indices.get(str(span.index))
        if not positions or len(positions) != len(span.text):
            result.append(span)
            continue
        text = ''.join(changes.get(str(index), char) for index,char in zip(positions, span.text))
        if text: result.append(replace(span, text=text))
    known = {s.index for s in result}
    result.extend(TextSpan(**dict(h,box=tuple(h['box']))) for h in hyphens.values() if h['index'] not in known)
    result.sort(key=lambda s:s.index)
    return result
