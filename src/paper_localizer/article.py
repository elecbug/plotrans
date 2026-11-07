"""Scholarly article structure shared by Markdown and offline HTML exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from importlib.resources import files
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt

from .markup import FIGURE_LINK, image_reference, inline_image_tag
from .math_markup import LATEX, expression, mathml, render_math
from .tables import table_html
from .source_links import assign_source_links, web_target


_MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_ABSTRACT = re.compile(r"^(?:Abstract|초록)\s*[.：:\-—]?\s*", re.I)
_CAPTION = re.compile(r"^(?:-\s*)?(?:Fig(?:ure)?\.?|Table|그림|표)\s*\d+", re.I)


@dataclass
class Unit:
    source: str
    translation: str = ""
    kind: str = "text"
    pages: tuple[int, ...] = ()
    caption: Unit | None = None
    anchor: str = ""
    paragraph_id: str = ""
    table_data: dict | None = None
    links: list[dict] = field(default_factory=list)
    image_size: tuple[int, ...] = ()
    translated_table_data: dict | None = None


@dataclass
class Section:
    title: str
    anchor: str
    level: int = 2
    units: list[Unit] = field(default_factory=list)
    translation: str = ""


@dataclass
class Article:
    title: str
    filename: str
    pages: int | str
    language: str
    author: str = ""
    front: list[Unit] = field(default_factory=list)
    abstract: list[Unit] = field(default_factory=list)
    keywords: list[Unit] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    translated: bool = False
    title_markup: str = ""
    title_translation: str = ""
    warnings: list[str] = field(default_factory=list)
    ocr_sources: list[dict] = field(default_factory=list)
    vision_sources: list[dict] = field(default_factory=list)
    identifiers: list[dict] = field(default_factory=list)
    source_links: list[dict] = field(default_factory=list)


def _plain(text: str) -> str:
    return FIGURE_LINK.sub("", text).strip().lstrip("# ")


def _slug(text: str, used: set[str]) -> str:
    slug = re.sub(r"[^\w\- ]", "", _plain(text).lower()).strip().replace(" ", "-") or "section"
    anchor, count = slug, 1
    while anchor in used:
        count += 1; anchor = f"{slug}-{count}"
    used.add(anchor)
    return anchor


def build_article(document: dict, title: str, filename: str, chunks, translations=None,
                  language: str = "English") -> Article:
    if translations is not None and len(translations) != len(chunks):
        raise ValueError("The source and translation chunk counts do not match.")
    article = Article(title, filename, document.get("number of pages", "?"), language,
                      str(document.get("author") or ""), translated=translations is not None)
    if issues := document.get('translation issues'):
        excerpts = sum(issue['kind'] == 'excerpt' for issue in issues)
        cells = sum(issue['kind'] == 'table-cell' for issue in issues)
        article.warnings.append(f"Translation validation retained {excerpts} source excerpt(s) and {cells} source table cell(s). "
                                "See the accompanying .translation.json report for affected locations.")
    if pages := document.get("source image pages"):
        article.warnings.append("Pages " + ", ".join(map(str, pages)) +
                                " use an undecodable font. Complete source pages are preserved as images; OCR is required to translate their text.")
    article.ocr_sources = document.get("ocr sources", [])
    article.vision_sources = document.get("vision sources", [])
    if vision := document.get("vision ocr"):
        rejected = sum(not region["accepted"] for region in vision["regions"])
        article.warnings.append(f"Local vision OCR: {vision['model']}. {rejected} region(s) failed consistency checks and retain their source representation. OCR consistency does not guarantee correctness.")
    if pages := document.get("ocr pages"):
        article.warnings.append("Local CPU OCR was used on pages " + ", ".join(map(str,pages)) +
            ". OCR can misread text. Original pages are available below; uncertain text and notation are preserved as source images.")
    if pages := document.get("ocr failures"):
        article.warnings.append("OCR found no readable text on pages " + ", ".join(map(str,pages)) + ". Check the original pages.")
    current: Section | None = None
    abstract_mode = False
    pending_figure: Unit | None = None
    used = {"abstract", "full-text", "figures", "ocr-sources"}
    figure_count = 0
    classified_abstract_seen = False
    image_sizes = {asset["source"]: asset.get("pixel size", ()) for asset in document.get("figures", [])}

    def append(unit: Unit) -> None:
        nonlocal current
        if current is None and classified_abstract_seen and not abstract_mode:
            # Some papers start their body without an Introduction heading.
            # Explicit source metadata still goes to article.front above.
            current = Section('Full text',_slug('Full text',used))
            article.sections.append(current)
        target = article.abstract if abstract_mode else current.units if current else article.front
        previous = next((other for other in reversed(target)
                         if unit.paragraph_id and other.paragraph_id == unit.paragraph_id), None)
        if (previous and unit.kind == previous.kind == "text"
                and bool(previous.translation) == bool(unit.translation)):
            previous.source += " " + unit.source
            if unit.translation:
                previous.translation = (previous.translation + " " + unit.translation).strip()
            previous.pages = tuple(sorted(set(previous.pages + unit.pages)))
        else:
            target.append(unit)

    for index, chunk in enumerate(chunks):
        translated = translations[index] if translations is not None else ""
        if chunk.reference_section:
            translated = ""
        if chunk.kind == "frontmatter":
            article.front.append(Unit(chunk.source, pages=chunk.pages))
            pending_figure = None
            continue
        if chunk.kind == "keywords":
            strip_label = lambda value: re.sub(r"^(?:index terms|keywords)\s*[:.\-—]?\s*", "", value, flags=re.I)
            article.keywords.append(Unit(strip_label(chunk.source), strip_label(translated), pages=chunk.pages))
            abstract_mode = False
            pending_figure = None
            continue
        if chunk.kind in {"abstract", "abstract-image"}:
            classified_abstract_seen = True
            if chunk.source.strip():
                article.abstract.append(Unit(_ABSTRACT.sub("", chunk.source), _ABSTRACT.sub("", translated),
                                             kind="formula" if chunk.kind == "abstract-image" else "text", pages=chunk.pages))
            pending_figure = None
            continue
        if chunk.kind == "table" and chunk.table_data:
            append(Unit(chunk.source, translated, kind="native-table", pages=chunk.pages, table_data=chunk.table_data,
                        translated_table_data=document.get('translated native tables',{}).get(str(chunk.index))))
            pending_figure = None
            continue
        if chunk.kind in {"caption", "caption-image"}:
            # A source-only caption crop has no machine translation. Showing
            # it in both columns would duplicate the image under a false label.
            unit = Unit(chunk.source, translated if chunk.kind == "caption" else "", kind=chunk.kind, pages=chunk.pages)
            if pending_figure and chunk.caption_target:
                pending_figure.caption = unit
            else:
                append(unit)
            pending_figure = None
            continue
        if chunk.kind in {"figure", "table-image", "portrait", "formula", "code"}:
            if document.get("source image pages") and current is None:
                current = Section("Source pages", "source-pages")
                article.sections.append(current); abstract_mode = False
            unit = Unit(chunk.source, kind=chunk.kind, pages=chunk.pages)
            if image := next(FIGURE_LINK.finditer(chunk.source), None):
                size = image_sizes.get(image_reference(image).source, ())
                if len(size) == 2 and all(isinstance(value, int) and value > 0 for value in size):
                    unit.image_size = tuple(size)
            if chunk.kind == "table-image":
                image = next(FIGURE_LINK.finditer(chunk.source), None)
                if image:
                    unit.table_data = document.get("table transcriptions", {}).get(image_reference(image).source)
            if chunk.kind in {"figure", "table-image"}:
                figure_count += 1; unit.anchor = f"figure-{figure_count}"
            append(unit)
            pending_figure = unit if chunk.kind in {"figure", "table-image"} else None
            continue
        # Model/OCR text may itself contain blank lines. They are not the
        # boundaries used when assigning the source paragraph identifiers.
        source_parts = list(chunk.source_parts) if chunk.source_parts else chunk.source.split("\n\n")
        paragraph_ids = list(chunk.paragraph_ids) if chunk.paragraph_ids else [""] * len(source_parts)
        if len(paragraph_ids) != len(source_parts):
            # Accept older serialized chunks without inventing a correspondence
            # between their text fragments and a shorter identifier list.
            paragraph_ids = [""] * len(source_parts)
        target_parts = translated.split("\n\n") if translated else []
        heading = _HEADING.match(source_parts[0])
        if heading:
            heading_text = heading.group(2)
            heading_translation = ""
            source_parts.pop(0)
            paragraph_ids.pop(0)
            if target_parts and (translated_heading := _HEADING.match(target_parts[0])):
                heading_translation = translated_heading.group(2)
                target_parts.pop(0)
            if _plain(heading_text).casefold() != _plain(title).casefold():
                if _ABSTRACT.fullmatch(_plain(heading_text)):
                    abstract_mode = True
                else:
                    level = max(2, min(4, len(heading.group(1))))
                    current = Section(heading_text, _slug(heading_text, used), level)
                    current.translation = heading_translation
                    article.sections.append(current); abstract_mode = False
            else:
                article.title_markup = heading_text
                article.title_translation = heading_translation
            pending_figure = None
        if target_parts and len(target_parts) != len(source_parts):
            pairs = [("\n\n".join(source_parts), "\n\n".join(target_parts), "")]
        else:
            pairs = [(source, target_parts[i] if target_parts else "", paragraph_ids[i]) for i, source in enumerate(source_parts)]
        for source, target, paragraph_id in pairs:
            if not source.strip(): continue
            unit = Unit(source, target, pages=chunk.pages, paragraph_id=paragraph_id)
            if _ABSTRACT.match(source) and current is None:
                abstract_mode = True
                unit.source = _ABSTRACT.sub("", source, count=1)
                unit.translation = _ABSTRACT.sub("", target, count=1)
                append(unit)
            elif pending_figure and _CAPTION.match(source) and "\n\n" not in source:
                unit.source = re.sub(r"^-\s+", "", source)
                unit.translation = re.sub(r"^-\s+", "", target)
                pending_figure.caption = unit
            else:
                append(unit)
            pending_figure = None
    units = article.front + article.abstract + article.keywords + [unit for section in article.sections for unit in section.units]
    units += [unit.caption for unit in list(units) if unit.caption]
    article.identifiers, article.source_links = assign_source_links(document.get("source links", []), units)
    return article


def _safe_markdown(text: str, math_tokens: dict[str, str] | None = None,
                   math_originals: dict[str, str] | None = None) -> str:
    parts, cursor = [], 0
    token_prefix = "PLoTransOpaqueMath"
    while token_prefix in text:
        token_prefix += "X"
    tokens = [(match.start(), match, "image") for match in FIGURE_LINK.finditer(text)]
    tokens += [(match.start(), match, "math") for match in LATEX.finditer(text)
               if not any(image.start() <= match.start() < image.end() for image in FIGURE_LINK.finditer(text))]
    for _, match, kind in sorted(tokens, key=lambda item: item[0]):
        if match.start() < cursor: continue
        parts.append(escape(text[cursor:match.start()], quote=False))
        if kind == "math":
            try:
                rendered = mathml(expression(match), bool(match.group(1) or match.group(4)))
                rendered = '<span class="math-scroll">' + rendered + '</span>'
                if math_tokens is not None:
                    token = f"{token_prefix}{len(math_tokens)}End"
                    math_tokens[token] = rendered
                    if math_originals is not None:
                        math_originals[token] = match.group(0)
                    parts.append(token)
                else:
                    parts.append(rendered)
            except Exception: parts.append(escape(match.group(0)))
            cursor = match.end()
            continue
        ref = image_reference(match)
        if ref.inline:
            parts.append(f'<span class="inline-math-scroll" style="vertical-align:{ref.align_em:.3f}em">' +
                         inline_image_tag(ref.source, ref.alt, ref.height_em, 0, ref.aspect_ratio) + '</span>')
        else:
            parts.append(f"![{escape(ref.alt)}]({ref.source})")
        cursor = match.end()
    parts.append(escape(text[cursor:], quote=False))
    return "".join(parts)


def _md_html(text: str, inline: bool = False) -> str:
    math_tokens: dict[str, str] = {}
    math_originals: dict[str, str] = {}
    safe = _safe_markdown(text, math_tokens, math_originals)
    rendered = _MARKDOWN.renderInline(safe) if inline else _MARKDOWN.render(safe)
    # MathML must remain opaque to Markdown's emphasis/link parsers. Literal
    # stars in two equations otherwise produce crossed <em>/<mo> elements.
    if math_tokens:
        pattern = re.compile("|".join(map(re.escape, math_tokens)))
        parts = re.split(r"(<[^>]*>)", rendered)
        for i,part in enumerate(parts):
            # A dollar-containing URL or link title must remain an attribute,
            # rather than receiving an HTML subtree inside its quoted value.
            parts[i] = pattern.sub(lambda match: escape(math_originals[match.group()], quote=True)
                                   if i%2 else math_tokens[match.group()], part)
        rendered = "".join(parts)
    return rendered


def _unit_markdown(unit: Unit) -> str:
    page = ", ".join(map(str, unit.pages))
    if unit.kind == "native-table" and unit.table_data:
        text = f"<!-- p.{page} -->\n\n" + table_html(unit.table_data["rows"], lambda text: _md_html(text, inline=True))
        if unit.translation:
            target = (table_html(unit.translated_table_data['rows'], lambda text: _md_html(text, inline=True))
                      if unit.translated_table_data else render_math(unit.translation))
            text += "\n\n<details>\n<summary>Translation</summary>\n\n" + target + "\n\n</details>"
        return text + ("\n\n" + _links_markdown(unit.links) if unit.links else "")
    text = f"<!-- p.{page} -->\n\n" + (unit.source if unit.kind == "code" else render_math(unit.source))
    if unit.translation:
        text += "\n\n<details>\n<summary>Translation</summary>\n\n" + render_math(unit.translation) + "\n\n</details>"
    if unit.caption:
        if unit.kind == "table-image":
            text = _unit_markdown(unit.caption) + "\n\n" + text
        else:
            text += "\n\n" + _unit_markdown(unit.caption)
    if unit.table_data:
        text += "\n\n" + _table_transcription(unit)
    if unit.links:
        text += "\n\n" + _links_markdown(unit.links)
    return text


def _links_markdown(links: list[dict]) -> str:
    return " · ".join(f"[{link['label']}]({target})" for link in links if (target := web_target(link.get("url"))))


def _links_html(links: list[dict]) -> str:
    anchors = [f'<a href="{escape(target)}" target="_blank" rel="noopener noreferrer">{escape(link["label"])}</a>'
               for link in links if (target := web_target(link.get("url")))]
    return '<p class="source-links">' + " · ".join(anchors) + '</p>' if anchors else ""


def _table_transcription(unit: Unit) -> str:
    if not unit.table_data: return ""
    native = unit.table_data.get('verification', {}).get('method') == 'native-geometry'
    explanation = ('Native table cells checked against positioned source text with independent PDF engines. '
                   if native else 'OCR transcription checked across rendering resolutions and against independent OCR. ')
    return ('<details class="table-transcription"><summary>Searchable table transcription</summary>'
            '<p>' + explanation +
            'The original image remains the reference for values and notation.</p>'
            + table_html(unit.table_data["rows"], lambda value: _md_html(value, inline=True)) + '</details>')


def article_markdown(article: Article) -> str:
    lines = [f"# {article.title_markup or article.title}", ""]
    if article.title_translation:
        lines.extend(['<details><summary>Title translation</summary>', '', render_math(article.title_translation), '', '</details>', ''])
    if article.author: lines.extend([article.author, ""])
    lines.extend([f"- Source file: `{article.filename}`", f"- Pages: {article.pages}", ""])
    if article.identifiers:
        lines.extend(["Source identifiers: " + _links_markdown(article.identifiers), ""])
    for warning in article.warnings:
        lines.extend(["> Parsing note: " + warning, ""])
    if article.front:
        if article.sections or article.abstract:
            lines.extend(["<details>", "<summary>Authors and source information</summary>", ""])
            lines.extend(_unit_markdown(unit) + "\n" for unit in article.front)
            lines.extend(["</details>", ""])
        else:
            lines.extend(_unit_markdown(unit) + "\n" for unit in article.front)
    if article.abstract:
        lines.extend(["## Abstract", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.abstract)
    if article.keywords:
        lines.extend(["## Index terms", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.keywords)
    if article.sections:
        lines.extend(["## Contents", ""])
        lines.extend(f"{'  ' * (section.level - 2)}- [{_plain(section.title)}](#{section.anchor})" for section in article.sections)
        lines.append("")
    for section in article.sections:
        lines.extend([f"{'#' * section.level} {section.title}", ""])
        if section.translation:
            lines.extend(['<details><summary>Heading translation</summary>', '', render_math(section.translation), '', '</details>', ''])
        lines.extend(_unit_markdown(unit) + "\n" for unit in section.units)
    if article.ocr_sources:
        lines.extend(["<details>", "<summary>Original pages used for OCR</summary>", ""])
        for source in article.ocr_sources:
            lines.extend([f"![Original source page {source['page']}]({source['source']})", ""])
        lines.extend(["</details>", ""])
    if article.vision_sources:
        lines.extend(["<details>", "<summary>Source regions transcribed by vision OCR</summary>", ""])
        for source in article.vision_sources:
            lines.extend([f"![Source region p.{source['page']}]({source['source']})", ""])
        lines.extend(["</details>", ""])
    if article.source_links:
        lines.extend(["<details>", "<summary>Other links from the source PDF</summary>", ""])
        for link in article.source_links:
            lines.extend([f"- [{link['url']}]({link['url']}) · p." + ", ".join(map(str, link.get("pages", []))), ""])
        lines.extend(["</details>", ""])
    lines.extend(["---", "", "Extracted from the source PDF. Equation images preserve the original notation.", ""])
    return "\n".join(lines)


def _unit_html(unit: Unit) -> str:
    pages = ", ".join(map(str, unit.pages))
    if unit.kind == "native-table" and unit.table_data:
        source = '<div class="native-table-scroll">' + table_html(unit.table_data["rows"], lambda text: _md_html(text, inline=True)) + '</div>'
        target = ('<div class="native-table-scroll">' + table_html(unit.translated_table_data['rows'], lambda text: _md_html(text, inline=True)) + '</div>'
                  if unit.translated_table_data else _md_html(unit.translation))
        translated = '<div class="translation">' + target + '</div>' if unit.translation else ""
        return (f'<div class="reading-pair{" has-translation" if translated else ""}"><div class="source">{source}</div>'
                + translated + f'</div><span class="page-reference">p.{pages}</span>' + _links_html(unit.links))
    dimensions = f' width="{unit.image_size[0]}" height="{unit.image_size[1]}"' if unit.image_size else ""
    if unit.kind == "code":
        source = re.sub(r"^```[^\n]*\n|\n```$", "", unit.source)
        return '<pre><code>' + escape(source) + f'</code></pre><span class="page-reference">p.{pages}</span>'
    if unit.kind == "portrait":
        match = next(FIGURE_LINK.finditer(unit.source), None)
        if not match: return ""
        ref = image_reference(match)
        return (f'<figure class="author-portrait"><a href="{escape(ref.source)}" target="_blank" rel="noopener">'
                f'<img src="{escape(ref.source)}" alt="{escape(ref.alt)}"{dimensions} loading="lazy"></a></figure>')
    if unit.kind in {"figure", "table-image"}:
        match = next(FIGURE_LINK.finditer(unit.source), None)
        if not match: return ""
        ref = image_reference(match)
        caption = _pair_html(unit.caption, caption=True) if unit.caption else f"<span>Fig. · p.{pages}</span>"
        label = f'<figcaption>{caption}</figcaption>'
        is_table = unit.kind == "table-image"
        return (f'<figure class="paper-figure{ " paper-table" if is_table else ""}" id="{unit.anchor}">'
                f'{label if is_table else ""}<a href="{escape(ref.source)}" target="_blank" '
                f'rel="noopener"><img src="{escape(ref.source)}" alt="{escape(ref.alt)}"{dimensions} loading="lazy"></a>'
                f'{"" if is_table else label}<a class="figure-open" href="{escape(ref.source)}" '
                f'target="_blank" rel="noopener">View full size ↗</a>{_table_transcription(unit)}</figure>')
    if unit.kind == "formula":
        return '<div class="equation-block">' + _md_html(unit.source) + '</div>'
    return _pair_html(unit)


def _pair_html(unit: Unit, caption: bool = False) -> str:
    page = ", ".join(map(str, unit.pages))
    source = _md_html(unit.source, inline=caption)
    translated = _md_html(unit.translation, inline=caption) if unit.translation else ""
    target = f'<div class="translation">{translated}</div>' if translated else ""
    return (f'<div class="reading-pair{ " has-translation" if translated else ""}"><div class="source">{source}</div>'
            f'{target}</div><span class="page-reference">p.{page}</span>' + _links_html(unit.links))


def _heading_html(source: str, translation: str) -> str:
    original = _md_html(source, True)
    if not translation:
        return original
    return ('<span class="heading-source">' + original + '</span>'
            '<span class="heading-translation">' + _md_html(translation, True) + '</span>')


def article_html(article: Article, markdown_name: str) -> str:
    navigation = []
    if article.ocr_sources: navigation.append('<a href="#ocr-sources">OCR source pages</a>')
    if article.abstract: navigation.append('<a href="#abstract">Abstract</a>')
    navigation.extend(f'<a class="depth-{section.level}" href="#{section.anchor}">{_heading_html(_plain(section.title), _plain(section.translation))}</a>'
                      for section in article.sections)
    figures = [unit for section in article.sections for unit in section.units if unit.kind in {"figure", "table-image"}]
    if figures:
        navigation.append('<h3>Figures and tables</h3>')
        for number, unit in enumerate(figures, 1):
            label = f"Fig. {number}"
            if unit.caption and (match := _CAPTION.match(unit.caption.source)): label = match.group(0).lstrip("- ")
            navigation.append(f'<a href="#{unit.anchor}">{escape(label)} · p.{unit.pages[0]}</a>')
    abstract = ('<section class="abstract-card" id="abstract"><h2>Abstract</h2>' +
                "".join(_unit_html(unit) for unit in article.abstract) + '</section>') if article.abstract else ""
    keywords = ('<section class="keywords"><h2>Index terms</h2>' +
                "".join(_unit_html(unit) for unit in article.keywords) + '</section>') if article.keywords else ""
    front = "".join(_unit_html(unit) for unit in article.front)
    if front and (article.sections or article.abstract):
        front = '<details class="author-details"><summary>Authors and source information</summary>' + front + '</details>'
    front = "".join('<aside class="parsing-note">' + escape(warning) + '</aside>' for warning in article.warnings) + front
    body = "".join(f'<section class="paper-section" id="{section.anchor}"><h{section.level}>{_heading_html(section.title, section.translation)}</h{section.level}>'
                   + "".join(_unit_html(unit) for unit in section.units) + '</section>' for section in article.sections)
    if article.ocr_sources:
        body += '<details class="ocr-sources" id="ocr-sources"><summary>Original pages used for OCR</summary>'
        for source in article.ocr_sources:
            body += (f'<figure><a href="{escape(source["source"])}"><img src="{escape(source["source"])}" '
                     f'alt="Original source page {source["page"]}" loading="lazy"></a><figcaption>Source page {source["page"]}</figcaption></figure>')
        body += '</details>'
    if article.vision_sources:
        body += '<details class="ocr-sources"><summary>Source regions transcribed by vision OCR</summary>'
        for source in article.vision_sources:
            body += (f'<figure><img src="{escape(source["source"])}" alt="Source region p.{source["page"]}" loading="lazy">'
                     f'<figcaption>Source page {source["page"]}</figcaption></figure>')
        body += '</details>'
    if article.source_links:
        body += '<details class="source-link-list"><summary>Other links from the source PDF</summary><ul>'
        for link in article.source_links:
            pages = ", ".join(map(str, link.get("pages", [])))
            body += f'<li><a href="{escape(link["url"])}" target="_blank" rel="noopener noreferrer">{escape(link["url"])}</a> · p.{pages}</li>'
        body += '</ul></details>'
    checked_source = "" if article.translated else "checked"
    checked_both = "checked" if article.translated else ""
    controls = ('<label for="view-source">Original</label><label for="view-translated">Translation</label>'
                '<label for="view-both">Parallel view</label>') if article.translated else '<span>Original</span>'
    top_links = (['<a href="#abstract">Abstract</a>'] if article.abstract else []) + ['<a href="#full-text">Full text</a>']
    for section in article.sections:
        if re.search(r"references|참고문헌", _plain(section.title), re.I):
            top_links.append(f'<a href="#{section.anchor}">References</a>'); break
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(article.title)}</title><link rel="stylesheet" href="assets/article.css"></head>
<body><div class="masthead"><a href="#article-title">PLoTrans</a><span>LOCAL RESEARCH LIBRARY</span></div>
<div class="article-shell"><input class="view-input" type="radio" name="view" id="view-source" {checked_source}>
<input class="view-input" type="radio" name="view" id="view-translated"><input class="view-input" type="radio" name="view" id="view-both" {checked_both}>
<header class="article-header"><p class="eyebrow">RESEARCH ARTICLE · {article.pages} PAGES</p>
<h1 id="article-title">{_heading_html(article.title_markup or article.title, article.title_translation)}</h1><div class="authors">{_md_html(article.author, True)}</div>{_links_html(article.identifiers)}
<div class="document-meta"><span>{escape(article.filename)}</span><a href="{escape(markdown_name)}" download>Download Markdown ↓</a></div>
<div class="article-toolbar"><nav>{''.join(top_links)}</nav>
<div class="view-switch" aria-label="Reading view">{controls}</div></div></header>
<div class="article-layout"><aside class="contents"><details open><summary>Document contents</summary><nav>{''.join(navigation)}</nav></details></aside>
<main class="paper-body" id="full-text">{front}{abstract}{keywords}{body}
<footer>A local reading edition extracted from the source PDF. Equation images preserve the original notation. Verify translations against the source.</footer>
</main></div></div></body></html>'''


def write_article(document: dict, title: str, filename: str, chunks, translations,
                  language: str, output: Path) -> Path:
    article = build_article(document, title, filename, chunks, translations, language)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / "assets"; assets.mkdir(exist_ok=True)
    css = files("paper_localizer").joinpath("assets/article.css").read_text(encoding="utf-8")
    (assets / "article.css").write_text(css, encoding="utf-8")
    for destination, text in [(output, article_markdown(article)),
                              (output.with_suffix(".html"), article_html(article, output.name))]:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8"); temporary.replace(destination)
    ocr_report = output.with_suffix(".ocr.json")
    vision = document.get('vision ocr', {})
    if vision or document.get('native table transcriptions'):
        ocr_report.write_text(json.dumps({"vision": vision, "equations": document.get("math transcriptions", {}),
                                         "native_tables": document.get('native table transcriptions', []),
                                         "tables": document.get("table transcriptions", {})},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        ocr_report.unlink(missing_ok=True)
    from .bundle import build_bundle
    output.with_suffix(".zip").write_bytes(build_bundle(output))
    return output
