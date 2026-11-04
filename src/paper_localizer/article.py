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


@dataclass
class Section:
    title: str
    anchor: str
    level: int = 2
    units: list[Unit] = field(default_factory=list)


@dataclass
class Article:
    title: str
    filename: str
    pages: int | str
    language: str
    author: str = ""
    front: list[Unit] = field(default_factory=list)
    abstract: list[Unit] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    summary: str = ""
    translated: bool = False
    title_markup: str = ""
    warnings: list[str] = field(default_factory=list)
    ocr_sources: list[dict] = field(default_factory=list)
    vision_sources: list[dict] = field(default_factory=list)


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
                  summary: str = "", language: str = "English") -> Article:
    if translations is not None and len(translations) != len(chunks):
        raise ValueError("The source and translation chunk counts do not match.")
    article = Article(title, filename, document.get("number of pages", "?"), language,
                      str(document.get("author") or ""), summary=summary, translated=translations is not None)
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
    used = {"abstract", "summary", "full-text", "figures", "ocr-sources"}
    figure_count = 0

    def append(unit: Unit) -> None:
        if abstract_mode: article.abstract.append(unit)
        elif current: current.units.append(unit)
        else: article.front.append(unit)

    for index, chunk in enumerate(chunks):
        translated = translations[index] if translations is not None else ""
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
        if chunk.kind in {"figure", "formula", "code"}:
            if document.get("source image pages") and current is None:
                current = Section("Source pages", "source-pages")
                article.sections.append(current); abstract_mode = False
            unit = Unit(chunk.source, kind=chunk.kind, pages=chunk.pages)
            if chunk.kind == "figure":
                figure_count += 1; unit.anchor = f"figure-{figure_count}"
            append(unit)
            pending_figure = unit if chunk.kind == "figure" else None
            continue
        source_parts = chunk.source.split("\n\n")
        target_parts = translated.split("\n\n") if translated else []
        heading = _HEADING.match(source_parts[0])
        if heading:
            heading_text = heading.group(2)
            source_parts.pop(0)
            if target_parts and _HEADING.match(target_parts[0]): target_parts.pop(0)
            if _plain(heading_text).casefold() != _plain(title).casefold():
                if _ABSTRACT.fullmatch(_plain(heading_text)):
                    abstract_mode = True
                else:
                    numbering = re.match(r"^(\d+(?:\.\d+)*)[.\s]", heading_text)
                    level = min(4, 2 + numbering.group(1).count(".")) if numbering else 2
                    current = Section(heading_text, _slug(heading_text, used), level)
                    article.sections.append(current); abstract_mode = False
            else:
                article.title_markup = heading_text
            pending_figure = None
        if target_parts and len(target_parts) != len(source_parts):
            pairs = [("\n\n".join(source_parts), "\n\n".join(target_parts))]
        else:
            pairs = [(source, target_parts[i] if target_parts else "") for i, source in enumerate(source_parts)]
        for source, target in pairs:
            if not source.strip(): continue
            unit = Unit(source, target, pages=chunk.pages)
            if _ABSTRACT.match(source) and current is None:
                abstract_mode = True
                unit.source = _ABSTRACT.sub("", source, count=1)
                unit.translation = _ABSTRACT.sub("", target, count=1)
                article.abstract.append(unit)
            elif pending_figure and _CAPTION.match(source) and "\n\n" not in source:
                unit.source = re.sub(r"^-\s+", "", source)
                unit.translation = re.sub(r"^-\s+", "", target)
                pending_figure.caption = unit
            else:
                append(unit)
            pending_figure = None
    if not article.author and article.front:
        article.author = article.front[0].source
    return article


def _safe_markdown(text: str) -> str:
    parts, cursor = [], 0
    tokens = [(match.start(), match, "image") for match in FIGURE_LINK.finditer(text)]
    tokens += [(match.start(), match, "math") for match in LATEX.finditer(text)
               if not any(image.start() <= match.start() < image.end() for image in FIGURE_LINK.finditer(text))]
    for _, match, kind in sorted(tokens, key=lambda item: item[0]):
        if match.start() < cursor: continue
        parts.append(escape(text[cursor:match.start()], quote=False))
        if kind == "math":
            try: parts.append(mathml(expression(match), bool(match.group(1) or match.group(4))))
            except Exception: parts.append(escape(match.group(0)))
            cursor = match.end()
            continue
        ref = image_reference(match)
        if ref.inline:
            parts.append(inline_image_tag(ref.source, ref.alt, ref.height_em, ref.align_em, ref.aspect_ratio))
        else:
            parts.append(f"![{escape(ref.alt)}]({ref.source})")
        cursor = match.end()
    parts.append(escape(text[cursor:], quote=False))
    return "".join(parts)


def _md_html(text: str, inline: bool = False) -> str:
    safe = _safe_markdown(text)
    return _MARKDOWN.renderInline(safe) if inline else _MARKDOWN.render(safe)


def _unit_markdown(unit: Unit) -> str:
    page = ", ".join(map(str, unit.pages))
    text = f"<!-- p.{page} -->\n\n" + (unit.source if unit.kind == "code" else render_math(unit.source))
    if unit.translation:
        text += "\n\n<details>\n<summary>Translation</summary>\n\n" + render_math(unit.translation) + "\n\n</details>"
    if unit.caption:
        text += "\n\n" + _unit_markdown(unit.caption)
    return text


def article_markdown(article: Article) -> str:
    lines = [f"# {article.title_markup or article.title}", ""]
    if article.author: lines.extend([article.author, ""])
    lines.extend([f"- Source file: `{article.filename}`", f"- Pages: {article.pages}", ""])
    for warning in article.warnings:
        lines.extend(["> Parsing note: " + warning, ""])
    if article.front:
        lines.extend(["<details>", "<summary>Authors and source information</summary>", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.front)
        lines.extend(["</details>", ""])
    if article.abstract:
        lines.extend(["## Abstract", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.abstract)
    if article.summary: lines.extend(["## Article summary", "", article.summary, ""])
    if article.sections:
        lines.extend(["## Contents", ""])
        lines.extend(f"{'  ' * (section.level - 2)}- [{_plain(section.title)}](#{section.anchor})" for section in article.sections)
        lines.append("")
    for section in article.sections:
        lines.extend([f"{'#' * section.level} {section.title}", ""])
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
    lines.extend(["---", "", "Extracted from the source PDF. Equation images preserve the original notation.", ""])
    return "\n".join(lines)


def _unit_html(unit: Unit) -> str:
    pages = ", ".join(map(str, unit.pages))
    if unit.kind == "code":
        source = re.sub(r"^```[^\n]*\n|\n```$", "", unit.source)
        return '<pre><code>' + escape(source) + f'</code></pre><span class="page-reference">p.{pages}</span>'
    if unit.kind == "figure":
        match = next(FIGURE_LINK.finditer(unit.source), None)
        if not match: return ""
        ref = image_reference(match)
        caption = _pair_html(unit.caption, caption=True) if unit.caption else f"<span>Fig. · p.{pages}</span>"
        return (f'<figure class="paper-figure" id="{unit.anchor}"><a href="{escape(ref.source)}" target="_blank" '
                f'rel="noopener"><img src="{escape(ref.source)}" alt="{escape(ref.alt)}" loading="lazy"></a>'
                f'<figcaption>{caption}</figcaption><a class="figure-open" href="{escape(ref.source)}" '
                f'target="_blank" rel="noopener">View full size ↗</a></figure>')
    if unit.kind == "formula":
        return '<div class="equation-block">' + _md_html(unit.source) + '</div>'
    return _pair_html(unit)


def _pair_html(unit: Unit, caption: bool = False) -> str:
    page = ", ".join(map(str, unit.pages))
    source = _md_html(unit.source, inline=caption)
    translated = _md_html(unit.translation, inline=caption) if unit.translation else ""
    target = f'<div class="translation">{translated}</div>' if translated else ""
    return (f'<div class="reading-pair{ " has-translation" if translated else ""}"><div class="source">{source}</div>'
            f'{target}</div><span class="page-reference">p.{page}</span>')


def article_html(article: Article, markdown_name: str) -> str:
    navigation = []
    if article.ocr_sources: navigation.append('<a href="#ocr-sources">OCR source pages</a>')
    if article.abstract: navigation.append('<a href="#abstract">Abstract</a>')
    if article.summary: navigation.append('<a href="#summary">Article summary</a>')
    navigation.extend(f'<a class="depth-{section.level}" href="#{section.anchor}">{escape(_plain(section.title))}</a>'
                      for section in article.sections)
    figures = [unit for section in article.sections for unit in section.units if unit.kind == "figure"]
    if figures:
        navigation.append('<h3>Figures</h3>')
        for number, unit in enumerate(figures, 1):
            label = f"Fig. {number}"
            if unit.caption and (match := _CAPTION.match(unit.caption.source)): label = match.group(0).lstrip("- ")
            navigation.append(f'<a href="#{unit.anchor}">{escape(label)} · p.{unit.pages[0]}</a>')
    abstract = ('<section class="abstract-card" id="abstract"><h2>Abstract</h2>' +
                "".join(_unit_html(unit) for unit in article.abstract) + '</section>') if article.abstract else ""
    summary = ('<section class="summary-card" id="summary"><h2>Article summary</h2>' + _md_html(article.summary) + '</section>') if article.summary else ""
    front = ('<details class="author-details"><summary>Authors and source information</summary>' +
             "".join(_unit_html(unit) for unit in article.front) + '</details>') if article.front else ""
    front = "".join('<aside class="parsing-note">' + escape(warning) + '</aside>' for warning in article.warnings) + front
    body = "".join(f'<section class="paper-section" id="{section.anchor}"><h{section.level}>{_md_html(section.title, True)}</h{section.level}>'
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
<h1 id="article-title">{_md_html(article.title_markup or article.title, True)}</h1><div class="authors">{_md_html(article.author, True)}</div>
<div class="document-meta"><span>{escape(article.filename)}</span><a href="{escape(markdown_name)}" download>Download Markdown ↓</a></div>
<div class="article-toolbar"><nav>{''.join(top_links)}</nav>
<div class="view-switch" aria-label="Reading view">{controls}</div></div></header>
<div class="article-layout"><aside class="contents"><details open><summary>Document contents</summary><nav>{''.join(navigation)}</nav></details></aside>
<main class="paper-body" id="full-text">{front}{abstract}{summary}{body}
<footer>A local reading edition extracted from the source PDF. Equation images preserve the original notation. Verify translations and summaries against the source.</footer>
</main></div></div></body></html>'''


def write_article(document: dict, title: str, filename: str, chunks, translations, summary: str,
                  language: str, output: Path) -> Path:
    article = build_article(document, title, filename, chunks, translations, summary, language)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / "assets"; assets.mkdir(exist_ok=True)
    css = files("paper_localizer").joinpath("assets/article.css").read_text(encoding="utf-8")
    (assets / "article.css").write_text(css, encoding="utf-8")
    for destination, text in [(output, article_markdown(article)),
                              (output.with_suffix(".html"), article_html(article, output.name))]:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8"); temporary.replace(destination)
    ocr_report = output.with_suffix(".ocr.json")
    if vision := document.get("vision ocr"):
        ocr_report.write_text(json.dumps({"vision": vision, "equations": document.get("math transcriptions", {})},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        ocr_report.unlink(missing_ok=True)
    from .bundle import build_bundle
    output.with_suffix(".zip").write_bytes(build_bundle(output))
    return output
