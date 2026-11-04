"""Scholarly article structure shared by Markdown and offline HTML exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from importlib.resources import files
from pathlib import Path
import re

from markdown_it import MarkdownIt

from .markup import FIGURE_LINK, image_reference, inline_image_tag


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
                  summary: str = "", language: str = "Korean") -> Article:
    if translations is not None and len(translations) != len(chunks):
        raise ValueError("원문과 번역 청크 수가 일치하지 않습니다.")
    article = Article(title, filename, document.get("number of pages", "?"), language,
                      str(document.get("author") or ""), summary=summary, translated=translations is not None)
    current: Section | None = None
    abstract_mode = False
    pending_figure: Unit | None = None
    used = {"abstract", "summary", "full-text", "figures"}
    figure_count = 0

    def append(unit: Unit) -> None:
        if abstract_mode: article.abstract.append(unit)
        elif current: current.units.append(unit)
        else: article.front.append(unit)

    for index, chunk in enumerate(chunks):
        translated = translations[index] if translations is not None else ""
        if chunk.kind in {"figure", "formula"}:
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
    for match in FIGURE_LINK.finditer(text):
        parts.append(escape(text[cursor:match.start()], quote=False))
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
    text = f"<!-- p.{page} -->\n\n" + unit.source
    if unit.translation:
        text += "\n\n<details>\n<summary>번역</summary>\n\n" + unit.translation + "\n\n</details>"
    if unit.caption:
        text += "\n\n" + _unit_markdown(unit.caption)
    return text


def article_markdown(article: Article) -> str:
    lines = [f"# {article.title}", ""]
    if article.author: lines.extend([article.author, ""])
    lines.extend([f"- 원본 파일: `{article.filename}`", f"- 페이지: {article.pages}", ""])
    if article.front:
        lines.extend(["<details>", "<summary>저자 및 원문 정보</summary>", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.front)
        lines.extend(["</details>", ""])
    if article.abstract:
        lines.extend(["## Abstract", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in article.abstract)
    if article.summary: lines.extend(["## 논문 요약", "", article.summary, ""])
    if article.sections:
        lines.extend(["## 목차", ""])
        lines.extend(f"{'  ' * (section.level - 2)}- [{_plain(section.title)}](#{section.anchor})" for section in article.sections)
        lines.append("")
    for section in article.sections:
        lines.extend([f"{'#' * section.level} {section.title}", ""])
        lines.extend(_unit_markdown(unit) + "\n" for unit in section.units)
    lines.extend(["---", "", "원문 PDF에서 추출한 문서입니다. 이미지 수식은 원본 모양을 보존합니다.", ""])
    return "\n".join(lines)


def _unit_html(unit: Unit) -> str:
    pages = ", ".join(map(str, unit.pages))
    if unit.kind == "figure":
        match = next(FIGURE_LINK.finditer(unit.source), None)
        if not match: return ""
        ref = image_reference(match)
        caption = _pair_html(unit.caption, caption=True) if unit.caption else f"<span>그림 · p.{pages}</span>"
        return (f'<figure class="paper-figure" id="{unit.anchor}"><a href="{escape(ref.source)}" target="_blank" '
                f'rel="noopener"><img src="{escape(ref.source)}" alt="{escape(ref.alt)}" loading="lazy"></a>'
                f'<figcaption>{caption}</figcaption><a class="figure-open" href="{escape(ref.source)}" '
                f'target="_blank" rel="noopener">원본 크기 보기 ↗</a></figure>')
    if unit.kind == "formula":
        return '<div class="equation-block">' + _md_html(unit.source) + '</div>'
    return _pair_html(unit)


def _pair_html(unit: Unit, caption: bool = False) -> str:
    page = ", ".join(map(str, unit.pages))
    source = _md_html(unit.source, inline=caption)
    translated = _md_html(unit.translation, inline=caption) if unit.translation else ""
    target = f'<div class="translation" lang="ko">{translated}</div>' if translated else ""
    return (f'<div class="reading-pair{ " has-translation" if translated else ""}"><div class="source">{source}</div>'
            f'{target}</div><span class="page-reference">p.{page}</span>')


def article_html(article: Article, markdown_name: str) -> str:
    navigation = []
    if article.abstract: navigation.append('<a href="#abstract">Abstract</a>')
    if article.summary: navigation.append('<a href="#summary">논문 요약</a>')
    navigation.extend(f'<a class="depth-{section.level}" href="#{section.anchor}">{escape(_plain(section.title))}</a>'
                      for section in article.sections)
    figures = [unit for section in article.sections for unit in section.units if unit.kind == "figure"]
    if figures:
        navigation.append('<h3>그림</h3>')
        for number, unit in enumerate(figures, 1):
            label = f"그림 {number}"
            if unit.caption and (match := _CAPTION.match(unit.caption.source)): label = match.group(0).lstrip("- ")
            navigation.append(f'<a href="#{unit.anchor}">{escape(label)} · p.{unit.pages[0]}</a>')
    abstract = ('<section class="abstract-card" id="abstract"><h2>Abstract</h2>' +
                "".join(_unit_html(unit) for unit in article.abstract) + '</section>') if article.abstract else ""
    summary = ('<section class="summary-card" id="summary"><h2>논문 요약</h2>' + _md_html(article.summary) + '</section>') if article.summary else ""
    front = ('<details class="author-details"><summary>저자 및 원문 정보</summary>' +
             "".join(_unit_html(unit) for unit in article.front) + '</details>') if article.front else ""
    body = "".join(f'<section class="paper-section" id="{section.anchor}"><h{section.level}>{_md_html(section.title, True)}</h{section.level}>'
                   + "".join(_unit_html(unit) for unit in section.units) + '</section>' for section in article.sections)
    checked_source = "" if article.translated else "checked"
    checked_both = "checked" if article.translated else ""
    controls = ('<label for="view-source">원문</label><label for="view-translated">번역</label>'
                '<label for="view-both">함께 보기</label>') if article.translated else '<span>원문</span>'
    top_links = (['<a href="#abstract">초록</a>'] if article.abstract else []) + ['<a href="#full-text">본문</a>']
    for section in article.sections:
        if re.search(r"references|참고문헌", _plain(section.title), re.I):
            top_links.append(f'<a href="#{section.anchor}">참고문헌</a>'); break
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(article.title)}</title><link rel="stylesheet" href="assets/article.css"></head>
<body><div class="masthead"><a href="#article-title">PAPER READER</a><span>LOCAL RESEARCH LIBRARY</span></div>
<div class="article-shell"><input class="view-input" type="radio" name="view" id="view-source" {checked_source}>
<input class="view-input" type="radio" name="view" id="view-translated"><input class="view-input" type="radio" name="view" id="view-both" {checked_both}>
<header class="article-header"><p class="eyebrow">RESEARCH ARTICLE · {article.pages} PAGES</p>
<h1 id="article-title">{escape(article.title)}</h1><div class="authors">{_md_html(article.author, True)}</div>
<div class="document-meta"><span>{escape(article.filename)}</span><a href="{escape(markdown_name)}" download>Markdown 다운로드 ↓</a></div>
<div class="article-toolbar"><nav>{''.join(top_links)}</nav>
<div class="view-switch" aria-label="읽기 언어">{controls}</div></div></header>
<div class="article-layout"><aside class="contents"><details open><summary>문서 목차</summary><nav>{''.join(navigation)}</nav></details></aside>
<main class="paper-body" id="full-text">{front}{abstract}{summary}{body}
<footer>원문 PDF에서 추출한 로컬 문서입니다. 수식 이미지는 원본 모양을 보존하며, 번역과 요약은 원문 대조가 필요합니다.</footer>
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
    from .bundle import build_bundle
    output.with_suffix(".zip").write_bytes(build_bundle(output))
    return output
