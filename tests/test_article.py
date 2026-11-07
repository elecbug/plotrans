from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from paper_localizer.article import article_html, build_article, write_article, _md_html
from paper_localizer.bundle import FIGURE_LINK, standalone_html
from paper_localizer.core import Block, Chunk, make_chunks
from paper_localizer.markup import image_reference, inline_image_tag


class Audit(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.ids = set(); self.links = []; self.scripts = 0
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs: self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"): self.links.append(attrs["href"][1:])
        if tag == "script": self.scripts += 1


class ArticleTests(unittest.TestCase):
    def test_reference_entries_ignore_model_generated_page_ranges(self):
        chunks=make_chunks([Block(1,'heading','## References'),
                            Block(1,'paragraph','[14] A paper. pp. 2142–2160.')])
        article=build_article({},'Paper','paper.pdf',chunks,['## 참고문헌\n\n[14] pp. 25–2160.'])
        html=article_html(article,'paper.md')
        self.assertIn('2142–2160',html)
        self.assertNotIn('25–2160',html)

    def test_body_without_an_abstract_or_section_is_not_collapsed_as_author_metadata(self):
        chunks=[Chunk(1,(1,),'# Short paper\n\nEach node forwards a message to two peers.')]
        article=build_article({},'Short paper','paper.pdf',chunks,
                              ['# 짧은 논문\n\n각 노드는 두 개의 피어에게 메시지를 전달합니다.'])
        html=article_html(article,'paper.md')
        self.assertNotIn('class="author-details"',html)
        self.assertIn('각 노드는 두 개의 피어에게 메시지를 전달합니다.',html)
        from paper_localizer.article import article_markdown
        self.assertNotIn('<summary>Authors and source information</summary>',article_markdown(article))

    def test_retained_source_segment_is_not_hidden_inside_a_translated_paragraph(self):
        chunks=[Chunk(1,(1,),'## Body'),
                Chunk(2,(1,),'First part.',paragraph_ids=('shared',),source_parts=('First part.',)),
                Chunk(3,(1,),'Retained original.',paragraph_ids=('shared',),source_parts=('Retained original.',))]
        article=build_article({},'Paper','paper.pdf',chunks,['## 본문','첫 부분.',''])
        units=article.sections[0].units
        self.assertEqual(len(units),2)
        self.assertEqual(units[1].source,'Retained original.')
        self.assertFalse(units[1].translation)
        html=article_html(article,'paper.md')
        self.assertIn('<div class="reading-pair"><div class="source"><p>Retained original.</p>',html)

    def test_original_summary_section_remains_without_a_generated_summary(self):
        article=build_article({},'Paper','paper.pdf',[Chunk(1,(1,),'## Summary\n\nAn original conclusion.')])
        html=article_html(article,'paper.md')
        self.assertIn('An original conclusion.',html)
        self.assertIn('id="summary"',html)
        self.assertNotIn('summary-card',html)
        self.assertNotIn('Article summary',html)

    def test_translated_title_and_section_headings_are_retained_in_exports(self):
        chunks=[Chunk(1,(1,),'# Network Study'),Chunk(2,(1,),'## 1 Introduction\n\nBody.')]
        article=build_article({},'Network Study','paper.pdf',chunks,
                              ['# 네트워크 연구','## 1 서론\n\n본문.'],language='Korean')
        self.assertEqual(article.title_translation,'네트워크 연구')
        self.assertEqual(article.sections[0].translation,'1 서론')
        html=article_html(article,'paper.md')
        self.assertIn('<span class="heading-source">Network Study</span>',html)
        self.assertIn('<span class="heading-translation">네트워크 연구</span>',html)
        self.assertIn('<span class="heading-translation">1 서론</span>',html)
        from paper_localizer.article import article_markdown
        md=article_markdown(article)
        self.assertIn('Title translation',md)
        self.assertIn('Heading translation',md)
        self.assertIn('네트워크 연구',md)

    def test_fixed_translated_cells_are_rendered_without_model_generated_table_structure(self):
        rows=[[{'text':'Header','rowspan':1,'colspan':2,'header':True}],
              [{'text':'Left','rowspan':1,'colspan':1,'header':False},
               {'text':'99.9%','rowspan':1,'colspan':1,'header':False}]]
        target=[[{'text':'머리글','rowspan':1,'colspan':2,'header':True}],
                [{'text':'왼쪽','rowspan':1,'colspan':1,'header':False},
                 {'text':'99.9%','rowspan':1,'colspan':1,'header':False}]]
        doc={'translated native tables':{'1':{'rows':target}}}
        chunks=[Chunk(1,(1,),'source table','table',table_data={'rows':rows})]
        article=build_article(doc,'Paper','paper.pdf',chunks,['untrusted model table text'])
        html=article_html(article,'paper.md')
        self.assertEqual(html.count('colspan="2"'),2)
        self.assertIn('머리글',html)
        self.assertIn('왼쪽',html)
        self.assertNotIn('untrusted model table text',html)

    def test_embedded_ocr_paragraphs_preserve_chunk_ownership(self):
        blocks = [Block(1, "heading", "## Body"),
                  Block(1, "paragraph", "First paragraph.\n\nSecond paragraph.", paragraph_id="original"),
                  Block(1, "paragraph", "Another source block.", paragraph_id="next")]
        article = build_article({}, "Paper", "paper.pdf", make_chunks(blocks))
        self.assertEqual([u.paragraph_id for u in article.sections[0].units], ["original", "next"])
        self.assertIn("Second paragraph.", article.sections[0].units[0].source)

    def test_legacy_chunks_with_extra_blank_lines_do_not_lose_text(self):
        chunks = [Chunk(1, (1,), "## Body\n\nFirst.\n\nSecond.", paragraph_ids=("one", "two"))]
        article = build_article({}, "Paper", "paper.pdf", chunks)
        self.assertEqual([u.source for u in article.sections[0].units], ["First.", "Second."])

    def test_starred_math_cannot_open_markdown_emphasis_across_equations(self):
        import xml.etree.ElementTree as ET
        html = _md_html(r"Call $N^{*}$ for some set $N^{*}$, then retain the following text.")
        self.assertNotIn("<em>", html)
        root = ET.fromstring(html.strip())
        self.assertEqual(len(root.findall(".//{http://www.w3.org/1998/Math/MathML}math")), 2)
        self.assertIn("retain the following text.", "".join(root.itertext()))

    def test_currency_amounts_do_not_turn_intervening_prose_into_math(self):
        html = _md_html("FileCoin is $1.4B USD. Eth2.0 has a market cap of $143B USD.")
        self.assertNotIn("<math", html)
        self.assertIn("$1.4B USD", html)
        self.assertIn("market cap of $143B USD", html)

    def test_math_like_url_remains_an_attribute_without_html_injection(self):
        html=_md_html('[source](https://example.org/$x$) with $y^2$.')
        self.assertIn('href="https://example.org/$x$"', html)
        self.assertEqual(html.count('<math '), 1)
        import xml.etree.ElementTree as ET
        root=ET.fromstring(html.strip())
        self.assertEqual(root.find('a').attrib['href'], 'https://example.org/$x$')

    def test_graphics_reserve_their_source_aspect_ratio_before_lazy_loading(self):
        document = {"figures": [
            {"source": "figs/paper/plot.png", "pixel size": [1200, 600]},
            {"source": "figs/paper/portrait.png", "pixel size": [300, 450]}]}
        chunks = [Chunk(1, (1,), "## Body"),
                  Chunk(2, (1,), "![Plot](figs/paper/plot.png)", "figure"),
                  Chunk(3, (1,), "![Author](figs/paper/portrait.png)", "portrait")]
        html = article_html(build_article(document, "Paper", "paper.pdf", chunks), "paper.md")
        self.assertIn('alt="Plot" width="1200" height="600" loading="lazy"', html)
        self.assertIn('alt="Author" width="300" height="450" loading="lazy"', html)

    def test_source_only_caption_image_is_not_duplicated_as_a_translation(self):
        image="![Original caption](figs/paper/caption.png)"
        chunks=[Chunk(1,(1,),"![Plot](figs/paper/plot.png)","figure"),
                Chunk(2,(1,),image,"caption-image","figs/paper/plot.png")]
        article=build_article({},"Paper","paper.pdf",chunks,[chunks[0].source,image])
        html=article_html(article,"paper.md")
        self.assertEqual(html.count('src="figs/paper/caption.png"'),1)
        self.assertFalse(article.front[0].caption.translation)

    def test_article_exports_structure_inline_equations_and_all_portable_assets(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); assets = root / "figs/paper"; assets.mkdir(parents=True)
            for name in ["figure.png", "page-001-math-001.png"]:
                Image.new("RGB", (50, 20), "white").save(assets / name)
            equation = inline_image_tag("figs/paper/page-001-math-001.png", "equation", 1.1, -0.2, 2.5)
            sources = ["# A Routing Study\n\nJane Example", "Abstract. An abstract with evidence.",
                       f"## 1 Introduction\n\nAn equation {equation} in a sentence. <script>bad()</script>",
                       "![Plot](figs/paper/figure.png)", "Fig.1. A routing plot.\n\nFurther discussion.",
                       "## References\n\n[1] A referenced paper."]
            chunks = [Chunk(i + 1, (1,), source, "figure" if i == 3 else "text") for i, source in enumerate(sources)]
            translations = ["# 라우팅 연구\n\n저자", "초록. 근거를 포함한 초록.",
                            f"## 1 서론\n\n문장 속 수식 {equation}입니다.", "", "그림 1. 라우팅 그림.\n\n후속 설명.",
                            "## 참고문헌\n\n[1] 인용 논문."]
            output = write_article({"number of pages": 1}, "A Routing Study", "paper.pdf", chunks,
                                   translations, "Korean", root / "paper.md")
            markdown = output.read_text(encoding="utf-8")
            html = output.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn("## Abstract", markdown)
            self.assertIn("## 1 Introduction", markdown)
            self.assertNotIn("청크", markdown)
            self.assertIn('<figcaption>', html)
            self.assertIn('id="view-both" checked', html)
            self.assertIn('id="abstract"', html)
            self.assertIn('class="inline-math"', html)
            self.assertIn('>PLoTrans</a>', html)
            self.assertNotIn('PAPER READER', html)
            self.assertIn('Document contents', html)
            self.assertIn('Parallel view', html)
            self.assertNotIn('<script>', html)
            audit = Audit(html)
            self.assertEqual(audit.scripts, 0)
            self.assertTrue(all(link in audit.ids for link in audit.links))
            self.assertTrue((root / "assets/article.css").is_file())
            with ZipFile(output.with_suffix(".zip")) as archive:
                self.assertEqual(set(archive.namelist()), {"paper.md", "paper.html", "assets/article.css",
                                                         "figs/paper/figure.png", "figs/paper/page-001-math-001.png"})
                unpacked = root / "unpacked"; archive.extractall(unpacked)
                for match in FIGURE_LINK.finditer(html):
                    self.assertTrue((unpacked / image_reference(match).source).is_file())
            preview = standalone_html(output.with_suffix(".html"))
            self.assertIn('<style>', preview)
            self.assertIn('src="data:image/png;base64,', preview)
            self.assertNotIn('href="assets/', preview)


if __name__ == "__main__": unittest.main()
