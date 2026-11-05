from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from paper_localizer.article import article_html, build_article, write_article
from paper_localizer.bundle import FIGURE_LINK, standalone_html
from paper_localizer.core import Chunk
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
                                   translations, "A grounded summary.", "Korean", root / "paper.md")
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
