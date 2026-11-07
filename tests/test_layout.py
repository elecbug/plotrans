from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paper_localizer.article import build_article, write_article
from paper_localizer.core import extract_blocks, make_chunks, prepare_document
from paper_localizer.formulas import _reconcile_snapshots
from paper_localizer.layout import caption_label, prepare_layout


class LayoutTests(unittest.TestCase):
    def test_bulleted_and_appendix_captions_are_not_lost(self):
        self.assertEqual(caption_label("■ Figure 5. A complete caption."), "Fig. 5")
        self.assertEqual(caption_label("Fig. A.4. The appendix result."), "Fig. A.4")
        self.assertEqual(caption_label("Figure 5.1 shows a result discussed in the body."), "")

    def test_unmatched_table_caption_is_not_attached_to_the_previous_figure(self):
        from paper_localizer.core import Chunk
        chunks = [Chunk(1,(1,),"![Plot](figs/paper/plot.png)","figure"),
                  Chunk(2,(1,),"Table 1. Independent table caption.","caption")]
        article = build_article({}, "Paper", "paper.pdf", chunks)
        self.assertIsNone(article.front[0].caption)
        self.assertEqual(article.front[1].source, "Table 1. Independent table caption.")

    def test_piecewise_snapshot_owns_all_overlapping_source_fragments(self):
        reference = "figs/paper/equation.png"
        nodes = [
            {"type": "paragraph", "content": "qz={", "bounding box": [125, 141, 149, 178]},
            {"type": "paragraph", "content": "(q/p)z if p>q}", "bounding box": [148, 141, 227, 178],
             "formula snapshot": f"![Equation]({reference})"},
            {"type": "paragraph", "content": "1 if p≤q", "bounding box": [160, 160, 221, 174]},
            {"type": "paragraph", "content": "Following explanation.", "bounding box": [50, 100, 350, 120]},
        ]
        _reconcile_snapshots(nodes, [], [{"source": reference, "bounding box": [124, 140, 228, 179], "block": True}])
        blocks = extract_blocks({"kids": nodes})
        self.assertEqual([b.text for b in blocks], [f"![Equation]({reference})", "Following explanation."])
        nodes[1]["content"] = ""
        self.assertEqual(extract_blocks({"kids": nodes})[0].text, f"![Equation]({reference})")

    def test_caption_continuation_can_precede_caption_in_parser_order(self):
        continuation = {"type": "heading", "page number": 1, "font": "Times-Bold", "font size": 9,
                        "bounding box": [50, 549, 556, 571], "content": "node network. The complete final sentence."}
        caption = {"type": "heading", "page number": 1, "font": "Times-Bold", "font size": 9,
                   "bounding box": [50, 560, 411, 571], "content": "Figure 8: Keys stored per node in a"}
        body = {"type": "paragraph", "page number": 1, "font": "Times-Roman", "font size": 9,
                "bounding box": [50, 530, 295, 541], "content": "Figure 8 shows the body discussion."}
        document = prepare_layout({"kids": [continuation, caption, body]})
        self.assertEqual(caption["content"], "Fig. 8: Keys stored per node in a node network. The complete final sentence.")
        self.assertEqual(continuation["content"], "")
        self.assertEqual(body["type"], "paragraph")
        self.assertEqual(len(document["captions"]), 1)

    def test_two_column_multiplots_and_complete_captions_survive_exports(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); source = root / "columns.pdf"
            writer = PdfWriter(); page = writer.add_blank_page(width=720, height=800)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            operations = []
            for x, base in [(50, 400), (400, 350)]:
                for y in [base, base+100, base+200]:
                    operations.append(f"{x} {y} 240 60 re S")
            # A caption bullet touching the first frame must not drag half of
            # its caption into the crop during connected-component detection.
            operations.append("50 374 2 28 re f")
            operations.extend(["BT /F1 8 Tf 50 375 Td (Fig. 1. Three panels with) Tj ET",
                               "BT /F1 8 Tf 140 389 Td (\\(a\\) Capped) Tj ET",
                               "BT /F1 8 Tf 50 364 Td (the full caption ending.) Tj ET",
                               "BT /F1 8 Tf 400 325 Td (Fig. 2. The second complete caption.) Tj ET",
                               "BT /F1 12 Tf 50 300 Td (Body explanation stays editable.) Tj ET"])
            stream = DecodedStreamObject(); stream.set_data(" ".join(operations).encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            with source.open("wb") as file: writer.write(file)
            writer.close()
            document = {"number of pages": 1, "kids": [
                {"type": "heading", "content": "Two columns", "page number": 1, "bounding box": [50, 740, 300, 760]},
                {"type": "heading", "content": "Figure 1. Three panels with", "font": "Helvetica", "font size": 8, "page number": 1, "bounding box": [50, 373, 290, 398]},
                {"type": "paragraph", "content": "the full caption ending.", "font": "Helvetica", "font size": 8, "page number": 1, "bounding box": [50, 362, 290, 371]},
                {"type": "paragraph", "content": "Fig. 2. The second complete caption.", "font": "Helvetica", "font size": 8, "page number": 1, "bounding box": [400, 323, 640, 332]},
                {"type": "paragraph", "content": "Body explanation stays editable.", "font": "Helvetica", "font size": 12, "page number": 1, "bounding box": [50, 297, 350, 310]},
            ]}
            document = prepare_document(source, document, root)
            self.assertEqual(len(document["figures"]), 2)
            self.assertEqual([f["caption label"] for f in document["figures"]], ["Fig. 1", "Fig. 2"])
            for figure in document["figures"]:
                self.assertGreater(figure["bounding box"][1], figure["caption node"]["bounding box"][3])
            self.assertLess(document["figures"][0]["caption node"]["bounding box"][3],385)
            chunks = make_chunks(extract_blocks(document), limit=200)
            self.assertEqual(sum(c.kind == "caption" for c in chunks), 2)
            article = build_article(document, "Two columns", source.name, chunks)
            figures = [u for u in article.front if u.kind == "figure"]
            self.assertEqual(len(figures), 2)
            self.assertIn("the full caption ending.", figures[0].caption.source)
            output = write_article(document, "Two columns", source.name, chunks, None, "English", root/"paper.md")
            for path in [output, output.with_suffix(".html")]:
                text = path.read_text(encoding="utf-8")
                self.assertEqual(text.count("the full caption ending."), 1)
                self.assertIn("Body explanation stays editable.", text)


if __name__ == "__main__": unittest.main()
