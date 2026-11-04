from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           FloatObject, NameObject, NumberObject)

from paper_localizer.bundle import FIGURE_LINK, build_bundle, inline_equation_preview
from paper_localizer.core import ProcessingError, _split_text, _translate_chunk, extract_blocks
from paper_localizer.formulas import recover_formulas
from paper_localizer.markup import image_reference, inline_image_tag
from io import BytesIO
from zipfile import ZipFile


def math_fixture(path: Path) -> dict:
    """A real Type3 equation without Unicode mapping, inline and displayed."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=700, height=250)
    normal = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                               NameObject("/Subtype"): NameObject("/Type1"),
                               NameObject("/BaseFont"): NameObject("/Helvetica")})
    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): normal})})
    glyph = DecodedStreamObject()
    glyph.set_data(b"2000 0 0 -200 2000 800 d1 BT /F1 1000 Tf 0 0 Td (b=4) Tj ET")
    symbol = DecodedStreamObject()
    symbol.set_data(b"700 0 0 -200 700 800 d1 BT /F1 1000 Tf 0 0 Td (N) Tj ET")
    math_font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type3"),
        NameObject("/FontBBox"): ArrayObject([NumberObject(n) for n in [0, -200, 2000, 800]]),
        NameObject("/FontMatrix"): ArrayObject([FloatObject(n) for n in [.001, 0, 0, .001, 0, 0]]),
        NameObject("/FirstChar"): NumberObject(1), NameObject("/LastChar"): NumberObject(2),
        NameObject("/Widths"): ArrayObject([NumberObject(2000), NumberObject(700)]),
        NameObject("/Encoding"): DictionaryObject({NameObject("/Differences"):
                                                    ArrayObject([NumberObject(1), NameObject("/equation"), NameObject("/symbolN")])}),
        NameObject("/CharProcs"): DictionaryObject({NameObject("/equation"): writer._add_object(glyph),
                                                   NameObject("/symbolN"): writer._add_object(symbol)}),
        NameObject("/Resources"): resources,
    })
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): normal, NameObject("/M0"): math_font})})
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 30 180 Td (Fig.7. Quality of routing tables, ) Tj "
        b"/M0 12 Tf <01> Tj /F1 12 Tf (, and 5,000 nodes.) Tj ET "
        b"BT /F1 12 Tf 30 130 Td (Normal prose with a legacy\002soft hyphen stays editable.) Tj ET "
        b"BT /F1 12 Tf 30 100 Td (Complexity is O\\(log ) Tj /M0 12 Tf <02> Tj "
        b"/F1 12 Tf (\\) + 1.) Tj ET "
        b"BT /M0 16 Tf 80 60 Td <01> Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as file: writer.write(file)
    writer.close()
    return {"kids": [
        {"type": "caption", "page number": 1, "bounding box": [30, 176, 500, 194],
         "content": "Fig.7. Quality of routing tables, , and 5,000 nodes."},
        {"type": "paragraph", "page number": 1, "bounding box": [30, 126, 500, 144],
         "content": "Normal prose with a legacy soft hyphen stays editable."},
        {"type": "paragraph", "page number": 1, "bounding box": [30, 96, 500, 114],
         "content": "Complexity is O(log ) + 1."},
    ]}


class FormulaTests(unittest.TestCase):
    def test_missing_inline_and_displayed_type3_equations_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); source = root / "formula.pdf"
            document = recover_formulas(source, math_fixture(source), root)
            self.assertEqual(len(document["formulas"]), 3)
            caption = next(n for n in document["kids"] if n.get("type") == "caption")["content"]
            self.assertRegex(caption, r'tables, <img\b[^>]+/>, and 5,000 nodes\.')
            reference = image_reference(next(FIGURE_LINK.finditer(caption)))
            self.assertTrue(reference.inline)
            self.assertGreater(reference.height_em, 0.3)
            self.assertLess(reference.height_em, 3)
            self.assertIn("vertical-align:", caption)
            self.assertTrue(all((root / f["source"]).is_file() for f in document["formulas"]))
            self.assertIn("Normal prose with a legacy soft hyphen stays editable.",
                          [b.text for b in extract_blocks(document)])
            complexity = next(n["content"] for n in document["kids"] if n.get("content", "").startswith("Complexity"))
            self.assertRegex(complexity, r'^Complexity is <img\b[^>]+/>\.$')
            self.assertEqual(len(list(FIGURE_LINK.finditer(complexity))), 1)
            self.assertTrue(any(b.kind == "formula" for b in extract_blocks(document)))
            preview = inline_equation_preview(root / "paper.md", caption)
            self.assertIn("data:image/png;base64,", preview)
            self.assertNotIn('src="figs/', preview)
            self.assertIn("inline-block", preview)
            markdown = root / "paper.md"
            markdown.write_text(caption + "\n\n" + next(
                n["content"] for n in document["kids"] if n.get("formula image")), encoding="utf-8")
            with ZipFile(BytesIO(build_bundle(markdown))) as archive:
                self.assertEqual(len(archive.namelist()), 3)
            unsafe = caption.replace('class="inline-math"', 'onerror="alert(1)" class="inline-math"')
            safe = inline_equation_preview(markdown, "<script>alert(1)</script>" + unsafe)
            self.assertNotIn("<script>", safe)
            self.assertNotIn("onerror", safe)
            self.assertIn("&lt;script&gt;", safe)

    def test_translation_retries_then_restores_equation_without_exposing_path(self):
        class Client:
            def __init__(self): self.prompts = []
            def chat(self, model, prompt, *, translation=False):
                self.prompts.append(prompt)
                return "누락" if len(self.prompts) == 1 else "수식 [[MATH_0001]] 입니다."
        client = Client()
        source = "Equation ![수식-p1-1](figs/paper/page-001-math-001.png)."
        result = _translate_chunk(client, "translate", source, "Korean")
        self.assertIn("![수식-p1-1](figs/paper/page-001-math-001.png)", result)
        self.assertEqual(len(client.prompts), 2)
        self.assertTrue(all("figs/" not in prompt for prompt in client.prompts))

    def test_translation_cannot_silently_drop_an_equation(self):
        class Client:
            def chat(self, *args, **kwargs): return "수식 누락"
        with self.assertRaises(ProcessingError):
            _translate_chunk(Client(), "translate", "![수식](figs/paper/page-001-math-001.png)", "Korean")

    def test_chunk_boundaries_do_not_split_inline_images(self):
        image = "![수식-p1-1](figs/paper/page-001-math-001.png)"
        source = "word " * 34 + image + " sentence " * 30
        parts = _split_text(source, 200)
        self.assertEqual(sum(len(FIGURE_LINK.findall(part)) for part in parts), 1)

    def test_html_equations_keep_their_positions_and_attributes_in_translation(self):
        class Client:
            def chat(self, *args, **kwargs): return "앞 [[MATH_0001]], 중간 [[MATH_0002]] 뒤."
        first = inline_image_tag("figs/paper/first.png", "첫 수식", 1.2, -0.2, 2)
        second = inline_image_tag("figs/paper/second.png", "두 번째 수식", 0.9, 0.1, 3)
        source = f"before {first}, middle {second} after."
        result = _translate_chunk(Client(), "translate", source, "Korean")
        self.assertEqual(result, f"앞 {first}, 중간 {second} 뒤.")
        # HTML attributes must not force a short sentence into separate chunks.
        self.assertEqual(_split_text(source, 200), [source])


if __name__ == "__main__": unittest.main()
