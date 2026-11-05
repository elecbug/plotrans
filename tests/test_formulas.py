from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           FloatObject, NameObject, NumberObject)

from paper_localizer.bundle import FIGURE_LINK, build_bundle, inline_equation_preview
from paper_localizer.core import ProcessingError, _split_text, _translate_chunk, extract_blocks
from paper_localizer.formulas import (TextSpan, _expand_expressions, _groups, _insertions,
                                      _plain_text_group, _remove_figure_duplicates, recover_formulas)
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
        b"BT /F1 12 Tf 30 100 Td (Complexity is Cost\\(log ) Tj /M0 12 Tf <02> Tj "
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
         "content": "Complexity is Cost(log ) + 1."},
    ]}


class FormulaTests(unittest.TestCase):
    def test_inline_crop_keeps_source_space_before_a_citation(self):
        from test_structure import fixture_pdf, paragraph
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'citation.pdf'
            fixture_pdf(source, [b'BT /F2 10 Tf 30 500 Td (Bound ) Tj ET '
                                 b'BT /F1 10 Tf 63 500 Td (f) Tj ET '
                                 b'BT /F2 10 Tf 74 500 Td ([22].) Tj ET'])
            document = {'kids': [paragraph('Bound f [22].', [30, 490, 280, 515])]}
            recover_formulas(source, document, root)
            self.assertRegex(document['kids'][0]['content'], r'<img\b[^>]+/> \[22\]\.\Z')

    def test_touching_punctuation_is_owned_once_by_the_source_crop(self):
        spans = [TextSpan(0, "Before", (10, 60, 35, 70), False, 10, 60, "Roman"),
                 TextSpan(1, "x", (40, 60, 47, 70), True, 10, 60, "Italic"),
                 TextSpan(2, ".", (47.2, 60, 49, 62), False, 10, 60, "Roman")]
        groups = _groups(spans)
        self.assertEqual([[s.text for s in g] for g in groups], [["x", "."]])
        self.assertEqual(_insertions("Before x.", spans, groups), [(7, 9)])

    def test_separated_citation_does_not_become_part_of_a_formula(self):
        spans = [TextSpan(0, "f", (10, 60, 16, 70), True, 10, 60, "Italic"),
                 TextSpan(1, "(", (16.5, 58, 20, 71), False, 10, 60, "Roman"),
                 TextSpan(2, "x", (20.5, 60, 27, 70), True, 10, 60, "Italic"),
                 TextSpan(3, ")", (27.5, 58, 31, 71), False, 10, 60, "Roman"),
                 TextSpan(4, "[", (35, 58, 38, 71), False, 10, 60, "Roman"),
                 TextSpan(5, "22", (38.5, 60, 49, 70), False, 10, 60, "Roman"),
                 TextSpan(6, "]", (49.5, 58, 53, 71), False, 10, 60, "Roman")]
        groups = _groups(_expand_expressions(spans))
        self.assertEqual(["".join(s.text for s in g) for g in groups], ["f(x)"])
        attached = [TextSpan(0, "A", (10, 60, 16, 70), True, 10, 60, "Italic")] + [
            TextSpan(s.index, s.text, (s.box[0]-18, *s.box[1:2], s.box[2]-18, s.box[3]),
                     s.math, s.font_size, s.baseline, s.font_name) for s in spans[4:]]
        self.assertEqual(["".join(s.text for s in g) for g in _groups(_expand_expressions(attached))], ["A[22]"])

    def test_prose_labels_and_acronyms_stay_text_but_scripts_stay_math(self):
        label = [TextSpan(0, "A", (10, 60, 17, 70), True, 10, 60, "Helvetica-Oblique"),
                 TextSpan(1, ".", (17.2, 60, 19, 62), False, 10, 60, "Helvetica-Oblique")]
        self.assertTrue(_plain_text_group(label, {"type": "heading", "content": "A. METHODS"}))
        self.assertTrue(_plain_text_group(label[:1], {"type": "heading", "content": "A. METHODS"}))
        self.assertFalse(_plain_text_group(label, {"type": "paragraph", "content": "A. METHODS"}))
        acronym = [TextSpan(0, "P", (10, 60, 17, 70), True, 10, 60, "Times-Italic"),
                   TextSpan(1, "2", (17.2, 60, 22, 70), True, 10, 60, "Times-Italic"),
                   TextSpan(2, "P", (22.2, 60, 29, 70), True, 10, 60, "Times-Italic")]
        self.assertTrue(_plain_text_group(acronym, {"content": "P2P systems."}))
        acronym[1] = TextSpan(1, "2", (17.2, 57, 22, 63), True, 7, 57, "Times-Italic")
        self.assertFalse(_plain_text_group(acronym, {"content": "P2P systems."}))
        bullet = [TextSpan(0, "•", (10, 60, 15, 65), True, 10, 60, "Symbol")]
        self.assertTrue(_plain_text_group(bullet, {"content": "• First requirement."}))
        self.assertFalse(_plain_text_group(bullet, {"content": "The operator • is used."}))

    def test_named_subscript_stays_with_its_math_identifier(self):
        spans = [TextSpan(0, "D", (30, 60, 39, 71), True, 10, 60, "MathMI"),
                 TextSpan(1, "lazy", (39, 57, 54, 63), False, 7, 57, "Roman"),
                 TextSpan(2, "values", (60, 60, 95, 71), False, 10, 60, "Roman")]
        groups = _groups(_expand_expressions(spans))
        self.assertEqual([[s.text for s in g] for g in groups], [["D", "lazy"]])

    def test_whole_page_of_undecodable_fonts_has_one_complete_source_image(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); source = root/"legacy.pdf"
            document = math_fixture(source)
            writer = PdfWriter(clone_from=source)
            stream = DecodedStreamObject()
            stream.set_data(b"BT /M0 16 Tf 40 180 Td <010201020102> Tj ET")
            writer.pages[0][NameObject("/Contents")] = writer._add_object(stream)
            with source.open("wb") as file: writer.write(file)
            writer.close()
            document = recover_formulas(source, document, root)
            self.assertEqual(document["source image pages"], [1])
            self.assertEqual(document["formulas"], [])
            blocks = extract_blocks(document)
            self.assertEqual(len(blocks), 1)
            self.assertIn("Source page 1", blocks[0].text)
            self.assertTrue((root / image_reference(next(FIGURE_LINK.finditer(blocks[0].text))).source).is_file())

    def test_embedded_figure_labels_are_removed_without_removing_neighboring_prose(self):
        spans = [TextSpan(0, "Before prose.", (0, 60, 40, 70), False, 10, 60),
                 TextSpan(1, "Input", (10, 30, 30, 40), False, 10, 30),
                 TextSpan(2, "Output", (40, 30, 60, 40), False, 10, 30),
                 TextSpan(3, "After prose.", (0, 0, 40, 10), False, 10, 0)]
        node = {"type": "paragraph", "content": "Before prose. Input Output After prose.",
                "bounding box": [0, 0, 80, 80]}
        _remove_figure_duplicates([node], spans, [(0, 20, 80, 50)])
        self.assertEqual(node["content"], "Before prose. After prose.")

    def test_uncertain_source_snapshot_is_not_also_emitted_as_duplicate_text(self):
        snapshot = "![Source block](figs/paper/page-001-math-001.png)"
        document = {"kids": [{"type": "paragraph", "page number": 1,
                               "content": "Incomplete equation and text.", "formula snapshot": snapshot}]}
        blocks = extract_blocks(document)
        self.assertEqual([block.text for block in blocks], [snapshot])
        self.assertEqual(blocks[0].kind, "formula")

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
