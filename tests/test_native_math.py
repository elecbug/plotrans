from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium

from paper_localizer.formulas import recover_formulas
from paper_localizer.native_math import native_expression
from paper_localizer.math_markup import mathml
from test_structure import fixture_pdf


class NativeMathTests(unittest.TestCase):
    def unicode_expression(self, value, *, mapping_errors=None, box=None):
        textpage = Mock()
        textpage.count_chars.return_value = len(value)
        textpage.get_charbox.side_effect = lambda index: (10+index*8, 398, 17+index*8, 410)
        obj = textpage.get_textobj.return_value
        obj.get_font.return_value.get_base_name.return_value = "Times-Italic"
        obj.get_font_size.return_value = 12
        obj.get_matrix.return_value = SimpleNamespace(c=0, d=1, f=400)
        page = Mock()
        page.get_textpage.return_value = textpage
        with patch("paper_localizer.native_math.pdfium.raw.FPDFText_GetUnicode",
                   side_effect=lambda _, index: ord(value[index])), \
             patch("paper_localizer.native_math.pdfium.raw.FPDFText_HasUnicodeMapError",
                   side_effect=lambda _, index: (mapping_errors or {}).get(index, 0)):
            return native_expression(page, box or [0, 390, 1000, 420])

    def test_invalid_unicode_mapping_never_produces_a_partial_expression(self):
        self.assertIsNone(self.unicode_expression("x", mapping_errors={0: 1}))
        self.assertIsNone(self.unicode_expression("x ", mapping_errors={1: 1}))
        self.assertIsNone(self.unicode_expression("x", mapping_errors={0: -1}))
        self.assertEqual(self.unicode_expression("xy", mapping_errors={1: 1}, box=[0, 390, 18, 420]), "x")

    def test_visible_braces_and_underscores_do_not_become_tex_syntax(self):
        for value in ("x∈{a,b}", "x_y"):
            with self.subTest(value=value):
                result = self.unicode_expression(value)
                self.assertIsNotNone(result)
                rendered = "".join(ET.fromstring(mathml(result)).itertext())
                self.assertEqual("".join(rendered.split()), value)
        self.assertIsNone(self.unicode_expression("x^2"))

    def test_small_inline_fraction_is_not_mistaken_for_two_scripts(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (x=) Tj ET '
                                 b'BT /F1 8 Tf 60 405 Td (a) Tj ET '
                                 b'BT /F1 8 Tf 60 395 Td (b) Tj ET '
                                 b'59 402 m 67 402 l S', [39, 390, 70, 417])
        self.assertIsNone(result)

    def test_greek_variants_retain_their_actual_unicode_identity(self):
        for value in ("ε", "ϵ", "φ", "ϕ", "θ", "ϑ", "π", "ϖ", "ρ", "ϱ", "σ", "ς"):
            with self.subTest(value=value):
                result = self.unicode_expression(value)
                self.assertIsNotNone(result)
                self.assertEqual("".join(ET.fromstring(mathml(result)).itertext()), value)

    def test_unhandled_semantic_font_styles_and_greek_variants_defer_to_ocr(self):
        for value in ("𝕏", "𝖷", "𝔵", "𝓍", "ϐ"):
            with self.subTest(value=value):
                self.assertIsNone(self.unicode_expression(value))
        self.assertEqual(self.unicode_expression("ℝ"), r"\mathbb{R}")
        self.assertEqual(self.unicode_expression("ℓ"), r"\ell")

    def expression(self, operations, box):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pdf'
            fixture_pdf(source, [operations])
            with pdfium.PdfDocument(source) as pdf:
                page = pdf[0]
                try: return native_expression(page, box)
                finally: page.close()

    def test_uniform_source_characters_do_not_need_a_model_guess(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (s) Tj ET '
                                 b'BT /F2 12 Tf 48 400 Td (+ 1) Tj ET', [39, 395, 80, 415])
        self.assertEqual(result, 's+1')

    def test_named_subscript_retains_its_script_and_roman_style(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (D) Tj ET '
                                 b'BT /F2 8 Tf 49 397 Td (avg) Tj ET', [39, 392, 70, 415])
        self.assertEqual(result, r'D_{\mathrm{avg}}')

    def test_superscript_is_not_flattened_into_the_identifier(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (x) Tj ET '
                                 b'BT /F2 8 Tf 47 405 Td (2) Tj ET', [39, 395, 60, 418])
        self.assertEqual(result, 'x^{2}')

    def test_superscript_can_be_scaled_by_a_pdf_transformation_matrix(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (x) Tj ET '
                                 b'q 0.7 0 0 0.7 47 405 cm BT /F2 12 Tf 0 0 Td (2) Tj ET Q',
                                 [39, 395, 60, 420])
        self.assertEqual(result, 'x^{2}')

    def test_multiline_expression_defers_to_ocr(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (x) Tj ET '
                                 b'BT /F1 12 Tf 40 380 Td (y) Tj ET', [39, 375, 70, 415])
        self.assertIsNone(result)

    def test_unexplained_small_character_on_same_baseline_defers_to_ocr(self):
        result = self.expression(b'BT /F1 12 Tf 40 400 Td (x) Tj ET '
                                 b'BT /F2 7 Tf 49 400 Td (A) Tj ET', [39, 395, 70, 415])
        self.assertIsNone(result)

    def test_math_script_is_preserved_even_when_the_font_is_a_body_font(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.pdf'
            fixture_pdf(source, [b'BT /F1 12 Tf 40 400 Td (D) Tj ET '
                                 b'BT /F2 8 Tf 49 397 Td (avg) Tj ET '
                                 b'BT /F2 12 Tf 70 400 Td ( is the mean.) Tj ET'])
            document = {'kids': [{'type': 'paragraph', 'page number': 1,
                                  'font': 'Times-Roman', 'font size': 12,
                                  'bounding box': [39, 392, 160, 415], 'content': 'Davg is the mean.'}]}
            document = recover_formulas(source, document, root)
            self.assertEqual(len(document['formulas']), 1)
            self.assertIn('<img ', document['kids'][0]['content'])
            with pdfium.PdfDocument(source) as pdf:
                page = pdf[0]
                try:
                    self.assertEqual(native_expression(page, document['formulas'][0]['bounding box']), r'D_{\mathrm{avg}}')
                finally: page.close()


if __name__ == '__main__':
    unittest.main()
