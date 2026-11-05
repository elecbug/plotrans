from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (DecodedStreamObject, DictionaryObject,
                           NameObject, NumberObject)

from paper_localizer.bundle import build_bundle, figure_path
from paper_localizer.core import extract_blocks, make_chunks, process_pdf
from paper_localizer.figures import _connected_regions, _text_labels, attach_figures
import pypdfium2 as pdfium


def make_fixture(path: Path) -> dict:
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})

    def page_with(content: bytes):
        page = writer.add_blank_page(width=400, height=500)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject(); stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream)
        return page

    page_with(b"BT /F1 12 Tf 40 450 Td (Body stays readable.) Tj ET "
              b"BT /F1 8 Tf 40 322 Td (This is a normal sentence immediately above the diagram.) Tj ET "
              b"60 250 100 60 re S 210 250 100 60 re S "
              b"160 280 m 210 280 l S 203 284 m 210 280 l 203 276 l S "
              b"BT /F1 12 Tf 80 275 Td (Input) Tj ET "
              b"BT /F1 12 Tf 230 275 Td (Output) Tj ET "
              b"BT /F1 8 Tf 0 1 -1 0 45 250 Tm (Vertical label) Tj ET "
              b"BT /F1 12 Tf 60 225 Td (Figure 1.) Tj 55 0 Td (Vector flowchart.) Tj ET")
    page_with(b"60 250 240 90 re S 60 280 m 300 280 l S "
              b"60 310 m 300 310 l S 180 250 m 180 340 l S")
    raster_page = page_with(b"q 120 0 0 90 60 200 cm /Im0 Do Q")
    raster = DecodedStreamObject(); raster.set_data(bytes([20, 80, 220]) * (32 * 24))
    raster.update({NameObject("/Type"): NameObject("/XObject"),
                   NameObject("/Subtype"): NameObject("/Image"),
                   NameObject("/Width"): NumberObject(32),
                   NameObject("/Height"): NumberObject(24),
                   NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                   NameObject("/BitsPerComponent"): NumberObject(8)})
    raster_page["/Resources"][NameObject("/XObject")] = DictionaryObject({
        NameObject("/Im0"): writer._add_object(raster)})
    with path.open("wb") as file:
        writer.write(file)
    writer.close()
    return {"kids": [
        {"type": "paragraph", "page number": 1, "bounding box": [40, 446, 180, 462],
         "content": "Body stays readable."},
        {"type": "paragraph", "page number": 1, "bounding box": [40, 319, 350, 334],
         "content": "This is a normal sentence immediately above the diagram."},
        {"type": "paragraph", "page number": 1, "bounding box": [80, 272, 110, 285],
         "content": "Input"},
        {"type": "paragraph", "page number": 1, "bounding box": [230, 272, 270, 285],
         "content": "Output"},
        {"type": "caption", "page number": 1, "bounding box": [60, 222, 250, 235],
         "content": "Figure 1. Vector flowchart."},
        {"type": "table", "page number": 2, "bounding box": [60, 250, 300, 340],
         "rows": [{"cells": [{"kids": [{"content": "A"}]},
                              {"kids": [{"content": "B"}]}]} for _ in range(3)]},
        {"type": "image", "page number": 3, "bounding box": [60, 200, 180, 290],
         "source": "temporary/image.png"},
    ]}


class FigureTests(unittest.TestCase):
    def test_zero_dilation_can_inspect_unclaimed_graphic_pixels(self):
        mask = Image.new("L", (80, 60), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle((5, 6, 10, 11), fill=255)
        draw.rectangle((30, 40, 35, 45), fill=255)
        self.assertEqual(_connected_regions(mask, gap=0, min_width=1, min_height=1),
                         [(5, 6, 11, 12), (30, 40, 36, 46)])
        mask.close()

    def test_detached_small_vector_icon_stays_with_its_figure(self):
        from test_structure import fixture_pdf
        from paper_localizer.layout import prepare_layout
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "detached-icon.pdf"
            fixture_pdf(source, [
                b"110 300 140 150 re S 340 300 140 150 re S "
                b"q 0 0 1 rg 75 365 15 9 re f Q "
                b"BT /F2 10 Tf 80 275 Td (Figure 1. Left diagram.) Tj ET "
                b"BT /F2 10 Tf 330 275 Td (Figure 2. Right diagram.) Tj ET"])
            document = {"kids": [
                {"type": "caption", "content": "Figure 1. Left diagram.", "page number": 1,
                 "bounding box": [80, 272, 280, 286]},
                {"type": "caption", "content": "Figure 2. Right diagram.", "page number": 1,
                 "bounding box": [330, 272, 530, 286]}]}
            result = attach_figures(source, prepare_layout(document), root)
            self.assertEqual(len(result["figures"]), 2)
            left, right = sorted(result["figures"], key=lambda asset: asset["bounding box"][0])
            self.assertLess(left["bounding box"][0], 75)
            self.assertLess(left["bounding box"][2], 300)
            self.assertGreater(right["bounding box"][0], 300)
            with Image.open(root / left["source"]) as image:
                blue = sum(b > r + 100 and b > g + 100
                           for r, g, b in image.convert("RGB").get_flattened_data())
                self.assertGreater(blue, 500)

    def test_form_xobject_legend_uses_transformed_page_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); source = root/"transformed.pdf"
            writer = PdfWriter(); page = writer.add_blank_page(width=600, height=500)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"):font})})
            form = DecodedStreamObject(); form.set_data(b"0 20 100 50 re S BT /F1 8 Tf 105 25 Td (Transformed legend) Tj ET")
            from pypdf.generic import ArrayObject
            form.update({NameObject("/Type"):NameObject("/XObject"), NameObject("/Subtype"):NameObject("/Form"),
                         NameObject("/BBox"):ArrayObject([NumberObject(n) for n in [0,0,250,100]]), NameObject("/Resources"):resources})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"):DictionaryObject({NameObject("/Plot"):writer._add_object(form)})})
            stream = DecodedStreamObject(); stream.set_data(b"q 1.5 0 0 1.5 150 200 cm /Plot Do Q")
            page[NameObject("/Contents")] = writer._add_object(stream)
            with source.open("wb") as file: writer.write(file)
            writer.close()
            with pdfium.PdfDocument(source) as pdf:
                labels = _text_labels(pdf[0])
            legend = next(n for n in labels if "Transformed" in n["content"])
            self.assertGreater(legend["bounding box"][0], 300)
            self.assertGreater(legend["bounding box"][1], 230)
            document = attach_figures(source, {"kids":[]}, root)
            self.assertEqual(len(document["figures"]), 1)
            self.assertGreater(document["figures"][0]["bounding box"][2], legend["bounding box"][2])

    def test_vector_and_raster_crops_remove_labels_but_preserve_body_and_table(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / "fixture.pdf"
            document = make_fixture(source)
            result = attach_figures(source, document, root)
            self.assertEqual([f["page number"] for f in result["figures"]], [1, 3])
            self.assertLess(result["figures"][0]["bounding box"][0], 45)
            self.assertGreater(result["figures"][0]["bounding box"][1], 235)
            self.assertLess(result["figures"][0]["bounding box"][3], 319)
            blocks = extract_blocks(result)
            texts = [b.text for b in blocks]
            self.assertIn("Body stays readable.", texts)
            self.assertIn("Figure 1. Vector flowchart.", texts)
            self.assertNotIn("Input", texts)
            self.assertNotIn("Output", texts)
            self.assertTrue(any(b.kind == "table" for b in blocks))
            for figure in result["figures"]:
                with Image.open(root / figure["source"]) as image:
                    self.assertGreater(image.width, 250)
                    self.assertGreater(image.height, 100)

    def test_figures_are_not_sent_to_translation_and_bundle_is_portable(self):
        class Client:
            def __init__(self): self.prompts = []
            def check_models(self, *models): pass
            def chat(self, model, prompt, *, translation=False):
                self.prompts.append(prompt)
                return "번역문" if translation else "요약"

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            pdf = root / "fixture.pdf"
            document = attach_figures(pdf, make_fixture(pdf), root)
            chunks = make_chunks(extract_blocks(document), 200)
            self.assertEqual(sum(c.kind == "figure" for c in chunks), 2)
            client = Client()
            output = process_pdf(pdf, root / "result.md", parser=lambda path: document, client=client)
            markdown = output.read_text(encoding="utf-8")
            self.assertEqual(markdown.count("!["), 2)
            self.assertTrue(all("figs/" not in prompt for prompt in client.prompts))
            with ZipFile(BytesIO(build_bundle(output))) as archive:
                self.assertEqual(len(archive.namelist()), 5)
                self.assertIn("result.md", archive.namelist())
                self.assertIn("result.html", archive.namelist())
                self.assertIn("assets/article.css", archive.namelist())
                self.assertTrue(all(f["source"] in archive.namelist() for f in document["figures"]))

    def test_bundle_rejects_paths_outside_figs(self):
        with self.assertRaises(ValueError):
            figure_path(Path("outputs/paper.md"), "figs/../../private.txt")


if __name__ == "__main__":
    unittest.main()
