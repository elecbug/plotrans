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
from paper_localizer.figures import attach_figures


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
                self.assertEqual(len(archive.namelist()), 3)
                self.assertIn("result.md", archive.namelist())
                self.assertTrue(all(f["source"] in archive.namelist() for f in document["figures"]))

    def test_bundle_rejects_paths_outside_figs(self):
        with self.assertRaises(ValueError):
            figure_path(Path("outputs/paper.md"), "figs/../../private.txt")


if __name__ == "__main__":
    unittest.main()
