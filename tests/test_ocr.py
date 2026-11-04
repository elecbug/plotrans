"""OCR routing, geometric preservation and optional real offline inference."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter

from paper_localizer.article import write_article
from paper_localizer.core import extract_blocks, make_chunks, prepare_document, process_pdf
from paper_localizer.markup import FIGURE_LINK, image_reference
from paper_localizer.layout import prepare_layout
from paper_localizer.ocr import OcrOptions, _layout, _reading_order, apply_ocr


def scan_fixture(path):
    image = Image.new("RGB",(600,800),"white"); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    entries = [("1 Introduction",60,.99),("Local OCR restores readable prose.",110,.99),
               ("Processing stays on this computer.",137,.99),("x = y + 1",190,.99),
               ("if x > 0",211,.99),("Uncertain words need their original image.",280,.45)]
    lines = []
    for text,y,score in entries:
        draw.text((40,y),text,fill="black",font=font)
        l,t,r,b = draw.textbbox((40,y),text,font=font)
        lines.append({"text":text if score > .8 else "garbled recognition", "score":score,"box":[[l,t],[r,t],[r,b],[l,b]]})
    image.save(path,"PDF",resolution=72); image.close()
    def recognize(rendered):
        scale = rendered.width/600
        return [dict(line,box=[[x*scale,y*scale] for x,y in line["box"]]) for line in lines]
    return recognize


class OcrTests(unittest.TestCase):
    def test_options_reject_invalid_settings(self):
        for kwargs in [{"mode":"cloud"},{"dpi":20},{"min_confidence":1.1}]:
            with self.assertRaises(ValueError): OcrOptions(**kwargs)

    def test_off_and_blank_pages_do_not_load_ocr(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"blank.pdf"
            writer=PdfWriter();writer.add_blank_page(width=600,height=800)
            with source.open("wb") as file: writer.write(file)
            writer.close()
            with patch("paper_localizer.ocr._recognize",side_effect=AssertionError("OCR must not run")):
                for options in [OcrOptions(mode="off"),OcrOptions()]:
                    result=apply_ocr(source,{"kids":[]},root,options)
                    self.assertNotIn("ocr pages",result)

    def test_scan_text_survives_figure_extraction_and_uncertain_math_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"scan.pdf";recognize=scan_fixture(source)
            with patch("paper_localizer.ocr._recognize",side_effect=recognize):
                document=prepare_document(source,{"kids":[],"number of pages":1},root)
            blocks=extract_blocks(document);text="\n".join(b.text for b in blocks)
            self.assertIn("Local OCR restores readable prose.",text)
            self.assertNotIn("garbled recognition",text)
            self.assertNotIn("x = y + 1",text)
            self.assertEqual(document["ocr pages"],[1])
            self.assertNotIn("source image pages",document)
            self.assertEqual(len([b for b in blocks if b.kind=="formula"]),2)
            self.assertFalse(any((f["bounding box"][2]-f["bounding box"][0])>550 for f in document["figures"]))
            output=write_article(document,"OCR test",source.name,make_chunks(blocks),None,"","English",root/"paper.md")
            for path in [output,output.with_suffix(".html")]:
                body=path.read_text(encoding="utf-8")
                self.assertIn("Original pages used for OCR",body)
                self.assertIn("Local CPU OCR",body)
            with ZipFile(output.with_suffix(".zip")) as archive:
                for match in FIGURE_LINK.finditer(output.read_text(encoding="utf-8")):
                    self.assertIn(image_reference(match).source,archive.namelist())

    def test_ocr_cache_reuses_results_and_changes_with_pdf_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"scan.pdf";recognize=scan_fixture(source)
            with patch("paper_localizer.ocr._recognize",side_effect=recognize) as engine:
                first=apply_ocr(source,{"kids":[]},root)
                second=apply_ocr(source,{"kids":[]},root)
                self.assertEqual(engine.call_count,1)
                self.assertFalse(first["ocr details"][0]["cache hit"])
                self.assertTrue(second["ocr details"][0]["cache hit"])
                with source.open("ab") as file: file.write(b"\n% changed input\n")
                apply_ocr(source,{"kids":[]},root)
                self.assertEqual(engine.call_count,2)

    def test_ocr_page_replacement_keeps_other_pages_in_the_same_container(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"scan.pdf";recognize=scan_fixture(source)
            writer=PdfWriter(clone_from=source);writer.add_blank_page(width=600,height=800)
            with source.open("wb") as file:writer.write(file)
            writer.close()
            document={"kids":[{"type":"list","page number":1,"kids":[
                {"type":"paragraph","page number":1,"content":"unreadable native text"},
                {"type":"paragraph","page number":2,"content":"Native page two text must remain."}]}]}
            with patch("paper_localizer.ocr._recognize",side_effect=recognize):
                document=apply_ocr(source,document,root)
            blocks=extract_blocks(document)
            self.assertIn("Native page two text must remain.",[b.text for b in blocks])
            self.assertEqual(document["ocr pages"],[1])

    def test_columns_are_read_completely_before_the_next_column(self):
        lines=[{"text":"Title","bounding box":[40,750,560,780]}]
        for i in range(6):
            lines.extend([{"text":f"L{i}","bounding box":[40,700-i*20,260,715-i*20]},
                          {"text":f"R{i}","bounding box":[340,700-i*20,560,715-i*20]}])
        self.assertEqual([n["text"] for n in _reading_order(lines,600)],["Title"]+[f"L{i}" for i in range(6)]+[f"R{i}" for i in range(6)])

    def test_title_continuations_and_split_section_numbers_are_joined(self):
        lines=[{"text":"A Research Title","score":.99,"bounding box":[100,710,500,727]},
               {"text":"Its Second Line","score":.99,"bounding box":[200,696,400,708]},
               {"text":"An Author","score":.99,"bounding box":[230,660,370,671]},
               {"text":"Implementation notes","score":.99,"bounding box":[65,600,260,612]},
               {"text":"4","score":.99,"bounding box":[50,600,58,611]},
               {"text":"A body paragraph follows.","score":.99,"bounding box":[50,580,400,592]}]
        nodes=_layout(lines,1,600,800,OcrOptions())
        self.assertEqual(nodes[0]["content"],"A Research Title Its Second Line")
        self.assertTrue(any(n["type"]=="heading" and n["content"]=="4 Implementation notes" for n in nodes))

    def test_reference_numbers_remain_prose(self):
        lines=[{"text":"1 INTRODUCTION","score":.99,"bounding box":[50,750,250,762]},
               {"text":"References","score":.99,"bounding box":[50,700,180,712]},
               {"text":"1. Jane Author. A paper title and","score":.99,"bounding box":[50,680,400,692]},
               {"text":"its publication information.","score":.99,"bounding box":[50,666,400,678]}]
        nodes=_layout(lines,5,600,800,OcrOptions())
        nodes=prepare_layout({"kids":nodes})["kids"]
        self.assertEqual([n["content"] for n in nodes if n["type"]=="heading"],["1 INTRODUCTION","References"])

    def test_only_usable_ocr_prose_reaches_translation(self):
        class Client:
            def __init__(self):self.prompts=[]
            def check_models(self,*models):pass
            def chat(self,model,prompt,**kwargs):self.prompts.append(prompt);return "Translated prose."
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"scan.pdf";recognize=scan_fixture(source)
            with patch("paper_localizer.ocr._recognize",side_effect=recognize):
                document=prepare_document(source,{"kids":[],"number of pages":1},root)
            client=Client();process_pdf(source,root/"translated.md",client=client,parser=lambda p:document)
            self.assertTrue(any("Local OCR restores" in p for p in client.prompts))
            self.assertFalse(any("garbled recognition" in p or "figs/" in p for p in client.prompts))

    @unittest.skipUnless(os.environ.get("PLOTRANS_TEST_OCR")=="1","set PLOTRANS_TEST_OCR=1 for real bundled-model inference")
    def test_real_ocr_runs_offline_on_a_scanned_pdf(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);source=root/"scan.pdf";scan_fixture(source)
            with patch("socket.socket.connect",side_effect=AssertionError("OCR must remain offline")):
                document=prepare_document(source,{"kids":[],"number of pages":1},root)
            self.assertEqual(document["ocr pages"],[1])
            text=" ".join(b.text for b in extract_blocks(document))
            self.assertIn("Local OCR restores readable prose",text)


if __name__ == "__main__":unittest.main()
