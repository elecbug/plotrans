import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from paper_localizer.core import OllamaClient, extract_blocks, make_chunks, process_pdf, render_markdown


DOCUMENT = {
    "title": "Sample Paper",
    "number of pages": 2,
    "kids": [
        {"type": "heading", "page number": 1, "heading level": 1, "content": "Introduction"},
        {"type": "paragraph", "page number": 1, "content": "The method improved accuracy."},
        {"type": "table", "page number": 2, "rows": [
            {"cells": [
                {"column number": 1, "kids": [{"content": "Model"}]},
                {"column number": 2, "kids": [{"content": "Score"}]},
            ]},
            {"cells": [
                {"column number": 1, "kids": [{"content": "A"}]},
                {"column number": 2, "kids": [{"content": "0.9"}]},
            ]},
        ]},
        {"type": "footer", "page number": 2, "kids": [{"content": "Footer"}]},
    ],
}


class FakeClient:
    def __init__(self):
        self.calls = []

    def check_models(self, *models):
        self.models = models

    def chat(self, model, prompt, *, translation=False):
        self.calls.append((model, translation, prompt))
        if translation:
            return "번역된 내용"
        if "Summarize only" in prompt:
            return "- 정확도 향상 [p.1]"
        return "### 주요 결과\n- 정확도 향상 [p.1]"


class CoreTests(unittest.TestCase):
    def test_extraction_and_chunking_keep_pages_and_tables(self):
        blocks = extract_blocks(DOCUMENT)
        self.assertEqual([block.page for block in blocks], [1, 1, 2])
        self.assertIn("| Model | Score |", blocks[2].text)
        self.assertNotIn("Footer", " ".join(block.text for block in blocks))
        chunks = make_chunks(blocks, 200)
        self.assertEqual(chunks[0].pages, (1, 2))

    def test_pipeline_writes_grounded_aligned_markdown(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "paper.pdf"
            source.write_bytes(b"%PDF-fake test input")
            output = Path(directory) / "result.md"
            client = FakeClient()
            process_pdf(source, output, parser=lambda path: DOCUMENT, client=client)
            markdown = output.read_text(encoding="utf-8")
            self.assertIn("## 논문 요약", markdown)
            self.assertIn("The method improved accuracy.", markdown)
            self.assertIn("번역된 내용", markdown)
            self.assertIn("p.1, 2", markdown)
            self.assertFalse(output.with_suffix(".progress.json").exists())
            self.assertEqual(client.models[0], "kaelri/hy-mt2:7b-q4_K_M")

    def test_generic_pdf_metadata_title_uses_first_heading(self):
        document = dict(DOCUMENT, title="paper.dvi")
        result = render_markdown(document, "paper.pdf", [], [], "요약", "translation", "summary", "Korean")
        self.assertTrue(result.startswith("# Introduction\n"))

    def test_ollama_http_payload(self):
        requests = []

        def fake_open(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/api/tags"):
                return BytesIO(b'{"models":[{"name":"translate"},{"name":"summary"}]}')
            return BytesIO(b'{"message":{"content":"translated"}}')

        with patch("paper_localizer.core.urllib.request.urlopen", side_effect=fake_open):
            client = OllamaClient()
            client.check_models("translate", "summary")
            self.assertEqual(client.chat("translate", "source", translation=True), "translated")
        self.assertIn(b'"think": false', requests[1].data)
        self.assertIn(b'"num_predict": 4096', requests[1].data)


if __name__ == "__main__":
    unittest.main()
