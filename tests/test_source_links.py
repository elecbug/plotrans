from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter
from pypdf.generic import (ArrayObject, DictionaryObject, NameObject, NumberObject,
                           TextStringObject)

from paper_localizer.article import Unit, article_html, article_markdown, build_article
from paper_localizer.core import Chunk
from paper_localizer.source_links import (assign_source_links, read_source_links,
                                         restore_printed_targets, web_target)


class SourceLinkTests(unittest.TestCase):
    def test_pdf_annotation_targets_are_deduplicated_without_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "links.pdf"
            writer = PdfWriter()
            for targets in (["https://example.org/paper", "https://example.org/paper", "javascript:alert(1)"],
                            ["https://example.org/paper", "https://doi.org/10.1234/example"]):
                page = writer.add_blank_page(width=200, height=200)
                annotations = []
                for target in targets:
                    annotation = DictionaryObject({
                        NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Link"),
                        NameObject("/Rect"): ArrayObject([NumberObject(n) for n in [10, 20, 100, 30]]),
                        NameObject("/A"): DictionaryObject({NameObject("/S"): NameObject("/URI"),
                                                            NameObject("/URI"): TextStringObject(target)})})
                    annotations.append(writer._add_object(annotation))
                page[NameObject("/Annots")] = ArrayObject(annotations)
            with source.open("wb") as file:
                writer.write(file)
            writer.close()
            links = read_source_links(source)
            self.assertEqual(len(links), 2)
            self.assertEqual(links[0]["pages"], [1, 2])
            self.assertEqual(len(links[0]["rectangles"]), 3)

    def test_link_is_attached_only_to_a_unique_printed_target_on_its_page(self):
        unit = Unit("[1] Example. doi: 10.1234 / example.", pages=(8,))
        links = [{"url": "https://doi.org/10.1234/example", "pages": [8]},
                 {"url": "https://orcid.org/0000-0002-1825-0097", "pages": [1]}]
        identifiers, unmatched = assign_source_links(links, [unit])
        self.assertEqual(len(unit.links), 1)
        self.assertEqual(len(identifiers), 1)
        self.assertEqual(identifiers[0]["label"], "ORCID 0000-0002-1825-0097")
        self.assertEqual(unmatched, [])
        self.assertNotIn("author", identifiers[0])

    def test_ambiguous_or_wrong_page_target_remains_available_without_attribution(self):
        link = {"url": "https://example.org/paper", "pages": [4]}
        for units in ([Unit("https://example.org/paper", pages=(3,))],
                      [Unit("https://example.org/paper", pages=(4,)),
                       Unit("https://example.org/paper", pages=(4,))]):
            with self.subTest(units=units):
                _, unmatched = assign_source_links([link], units)
                self.assertEqual(len(unmatched), 1)
                self.assertFalse(any(unit.links for unit in units))

    def test_active_or_invalid_targets_are_never_exported(self):
        for value in ("javascript:alert(1)", "data:text/html,x", "file:///tmp/example",
                      "https://", "https://[invalid", "https://example.org:bad/path",
                      "https://example.org/\nscript", "https:\\example.org",
                      "https://example.org/\ud800"):
            with self.subTest(value=value):
                self.assertEqual(web_target(value), "")
        self.assertEqual(web_target("https://example.org/a(b)?q=<x>"),
                         "https://example.org/a%28b%29?q=%3Cx%3E")

    def test_html_and_markdown_keep_identifiers_reference_links_and_unmatched_links(self):
        document = {"source links": [
            {"url": "https://orcid.org/0000-0002-1825-0097", "pages": [1]},
            {"url": "https://doi.org/10.1234/example", "pages": [2]},
            {"url": "https://example.org/a(b)?a=1&b=2", "pages": [1]},
            {"url": "javascript:alert(1)", "pages": [1]}]}
        article = build_article(document, "Paper", "paper.pdf",
                                [Chunk(1, (2,), "## References\n\n[1] Example. doi: 10.1234/example.")])
        html = article_html(article, "paper.md")
        markdown = article_markdown(article)
        self.assertIn('href="https://doi.org/10.1234/example"', html)
        self.assertIn("ORCID 0000-0002-1825-0097", html)
        self.assertIn("Other links from the source PDF", html)
        self.assertIn("a%28b%29?a=1&amp;b=2", html)
        self.assertIn("[DOI](https://doi.org/10.1234/example)", markdown)
        self.assertNotIn("javascript:", html + markdown)

    def test_wrapped_target_repairs_require_native_and_annotation_agreement(self):
        links = [{"url": "https://doi.org/10.1007/978-0-387-09751-0_29"},
                 {"url": "https://example.org/a-long-url.pdf"}]
        text = 'Source doi: 10.1007/978-0-387-097510_29. Available: https://exa mple.org/a-long-url.pdf'
        native = 'Source doi: 10.1007/978-0-387-09751-\r\n0_29. Available: https://example.org/a-long-\r\nurl.pdf'
        result, repairs = restore_printed_targets(text, native, links)
        self.assertEqual(result, 'Source doi: 10.1007/978-0-387-09751-0_29. Available: https://example.org/a-long-url.pdf')
        self.assertEqual(len(repairs), 2)
        self.assertEqual(restore_printed_targets(text, native.replace('09751-', '09751'), links)[0],
                         'Source doi: 10.1007/978-0-387-097510_29. Available: https://example.org/a-long-url.pdf')
        self.assertEqual(restore_printed_targets(text, native, [])[0], text)

    def test_link_repair_never_changes_digits_case_other_punctuation_or_ambiguous_matches(self):
        target = 'https://example.org/Case-12.pdf'
        for text in (target.replace('12', '13'), target.replace('Case', 'case'),
                     target.replace('.pdf', '_pdf'), target.replace('-', '') + ' ' + target.replace('-', '')):
            with self.subTest(text=text):
                self.assertEqual(restore_printed_targets(text, target, [{"url": target}]), (text, []))

    def test_reference_assembly_repairs_cross_block_url_only_with_page_confirmation(self):
        target = 'https://example.org/a-long-url.pdf'
        chunks = [Chunk(1, (1,), '## References'),
                  Chunk(2, (1,), '[17] Available: https://example.org/a-long', paragraph_ids=('reference-17',)),
                  Chunk(3, (1,), '-url.pdf', paragraph_ids=('reference-17',))]
        for confirmed in (False, True):
            link = {'url': target, 'pages': [1]}
            if confirmed:
                link['printed targets'] = {'1': target}
            article = build_article({'source links': [link]}, 'Paper', 'paper.pdf', chunks)
            unit = article.sections[0].units[0]
            self.assertEqual(unit.source, '[17] Available: ' + (target if confirmed else target.replace('-url', ' -url')))
            self.assertEqual(bool(unit.links[0].get('text repairs')), confirmed)


if __name__ == "__main__":
    unittest.main()
