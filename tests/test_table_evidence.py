from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import pdfplumber
import pypdfium2 as pdfium
from pdfminer.psparser import PSException

from paper_localizer.article import Unit, _table_transcription, write_article
from paper_localizer.formulas import _spans
from paper_localizer.table_evidence import _checked_rows, supplement_source_tables
from paper_localizer.tables import TableError, table_grid
from paper_localizer.vision import refine_document
from test_structure import fixture_pdf


class NativeTableEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root/'source.pdf'
        operations = (b'30 550 m 280 550 l S 30 525 m 280 525 l S 30 470 m 280 470 l S '
                      b'BT /F3 10 Tf 40 535 Td (Label) Tj ET '
                      b'BT /F3 10 Tf 180 535 Td (Score) Tj ET '
                      b'BT /F2 10 Tf 40 510 Td (Alpha) Tj ET '
                      b'BT /F2 10 Tf 180 510 Td (1) Tj ET '
                      b'BT /F2 10 Tf 40 485 Td (Beta) Tj ET '
                      b'BT /F2 10 Tf 180 485 Td (2) Tj ET')
        fixture_pdf(self.path, [operations])
        self.box = (30, 470, 280, 550)
        self.asset = {'source': 'figs/table.png', 'asset kind': 'table', 'page number': 1,
                      'caption label': 'Table 1', 'bounding box': list(self.box)}
        self.document = {'figures': [self.asset], 'source table regions': [
            {'page number': 1, 'caption label': 'Table 1', 'bounding box': list(self.box)}]}
        self.pdf = pdfium.PdfDocument(self.path)
        self.addCleanup(self.pdf.close)
        self.page = self.pdf[0]
        self.addCleanup(self.page.close)
        self.spans = _spans(self.page)
        self.cells = [[(30, 50, 160, 75), (160, 50, 280, 75)],
                      [(30, 75, 160, 100), (160, 75, 280, 100)],
                      [(30, 100, 160, 130), (160, 100, 280, 130)]]

    def proposal(self, values=None, cells=None):
        proposal = Mock()
        proposal.extract.return_value = values or [['Label', 'Score'], ['Alpha', '1'], ['Beta', '2']]
        proposal.rows = [Mock(cells=row) for row in (cells or self.cells)]
        return proposal

    def test_real_table_adds_native_cells_and_keeps_source_image(self):
        original = deepcopy(self.asset)
        document = supplement_source_tables(self.path, self.document)
        self.assertEqual(document['figures'], [original])
        data = document['table transcriptions'][self.asset['source']]
        self.assertEqual(table_grid(data['rows']), (('Label', 'Score'), ('Alpha', '1'), ('Beta', '2')))
        self.assertEqual(data['verification']['cell_count'], 6)
        self.assertTrue(data['verification']['complete_span_ownership'])
        supplement_source_tables(self.path, document)
        self.assertEqual(len(document['native table transcriptions']), 1)

    def test_a_changed_number_or_dropped_word_cannot_be_approved(self):
        for value in ['10', '-1', '1.0', '']:
            with self.subTest(value=value), self.assertRaises(TableError):
                self._check(self.proposal([['Label', 'Score'], ['Alpha', value], ['Beta', '2']]))
        with self.assertRaises(TableError):
            self._check(self.proposal([['Label', 'Score'], ['Alph', '1'], ['Beta', '2']]))

    def _check(self, proposal):
        return _checked_rows(proposal, self.page, self.box, self.spans)

    def test_column_boundary_cutting_a_complete_word_is_rejected(self):
        cells = [[(30, 50, 195, 75), (195, 50, 280, 75)], *self.cells[1:]]
        with self.assertRaises(TableError):
            self._check(self.proposal([['Label Sco', 're'], ['Alpha', '1'], ['Beta', '2']], cells))

    def test_merged_and_private_use_cells_remain_source_images(self):
        with self.assertRaises(TableError):
            self._check(self.proposal(cells=[[self.cells[0][0], None], *self.cells[1:]]))
        with self.assertRaises(TableError):
            self._check(self.proposal([['Label', 'Score'], ['Alpha', '\ue123'], ['Beta', '2']]))

    def test_omitted_positioned_spans_in_a_column_gap_are_rejected(self):
        spans = deepcopy(self.spans)
        extra = deepcopy(spans[0]);extra.index = 9999;extra.box = (151, 500, 155, 510);extra.text = 'lost'
        cells = [[(30, 50, 150, 75), (160, 50, 280, 75)],
                 [(30, 75, 150, 100), (160, 75, 280, 100)],
                 [(30, 100, 150, 130), (160, 100, 280, 130)]]
        with self.assertRaisesRegex(TableError, 'omitted'):
            _checked_rows(self.proposal(cells=cells), self.page, self.box, spans+[extra])

    def test_pdfminer_parse_failure_retains_the_existing_image(self):
        with patch('paper_localizer.table_evidence.pdfplumber.open', side_effect=PSException('invalid table stream')):
            document = supplement_source_tables(self.path, self.document)
        self.assertEqual(document['figures'], [self.asset])
        self.assertFalse(document['table transcriptions'])
        self.assertFalse(document['native table transcriptions'][0]['accepted'])

    def test_verified_native_cells_do_not_trigger_table_model_requests(self):
        document = supplement_source_tables(self.path, self.document)
        calls = []
        def request(url, path, **kwargs):
            calls.append(path)
            if path == '/api/tags':
                return {'models': [{'name': 'glm-ocr:latest', 'digest': 'model'}]}
            self.fail('Native table unexpectedly invoked model OCR.')
        result = refine_document(self.path, document, self.root, model='glm-ocr:latest',
                                 url='http://127.0.0.1', requester=request)
        self.assertEqual(calls, ['/api/tags'])
        self.assertEqual(table_grid(result['table transcriptions'][self.asset['source']]['rows'])[1], ('Alpha', '1'))
        self.assertTrue(result['vision ocr']['regions'][0]['accepted'])

    def test_reader_distinguishes_native_and_model_verification(self):
        rows, evidence = self._check(self.proposal())
        native = Unit('', table_data={'rows': rows, 'verification': evidence})
        self.assertIn('Native table cells checked', _table_transcription(native))
        self.assertNotIn('OCR transcription checked', _table_transcription(native))
        model = Unit('', table_data={'rows': rows, 'verification': {'accepted': True}})
        self.assertIn('OCR transcription checked', _table_transcription(model))

    def test_native_subscripts_are_preserved_without_model_ocr(self):
        path = self.root/'scripts.pdf'
        fixture_pdf(path, [b'BT /F3 10 Tf 40 535 Td (Label) Tj ET '
                           b'BT /F1 10 Tf 180 535 Td (D) Tj ET '
                           b'BT /F2 7 Tf 187 532 Td (low) Tj ET '
                           b'BT /F2 10 Tf 40 510 Td (Alpha) Tj ET '
                           b'BT /F2 10 Tf 180 510 Td (1) Tj ET '
                           b'BT /F2 10 Tf 40 485 Td (Beta) Tj ET '
                           b'BT /F2 10 Tf 180 485 Td (2) Tj ET'])
        with pdfium.PdfDocument(path) as pdf:
            page = pdf[0]
            try:
                proposal = self.proposal([['Label', 'D\nlow'], ['Alpha', '1'], ['Beta', '2']])
                rows, _ = _checked_rows(proposal, page, self.box, _spans(page))
                self.assertEqual(rows[0][1]['text'], '$D_{\\mathrm{low}}$')
                with patch('paper_localizer.table_evidence.native_expression', return_value=None):
                    with self.assertRaisesRegex(TableError, 'notation'):
                        _checked_rows(proposal, page, self.box, _spans(page))
            finally:
                page.close()

    def test_native_evidence_is_saved_without_a_vision_model(self):
        document = supplement_source_tables(self.path, self.document)
        output = self.root/'paper.md'
        write_article(document, 'Native table', 'source.pdf', [], None, 'English', output)
        evidence = json.loads(output.with_suffix('.ocr.json').read_text())
        self.assertFalse(evidence['vision'])
        self.assertEqual(evidence['tables'][self.asset['source']]['verification']['cell_count'], 6)
        self.assertTrue(evidence['native_tables'][0]['accepted'])


if __name__ == '__main__':
    unittest.main()
