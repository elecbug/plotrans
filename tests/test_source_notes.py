from pathlib import Path
import tempfile
import unittest

from paper_localizer.core import extract_blocks
from paper_localizer.structure import repair_structure
from test_structure import fixture_pdf, paragraph


class SourceNoteTests(unittest.TestCase):
    def source(self, root, first='13The parameter does', last='not affect our findings.',
               size=8, x=30):
        source = root/'notes.pdf'
        operations = (b'BT /F2 10 Tf 30 500 Td '
                      b'(The ordinary body uses larger letters and preserves every complete sentence.) Tj ET '
                      + f'BT /F2 {size} Tf 30 100 Td ({first}) Tj ET '.encode()
                      + f'BT /F2 {size} Tf {x} 90 Td ({last}) Tj ET'.encode())
        fixture_pdf(source,[operations])
        return source

    def test_short_last_line_of_numbered_note_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            node = paragraph('13The parameter does',[29,98,280,109],font='Times-Roman',**{'font size':8})
            doc = repair_structure(source,{'kids':[node]})
            text = ' '.join(b.text for b in extract_blocks(doc))
            self.assertIn('13The parameter does not affect our findings.',text)
            repaired = next(n for n in doc['kids'] if n.get('source note continuations'))
            self.assertEqual(repaired['source note continuations'][0]['text'],'not affect our findings.')
            self.assertLess(repaired['bounding box'][1],95)
            self.assertEqual(text.count('not affect our findings.'),1)

    def test_source_mismatch_or_different_column_is_not_appended(self):
        for mismatch, x in [(True,30),(False,320)]:
            with self.subTest(mismatch=mismatch,x=x),tempfile.TemporaryDirectory() as directory:
                source = self.source(Path(directory),x=x)
                content = '13A different source sentence' if mismatch else '13The parameter does'
                node = paragraph(content,[29,98,280,109],font='Times-Roman',**{'font size':8})
                doc = repair_structure(source,{'kids':[node]})
                self.assertIn(content,' '.join(b.text for b in extract_blocks(doc)))
                self.assertFalse(any(n.get('source note continuations') for n in doc['kids']))

    def test_completed_note_and_normal_size_body_keep_their_boundaries(self):
        for first,size in [('13The parameter does.',8),('The parameter does',8),('13The parameter does',10)]:
            with self.subTest(first=first,size=size),tempfile.TemporaryDirectory() as directory:
                source = self.source(Path(directory),first=first,size=size)
                node = paragraph(first,[29,98,280,111],font='Times-Roman',**{'font size':size})
                doc = repair_structure(source,{'kids':[node]})
                self.assertIn(first,' '.join(b.text for b in extract_blocks(doc)))
                self.assertFalse(any(n.get('source note continuations') for n in doc['kids']))


class SourceLabelTests(unittest.TestCase):
    def test_short_bold_body_label_recovers_the_same_label_omitted_at_page_top(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.pdf'
            fixture_pdf(path,[
                b'BT /F3 10 Tf 30 500 Td (Preconditions) Tj ET',
                b'BT /F3 10 Tf 30 500 Td (Preconditions) Tj ET',
                b'BT /F3 10 Tf 30 570 Td (Preconditions) Tj ET '
                b'BT /F2 10 Tf 30 550 Td (The complete body follows the repeated source label.) Tj ET',
                b'BT /F3 8 Tf 30 570 Td (Preconditions) Tj ET',
            ])
            nodes = [paragraph('Preconditions',[29,496,150,510],page=number,
                               font='Helvetica-Bold',**{'font size':10}) for number in [1,2]]
            nodes.append(paragraph('The complete body follows the repeated source label.',[29,546,400,561],page=3,
                                   font='Times-Roman',**{'font size':10}))
            doc = repair_structure(path,{'kids':nodes})
            labels = [n for n in doc['kids'] if n.get('source label recovered')]
            self.assertEqual([(n['page number'],n['content']) for n in labels],[(3,'Preconditions')])
            self.assertEqual(labels[0]['type'],'paragraph')

    def test_a_repeated_running_header_without_body_examples_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'headers.pdf'
            fixture_pdf(path,[b'BT /F3 10 Tf 30 570 Td (Running Header) Tj ET']*3)
            nodes = [paragraph('Running Header',[29,566,150,580],page=n,
                               type='header',font='Helvetica-Bold',**{'font size':10}) for n in [1,2]]
            doc = repair_structure(path,{'kids':nodes})
            self.assertFalse(any(n.get('source label recovered') for n in doc['kids']))

    def test_recovering_a_label_does_not_strand_its_short_status_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.pdf'
            fixture_pdf(path,[b'BT /F3 10 Tf 30 500 Td (Verification) Tj ET',
                              b'BT /F3 10 Tf 30 500 Td (Verification) Tj ET',
                              b'BT /F3 10 Tf 30 570 Td (Verification) Tj ET '
                              b'BT /F2 10 Tf 30 555 Td (Unresolved.) Tj ET'])
            nodes = [paragraph('Verification',[29,496,150,510],page=n,
                               font='Helvetica-Bold',**{'font size':10}) for n in [1,2]]
            doc = repair_structure(path,{'kids':nodes})
            values = [b.text for b in extract_blocks(doc) if b.page==3]
            self.assertIn('Verification Unresolved.',' '.join(values))
            self.assertEqual(' '.join(values).count('Unresolved.'),1)



if __name__ == '__main__':
    unittest.main()
