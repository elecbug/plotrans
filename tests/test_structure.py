from pathlib import Path
import tempfile
import unittest
import textwrap
from unittest.mock import Mock

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paper_localizer.article import article_html, build_article
from paper_localizer.core import Block, extract_blocks, make_chunks
from paper_localizer.figures import attach_figures
from paper_localizer.layout import caption_label, prepare_layout
from paper_localizer.structure import (_restore_source_spacing, order_document,
                                       repair_structure)


def fixture_pdf(path, pages):
    writer = PdfWriter()
    fonts = {name: DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                    NameObject('/Subtype'): NameObject('/Type1'),
                                    NameObject('/BaseFont'): NameObject('/' + font)})
             for name, font in [('F1', 'Helvetica-Oblique'), ('F2', 'Times-Roman'), ('F3', 'Helvetica-Bold'),
                                ('F4', 'Courier'), ('F5', 'Symbol')]}
    for operations in pages:
        page = writer.add_blank_page(width=600, height=600)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({
            NameObject('/' + name): font for name, font in fonts.items()})})
        stream = DecodedStreamObject(); stream.set_data(operations)
        page[NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as target:
        writer.write(target)
    writer.close()


def paragraph(text, box, page=1, **kwargs):
    return {'type': 'paragraph', 'page number': page, 'content': text,
            'bounding box': box, **kwargs}


class StructureTests(unittest.TestCase):
    def test_centered_abstract_label_omitted_by_layout_uses_the_prose_column(self):
        abstract = ('This abstract explains how source geometry identifies a complete paragraph, '
                    'preserving its words and its boundaries while the reading edition keeps '
                    'metadata separate from the body. The experiment compares several network '
                    'designs and reports their behavior under changing conditions, with all '
                    'measurements retained for later source review.')
        title='A native source geometry study'
        operations=b'BT /F3 16 Tf 30 560 Td ('+title.encode()+b') Tj ET '
        operations+=b'BT /F3 14 Tf 260 480 Td (ABSTRACT) Tj ET '
        for i,line in enumerate(textwrap.wrap(abstract,72)):
            operations+=f'BT /F2 9 Tf 120 {460-i*11} Td ({line}) Tj ET '.encode()
        operations+=b'BT /F2 10 Tf 30 330 Td (The body starts here without a printed section title.) Tj ET'
        with tempfile.TemporaryDirectory() as directory:
            pdf=Path(directory)/'source.pdf';fixture_pdf(pdf,[operations])
            document=repair_structure(pdf,{'title':title,'kids':[
                paragraph(title,[30,550,570,580],font='Helvetica-Bold',**{'font size':16}),
                paragraph(abstract,[120,380,480,470],font='Times-Roman',**{'font size':9}),
                paragraph('The body starts here without a printed section title.',[30,325,285,340],font='Times-Roman',**{'font size':10})]})
            bodies=[n for n in document['kids'] if n.get('type')=='abstract' and n.get('content')]
            self.assertEqual(len(bodies),1)
            self.assertEqual(bodies[0]['content'],abstract)
            self.assertTrue(any(n.get('source role heading recovered') for n in document['kids']))
            self.assertTrue(any(n.get('type')=='paragraph' and n.get('content','').startswith('The body starts') for n in document['kids']))

    def test_large_prose_drop_cap_joins_its_first_word_but_math_font_does_not(self):
        for font in ['Helvetica-Bold','Helvetica-Oblique']:
            with self.subTest(font=font),tempfile.TemporaryDirectory() as directory:
                pdf=Path(directory)/'source.pdf'
                fixture_pdf(pdf,[b'BT '+(b'/F3' if font=='Helvetica-Bold' else b'/F1')+b' 48 Tf 30 484 Td (P) Tj ET '
                                 b'BT /F2 10 Tf 63 500 Td (eer-to-peer networks preserve the complete first word.) Tj ET'])
                document=repair_structure(pdf,{'kids':[
                    paragraph('P',[30,470,60,530],font=font,**{'font size':48}),
                    paragraph('eer-to-peer networks preserve the complete first word.',[63,480,285,512],font='Times-Roman',**{'font size':10})]})
                text=' '.join(n.get('content','') for n in document['kids'] if not n.get('suppressed'))
                self.assertEqual('Peer-to-peer' in text,font=='Helvetica-Bold')

    def test_column_order_is_independent_of_parser_list_nesting(self):
        nodes = [paragraph('Right upper.', [320, 450, 570, 530]),
                 paragraph('Left lower.', [30, 300, 280, 380]),
                 paragraph('Right lower.', [320, 300, 570, 380]),
                 paragraph('Left upper.', [30, 450, 280, 530])]
        document = {'kids': [{'type': 'list', 'page number': 1, 'list items': nodes}],
                    'page sizes': {'1': [600, 600]}}
        ordered = order_document(document)['kids']
        self.assertEqual([n['content'] for n in ordered],
                         ['Left upper.', 'Left lower.', 'Right upper.', 'Right lower.'])

    def test_full_width_bands_split_reading_order_into_column_groups(self):
        nodes = [paragraph('Bottom right.', [320, 100, 570, 180]),
                 paragraph('Top right.', [320, 430, 570, 520]),
                 paragraph('Top left.', [30, 430, 280, 520]),
                 paragraph('A full width band.', [30, 250, 570, 400]),
                 paragraph('Bottom left.', [30, 100, 280, 180])]
        document = {'kids': nodes, 'page sizes': {'1': [600, 600]}}
        self.assertEqual([n['content'] for n in order_document(document)['kids']],
                         ['Top left.', 'Top right.', 'A full width band.', 'Bottom left.', 'Bottom right.'])

    def test_run_in_heading_is_split_only_when_native_text_agrees(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pdf'
            fixture_pdf(source, [b'BT /F1 10 Tf 30 500 Td (C. NATIVE SECTION) Tj ET '
                                 b'BT /F2 10 Tf 30 480 Td (The body remains editable.) Tj ET'])
            original = paragraph('C. NATIVE SECTION The body remains editable.', [30, 476, 280, 510],
                                 **{'type': 'list item', 'font': 'Helvetica-Oblique', 'font size': 10})
            document = repair_structure(source, {'kids': [original]})
            self.assertEqual([n['type'] for n in document['kids']], ['heading', 'paragraph'])
            self.assertEqual(document['kids'][0]['content'], 'C. NATIVE SECTION')
            self.assertEqual(document['kids'][0]['heading level'], 3)
            self.assertEqual(document['kids'][1]['content'], 'The body remains editable.')
            self.assertLess(document['kids'][1]['bounding box'][3], document['kids'][0]['bounding box'][1])
            mismatch = dict(original, content='C. AN UNRELATED TITLE The body remains editable.')
            document = repair_structure(source, {'kids': [mismatch]})
            self.assertNotEqual(document['kids'][0]['type'], 'heading')

    def test_native_spacing_requires_exact_printable_character_agreement(self):
        node = paragraph('Alphabetagamma.', [30, 400, 280, 420])
        textpage = Mock()
        textpage.get_text_bounded.return_value = 'Alpha beta gamma.'
        self.assertTrue(_restore_source_spacing(node, textpage))
        self.assertEqual(node['content'], 'Alpha beta gamma.')
        node['content'] = 'Alphabetagamma.'
        textpage.get_text_bounded.return_value = 'Alpha beta changed.'
        self.assertFalse(_restore_source_spacing(node, textpage))
        self.assertEqual(node['content'], 'Alphabetagamma.')

    def test_native_discretionary_hyphens_do_not_introduce_control_characters(self):
        node = paragraph('A supportedword.', [30, 400, 280, 420])
        textpage = Mock()
        textpage.get_text_bounded.return_value = 'A supported\x02word.'
        self.assertFalse(_restore_source_spacing(node, textpage))
        self.assertNotIn('\x02', node['content'])
        node['content'] = 'Alphabetagamma-'
        textpage.get_text_bounded.return_value = 'Alpha beta gamma\x02'
        self.assertTrue(_restore_source_spacing(node, textpage))
        self.assertEqual(node['content'], 'Alpha beta gamma-')

    def test_native_spacing_does_not_restore_whitespace_inside_a_checked_url(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pdf'
            fixture_pdf(source, [b'BT /F2 10 Tf 30 500 Td (Available: https://example.org/a-long-) Tj ET '
                                 b'BT /F2 10 Tf 30 487 Td (url.pdf) Tj ET'])
            target = 'https://example.org/a-long-url.pdf'
            document = {'kids': [paragraph('Available: https://example.org/a-long- url.pdf',
                                           [30, 483, 280, 510])],
                        'source links': [{'url': target, 'pages': [1]}]}
            repair_structure(source, document)
            self.assertEqual(document['kids'][0]['content'], 'Available: ' + target)
            self.assertEqual(document['source link text repairs'][0]['after'], target)

    def test_body_references_are_not_promoted_to_captions(self):
        self.assertEqual(caption_label('Table 2 demonstrates an important limitation.'), '')
        self.assertEqual(caption_label('Figure 11 visualizes the measurements.'), '')
        self.assertEqual(caption_label('TABLE 2. Important limitations.'), 'Table 2')

    def test_caption_continuation_can_change_the_label_font(self):
        first = paragraph('TABLE 1. A caption with a bold label', [30, 480, 560, 492],
                          **{'font': 'Helvetica-Bold', 'font size': 8})
        second = paragraph('and its complete continuation.', [30, 468, 560, 480],
                           **{'font': 'Helvetica', 'font size': 8})
        document = prepare_layout({'kids': [first, second]})
        self.assertEqual(len(document['captions']), 1)
        self.assertIn('complete continuation.', document['captions'][0]['content'])

    def test_repeated_header_artwork_never_expands_a_real_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.pdf'
            operations = b'30 570 540 20 re S 30 563 m 570 563 l S 70 330 180 170 re S'
            fixture_pdf(source, [operations] * 4)
            document = {'kids': [paragraph(f'Figure {page}. A real drawing.', [70, 300, 250, 315], page,
                                           **{'font': 'Helvetica', 'font size': 8}) for page in range(1, 5)]}
            document = attach_figures(source, prepare_layout(document), root)
            self.assertEqual(len(document['figures']), 4)
            self.assertTrue(all(f['bounding box'][3] < 520 for f in document['figures']))
            self.assertEqual(len(document['excluded artwork']), 4)

    def test_long_paragraph_does_not_gain_breaks_from_model_chunk_size(self):
        text = ' '.join('Sentence ' + ' '.join(f'word{index}' for index in range(start, start + 30)) + '.'
                        for start in range(0, 180, 30))
        chunks = make_chunks([Block(1, 'heading', '## Introduction'), Block(1, 'paragraph', text)], 220)
        self.assertGreater(len(chunks), 3)
        article = build_article({}, 'Paper', 'source.pdf', chunks)
        self.assertEqual(len(article.sections[0].units), 1)
        self.assertEqual(article.sections[0].units[0].source, text)

    def test_distinct_paragraphs_remain_distinct_in_one_chunk(self):
        chunks = make_chunks([Block(1, 'heading', '## Introduction'),
                              Block(1, 'paragraph', 'A complete first paragraph.'),
                              Block(1, 'paragraph', 'A complete second paragraph.')])
        article = build_article({}, 'Paper', 'source.pdf', chunks)
        self.assertEqual([u.source for u in article.sections[0].units],
                         ['A complete first paragraph.', 'A complete second paragraph.'])

    def test_continuation_across_columns_and_a_float_is_one_reading_paragraph(self):
        document = {'page sizes': {'1': [600, 600]}, 'kids': [
            paragraph('The sentence continues with', [30, 50, 280, 100]),
            {'type': 'image', 'page number': 1, 'bounding box': [320, 400, 570, 530], 'source': 'figs/test/plot.png'},
            paragraph('the rest of the same sentence.', [320, 250, 570, 380]),
        ]}
        # Supply enough evidence for two columns, independently of the image.
        document['kids'].insert(0, paragraph('A separate opening paragraph.', [30, 300, 280, 380]))
        document = order_document(document)
        chunks = make_chunks(extract_blocks(document))
        article = build_article(document, 'Paper', 'source.pdf', chunks)
        sources = [u.source for u in article.front if u.kind == 'text']
        self.assertIn('The sentence continues with the rest of the same sentence.', sources)

    def test_author_metadata_and_keywords_do_not_become_body_sections(self):
        chunks = make_chunks([Block(1, 'frontmatter', 'Author One and Author Two'),
                              Block(1, 'paragraph', 'ABSTRACT The source abstract.'),
                              Block(1, 'keywords', 'INDEX TERMS Parsing, geometry.'),
                              Block(1, 'heading', '## I. INTRODUCTION'),
                              Block(1, 'paragraph', 'The first body paragraph.')])
        article = build_article({'author': 'Author One and Author Two'}, 'Paper', 'source.pdf', chunks)
        self.assertEqual(len(article.abstract), 1)
        self.assertEqual(article.keywords[0].source, 'Parsing, geometry.')
        self.assertEqual([s.title for s in article.sections], ['I. INTRODUCTION'])
        self.assertIn('id="abstract"', article_html(article, 'paper.md'))

    def test_reference_year_and_url_continuations_are_not_new_lists(self):
        document = {'page sizes': {'1': [600, 600]}, 'kids': [
            paragraph('REFERENCES', [30, 550, 280, 565], type='heading'),
            paragraph('[14] A. Author, A paper in CRYPTO', [30, 500, 280, 520], type='list item'),
            paragraph('2001. Berlin: Springer, 2001.', [48, 480, 280, 499]),
            paragraph('[15] A. Author. Available: https://example.org/a-long',
                      [30, 440, 280, 465], type='list item'),
            paragraph('-url.pdf', [48, 425, 150, 439]),
            paragraph('[16] A different reference.', [30, 390, 280, 415], type='list item'),
        ]}
        order_document(document)
        article = build_article(document, 'Paper', 'paper.pdf', make_chunks(extract_blocks(document)))
        sources = [unit.source for unit in article.sections[0].units]
        self.assertEqual(sources, [
            '[14] A. Author, A paper in CRYPTO 2001. Berlin: Springer, 2001.',
            '[15] A. Author. Available: https://example.org/a-long -url.pdf',
            '[16] A different reference.',
        ])
        self.assertNotIn('<ol start="2001">', article_html(article, 'paper.md'))

    def test_citations_and_source_confirmed_unindented_names_continue_paragraphs(self):
        document = {'page sizes': {'1': [600, 600]}, 'kids': [
            paragraph('An initial complete paragraph.', [30, 450, 280, 480]),
            paragraph('Selected design depends on the network', [30, 400, 280, 430]),
            paragraph('[6], [7], [8].', [30, 386, 140, 399]),
            paragraph('The term is Peer Reputation', [30, 40, 280, 70], **{'font size': 10}),
            paragraph('Management Systems in this context.', [320, 450, 570, 480],
                      **{'font size': 10, 'source first line indent': .1}),
        ]}
        order_document(document)
        article = build_article(document, 'Paper', 'paper.pdf', make_chunks(extract_blocks(document)))
        sources = [u.source for u in article.front]
        self.assertIn('Selected design depends on the network [6], [7], [8].', sources)
        self.assertIn('The term is Peer Reputation Management Systems in this context.', sources)

    def test_indented_or_unconfirmed_capitalized_column_start_is_not_joined(self):
        for indent in (None, 10):
            nodes = [paragraph('An initial complete paragraph.', [30, 450, 280, 480]),
                     paragraph('A prior fragment', [30, 40, 280, 70], **{'font size': 10}),
                     paragraph('Distinct paragraph.', [320, 450, 570, 480], **{'font size': 10})]
            if indent is not None: nodes[-1]['source first line indent'] = indent
            doc = {'page sizes': {'1': [600, 600]}, 'kids': nodes}
            order_document(doc)
            ordered = {node['content']: node for node in doc['kids']}
            self.assertNotEqual(ordered['A prior fragment']['paragraph id'],
                                ordered['Distinct paragraph.']['paragraph id'])

    def test_reference_continuation_can_cross_a_page_but_not_a_heading(self):
        document = {'page sizes': {'1': [600, 600], '2': [600, 600]}, 'kids': [
            paragraph('REFERENCES', [30, 550, 280, 565], type='heading'),
            paragraph('[1] A reference ending in', [30, 30, 280, 55], type='list item'),
            paragraph('2002. Its publication year.', [30, 500, 280, 520], page=2),
            paragraph('Appendix', [30, 460, 280, 480], page=2, type='heading'),
            paragraph('1. A real numbered item.', [30, 420, 280, 445], page=2, type='list item'),
        ]}
        order_document(document)
        article = build_article(document, 'Paper', 'paper.pdf', make_chunks(extract_blocks(document)))
        self.assertEqual(article.sections[0].units[0].source,
                         '[1] A reference ending in 2002. Its publication year.')
        self.assertEqual(article.sections[0].units[0].pages, (1, 2))
        self.assertIn('<ol>', article_html(article, 'paper.md'))


if __name__ == '__main__':
    unittest.main()
