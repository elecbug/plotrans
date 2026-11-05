from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from paper_localizer.article import article_html, build_article
from paper_localizer.core import _table_markdown, extract_blocks, make_chunks
from paper_localizer.layout import caption_for_region, caption_label, prepare_layout
from paper_localizer.figures import _pseudocode_region, attach_figures
from paper_localizer.formulas import TextSpan, _remove_figure_duplicates, _merge_legacy_equation_snapshots
from paper_localizer.structure import _mark_page_furniture, _restore_source_spacing, repair_structure
from paper_localizer.vision import _native_prose_preserved, refine_document, VisionOcrError
from test_structure import fixture_pdf, paragraph
from test_formulas import math_fixture


class ParserRegressionTests(unittest.TestCase):
    def test_native_merged_headers_keep_their_data_columns(self):
        def cell(text, column, **extra):
            return {'column number': column, 'kids': [{'content': text}], **extra}
        table = {'type': 'table', 'page number': 1, 'rows': [
            {'cells': [cell('Network', 1, **{'row span': 2, 'is_header': True}),
                       cell('Degree', 2, **{'column span': 2, 'is_header': True}),
                       cell('Diameter', 4, **{'row span': 2, 'is_header': True})]},
            {'cells': [cell('min', 2, is_header=True), cell('max', 3, is_header=True)]},
            {'cells': [cell('Testnet', 1), cell('1', 2), cell('418', 3), cell('5', 4)]}]}
        self.assertIn('| Network | Degree / min | Degree / max | Diameter |', _table_markdown(table))
        article = build_article({}, 'Paper', 'paper.pdf', make_chunks(extract_blocks({'kids': [table]})))
        html = article_html(article, 'paper.md')
        self.assertIn('rowspan="2"', html)
        self.assertIn('colspan="2"', html)
        self.assertIn('<td>418</td><td>5</td>', html)

    def test_page_counters_do_not_remove_equal_bibliographic_numbers(self):
        doc = {'page sizes': {str(i): [600, 800] for i in range(1, 6)}, 'kids': []}
        for i in range(1, 6):
            doc['kids'].append(paragraph(str(2150+i)+str(i), [290, 30, 310, 44], i))
        reference = paragraph('[5] A. Author, Paper 21533, pp. 12-23.', [30, 120, 280, 145], 3)
        doc['kids'].append(reference)
        _mark_page_furniture(doc)
        self.assertTrue(all(n['type']=='footer' for n in doc['kids'][:-1]))
        self.assertEqual(reference['type'], 'paragraph')
        self.assertIn('21533', extract_blocks(doc)[0].text)

    def test_only_source_confirmed_numeric_ranges_are_restored(self):
        node = paragraph('Proceedings pp. 1223, ACM.', [30, 400, 280, 420])
        textpage = Mock();textpage.get_text_bounded.return_value = 'Proceedings pp. 12-\r\n23, ACM.'
        self.assertTrue(_restore_source_spacing(node, textpage))
        self.assertIn('12-23', node['content'])
        textpage.get_text_bounded.return_value = 'Proceedings pp. 12-24, ACM.'
        self.assertFalse(_restore_source_spacing(node, textpage))
        textpage.get_text_bounded.return_value = 'Proceedings pp. 1223, ACM.'
        self.assertFalse(_restore_source_spacing(node, textpage))
        self.assertIn('12-23', node['content'])

    def test_native_prose_and_data_survive_consistent_ocr_omissions(self):
        source = 'Definition assumes a Poisson process assumption. There are 20 nodes.'
        self.assertFalse(_native_prose_preserved(source, 'Definition assumes a Poisson process. There are 20 nodes.'))
        self.assertFalse(_native_prose_preserved(source, source.replace('20', '30')))
        self.assertTrue(_native_prose_preserved(source, source))
        self.assertFalse(_native_prose_preserved(source, source+' Adjacent condition.'))
        self.assertFalse(_native_prose_preserved(source, source.replace('Poisson process', 'process Poisson')))

    def test_mid_document_ocr_disconnect_retries_once_then_keeps_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.pdf';math_fixture(source)
            text='The original source paragraph remains available for reading.'
            doc={'kids': [paragraph(text, [20, 30, 180, 50], font='OCR'),
                          paragraph(text, [20, 60, 180, 80], font='OCR')]}
            calls=[]
            def request(url, path, body=None, **kwargs):
                if path=='/api/tags': return {'models':[{'name':'glm-ocr:latest','digest':'model'}]}
                calls.append(path);raise VisionOcrError('Connection was interrupted')
            refine_document(source, doc, root, model='glm-ocr:latest', url='http://localhost', requester=request)
            self.assertEqual(len(calls), 2)
            self.assertEqual([n['content'] for n in doc['kids']], [text, text])
            self.assertTrue(all(not n['accepted'] for n in doc['vision ocr']['regions']))

    def test_two_resolutions_cannot_approve_a_missing_native_author_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.pdf';math_fixture(source)
            text='Alice Example Bob Example Carol Example Native authors describe a random distributed network.'
            candidate='Native authors describe a random distributed network.'
            node=paragraph(text, [20, 30, 180, 70], font='Times-Roman',
                           **{'formula snapshot': '![Source](figs/source.png)'})
            def request(url, path, body=None, **kwargs):
                if path=='/api/tags': return {'models':[{'name':'glm-ocr:latest','digest':'model'}]}
                return {'response': candidate, 'done_reason': 'stop'}
            doc=refine_document(source, {'kids':[node]}, root, model='glm-ocr:latest', url='http://localhost', requester=request)
            decision=doc['vision ocr']['regions'][0]
            self.assertEqual(decision['second'], candidate)
            self.assertFalse(decision['accepted'])
            self.assertEqual(node['content'], text)
            self.assertIn('formula snapshot', node)

    def test_native_page_top_continuation_is_recovered_in_its_own_page(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pdf'
            fixture_pdf(source, [b'BT /F2 10 Tf 30 550 Td (This sentence was omitted at the page top.) Tj ET '
                                 b'BT /F2 10 Tf 30 500 Td (The retained paragraph.) Tj ET'])
            doc = repair_structure(source, {'kids': [paragraph('The retained paragraph.', [30, 496, 280, 510])]})
            recovered = [n for n in doc['kids'] if n.get('source text recovered')]
            self.assertEqual([n['content'] for n in recovered], ['This sentence was omitted at the page top.'])
            self.assertEqual(recovered[0]['page number'], 1)

    def test_heading_bounds_can_include_a_word_from_the_preceding_paragraph(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.pdf'
            fixture_pdf(source, [b'BT /F2 10 Tf 30 530 Td (average) Tj ET '
                                 b'BT /F1 10 Tf 30 500 Td (2. RELATED WORK) Tj ET'])
            node=paragraph('2. RELATED WORK', [30,495,160,540])
            doc=repair_structure(source, {'kids':[node]})
            self.assertEqual(doc['kids'][0]['type'], 'heading')
            self.assertEqual(doc['kids'][0]['content'], '2. RELATED WORK')
            self.assertLess(doc['kids'][0]['bounding box'][3], 520)

    def test_table_below_caption_cannot_be_owned_by_a_later_figure(self):
        table = paragraph('Table 9. Parameters.', [30, 350, 570, 365], type='caption', **{'caption label': 'Table 9'})
        figure = paragraph('Fig. 2. A later figure.', [30, 100, 280, 115], type='caption', **{'caption label': 'Fig. 2'})
        self.assertIs(caption_for_region((30, 40, 570, 230), [table, figure], 600, []), table)
        self.assertEqual(caption_label('TABLE IV. Experimental settings.'), 'Table IV')
        self.assertEqual(caption_label('Table IV shows the results.'), '')

    def test_table_caption_survives_figure_duplicate_removal(self):
        node=paragraph('TABLE 1. Source data.', [30, 400, 280, 415], type='caption')
        span=TextSpan(0, node['content'], (30,400,280,415), False, 8, 402)
        _remove_figure_duplicates([node], [span], [(20,390,290,425)])
        self.assertEqual(node['content'], 'TABLE 1. Source data.')

    def test_legacy_equation_fragments_merge_with_their_number_and_keep_prose(self):
        nodes=[paragraph('![Eq](figs/full.png)',[80,450,220,480],type='formula',**{'formula image':True}),
               paragraph('![Eq](figs/tail.png)',[140,450,280,470],type='formula',**{'formula image':True}),
               paragraph('Ordinary prose must stay separate.',[285,450,450,470])]
        spans=[TextSpan(0,'T(u)=x',(80,450,220,480),True,10,460,'AdvP4C4E74'),
               TextSpan(1,'x(2)',(140,450,280,470),True,10,460,'AdvP4C4E74'),
               TextSpan(2,'Ordinary prose',(285,450,450,470),False,10,460,'Times-Roman')]
        assets=[{'source':'figs/full.png','bounding box':[80,450,220,480],'block':True},
                {'source':'figs/tail.png','bounding box':[140,450,280,470],'block':True}]
        saved=[]
        def save(box,**kwargs):saved.append(box);return '![Combined](figs/combined.png)'
        self.assertEqual(_merge_legacy_equation_snapshots(nodes,spans,assets,save),1)
        self.assertEqual(saved,[(80,450,280,480)])
        self.assertEqual(sum(not n.get('suppressed') for n in nodes[:2]),1)
        self.assertEqual(nodes[2]['content'],'Ordinary prose must stay separate.')

    def test_pseudocode_crop_stops_below_a_full_width_caption(self):
        caption=paragraph('Fig. 2. Pseudocode for lookup.', [340,350,540,365], type='caption', **{'caption label':'Fig. 2'})
        prior=paragraph('Fig. 1. A preceding diagram.', [30,540,570,555], type='caption', **{'caption label':'Fig. 1'})
        code=[paragraph(text, [320, top-12, 520, top]) for text,top in
              [('return result;',400),('if condition then',430),('while condition',460),('// Find a successor',510)]]
        region=_pseudocode_region(caption, [prior,*code,caption], 600, 600)
        self.assertIsNotNone(region)
        self.assertGreaterEqual(region[1], 600-540)
        self.assertEqual(region[2], 600)
        wide=dict(caption, **{'bounding box':[30,350,570,365]})
        self.assertIsNone(_pseudocode_region(wide, [prior,*code,wide], 600, 600))

    def test_margin_rule_touching_a_figure_does_not_capture_another_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.pdf'
            fixture_pdf(source, [b'30 573 m 570 573 l S 70 430 180 140 re S']*3)
            doc={'kids':[paragraph(f'Figure {i}. A diagram.', [70,400,250,415], i) for i in range(1,4)]}
            doc=attach_figures(source, prepare_layout(doc), root)
            self.assertEqual(len(doc['figures']), 3)
            self.assertTrue(all(f['bounding box'][2]<270 for f in doc['figures']))

    def test_abstract_is_bounded_by_its_column_and_not_the_opposite_body(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pdf';fixture_pdf(source, [b''])
            doc = repair_structure(source, {'kids': [
                paragraph('A sufficiently long paper title', [30, 550, 570, 575], **{'font size': 20}),
                paragraph('Abstract', [30, 500, 100, 515], type='heading'),
                paragraph('The abstract paragraph.', [30, 450, 280, 490]),
                paragraph('1 Introduction', [30, 400, 150, 415], type='heading'),
                paragraph('the continuation from the introduction.', [320, 460, 570, 510])]})
            article = build_article(doc, doc['title'], 'paper.pdf', make_chunks(extract_blocks(doc)))
            self.assertEqual([u.source for u in article.abstract], ['The abstract paragraph.'])
            self.assertTrue(any('continuation' in u.source for s in article.sections for u in s.units))

    def test_small_first_page_caption_and_reference_are_not_footnotes(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.pdf';fixture_pdf(source,[b''])
            caption=paragraph('Figure 1. A small caption in the lower body.',[30,100,280,115],**{'font size':8})
            reference=paragraph('[1] A referenced paper in the lower body.',[320,90,570,105],**{'font size':8})
            doc=repair_structure(source,{'kids':[
                paragraph('A sufficiently long paper title',[30,550,570,575],**{'font size':20}),
                paragraph('Abstract. Source abstract text.',[30,450,280,490],**{'font size':10}),
                paragraph('1 Introduction',[30,400,150,415],type='heading',**{'font size':12}),
                paragraph('A normal body paragraph.',[30,300,280,390],**{'font size':10}),caption,reference]})
            self.assertTrue(all(n['type']!='footnote' for n in doc['kids'] if n['content'] in [caption['content'],reference['content']]))


if __name__ == '__main__':
    unittest.main()
