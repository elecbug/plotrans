from pathlib import Path
import tempfile
import unittest

from paper_localizer.formulas import (TextSpan, _cell_owners, _groups,
                                      _remove_figure_duplicates, recover_formulas)
from paper_localizer.figures import attach_figures
from paper_localizer.layout import bind_captions, prepare_layout
from paper_localizer.core import extract_blocks
from paper_localizer.core import make_chunks
from paper_localizer.structure import _mark_page_furniture, order_document
from paper_localizer.article import build_article
from test_structure import fixture_pdf, paragraph


class FloatBoundaryTests(unittest.TestCase):
    def test_short_repeated_conference_header_does_not_interrupt_body(self):
        header='DEBS ’26, June 23–26, 2026, Lisbon, Portugal'
        doc={'page sizes':{str(n):[612,792] for n in range(1,5)},'kids':[
            paragraph(header,[431,723,558,730],page=n,**{'source text recovered':True})
            for n in range(1,5)]}
        doc['kids'].append(paragraph('A repeated description inside the body.',[54,600,296,620]))
        _mark_page_furniture(doc)
        self.assertTrue(all(n['type']=='header' for n in doc['kids'][:4]))
        self.assertEqual([b.text for b in extract_blocks(doc)],['A repeated description inside the body.'])

    def test_captioned_native_table_does_not_break_a_column_continuation(self):
        doc={'page sizes':{'1':[600,600]},'kids':[
            paragraph('A separate opening paragraph.',[30,300,280,380]),
            paragraph('The configuration improves all three',[30,50,280,100]),
            {'type':'table','page number':1,'caption label':'Table 1',
             'bounding box':[320,400,570,530],
             'rows':[{'cells':[{'kids':[{'content':'Metric'}]},{'kids':[{'content':'Value'}]}]},
                     {'cells':[{'kids':[{'content':'Delivery'}]},{'kids':[{'content':'99.9%'}]}]}]},
            paragraph('objectives simultaneously under churn.',[320,250,570,380])]}
        doc=order_document(doc)
        article=build_article(doc,'Paper','source.pdf',make_chunks(extract_blocks(doc)))
        texts=[u.source for u in article.front if u.kind=='text']
        self.assertIn('The configuration improves all three objectives simultaneously under churn.',texts)
        self.assertEqual(len([b for b in extract_blocks(doc) if b.kind=='table']),1)

    def test_centered_parenthesized_code_table_preserves_the_listing_as_one_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'code-table.pdf'
            lines=['(1) if (ready) {','(2) return result;','(3) } else {','(4) return failure; }']
            content=b''.join(('BT /F4 9 Tf 215 '+str(315-i*15)+' Td ('+text.replace('(',r'\(').replace(')',r'\)')+') Tj ET ').encode() for i,text in enumerate(lines))
            fixture_pdf(source,[content+b'BT /F2 10 Tf 200 245 Td (Table 1. Pseudo code for routing.) Tj ET'])
            doc={'kids':[
                paragraph('The value of the digit in the key is explained here.',[130,350,470,375]),
                *[paragraph(text,[215,310-i*15,395,325-i*15]) for i,text in enumerate(lines)],
                {'type':'caption','page number':1,'content':'Table 1. Pseudo code for routing.',
                 'bounding box':[200,240,400,255]},
                paragraph('The explanation resumes after the algorithm.',[130,210,470,225])]}
            result=attach_figures(source,prepare_layout(doc),root)
            self.assertEqual(len(result['figures']),1)
            asset=result['figures'][0]
            self.assertEqual(asset['caption label'],'Table 1')
            self.assertTrue(asset['source listing'])
            self.assertLess(asset['bounding box'][0],215)
            self.assertGreater(asset['bounding box'][2],395)
            self.assertGreater(asset['bounding box'][1],255)
            self.assertLess(asset['bounding box'][3],350)
            texts=[b.text for b in extract_blocks(result)]
            self.assertIn('The value of the digit in the key is explained here.',texts)
            self.assertIn('The explanation resumes after the algorithm.',texts)
            self.assertFalse(any(text in texts for text in lines))

    def test_detached_plot_panel_stays_with_the_neighboring_captioned_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'panels.pdf'
            fixture_pdf(source,[b'40 350 100 80 re S 160 350 100 80 re S '
                                b'BT /F2 10 Tf 145 360 Td (Shared legend) Tj ET '
                                b'BT /F2 10 Tf 180 315 Td (Figure 1. Panel comparison.) Tj ET'])
            doc={'kids':[{'type':'caption','page number':1,'content':'Figure 1. Panel comparison.',
                          'bounding box':[180,312,240,326]}]}
            result=attach_figures(source,prepare_layout(doc),root)
            self.assertEqual(len(result['figures']),1)
            asset=result['figures'][0]
            self.assertEqual(asset['caption label'],'Fig. 1')
            self.assertLess(asset['bounding box'][0],40)
            self.assertGreater(asset['bounding box'][2],260)

    def test_adjacent_tables_keep_separate_captions_and_surrounding_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'adjacent.pdf'
            fixture_pdf(source, [
                b'BT /F2 10 Tf 50 544 Td (The discussion before both tables.) Tj ET '
                b'60 460 240 60 re S 60 388 240 60 re S '
                b'BT /F2 10 Tf 80 490 Td (Alpha 147 194) Tj ET '
                b'BT /F2 6 Tf 60 452 Td (Table 1. First measurements.) Tj ET '
                b'BT /F2 10 Tf 80 418 Td (Beta 22.21 -4.5) Tj ET '
                b'BT /F2 6 Tf 60 378 Td (Table 2. Second measurements.) Tj ET '
                b'BT /F2 10 Tf 50 350 Td (The explanation after both tables.) Tj ET'])
            doc={'kids':[
                paragraph('The discussion before both tables.',[50,540,300,556]),
                {'type':'image','page number':1,'bounding box':[60,388,300,520]},
                {'type':'caption','page number':1,'content':'Table 1. First measurements.',
                 'bounding box':[60,450,300,459]},
                {'type':'caption','page number':1,'content':'Table 2. Second measurements.',
                 'bounding box':[60,376,300,387]},
                paragraph('The explanation after both tables.',[50,346,300,362])]}
            result=bind_captions(attach_figures(source,prepare_layout(doc),root))
            assets=result['figures']
            self.assertEqual([a['caption label'] for a in assets],['Table 1','Table 2'])
            self.assertTrue(all(a['asset kind']=='table' for a in assets))
            self.assertGreater(assets[0]['bounding box'][1],assets[0]['caption node']['bounding box'][3])
            self.assertLess(assets[1]['bounding box'][3],450)
            self.assertGreater(assets[1]['bounding box'][1],assets[1]['caption node']['bounding box'][3])
            self.assertTrue(all(a['caption node']['attached caption'] for a in assets))
            texts=[b.text for b in extract_blocks(result)]
            self.assertIn('The discussion before both tables.',texts)
            self.assertIn('The explanation after both tables.',texts)

    def test_table_caption_overlap_is_trimmed_before_caption_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'caption-overlap.pdf'
            fixture_pdf(source,[b'80 300 180 120 re S '
                                b'BT /F2 10 Tf 60 285 Td (Table 2. Measurements.) Tj ET'])
            doc={'kids':[
                {'type':'image','page number':1,'bounding box':[80,294,260,420]},
                {'type':'caption','page number':1,'content':'Table 2. Measurements.',
                 'bounding box':[60,278,295,300]}]}
            result=attach_figures(source,prepare_layout(doc),root)
            self.assertEqual(len(result['figures']),1)
            self.assertEqual(result['figures'][0]['caption label'],'Table 2')
            self.assertGreater(result['figures'][0]['bounding box'][1],doc['kids'][-1]['bounding box'][3])
            self.assertLess(result['figures'][0]['bounding box'][1],300)

    def test_adjacent_table_multipliers_do_not_move_outside_their_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root/'cells.pdf'
            fixture_pdf(source,[b'BT /F2 10 Tf 40 500 Td (8.6) Tj /F5 10 Tf (\264) Tj ET '
                                b'BT /F2 10 Tf 78 500 Td (1.1) Tj /F5 10 Tf (\264) Tj ET '
                                b'BT /F2 10 Tf 30 460 Td (The following explanation stays in the body.) Tj ET'])
            def cell(text,box,column):
                return {'type':'table cell','page number':1,'column number':column,
                        'row number':1,'bounding box':box,
                        'kids':[paragraph(text,box,font='Times-Roman',**{'font size':10})]}
            doc={'kids':[{'type':'table','page number':1,'bounding box':[30,490,115,515],
                          'rows':[{'type':'table row','cells':[
                              cell('8.6×',[30,490,70,515],1),cell('1.1×',[70,490,115,515],2)]}]},
                         paragraph('The following explanation stays in the body.',[30,450,350,475])]}
            recover_formulas(source,doc,root)
            texts=[c['kids'][0]['content'] for c in doc['kids'][0]['rows'][0]['cells']]
            self.assertEqual(texts,['8.6×','1.1×'])
            self.assertFalse(doc['formulas'])
            self.assertEqual(doc['kids'][1]['content'],'The following explanation stays in the body.')

    def test_math_group_overlap_cannot_rejoin_different_cell_owners(self):
        spans=[TextSpan(1,'x',(45,100,55,110),True,10,102,'Math'),
               TextSpan(2,'y',(53,100,63,110),True,10,102,'Math')]
        cells=[(30,95,54,115),(54,95,80,115)]
        owners=_cell_owners(spans,cells)
        self.assertEqual([len(g) for g in _groups(spans,owners)],[1,1])
        self.assertEqual(len(_groups(spans)),1)

    def test_a_display_image_spanning_cells_cannot_erase_a_cell(self):
        spans=[TextSpan(1,'8.6×',(40,100,60,110),True,10,102,'Math')]
        node=paragraph('8.6×',[30,95,70,115],**{'native table cell':True})
        self.assertEqual(_remove_figure_duplicates([node],spans,[(35,98,100,112)],False),0)
        self.assertEqual(node['content'],'8.6×')
        self.assertEqual(_remove_figure_duplicates([node],spans,[(35,98,65,112)],False),1)

    def test_duplicate_cell_label_uses_the_physical_cell_instead_of_tight_text_bounds(self):
        spans=[TextSpan(1,'t1',(40,100,48,110),True,10,102,'Math')]
        node=paragraph('t1',[40,100,48,110],**{'native table cell':True,
                       'native table cell box':[30,90,80,120]})
        self.assertEqual(_remove_figure_duplicates([node],spans,[(34,96,62,115)],False),1)
        self.assertEqual(node['content'],'')


if __name__=='__main__':unittest.main()
