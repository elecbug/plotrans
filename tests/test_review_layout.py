from pathlib import Path
import ctypes
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from paper_localizer.article import build_article
from paper_localizer.core import extract_blocks,make_chunks
from paper_localizer.core import _table_markdown,_native_table_rows
from paper_localizer.tables import table_grid
from paper_localizer.formulas import TextSpan,recover_formulas,_spans,_insertions
from paper_localizer.structure import repair_structure,_split_titles
from test_structure import fixture_pdf,paragraph


class ReviewLayoutTests(unittest.TestCase):
    def test_palatino_ital_font_marks_single_letter_variables(self):
        textpage=Mock();textpage.count_chars.return_value=1
        textpage.get_charbox.return_value=(30,400,38,410)
        obj=textpage.get_textobj.return_value;obj.raw=ctypes.c_void_p(1)
        obj.get_font.return_value.get_base_name.return_value='URWPalladioL-Ital'
        obj.get_font_size.return_value=10;obj.get_matrix.return_value=SimpleNamespace(c=0,d=1,f=400)
        page=Mock();page.get_textpage.return_value=textpage
        with patch('paper_localizer.formulas.pdfium.raw.FPDFText_GetUnicode',return_value=ord('D')):
            self.assertTrue(_spans(page)[0].math)

    def test_repeated_variables_align_with_native_label_and_positioned_accents(self):
        intro='The document preserves the complete ordinary source sentence. '*6
        content=intro+'(Dlow, D, Dhigh) in the Erd˝os model.'
        words=['Abstract',intro,'(Dlow,','D','Dhigh',') in the Erdos˝ model.']
        spans=[TextSpan(i,w,(30+i*20,400,45+i*20,410),i in [2,3,4],10,400,'Times-Roman') for i,w in enumerate(words)]
        starts=[content.index('(Dlow,'),content.index(', D,')+2,content.index('Dhigh')]
        expected=[(start,start+len(word)) for start,word in zip(starts,['(Dlow,','D','Dhigh'])]
        self.assertEqual(_insertions(content,spans,[[spans[2]],[spans[3]],[spans[4]]]),expected)

    def numbered_pdf(self,root,font=4,values=range(1,9)):
        p=root/'review.pdf';ops=[];text=[]
        for i,value in enumerate(values):
            y=540-i*25
            body=f'The measured value {i+1} and year 2026 remain valid.'
            ops.append(f'BT /F2 10 Tf 120 {y} Td ({body}) Tj ET BT /F2 {font} Tf 580 {y} Td ({value}) Tj ET'.encode())
            text.append(body+' '+str(value))
        fixture_pdf(p,[b' '.join(ops)])
        return p,paragraph(' '.join(text),[119,360,590,555],font='Times-Roman',**{'font size':10})

    def test_margin_counters_leave_equal_body_values_and_years_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            p,n=self.numbered_pdf(Path(directory));d=repair_structure(p,{'kids':[n]})
            visible=' '.join(b.text for b in extract_blocks(d))
            self.assertEqual(len(d['review line numbers']),8)
            self.assertEqual(visible.count('2026'),8)
            for number in range(1,9):self.assertIn(f'value {number} and year',visible)
            self.assertNotIn('valid. 1',visible)
            self.assertLess(d['kids'][0]['bounding box'][2],580)

    def test_equal_size_numeric_data_and_nonsequential_labels_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            for font,values in [(10,range(1,9)),(4,range(1,17,2))]:
                p,n=self.numbered_pdf(Path(directory),font,values);original=n['content']
                d=repair_structure(p,{'kids':[n]})
                self.assertFalse(d.get('review line numbers'))
                self.assertEqual(d['kids'][0]['content'],original)

    def test_code_line_numbers_and_restarted_sequences_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p,n=self.numbered_pdf(root)
            operations=[]
            for i in range(8):
                y=540-i*25
                operations.append(f'BT /F4 10 Tf 120 {y} Td (The measured value {i+1} and year 2026 remain valid.) Tj ET BT /F2 4 Tf 580 {y} Td ({i+1}) Tj ET'.encode())
            fixture_pdf(p,[b' '.join(operations)])
            d=repair_structure(p,{'kids':[n]})
            self.assertFalse(d.get('review line numbers'))
            p,n=self.numbered_pdf(root,values=[1,2,3,4,1,2,3,4])
            d=repair_structure(p,{'kids':[n]})
            self.assertFalse(d.get('review line numbers'))

    def test_regular_hierarchical_heading_with_positioned_accent(self):
        def s(i,text,x):return TextSpan(i,text,(x,400,x+25,410),False,10,400,'Times-Roman')
        spans=[s(0,'5.1.1.',30),s(1,'Erd',65),s(2,'os',95),s(3,'˝',100),s(4,'Graph',130)]
        n=paragraph('5.1.1. Erd˝os Graph',[30,398,180,412],pdfua_tag='LI')
        result=_split_titles(n,spans)
        self.assertEqual(result[0]['type'],'heading')
        self.assertEqual(result[0]['heading level'],4)
        self.assertEqual(result[0]['content'],'5.1.1. Erd˝os Graph')

    def test_abstract_and_keywords_have_separate_source_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'abstract.pdf'
            fixture_pdf(p,[b'BT /F3 18 Tf 30 560 Td (A Research Paper About Source Geometry) Tj ET '
                b'BT /F3 10 Tf 120 500 Td (Abstract) Tj ET '
                b'BT /F2 10 Tf 120 480 Td (The complete abstract preserves the original source.) Tj ET '
                b'BT /F3 10 Tf 120 450 Td (Keywords: Parsing; Geometry) Tj ET '
                b'BT /F3 12 Tf 120 400 Td (1. Introduction) Tj ET'])
            d=repair_structure(p,{'title':'A Research Paper About Source Geometry','kids':[
                paragraph('A Research Paper About Source Geometry',[30,555,570,580],type='heading',**{'font size':18}),
                paragraph('Abstract The complete abstract preserves the original source. Keywords: Parsing; Geometry',
                          [119,447,570,512],**{'font size':10}),
                paragraph('1. Introduction',[120,397,330,415],type='heading',**{'font size':12})]})
            a=build_article(d,d['title'],'source.pdf',make_chunks(extract_blocks(d)))
            self.assertEqual([u.source for u in a.abstract],['The complete abstract preserves the original source.'])
            self.assertEqual([u.source for u in a.keywords],['Parsing; Geometry'])

    def test_fraction_and_sum_fragments_form_one_complete_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'equation.pdf'
            fixture_pdf(p,[b'BT /F2 10 Tf 30 200 Td (This is an explanatory paragraph with enough ordinary source words to identify the main body column.) Tj ET '
                b'BT /F2 10 Tf 250 450 Td (ASPL =) Tj ET '
                b'BT /F1 10 Tf 310 458 Td (N) Tj ET '
                b'BT /F1 10 Tf 306 439 Td (N) Tj ET '
                b'BT /F1 10 Tf 345 450 Td (d(i,j)) Tj ET '
                b'306 452 m 330 452 l S'])
            d={'kids':[paragraph('This is an explanatory paragraph with enough ordinary source words to identify the main body column.',[30,195,570,212]),
                       paragraph('ASPL =',[250,447,290,460]),paragraph('N',[310,456,320,469]),
                       paragraph('N',[306,436,320,450]),paragraph('d(i,j)',[345,447,375,460])]}
            recover_formulas(p,d,root)
            displays=[n for n in d['kids'] if n.get('source display equation')]
            self.assertEqual(len(displays),1)
            self.assertEqual(displays[0]['display fragments'],4)
            self.assertTrue(displays[0]['formula image'])
            self.assertEqual(len([f for f in d['formulas'] if f['block']]),1)
            visible=extract_blocks(d)
            self.assertEqual(sum(b.kind=='formula' for b in visible),1)
            self.assertTrue(any('explanatory paragraph' in b.text for b in visible))

    def test_page_header_rules_do_not_join_a_ruled_table(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'table.pdf'
            fixture_pdf(p,[b'30 580 m 570 580 l S 30 578 m 570 578 l S '
                b'30 342 m 570 342 l S 30 320 m 570 320 l S 30 280 m 570 280 l S '
                b'BT /F3 10 Tf 40 330 Td (Method) Tj ET BT /F3 10 Tf 300 330 Td (Score) Tj ET '
                b'BT /F2 10 Tf 40 307 Td (Peri) Tj ET BT /F2 10 Tf 300 307 Td (0.67) Tj ET '
                b'BT /F2 10 Tf 40 294 Td (Baseline) Tj ET BT /F2 10 Tf 300 294 Td (0.08) Tj ET'])
            d=repair_structure(p,{'kids':[paragraph('Table 1. Source results.',[30,350,570,365]),
                paragraph('Method Score',[40,326,350,340]),paragraph('Peri 0.67',[40,303,350,318]),
                paragraph('Baseline 0.08',[40,290,350,302])]})
            tables=[n for n in d['kids'] if n.get('source reconstructed table')]
            self.assertEqual(len(tables),1)
            self.assertIn('| Peri | 0.67 |',_table_markdown(tables[0]))
            self.assertLess(tables[0]['bounding box'][3],350)

    def test_display_recovery_cannot_join_touching_table_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'cell-boundaries.pdf'
            body='The paragraph identifies the body column and keeps ordinary source words editable for the reader.'
            fixture_pdf(p,[b'BT /F3 10 Tf 40 330 Td (Method) Tj ET BT /F3 10 Tf 150 330 Td (Successful Victim) Tj ET '
                b'BT /F3 10 Tf 150 318 Td (Connections) Tj ET '
                b'BT /F2 10 Tf 40 295 Td (Peri) Tj ET BT /F1 10 Tf 150 295 Td (x + y = z) Tj ET '
                +f'BT /F2 10 Tf 30 150 Td ({body}) Tj ET'.encode()])
            def cell(text,r,c,box,span=1):
                return {'type':'table cell','page number':1,'row number':r,'column number':c,'row span':span,
                        'column span':1,'bounding box':box,'is_header':r<3,
                        'kids':[paragraph(text,box,font='Times-Roman',**{'font size':10,'native table cell':True})]}
            table={'type':'table','page number':1,'bounding box':[30,280,280,342], 'rows':[
                {'row number':1,'cells':[cell('Method',1,1,[30,309,140,342],2),cell('Successful Victim',1,2,[140,326,280,342])]},
                {'row number':2,'cells':[cell('Connections',2,2,[140,309,280,326])]},
                {'row number':3,'cells':[cell('Peri',3,1,[30,280,140,309]),cell('x + y = z',3,2,[140,280,280,309])]}]}
            d={'kids':[table,paragraph(body,[30,145,570,163])]};recover_formulas(p,d,root)
            grid=table_grid(_native_table_rows(table))
            self.assertEqual(grid[0],('Method','Successful Victim'))
            self.assertEqual(grid[1],('Method','Connections'))
            self.assertFalse(d.get('completed display equations'))

    def test_display_definition_keeps_words_and_native_line_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'definition.pdf'
            body='The paragraph identifies the body column and keeps ordinary source words editable for the reader.'
            fixture_pdf(p,[b'BT /F1 10 Tf 150 400 Td (N) Tj ET '
                b'BT /F2 10 Tf 160 400 Td (= total number of nodes in the network) Tj ET '
                +f'BT /F2 10 Tf 30 150 Td ({body}) Tj ET'.encode()])
            original='N = total number of nodes in the network'
            node=paragraph(original,[149,397,400,412],font='Times-Roman',**{'font size':10})
            d={'kids':[node,paragraph(body,[30,145,570,163])]};recover_formulas(p,d,root)
            self.assertEqual(node['content'],original)
            self.assertTrue(node.get('source display definition'))
            self.assertIn('![Eq.',node['formula snapshot'])

    def test_first_page_footer_matches_headers_omitted_by_layout_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'headers.pdf';label=b'Version November 1, 2026 submitted to Test Journal.'
            fixture_pdf(p,[b'BT /F2 8 Tf 30 30 Td ('+label+b') Tj ET',
                           b'BT /F2 8 Tf 30 570 Td ('+label+b') Tj ET',
                           b'BT /F2 8 Tf 30 570 Td ('+label+b') Tj ET'])
            d=repair_structure(p,{'kids':[paragraph(label.decode(),[30,27,300,40],**{'font size':8})]})
            margins=[n for n in d['kids'] if n.get('content','').startswith('Version')]
            self.assertEqual([n['type'] for n in margins],['footer','header','header'])
            self.assertEqual(extract_blocks(d),[])

    def test_publisher_notice_is_retained_outside_numbered_references(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'notice.pdf';notice="Disclaimer/Publisher's Note: Claims belong to the authors and the source text is retained."
            fixture_pdf(p,[b'BT /F3 12 Tf 30 500 Td (References) Tj ET '
                           b'BT /F2 10 Tf 30 470 Td (1. A source reference, pp. 12-23.) Tj ET '
                           +f'BT /F2 10 Tf 30 400 Td ({notice}) Tj ET'.encode()])
            d=repair_structure(p,{'kids':[paragraph('References',[30,497,300,515],type='heading'),
                paragraph('1. A source reference, pp. 12-23.',[30,467,400,482]),
                paragraph(notice,[30,397,570,412])]})
            a=build_article(d,'Paper','source.pdf',make_chunks(extract_blocks(d)))
            self.assertTrue(any(notice in u.source for u in a.front))
            self.assertFalse(any(notice in u.source for s in a.sections for u in s.units))


if __name__=='__main__':unittest.main()
