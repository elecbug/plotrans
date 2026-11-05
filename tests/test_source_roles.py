from pathlib import Path
import ctypes
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper_localizer.core import _table_markdown, extract_blocks
from paper_localizer.formulas import TextSpan, _spans, _insertions, recover_formulas
from paper_localizer.figures import attach_figures
from paper_localizer.layout import prepare_layout
from paper_localizer.source_roles import recover_authors, split_captions
from paper_localizer.structure import repair_structure
from test_structure import fixture_pdf, paragraph


def span(index,text,box,size=12,baseline=500,font='Times-Roman'):
    return TextSpan(index,text,tuple(box),False,size,baseline,font)


class SourceRoleTests(unittest.TestCase):
    def test_repeated_native_variable_does_not_remain_beside_its_image(self):
        variable='𝑑𝑖𝑛';content=f'uncapped {variable}, when {variable} is capped.'
        spans=[span(0,'uncapped',[10,100,45,110]),
               TextSpan(1,variable,(50,100,65,110),True,10,100,'Math'),
               span(2,', when',[66,100,100,110]),
               TextSpan(3,variable,(105,100,120,110),True,10,100,'Math'),
               span(4,'is capped.',[125,100,165,110])]
        first=content.index(variable);second=content.index(variable,first+1)
        self.assertEqual(_insertions(content,spans,[[spans[1]],[spans[3]]]),
                         [(first,first+len(variable)),(second,second+len(variable))])

    def test_pdfium_surrogate_pair_is_one_mathematical_glyph(self):
        textpage=Mock();textpage.count_chars.return_value=2
        textpage.get_charbox.return_value=(10,98,17,110)
        obj=textpage.get_textobj.return_value;obj.raw=ctypes.c_void_p(1)
        obj.get_font.return_value.get_base_name.return_value='LibertinusMath-Regular'
        obj.get_font_size.return_value=10
        obj.get_matrix.return_value=SimpleNamespace(c=0,d=1,f=100)
        page=Mock();page.get_textpage.return_value=textpage
        with patch('paper_localizer.formulas.pdfium.raw.FPDFText_GetUnicode',side_effect=lambda _,i:[0xD835,0xDC51][i]):
            spans=_spans(page)
        self.assertEqual([s.text for s in spans],['𝑑'])
        self.assertEqual(spans[0].index,0)
        self.assertTrue(spans[0].math)

    def document(self,author='Alice Example'):
        return {'author':author,'title bounds':[30,550,570,580],
                'kids':[paragraph('Source abstract.',[30,350,570,400],type='abstract')]}

    def test_adds_printed_coauthors_without_affiliation_markers(self):
        document=self.document()
        glyphs=[span(0,'Alice',[100,499,125,510]),span(1,'Example',[130,499,180,510]),
                span(2,'a',[185,505,190,513],8,505),
                span(3,'Priya',[320,499,350,510]),span(4,'Test',[355,499,380,510]),
                span(5,'University',[100,480,180,490],10,480)]
        recover_authors(document,glyphs)
        self.assertEqual(document['authors'],['Alice Example','Priya Test'])
        self.assertEqual(document['author recovery']['original'],'Alice Example')

    def test_institution_and_unrelated_names_cannot_augment_metadata(self):
        document=self.document()
        glyphs=[span(0,'Bob Jones',[100,499,180,510]),span(1,'ETH Zurich',[320,499,390,510])]
        recover_authors(document,glyphs)
        self.assertEqual(document['author'],'Alice Example')
        document=self.document()
        glyphs[0]=span(0,'Alice Example',[100,499,180,510])
        recover_authors(document,glyphs)
        self.assertEqual(document['author'],'Alice Example')

    def test_separates_two_names_with_a_source_em_space(self):
        document=self.document('')
        glyphs=[span(0,'Sorav',[238,499,266,510]),span(1,'Bansal',[270,499,304,510]),
                span(2,'Mary',[324,499,351,510]),span(3,'Baker',[355,499,384,510])]
        recover_authors(document,glyphs)
        self.assertEqual(document['authors'],['Sorav Bansal','Mary Baker'])

    def test_splits_only_native_confirmed_captions_in_different_columns(self):
        text='Table 2. First table. Fig. 2. A separate plot.'
        node=paragraph(text,[30,390,570,420],type='caption')
        glyphs=[span(0,'Table 2. First table.',[30,400,200,420]),span(1,'Fig. 2. A separate plot.',[320,390,570,410])]
        parts=split_captions(node,glyphs,600)
        self.assertEqual([n['content'] for n in parts],['Table 2. First table.','Fig. 2. A separate plot.'])
        glyphs[1]=span(1,'Fig. 2. A separate plot.',[50,370,270,390])
        self.assertEqual(split_captions(node,glyphs,600),[node])

    def test_overlapping_caption_continuation_keeps_its_full_sentence(self):
        first=paragraph('Fig. 9. These plots are for',[30,405,280,435],font='Times-Bold',**{'font size':8})
        second=paragraph('uncapped values and complete convergence.',[30,390,280,420],type='caption',font='Times-Roman',**{'font size':8})
        result=prepare_layout({'kids':[first,second]})
        self.assertIn('complete convergence.',result['captions'][0]['content'])
        first=paragraph('Fig. 9. A complete caption.',[30,405,280,435],font='Times-Bold',**{'font size':8})
        second=paragraph('An unrelated body paragraph.',[30,390,280,420],font='Times-Roman',**{'font size':8})
        result=prepare_layout({'kids':[first,second]})
        self.assertEqual(result['captions'][0]['content'],'Fig. 9. A complete caption.')

    def test_consumed_caption_cannot_clip_the_owner_caption_out_of_figure_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'caption.pdf'
            fixture_pdf(source,[b'80 300 170 180 re S '
                                 b'BT /F2 8 Tf 60 286 Td (Fig. 1. A caption.) Tj ET '
                                 b'BT /F2 8 Tf 220 286 Td (All three) Tj ET'])
            caption=paragraph('Fig. 1. A caption. All three',[60,278,300,296],type='caption',**{'caption label':'Fig. 1','font size':8})
            consumed=paragraph('',[60,276,299.999,296],type='caption')
            result=attach_figures(source,{'kids':[caption,consumed]},root)
            figure=result['figures'][0]
            self.assertEqual(figure['caption label'],'Fig. 1')
            self.assertGreater(figure['bounding box'][1],caption['bounding box'][3])

    def ruled_table(self,root,wrong_value=False):
        source=root/'table.pdf'
        fixture_pdf(source,[b'30 342 m 280 342 l S 30 309 m 280 309 l S 30 274 m 280 274 l S '
            b'BT /F3 10 Tf 40 330 Td (Method) Tj ET BT /F3 10 Tf 150 330 Td (Successful Victim) Tj ET '
            b'BT /F3 10 Tf 150 318 Td (Connections) Tj ET '
            b'BT /F2 10 Tf 40 295 Td (Peri) Tj ET BT /F2 10 Tf 150 295 Td (47/70 = 67.1%) Tj ET '
            b'BT /F2 10 Tf 40 283 Td (Baseline) Tj ET BT /F2 10 Tf 150 283 Td (6/70 = 8.6%) Tj ET'])
        nodes=[paragraph('Table 2. Rate of connections.',[30,350,280,365],type='caption'),
               paragraph('Method Successful Victim',[40,326,260,340],type='heading'),
               paragraph('Connections Peri 47/70 = '+('66%' if wrong_value else '67.1%'),[40,291,260,325]),
               paragraph('Baseline 6/70 = 8.6%',[40,279,260,290])]
        return repair_structure(source,{'kids':nodes})

    def test_source_table_caption_cannot_be_reused_by_the_following_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'neighbor.pdf'
            fixture_pdf(source,[b'150 550 100 30 re S 50 350 350 150 re S '
                                b'BT /F2 10 Tf 160 560 Td (1.0 0.5) Tj ET'])
            table=paragraph('Table 4. Pairwise values.',[30,520,450,540],type='caption',
                            **{'caption label':'Table 4','font size':10})
            plot=paragraph('Fig. 8. A neighboring plot.',[30,310,450,330],type='caption',
                           **{'caption label':'Fig. 8','font size':10})
            result=attach_figures(source,{'kids':[table,plot],
                'source table regions':[{'page number':1,'bounding box':[149,549,251,581],
                                         'caption label':'Table 4'}]},root)
            figures={f.get('caption label'):f for f in result['figures']}
            self.assertEqual(set(figures),{'Table 4','Fig. 8'})
            self.assertGreater(figures['Table 4']['bounding box'][1],540)
            self.assertLess(figures['Fig. 8']['bounding box'][3],520)

    def test_numbered_text_only_definition_is_one_source_figure(self):
        from paper_localizer.figures import _pseudocode_region
        caption=paragraph('Fig. 4. Property 1 definition in ACL2s.',[60,300,270,310],type='caption',
                          **{'caption label':'Fig. 4'})
        lines=['1 (property (p :peer)', '2 :hyps (valid p)', '3 (> (score p)', '4 0))']
        nodes=[paragraph(text,[40,330+i*12,280,340+i*12]) for i,text in enumerate(reversed(lines))]
        box=_pseudocode_region(caption,nodes+[caption],600,800)
        self.assertIsNotNone(box)
        self.assertEqual(box,(0,422,300.0,472))
        for i,n in enumerate(nodes):n['content']=f'{len(nodes)-i} Plain prose item'
        self.assertIsNone(_pseudocode_region(caption,nodes+[caption],600,800))

    def test_ruled_table_keeps_wrapped_headers_and_numeric_cell_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            document=self.ruled_table(Path(directory))
            table=next(n for n in document['kids'] if n.get('source reconstructed table'))
            markdown=_table_markdown(table)
            self.assertIn('| Method | Successful Victim / Connections |',markdown)
            self.assertIn('| Peri | 47/70 = 67.1% |',markdown)
            self.assertNotIn('Method Successful Victim',[b.text.lstrip('# ') for b in extract_blocks(document) if b.kind=='heading'])
            self.assertEqual(table['rows'][0]['cells'][0]['row span'],2)
            recover_formulas(Path(directory)/'table.pdf',document,Path(directory))
            self.assertIn('| Peri | 47/70 = 67.1% |',_table_markdown(table))

    def test_disagreeing_table_text_uses_source_region_instead_of_invented_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            document=self.ruled_table(Path(directory),wrong_value=True)
            self.assertFalse(any(n.get('source reconstructed table') for n in document['kids']))
            self.assertEqual(document['source table regions'][0]['caption label'],'Table 2')


if __name__=='__main__':unittest.main()
