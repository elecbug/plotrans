import unittest
from unittest.mock import Mock

from paper_localizer.article import article_html,build_article,_md_html
from paper_localizer.markup import FIGURE_LINK,image_reference
from paper_localizer.core import Block,_native_table_rows,extract_blocks,make_chunks
from paper_localizer.formulas import TextSpan,_plain_text_group
from paper_localizer.formulas import _merge_overlapping_prose_snapshots
from paper_localizer.formulas import _complete_prose_words
from paper_localizer.formulas import _reconcile_snapshots


class SnapshotExportTests(unittest.TestCase):
    def test_ocr_source_region_is_not_reclassified_as_an_inline_equation(self):
        source = ('<img src="figs/source.png" alt="Uncertain source text" class="ocr-region" '
                  'width="600" height="30" style="width:30em;max-width:100%;height:auto;" />')
        ref = image_reference(FIGURE_LINK.search(source))
        self.assertTrue(ref.region)
        self.assertFalse(ref.inline)
        html = _md_html(source)
        self.assertIn('class="ocr-region"',html)
        self.assertIn('width:30.000em;max-width:100%;height:auto',html)
        self.assertNotIn('inline-math-scroll',html)

    def test_source_region_reading_size_does_not_depend_on_raster_resolution(self):
        template = ('<img src="figs/source.png" class="ocr-region" width="{}" height="{}" '
                    'style="width:30em;max-width:100%;height:auto;" />')
        self.assertEqual(_md_html(template.format(600,30)),_md_html(template.format(1200,60)))

    def test_source_region_export_normalizes_untrusted_html_attributes(self):
        source = ('<img src="figs/source.png" alt="Source" class="ocr-region" width="600" height="30" '
                  'onerror="alert(1)" style="width:30em;position:fixed;" />')
        html = _md_html(source)
        self.assertIn('figs/source.png',html)
        self.assertNotIn('onerror',html)
        self.assertNotIn('position:fixed',html)
        oversized = source.replace('width:30em', 'height:1' + '0' * 100 + 'em')
        self.assertIn('class="ocr-region"',_md_html(oversized))

    def test_contained_inline_crop_retires_without_deleting_outside_prose(self):
        owner={'type':'paragraph','content':'Source prose with notation.','bounding box':[30,100,280,140],
               'formula snapshot':'![Source](figs/whole.png)'}
        outside={'type':'paragraph','content':'<img src="figs/inside.png" alt="x" /> The outside prose remains.',
                 'bounding box':[30,0,280,120]}
        assets=[{'source':'figs/whole.png','bounding box':[30,100,280,140],'block':True},
                {'source':'figs/inside.png','bounding box':[40,110,50,120],'block':False}]
        _reconcile_snapshots([owner,outside],[],assets)
        self.assertNotIn('figs/inside.png',outside['content'])
        self.assertIn('The outside prose remains.',outside['content'])
        self.assertFalse(outside.get('suppressed'))
        self.assertEqual(outside['source inline regions covered by snapshot'][0]['source'],'figs/inside.png')

    def test_prose_snapshot_does_not_cut_a_word_that_duplicate_removal_owns(self):
        box=(30,100,280,140)
        word=TextSpan(0,'connections',(20,98,80,108),False,10,100,'Times-Roman')
        actual=_complete_prose_words(box,[word])
        self.assertEqual(actual,(20,98,280,140))
        self.assertEqual(_complete_prose_words(box,[TextSpan(0,'connections',word.box,True,10,100,'Times-Italic')]),box)
        large=TextSpan(0,'connections',(-200,98,600,108),False,10,100,'Times-Roman')
        self.assertEqual(_complete_prose_words(box,[large]),box)

    def test_body_without_a_printed_heading_does_not_become_author_metadata(self):
        blocks=[Block(1,'frontmatter','Jane Example, Example University.'),
                Block(1,'abstract','This is the explicitly classified abstract.'),
                Block(1,'paragraph','The body begins without an Introduction heading.'),
                Block(1,'frontmatter','Copyright and source information.'),
                Block(2,'heading','## Later section'),
                Block(2,'paragraph','The later body section remains in order.')]
        article=build_article({},'Paper title','source.pdf',make_chunks(blocks))
        self.assertEqual([s.title for s in article.sections],['Full text','Later section'])
        self.assertIn('body begins',article.sections[0].units[0].source)
        self.assertEqual(len(article.front),2)
        self.assertTrue(all('body begins' not in u.source for u in article.front))
        self.assertEqual(len(article.abstract),1)

    def overlapping_paragraphs(self):
        owner = {'type':'paragraph','content':'First source paragraph with notation.',
                 'bounding box':[30,100,280,140],'font size':10,
                 'formula snapshot':'![Source](figs/old.png)'}
        continuation = {'type':'paragraph','content':'<img src="figs/inline.png" alt="x" /> and the outside prose.',
                        'bounding box':[30,80,280,112],'font size':10}
        assets = [{'source':'figs/old.png','bounding box':[29.6,99,280.4,141],'block':True},
                  {'source':'figs/inline.png','bounding box':[35,103,45,110],'block':False}]
        return owner,continuation,assets

    def test_overlapping_inline_crop_does_not_leave_duplicate_or_lose_outside_prose(self):
        owner,continuation,assets = self.overlapping_paragraphs()
        tp = Mock()
        tp.get_text_bounded.return_value = 'The first source paragraph and\r\nthe outside prose remain visible.'
        save = Mock(return_value='![Complete source](figs/merged.png)')
        changed = _merge_overlapping_prose_snapshots([owner,continuation],[],assets,save,tp)
        self.assertEqual(changed,1)
        self.assertTrue(continuation['suppressed'])
        self.assertLessEqual(owner['bounding box'][1],80)
        self.assertIn('outside prose remain visible.',owner['content'])
        self.assertEqual(len(owner['source merged paragraph fragments']),2)
        self.assertEqual([b.text for b in extract_blocks({'kids':[owner,continuation]})],
                         ['![Complete source](figs/merged.png)'])

    def test_source_snapshot_merge_respects_columns_headings_and_table_cells(self):
        for variant in ['column','heading','table','area']:
            with self.subTest(variant=variant):
                owner,continuation,assets = self.overlapping_paragraphs()
                nodes = [owner,continuation]
                if variant=='column':continuation['bounding box']=[320,80,570,112]
                if variant=='heading':nodes.append({'type':'heading','content':'New section','bounding box':[30,85,280,95]})
                if variant=='table':continuation['native table cell']=True
                if variant=='area':continuation['bounding box']=[30,0,280,112]
                save = Mock()
                self.assertEqual(_merge_overlapping_prose_snapshots(nodes,[],assets,save,Mock()),0)
                save.assert_not_called()
                self.assertFalse(continuation.get('suppressed'))

    def test_an_empty_spurious_table_does_not_block_preserving_a_complete_proof(self):
        owner,continuation,assets = self.overlapping_paragraphs()
        empty = {'type':'table','rows':[],'bounding box':[260,84,265,89]}
        tp = Mock()
        tp.get_text_bounded.return_value = 'The complete source proof remains visible.'
        save = Mock(return_value='![Proof](figs/proof.png)')
        self.assertEqual(_merge_overlapping_prose_snapshots([owner,continuation,empty],[],assets,save,tp),1)
        self.assertTrue(continuation['suppressed'])

    def test_source_information_and_notes_use_their_preserved_source_snapshot(self):
        front = '![Source authors and both affiliations](figs/source/front.png)'
        note = '![Complete source note](figs/source/note.png)'
        doc = {'kids':[
            {'type':'frontmatter','page number':1,'content':'Incomplete affiliation','formula snapshot':front},
            {'type':'footnote','page number':2,'content':'Incomplete note','formula snapshot':note},
        ]}
        blocks = extract_blocks(doc)
        self.assertIn(front,[b.text for b in blocks])
        self.assertIn(note,[b.text for b in blocks])
        html = article_html(build_article(doc,'Title','source.pdf',make_chunks(blocks)),'source.md')
        self.assertIn('figs/source/front.png',html)
        self.assertIn('figs/source/note.png',html)
        self.assertNotIn('Incomplete affiliation',html)
        self.assertNotIn('Incomplete note',html)

    def test_a_table_cell_uses_its_snapshot_instead_of_incomplete_child_text(self):
        source = '![Source fraction](figs/source/fraction.png)'
        table = {'rows':[{'cells':[{'kids':[
            {'content':'12','formula snapshot':source,'kids':[{'content':'duplicate'}]},
            {'content':'suppressed duplicate','suppressed':True},
        ]}]}]}
        self.assertEqual(_native_table_rows(table)[0][0]['text'],source)

    def test_source_matching_numeric_arrow_cell_stays_editable(self):
        spans = [TextSpan(0,'3.659',(30,100,50,108),True,9,100,'Roman'),
                 TextSpan(5,'→',(52,100,60,108),True,9.3,100,'CMSY10'),
                 TextSpan(6,'2.301',(62,100,82,108),True,9,100,'Roman')]
        cell = {'content':'3.659→2.301','native table cell':True}
        self.assertTrue(_plain_text_group(spans,cell))
        self.assertFalse(_plain_text_group(spans,dict(cell,content='3.659→2.300')))
        from dataclasses import replace
        raised = spans[:-1]+[replace(spans[-1],baseline=104,font_size=6)]
        self.assertFalse(_plain_text_group(raised,cell))


if __name__ == '__main__':
    unittest.main()
