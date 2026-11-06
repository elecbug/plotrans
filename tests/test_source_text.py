from dataclasses import replace
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper_localizer.formulas import TextSpan
from paper_localizer.source_text import (
    _span_indices, normalize_accents, normalized_spans, positioned_accents,
    _can_join, normalize_line_wraps,
    restore_local_spacing,
    restore_prose_dashes,
    geometric_word_spaces, normalize_geometric_spaces,
    _known_inflection, _separate_words,
    restore_prose_omissions,
)
from paper_localizer.structure import repair_structure
from test_structure import fixture_pdf, paragraph


class SourceTextTests(unittest.TestCase):
    def test_small_prose_omission_is_restored_between_exact_native_anchors(self):
        source = ('2 In reality, information of new nodes in the network needs to propagate and, '
                  'though this generally happens quickly, it is possible that propagation of '
                  'information on new nodes impacts how nodes cluster in the network.')
        target = source.replace('information of new nodes','of',1)
        tp = Mock();tp.get_text_bounded.return_value = source
        spans = [TextSpan(0,'In reality',(30,100,70,107),False,10,100,'Times-Roman')]
        node = {'type':'paragraph','content':target,'bounding box':[20,90,300,120]};document={}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,document))
        self.assertEqual(node['content'],source)
        self.assertEqual(document['source prose insertions'][0]['restored'],' information of new nodes ')
        # Repeated later words must not hide an earlier omitted occurrence.
        self.assertEqual(node['content'].count('information'),2)

    def test_prose_recovery_excludes_changes_to_data_math_code_links_and_outer_text(self):
        source = 'The complete original paragraph describes how the network keeps all source values unchanged.'
        tp = Mock();spans=[TextSpan(0,'source',(30,100,70,107),False,10,100,'Times-Roman')]
        for native,target,selected in [
            (source,source.replace('source values','different values'),spans),
            (source+' More outside words.',source,spans),
            (source.replace('all','all 42'),source,spans),
            (source,source.replace('original ','word '),[replace(spans[0],math=True)]),
            (source,source.replace('original ',''),[replace(spans[0],font_name='Courier')]),
            (source,source.replace('original ',''),[replace(spans[0],font_name='')]),
        ]:
            tp.get_text_bounded.return_value=native
            node={'type':'paragraph','content':target,'bounding box':[20,90,300,120]}
            self.assertFalse(restore_prose_omissions(node,tp,selected,1,{}))
            self.assertEqual(node['content'],target)
        native='The complete original paragraph links https://example.org/source/complete/path and keeps everything intact.'
        tp.get_text_bounded.return_value=native;target=native.replace('/complete','')
        node={'type':'paragraph','content':target,'bounding box':[20,90,300,120]}
        self.assertFalse(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],target)

    def test_prose_recovery_can_own_letters_separately_from_neighboring_math(self):
        prefix='The source formula '
        symbol='score'
        suffix=' appears here and all nodes in the complete network remain connected to their peers.'
        source=prefix+symbol+suffix
        tp=Mock();tp.get_text_bounded.return_value=source
        spans=[TextSpan(0,prefix,(30,100,70,107),False,10,100,'Times-Roman'),
               TextSpan(20,symbol,(72,100,85,107),True,10,100,'Times-Italic'),
               TextSpan(25,suffix,(86,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'paragraph','content':source.replace('network ',''),'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)
        node['content']=source.replace(symbol,'')
        self.assertFalse(restore_prose_omissions(node,tp,spans,1,{}))

    def test_local_font_projection_handles_clipped_text_outside_the_repaired_gap(self):
        prefix='The source formula '
        suffix=' appears here and all nodes in the complete network remain connected to their peers.'
        source=prefix+'score'+suffix
        tp=Mock();tp.get_text_bounded.return_value=source
        spans=[TextSpan(0,'outside '+prefix,(30,100,70,107),False,10,100,'Times-Roman'),
               TextSpan(20,'score',(72,100,85,107),True,10,100,'Times-Italic'),
               TextSpan(25,suffix,(86,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'paragraph','content':source.replace('network ',''),'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)
        node['content']=source.replace('score','')
        self.assertFalse(restore_prose_omissions(node,tp,spans,1,{}))

    def test_short_source_words_and_bibliography_initials_need_explicit_context(self):
        samples=[('Proof. From Proposition 6 we get that all miners will eventually form a stable clique in the complete network.','we '),
                 ('[14] L. Kiffer, J. Neu, S. Sridhar, A. Zohar, D. Tse, Security of blockchains at capacity, (2023).','A. ')]
        for source,omitted in samples:
            with self.subTest(omitted=omitted):
                tp=Mock();tp.get_text_bounded.return_value=source
                spans=[TextSpan(0,source,(30,100,280,107),False,10,100,'Times-Roman')]
                node={'type':'paragraph','content':source.replace(omitted,'',1),'bounding box':[20,90,300,120]}
                self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
                self.assertEqual(node['content'],source)
        source='The complete source describes a network with a stable and predictable topology for every participant.'
        tp=Mock();tp.get_text_bounded.return_value=source
        node={'type':'paragraph','content':source.replace('network','netrk'),'bounding box':[20,90,300,120]}
        spans=[TextSpan(0,source,(30,100,280,107),False,10,100,'Times-Roman')]
        self.assertFalse(restore_prose_omissions(node,tp,spans,1,{}))

    def test_neighboring_reference_rows_in_the_native_box_do_not_hide_an_initial(self):
        source='[14] L. Kiffer, J. Neu, S. Sridhar, A. Zohar, D. Tse, Security of blockchains at capacity, (2023).'
        native=('An unrelated neighboring reference remains outside the current paragraph. '+source+
                ' Another unrelated neighboring reference also stays outside the output.')
        tp=Mock();tp.get_text_bounded.return_value=native
        spans=[TextSpan(0,native,(30,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'paragraph','content':source.replace('A. ',''),'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)

    def test_a_missing_letter_requires_a_complete_known_native_word_and_font_ownership(self):
        prefix='The researcher previously worked at '
        word='Georgia'
        suffix=' before joining the current university and continuing the same research.'
        source=prefix+word+suffix
        tp=Mock();tp.get_text_bounded.return_value=source
        spans=[TextSpan(0,prefix,(30,100,80,107),False,10,100,'Times-Roman'),
               TextSpan(35,word,(82,100,115,107),False,10,100,'Times-Roman'),
               TextSpan(42,suffix,(116,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'paragraph','content':source.replace('Georgia','Georgi'),'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)
        spans[1]=replace(spans[1],math=True)
        node['content']=source.replace('Georgia','Georgi')
        self.assertFalse(restore_prose_omissions(node,tp,spans,1,{}))

    def test_a_missing_source_word_keeps_its_attached_prose_comma(self):
        source='Proof. Assume there is a stable topology, this means all edges remain stable and the miners form a clique.'
        tp=Mock();tp.get_text_bounded.return_value=source
        spans=[TextSpan(0,source,(30,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'paragraph','content':source.replace('topology, ',''),'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)

    def test_symbol_font_bullet_does_not_hide_a_missing_repeated_prose_word(self):
        source='• We compare average distance, diameter, and average eccentricity, keeping all measurements in the report.'
        target=source.replace('and average','and')
        tp=Mock();tp.get_text_bounded.return_value=source
        spans=[TextSpan(0,'•',(30,100,33,107),False,10,100,'Symbol'),
               TextSpan(2,source[2:],(34,100,280,107),False,10,100,'Times-Roman')]
        node={'type':'list item','content':target,'bounding box':[20,90,300,120]}
        self.assertTrue(restore_prose_omissions(node,tp,spans,1,{}))
        self.assertEqual(node['content'],source)

    def test_independent_words_do_not_split_inflections_or_short_names(self):
        for word in ['leverages','computationally','bootstrapping','reported','reporting']:
            self.assertTrue(_known_inflection(word),word)
        for left,right in [('lever','ages'),('computation','ally'),('bootstrap','ping'),('Vi','tis')]:
            self.assertFalse(_separate_words(left,right),(left,right))
        self.assertTrue(_separate_words('Model','checking'))

    def test_prose_font_change_with_a_real_word_gap_can_restore_a_missing_space(self):
        spans = [TextSpan(0,'its',(30,100,40,107),False,10,100,'Times-Roman'),
                 TextSpan(3,'local',(43,100,63,107),False,10,100,'Times-Italic')]
        records = geometric_word_spaces(spans)
        self.assertEqual(len(records),1)
        node = {'content':'forming itslocal mesh. https://example.org/itslocal',
                'bounding box':[20,90,250,115]}
        # Two occurrences cannot be resolved from one geometry record.
        normalize_geometric_spaces(node,records,1,{})
        self.assertIn('forming itslocal',node['content'])
        node['content'] = 'forming itslocal mesh.'
        document = {}
        normalize_geometric_spaces(node,records,1,document)
        self.assertEqual(node['content'],'forming its local mesh.')
        self.assertEqual(document['source typography repairs'][0]['kind'],'geometric-spacing')
        for content in ['https://example.org/itslocal','itslocal itslocal']:
            node['content'] = content
            normalize_geometric_spaces(node,records,1,{})
            self.assertEqual(node['content'],content)
        for changed in [replace(spans[1],box=(41,100,61,107)),
                        replace(spans[1],baseline=95),replace(spans[1],math=True),
                        replace(spans[1],font_name='Courier'),
                        replace(spans[1],font_name='Times-Roman')]:
            self.assertEqual(geometric_word_spaces([spans[0],changed]),[])
        words = [replace(spans[0],text='an'),replace(spans[1],text='other')]
        self.assertEqual(geometric_word_spaces(words),[])

    def test_a_printed_compound_hyphen_can_be_restored_for_two_independent_words(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'compound.pdf'
            fixture_pdf(path,[b'BT /F2 10 Tf 30 500 Td (Model-) Tj ET '
                              b'BT /F2 10 Tf 30 487 Td (checking verification remains readable.) Tj ET'])
            node = paragraph('Modelchecking verification remains readable.',[29,483,280,510],font='Times-Roman')
            document = repair_structure(path,{'kids':[node]})
            self.assertIn('Model-checking verification', ' '.join(n.get('content','') for n in document['kids']))
            self.assertTrue(any(r.get('inserted hyphen') for r in document['source typography repairs']))

    def test_existing_unbroken_words_and_inflections_do_not_gain_hyphens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrap.pdf'
            fixture_pdf(path,[b'BT /F2 10 Tf 30 500 Td (The lever-) Tj ET '
                              b'BT /F2 10 Tf 30 487 Td (ages remain unchanged.) Tj ET '
                              b'BT /F2 10 Tf 30 450 Td (The multi-) Tj ET '
                              b'BT /F2 10 Tf 30 437 Td (cast remains unchanged.) Tj ET '
                              b'BT /F2 10 Tf 30 400 Td (Another multicast example.) Tj ET'])
            nodes = [paragraph('The leverages remain unchanged.',[29,483,280,510]),
                     paragraph('The multicast remains unchanged.',[29,433,280,460])]
            document = repair_structure(path,{'kids':nodes})
            text = ' '.join(n.get('content','') for n in document['kids'])
            self.assertIn('The leverages remain unchanged.',text)
            self.assertIn('The multicast remains unchanged.',text)
            self.assertFalse(any(r.get('inserted hyphen') for r in document.get('source typography repairs',[])))

    def native_name(self, text='Micic´´', font='Times-Roman'):
        boxes = [(30+i*5, 100, 34+i*5, 107) for i in range(5)]
        boxes += [(41, 106, 43, 108), (51, 106, 53, 108)]
        spans = [TextSpan(0, 'Mici', (30,100,49,107), False,10,100,font),
                 TextSpan(4, 'c', boxes[4], False,10,100,font),
                 TextSpan(5, '´', boxes[5], False,10,100,font),
                 TextSpan(6, '´', boxes[6], False,10,100,font)]
        tp = Mock()
        tp.count_chars.return_value = len(text)
        tp.get_charbox.side_effect = boxes.__getitem__
        page = Mock()
        page.get_textpage.return_value = tp
        return page, tp, spans, text

    def test_two_accent_marks_can_follow_the_whole_native_word(self):
        page, tp, spans, text = self.native_name()
        with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                   side_effect=lambda _, i: ord(text[i])):
            records = positioned_accents(page, spans)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['normalized'], 'Mićić')
        node = {'content': 'J. Mi´ci´c, author.', 'bounding box': [20,95,150,115]}
        document = {}
        normalize_accents(node, records, 1, document)
        self.assertEqual(node['content'], 'J. Mićić, author.')
        projected = normalized_spans(spans, document, 1)
        self.assertEqual(''.join(s.text for s in projected), 'Mićić')
        self.assertEqual(projected[0].index, spans[0].index)
        self.assertEqual(projected[0].box, spans[0].box)

    def test_marks_in_code_or_math_fonts_are_not_name_accents(self):
        for font in ['Courier', 'CMMI10']:
            page, tp, spans, text = self.native_name(font=font)
            with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                       side_effect=lambda _, i: ord(text[i])):
                self.assertEqual(positioned_accents(page, spans), [])

    def test_existing_accented_letters_do_not_hide_a_separate_mark(self):
        text = 'ErdosRényi˝'
        tp = Mock()
        tp.count_chars.return_value = len(text)
        boxes = [(30+i*5,100,34+i*5,107) for i in range(10)] + [(46,106,49,108)]
        tp.get_charbox.side_effect = boxes.__getitem__
        page = Mock()
        page.get_textpage.return_value = tp
        spans = [TextSpan(0,text[:-1],(30,100,79,107),False,10,100,'Times-Roman'),
                 TextSpan(10,'˝',boxes[10],False,10,100,'Times-Roman')]
        with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                   side_effect=lambda _,i: ord(text[i])):
            records = positioned_accents(page,spans)
            self.assertEqual(records[0]['normalized'],'ErdősRényi')
            self.assertEqual(positioned_accents(page,[replace(s,math=True) for s in spans]),[])

    def single_accent(self,word,suffix='',base=0,mark_baseline=100):
        text = word+suffix+'´'
        boxes = [(30+i*5,100,34+i*5,107) for i in range(len(word)+len(suffix))]
        boxes.append((31+base*5,106,33+base*5,108))
        tp = Mock()
        tp.count_chars.return_value = len(text)
        tp.get_charbox.side_effect = boxes.__getitem__
        page = Mock()
        page.get_textpage.return_value = tp
        spans = [TextSpan(0,word,(30,100,29+len(word)*5,107),False,10,100,'Times-Roman'),
                 TextSpan(len(text)-1,'´',boxes[-1],False,10,mark_baseline,'Times-Roman')]
        with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                   side_effect=lambda _,i:ord(text[i])):
            return positioned_accents(page,spans)

    def test_dotless_i_under_an_acute_mark_becomes_the_source_accented_letter(self):
        records = self.single_accent('Dıaz',base=1)
        self.assertEqual(records[0]['normalized'],'Díaz')
        node = {'content':'G. D´ıaz, author.','bounding box':[20,90,200,120]}
        normalize_accents(node,records,1,{})
        self.assertEqual(node['content'],'G. Díaz, author.')

    def test_accented_initial_needs_bibliography_and_explicit_period_and_mark(self):
        records = self.single_accent('E',suffix='.',mark_baseline=102.2)
        self.assertTrue(records[0]['initial'])
        reference = '[28] P.-E.´ Dagand, Opis, 2010. https://example.org/E.´'
        node = {'content':reference,'bounding box':[20,90,300,120]}
        normalize_accents(node,records,1,{})
        self.assertEqual(node['content'],reference.replace('P.-E.´','P.-É.'))
        for text in ['A formula E.´ remains unchanged.','[28] P.-E. Dagand, Opis, 2010.']:
            node['content'] = text
            normalize_accents(node,records,1,{})
            self.assertEqual(node['content'],text)

    def test_low_or_wide_marks_are_not_composed(self):
        for box in [(40,101,43,104), (35,106,50,108)]:
            page, tp, spans, text = self.native_name()
            spans = spans[:2] + [replace(spans[2], box=box)]
            with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                       side_effect=lambda _, i: ord(text[i])):
                self.assertEqual(positioned_accents(page, spans), [])

    def test_missing_mark_requires_unique_source_and_target_word(self):
        records = [{'original': 'Erdos˝', 'normalized': 'Erdős', 'box': [30,100,60,110],
                    'glyph_overrides': {}, 'span_indices': {}}]
        tp = Mock()
        tp.get_text_bounded.return_value = 'The Erdos˝ graph.'
        node = {'content': 'The Erdos graph.', 'bounding box': [20,90,200,120]}
        normalize_accents(node, records, 1, {}, tp)
        self.assertEqual(node['content'], 'The Erdős graph.')
        for native, target in [('Erdos˝ and Erdos', 'Erdos and Erdos'),
                               ('Erdos˝ and Erdos', 'Erdos'),
                               ('Erdos˝', 'Erdos and Erdos')]:
            tp.get_text_bounded.return_value = native
            node['content'] = target
            normalize_accents(node, records, 1, {}, tp)
            self.assertEqual(node['content'], target)

    def test_links_and_unrelated_or_outside_words_are_preserved(self):
        record = {'original': 'Erdos˝', 'normalized': 'Erdős', 'box': [30,100,60,110],
                  'glyph_overrides': {}, 'span_indices': {}}
        text = 'Erd˝os https://example.org/Erd˝os author@Erd˝os.org unrelated´'
        node = {'content': text, 'bounding box': [20,90,400,120]}
        normalize_accents(node, [record], 1, {})
        self.assertEqual(node['content'], text.replace('Erd˝os ', 'Erdős ', 1))
        node = {'content': 'Erd˝os', 'bounding box': [200,90,400,120]}
        normalize_accents(node, [record], 1, {})
        self.assertEqual(node['content'], 'Erd˝os')

    def test_span_indices_follow_surrogate_pairs_and_control_gaps(self):
        codes = [0xd835,0xdc00,2,ord('a'),ord('´')]
        tp = Mock()
        tp.count_chars.return_value = len(codes)
        tp.get_charbox.return_value = (30,100,35,110)
        spans = [TextSpan(0, '𝐀a', (30,100,40,110),False,10,100),
                 TextSpan(4, '´', (35,107,37,110),False,10,100)]
        with patch('paper_localizer.source_text.pdfium.raw.FPDFText_GetUnicode',
                   side_effect=lambda _, i: codes[i]):
            indices = _span_indices(tp, spans, {3,4})
        self.assertEqual(indices, {'0': [0,3], '4': [4]})
        d = {'source glyph overrides': {'1': {'3': 'á', '4': ''}},
             'source span glyph indices': {'1': indices}}
        self.assertEqual([s.text for s in normalized_spans(spans,d,1)], ['𝐀á'])

    def test_hard_hyphen_requires_repeated_native_word_and_valid_break(self):
        self.assertTrue(_can_join('com', 'munication', Counter(communication=2)))
        for left, right, count in [('Gos','sipSub',1), ('comm','unication',3),
                                   ('re','sign',5), ('sub','section',5), ('foo','bar',0)]:
            self.assertFalse(_can_join(left,right,Counter({(left+right).casefold():count})))

    def test_lexical_words_still_require_a_valid_noncompound_break(self):
        for left,right in [('report','ing'),('ex','periences'),('in','troducing'),('per','sistently')]:
            self.assertTrue(_can_join(left,right,Counter()))
        for left,right in [('long','term'),('one','time'),('well','being'),('topology','aware'),
                           ('nine','dimensional'),('GossipSub','based')]:
            self.assertFalse(_can_join(left,right,Counter()))
        self.assertFalse(_can_join('com','munication',Counter({'communication':5,'com-munication':1})))

    def test_source_line_break_is_repaired_without_touching_compound_or_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrap.pdf'
            fixture_pdf(path, [
                b'BT /F2 10 Tf 30 500 Td (The source com-) Tj ET '
                b'BT /F2 10 Tf 30 487 Td (munication remains readable.) Tj ET '
                b'BT /F2 10 Tf 30 450 Td (The topology-) Tj ET '
                b'BT /F2 10 Tf 30 437 Td (aware system remains readable.) Tj ET '
                b'BT /F2 10 Tf 30 400 Td (https://example.org/com-) Tj ET '
                b'BT /F2 10 Tf 30 387 Td (munication) Tj ET',
                b'BT /F2 10 Tf 30 500 Td (communication communication) Tj ET',
            ])
            nodes = [paragraph('The source com- munication remains readable.', [29,483,280,510]),
                     paragraph('The topology- aware system remains readable.', [29,433,280,460]),
                     paragraph('https://example.org/com- munication', [29,383,280,410])]
            doc = repair_structure(path, {'kids': nodes})
            text = ' '.join(n.get('content','') for n in doc['kids'])
            self.assertIn('source communication remains', text)
            self.assertIn('topology-aware system', text)
            self.assertIn('https://example.org/com- munication', text)
            changes = [r for r in doc['source typography repairs'] if r['kind']=='line-wrap']
            self.assertEqual([r['removed hyphen'] for r in changes], [True, False])

    def test_a_nearby_line_in_another_column_is_not_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'columns.pdf'
            fixture_pdf(path, [b'BT /F2 10 Tf 30 500 Td (The source com-) Tj ET '
                               b'BT /F2 10 Tf 320 487 Td (munication remains in another column.) Tj ET '
                               b'BT /F2 10 Tf 30 450 Td (communication communication) Tj ET'])
            node = paragraph('The source com- munication remains in another column.', [29,483,580,510])
            doc = repair_structure(path, {'kids':[node]})
            self.assertIn('com- munication',doc['kids'][0]['content'])
            self.assertFalse(doc.get('source typography repairs'))

    def test_known_compound_recovers_a_hyphen_lost_at_a_line_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'compound.pdf'
            fixture_pdf(path,[b'BT /F2 10 Tf 30 500 Td (The proof-) Tj ET '
                              b'BT /F2 10 Tf 30 487 Td (of-work remains valid.) Tj ET '
                              b'BT /F2 10 Tf 30 450 Td (A proof-of-work example.) Tj ET'])
            node = paragraph('The proofof-work remains valid.',[29,483,280,510],font='Times-Roman')
            doc = repair_structure(path,{'kids':[node]})
            self.assertIn('The proof-of-work remains valid.', ' '.join(n.get('content','') for n in doc['kids']))
            self.assertTrue(any(r.get('inserted hyphen') for r in doc.get('source typography repairs',[])))
            from paper_localizer.formulas import _spans
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(path) as pdf:
                raw = _spans(pdf[0])
                projected = normalized_spans(raw,doc,1)
            self.assertGreater(len(projected),len(raw))
            self.assertEqual(sum(s.text=='-' for s in projected),4)

    def test_uncovered_or_code_line_wraps_are_preserved(self):
        r = {'left':'com','right':'munication','join':True,'source occurrences':3,
             'boxes':[[50,100,70,110],[70,100,73,110],[30,87,80,97]],'hyphen_index':10}
        for box, font in [([30,98,100,115],'Times-Roman'), ([20,80,120,115],'Courier')]:
            node = {'content':'com- munication','bounding box':box,'font':font}
            normalize_line_wraps(node,[r],1,{})
            self.assertEqual(node['content'],'com- munication')

    def test_prose_spacing_can_be_restored_around_different_math_glyphs(self):
        text = 'Complete ordinary words stay visible ? and the localcharacteristics remain clear.'
        native = 'Complete ordinary words stay visible α and the local characteristics remain clear.'
        node = {'content':text,'bounding box':[20,100,400,200]}
        tp = Mock()
        tp.get_text_bounded.return_value = native
        doc = {}
        self.assertTrue(restore_local_spacing(node,tp,1,doc))
        self.assertEqual(node['content'],text.replace('localcharacteristics','local characteristics'))
        self.assertEqual(''.join(node['content'].split()),''.join(text.split()))
        self.assertEqual(doc['source typography repairs'][0]['inserted spaces'],1)

    def test_partial_spacing_does_not_change_urls_code_or_short_matches(self):
        examples = [
            ('For source ? see https://example.org/localcharacteristics and keep the exact target.',
             'For source α see https://example.org/local characteristics and keep the exact target.','Times-Roman'),
            ('Complete ordinary words ? and the localcharacteristics remain clear.',
             'Complete ordinary words α and the local characteristics remain clear.','Courier'),
            ('x localword y','z local word y','Times-Roman'),
        ]
        for text,native,font in examples:
            tp = Mock()
            tp.get_text_bounded.return_value = native
            node = {'content':text,'bounding box':[20,100,400,200],'font':font}
            self.assertFalse(restore_local_spacing(node,tp,1,{}))
            self.assertEqual(node['content'],text)

    def test_prose_dash_dropped_at_a_line_break_is_restored_between_exact_anchors(self):
        text = 'The architectural organization of modern systems particularly the separation of responsibilities remains important.'
        node = {'content':text,'bounding box':[20,100,400,200]}
        tp = Mock()
        tp.get_text_bounded.return_value = text.replace('systems particularly','systems—\r\nparticularly')
        self.assertTrue(restore_prose_dashes(node,tp,1,{}))
        self.assertIn('systems—particularly',node['content'])
        self.assertEqual(node['content'].replace('—',' '),text)

    def test_prose_dash_requires_long_anchors_and_preserves_code_and_urls(self):
        for text,native,font in [
            ('systems particularly','systems—particularly','Times-Roman'),
            ('The native variable x2 retains every other ordinary source word.',
             'The native variable x—2 retains every other ordinary source word.','Times-Roman'),
            ('The architectural organization of modern systems particularly remains clear.',
             'The architectural organization of modern systems—particularly remains clear.','Courier'),
        ]:
            node = {'content':text,'bounding box':[20,100,400,200],'font':font}
            tp = Mock()
            tp.get_text_bounded.return_value = native
            self.assertFalse(restore_prose_dashes(node,tp,1,{}))
            self.assertEqual(node['content'],text)


if __name__ == '__main__':
    unittest.main()
