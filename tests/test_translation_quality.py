import unittest

from paper_localizer.core import (
    TRANSLATION_MODEL, TranslationValidationError, _sentence_boundaries, _split_text, extract_blocks, make_chunks,
    _translate_chunk, _translate_native_table, _translation_segments,
)
from paper_localizer.terminology import translation_terms


class TerminologyTests(unittest.TestCase):
    def test_longer_source_phrases_take_precedence(self):
        terms = translation_terms('Degree assortativity and degree distributions.', 'Graph topology', 'Korean')
        self.assertEqual(terms, (('degree distributions', '차수 분포'), ('Degree assortativity', '차수 동류성')))

    def test_terms_must_appear_in_source_not_only_context(self):
        self.assertEqual(translation_terms('An experiment.', 'GossipSub degree and churn', 'Korean'), ())

    def test_funder_full_name_is_disambiguated_without_guessing_acronyms(self):
        self.assertEqual(translation_terms('National Research Foundation of Korea (NRF).', '', 'Korean'),
                         (('National Research Foundation of Korea', '한국연구재단'),))
        self.assertEqual(translation_terms('An NRF grant.', 'Networking', 'Korean'), ())

    def test_language_and_subject_scope(self):
        self.assertEqual(translation_terms('The degree increased.', '', 'Korean'), ())
        self.assertEqual(translation_terms('The graph degree increased.', '', 'Japanese'), ())
        self.assertIn(('degree', '차수'), translation_terms('The graph degree increased.', '', 'ko-KR'))

    def test_non_graph_senses_are_not_overridden_by_paper_context(self):
        for source in ['The temperature rose by 5 degrees.', 'There are three degrees of freedom.',
                       'A rotation of 90 degrees.', 'The degree of confidence increased.',
                       'She received a doctoral degree.']:
            with self.subTest(source=source):
                self.assertEqual(translation_terms(source, 'Graph network', 'Korean'), ())

    def test_business_and_shipping_senses_are_not_network_terms(self):
        for source in ['Customer churn rose.', 'Delivery of goods increased.', 'Data miners found clusters.']:
            with self.subTest(source=source):
                self.assertEqual(translation_terms(source, 'Peer-to-peer network', 'Korean'), ())

    def test_delivery_rate_and_churn_are_selected_in_network_context(self):
        self.assertEqual(translation_terms('Delivery rate under churn.', 'P2P network', '한국어'),
                         (('Delivery rate', '메시지 전달 성공률'), ('churn', '노드 변동')))

    def test_short_headers_do_not_override_business_context(self):
        self.assertEqual(translation_terms('Churn', 'Customer retention in mobile networks', 'Korean'), ())
        self.assertEqual(translation_terms('Delivery', 'Routing in a shipping network', 'Korean'), ())

    def test_server_roles_require_explicit_physical_server_counts(self):
        self.assertEqual(translation_terms('6 Physical Servers (1 Manager, 5 Workers)', '', 'Korean'),
                         (('Physical Servers', '물리 서버'), ('Manager', '관리 노드'), ('Workers', '작업 노드')))
        for source in ['The manager supervised 5 workers.', '6 servers installed by workers.',
                       'One manager operates the physical servers.',
                       '6 physical servers were installed by 1 manager and 5 workers.']:
            self.assertEqual(translation_terms(source, 'Networking', 'Korean'), ())

    def test_delivery_throughput_is_not_forced_to_mean_success_rate(self):
        for source in ['The delivery rate is 10 messages per second.',
                       'The delivery rate is 100 Mbps.', 'The delivery rate is 100Mbps.',
                       'A delivery rate of 10 packets/s.', 'A delivery rate of 10packets/s.']:
            self.assertNotIn('메시지 전달 성공률', dict(translation_terms(source, 'P2P network', 'Korean')).values())

    def test_non_graph_degree_of_a_property_is_not_a_node_degree(self):
        self.assertEqual(translation_terms('A higher degree of cooperation.', 'Graph networks', 'Korean'), ())
        self.assertIn(('degree', '차수'), translation_terms('The degree of a node increased.', 'Graph networks', 'Korean'))

    def test_miners_alone_do_not_establish_a_blockchain_context(self):
        for source,context in [('The miners work underground.', ''), ('The coal miners are connected.', 'Network'),
                               ('The miners communicate.', 'Social networks')]:
            self.assertEqual(translation_terms(source, context, 'Korean'), ())
        self.assertIn(('miners', '채굴 노드'), translation_terms('Distance to miners.', 'Bitcoin network', 'Korean'))

    def test_only_source_matched_terms_reach_the_native_model(self):
        class Client:
            def chat(self, model, prompt, **kwargs):
                self.prompt = prompt
                return '차수 제어가 메시를 유지한다.'
        client = Client()
        _translate_chunk(client, TRANSLATION_MODEL, 'Degree-control maintains the mesh.', 'Korean',
                         context='Private paper title. Network churn and delivery rate.')
        self.assertIn('Degree-control translates to 차수 제어', client.prompt)
        self.assertNotIn('Private paper title', client.prompt)
        self.assertNotIn('churn translates to', client.prompt)
        self.assertNotIn('degree translates to', client.prompt)


class PassageTests(unittest.TestCase):
    def test_document_chunking_does_not_cut_off_the_rest_of_a_sentence(self):
        sentence = 'We first define a precise notion of closeness, ' * 8 + 'even where no node shares a unique prefix.'
        self.assertEqual(_split_text(sentence, 200), [sentence])
        self.assertEqual(_split_text(sentence + ' Another complete sentence.', 200),
                         [sentence, 'Another complete sentence.'])

    def test_document_chunking_counts_protected_math_not_html_attributes(self):
        equation = '<img src="figs/formula.png" class="inline-math" alt="' + 'x' * 800 + '" />'
        source = 'This complete sentence contains ' + equation + '. '
        pieces = _split_text(source * 8, 200)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(piece.endswith('.') and piece.count(equation) >= 1 for piece in pieces))
        self.assertEqual(' '.join(pieces), (source * 8).strip())

    def test_segments_reconstruct_exact_source_including_blank_lines(self):
        for source in ['\n\nFirst sentence. Next sentence.\n\n\nLast paragraph.\n\n',
                       'One sentence.\n \nSecond sentence.\n\t\n',
                       'A very long clause without any complete sentence boundary',
                       '\n\n\n', 'Short text.']:
            with self.subTest(source=source):
                pieces = _translation_segments(source, limit=15)
                self.assertEqual(''.join(piece + separator for piece, separator in pieces), source)

    def test_a_long_single_sentence_is_never_cut_at_an_arbitrary_word(self):
        source = 'Although ' + 'a very long dependent clause ' * 20 + 'the result holds.'
        self.assertEqual(_translation_segments(source, limit=100), [(source, '')])

    def test_abbreviations_and_author_initials_are_not_sentence_ends(self):
        source = 'A. Smith et al. Explain Fig. A and Eq. B, e.g. Other examples. Next sentence.'
        boundaries = _sentence_boundaries(source)
        self.assertEqual([source[:a] for a, b in boundaries], [source[:source.index(' Next')]])

    def test_closing_quotes_stay_with_the_sentence(self):
        source = 'They said "It works." Another sentence.'
        boundary, = _sentence_boundaries(source)
        self.assertEqual(source[:boundary[0]], 'They said "It works."')

    def test_quoted_sentences_ellipses_and_numbered_items_stay_intact(self):
        for source in ['They wrote “This is optional... By default, use both.” Next sentence.',
                       'They wrote "One sentence. Another sentence." Next sentence.',
                       'An aside (First sentence. Another sentence.) Next sentence.']:
            with self.subTest(source=source):
                boundaries = _sentence_boundaries(source)
                self.assertEqual(len(boundaries), 1)
                self.assertTrue(source[boundaries[0][1]:].startswith('Next sentence'))
        self.assertEqual(_sentence_boundaries('1. Initialize the graph.'), [])

    def test_failed_multi_sentence_passage_retries_complete_sentences(self):
        class Client:
            def __init__(self):
                self.sources = []
            def chat(self, model, prompt, **kwargs):
                piece = prompt.split('Do not add or omit information:\n\n', 1)[1]
                self.sources.append(piece)
                return piece.replace('[7]', '') if 'Second' in piece and 'First' in piece else piece
        source = 'First result [7]. Second result [8].'
        client = Client()
        self.assertEqual(_translate_chunk(client, TRANSLATION_MODEL, source, 'Korean'), source)
        self.assertEqual(len(client.sources), 4)
        self.assertEqual(client.sources[-2:], ['First result [7].', 'Second result [8].'])

    def test_sentences_are_rejoined_without_new_paragraphs(self):
        source = 'First sentence. Second sentence. Third sentence.'
        pieces = _translation_segments(source, limit=20)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all('\n' not in separator for piece, separator in pieces))

    def test_each_passage_validates_only_its_own_math_tokens(self):
        first = 'Use $x^2$ with 10 nodes [7]. ' + 'The graph remains connected. ' * 27
        second = 'Use $y^2$ with 20 nodes [8]. ' + 'The graph remains sparse. ' * 27
        source = first + '\n\n' + second
        class EchoClient:
            def __init__(self):
                self.sources = []
            def chat(self, model, prompt, **kwargs):
                piece = prompt.split('Do not add or omit information:\n\n', 1)[1]
                self.sources.append(piece)
                return piece
        client = EchoClient()
        self.assertEqual(_translate_chunk(client, TRANSLATION_MODEL, source, 'Korean'), source)
        self.assertGreater(len(client.sources), 2)
        self.assertFalse(any('[[MATH_0001]]' in text and '[[MATH_0002]]' in text for text in client.sources))

    def test_missing_citation_in_one_passage_rejects_the_chunk(self):
        class Client:
            def chat(self, model, prompt, **kwargs):
                return prompt.split('Do not add or omit information:\n\n', 1)[1].replace('[7]', '7')
        with self.assertRaises(TranslationValidationError):
            _translate_chunk(Client(), TRANSLATION_MODEL, 'First paragraph.\n\n' + 'Text [7]. ' * 90, 'Korean')

    def test_technical_headers_translate_but_protocol_identifiers_stay_fixed(self):
        rows = [[{'text': text, 'rowspan': 1, 'colspan': 1, 'header': True}
                 for text in ['Churn', 'GS Delivery', 'Pastry', '99.2%']]]
        class Client:
            def __init__(self):
                self.prompts = []
            def chat(self, model, prompt, **kwargs):
                self.prompts.append(prompt)
                return '노드 변동' if prompt.endswith('\n\nChurn') else 'GS 전달 성공률'
        client = Client()
        translated = _translate_native_table(client, TRANSLATION_MODEL, rows, 'Korean', context='GossipSub network')
        self.assertEqual([cell['text'] for cell in translated[0]], ['노드 변동', 'GS 전달 성공률', 'Pastry', '99.2%'])
        self.assertEqual(len(client.prompts), 2)
        for before, after in zip(rows[0], translated[0]):
            self.assertEqual({k: v for k, v in before.items() if k != 'text'},
                             {k: v for k, v in after.items() if k != 'text'})

    def test_generic_table_headers_translate_outside_the_korean_glossary(self):
        rows = [[{'text': text, 'rowspan': 1, 'colspan': 1, 'header': True}
                 for text in ['Component', 'Specification', 'Pastry']]]
        class Client:
            def chat(self, model, prompt, **kwargs):
                return '構成要素' if prompt.endswith('\n\nComponent') else '仕様'
        result = _translate_native_table(Client(), TRANSLATION_MODEL, rows, 'Japanese')
        self.assertEqual([cell['text'] for cell in result[0]], ['構成要素', '仕様', 'Pastry'])

    def test_metric_header_cannot_become_the_metric_system(self):
        rows = [[{'text': 'Metric', 'rowspan': 1, 'colspan': 1, 'header': True}]]
        class Client:
            def chat(self, *args, **kwargs):
                raise AssertionError('Unambiguous Korean table labels do not need inference.')
        target = _translate_native_table(Client(), TRANSLATION_MODEL, rows, 'Korean')
        self.assertEqual(target[0][0]['text'], '지표')

    def test_header_dictionary_cannot_remove_an_accompanying_equation(self):
        rows = [[{'text': 'Metric $n$', 'rowspan': 1, 'colspan': 1, 'header': True}]]
        class Client:
            def chat(self, model, prompt, **kwargs):
                return '지표 [[MATH_0001]]'
        target = _translate_native_table(Client(), TRANSLATION_MODEL, rows, 'Korean')
        self.assertEqual(target[0][0]['text'], '지표 $n$')


class FigureContinuationTests(unittest.TestCase):
    def document(self, ending='where no', next_kind='paragraph', next_text='node shares a unique prefix.'):
        return {'kids': [
            {'type': 'paragraph', 'page number': 2,
             'content': 'We define a lookup protocol that works even in the difficult cases ' + ending},
            {'type': 'image', 'page number': 3, 'source': 'figs/tree.png'},
            {'type': 'caption', 'page number': 3, 'content': 'Fig. 1: Lookup tree.'},
            {'type': next_kind, 'page number': 4, 'content': next_text},
        ]}

    def test_sentence_continues_across_a_figure_only_page(self):
        blocks = extract_blocks(self.document())
        self.assertEqual([b.kind for b in blocks], ['paragraph', 'image', 'caption'])
        self.assertTrue(blocks[0].text.endswith('where no node shares a unique prefix.'))
        self.assertEqual(blocks[0].source_pages, (2, 4))
        self.assertEqual(make_chunks(blocks)[0].pages, (2, 4))

    def test_complete_sentences_and_headings_do_not_join_across_floats(self):
        for document in [self.document(ending='are handled.'), self.document(next_kind='heading'),
                         self.document(next_text='Nodes have unique prefixes.')]:
            with self.subTest(document=document):
                self.assertEqual(len(extract_blocks(document)), 4)

    def test_equation_snapshots_and_body_text_block_the_join(self):
        for node in [{'type': 'paragraph', 'content': 'An intervening body paragraph.', 'page number': 3},
                     {'type': 'paragraph', 'formula image': True, 'content': 'equation snapshot', 'page number': 3}]:
            document = self.document()
            document['kids'].insert(2, node)
            self.assertEqual(len(extract_blocks(document)), 5)

    def test_an_unrepresented_missing_page_is_not_assumed_to_be_empty(self):
        document = self.document()
        for node in document['kids'][1:3]:
            node['page number'] = 2
        self.assertEqual(len(extract_blocks(document)), 4)


if __name__ == '__main__':
    unittest.main()
