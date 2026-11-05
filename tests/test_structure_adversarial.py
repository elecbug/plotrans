from copy import deepcopy
import random
import unittest

from paper_localizer.figures import _portrait_label
from paper_localizer.structure import order_document
from paper_localizer.layout import prepare_layout
from paper_localizer.core import extract_blocks


def block(text, box, **kwargs):
    return {'type': 'paragraph', 'page number': 1, 'content': text, 'bounding box': box, **kwargs}


class AdversarialStructureTests(unittest.TestCase):
    def test_disjoint_indented_paragraphs_do_not_imply_two_columns(self):
        nodes = [block('Lower paragraph.', [30, 200, 260, 280]),
                 block('An earlier indented paragraph.', [340, 400, 570, 480])]
        doc = {'kids': nodes, 'page sizes': {'1': [600, 600]}}
        result = order_document(doc)
        self.assertEqual([n['content'] for n in result['kids']],
                         ['An earlier indented paragraph.', 'Lower paragraph.'])

    def test_shuffled_two_column_layouts_preserve_the_known_writer_order(self):
        randomizer = random.Random(20261105)
        for case in range(80):
            left, right = [], []
            gutter = randomizer.uniform(22, 38)
            left_x = randomizer.uniform(20, 40)
            width = randomizer.uniform(210, 235)
            right_x = left_x + width + gutter
            for index in range(5):
                top = 550-index*85+randomizer.uniform(-5, 5)
                left.append(block(f'Left paragraph {index}.', [left_x, top-55, left_x+width, top]))
                right.append(block(f'Right paragraph {index}.', [right_x, top-55, right_x+width, top]))
            shuffled = deepcopy(left+right); randomizer.shuffle(shuffled)
            doc = {'kids': shuffled, 'page sizes': {'1': [600, 600]}}
            with self.subTest(case=case):
                self.assertEqual([n['content'] for n in order_document(doc)['kids']],
                                 [n['content'] for n in left+right])

    def test_different_body_font_is_not_eaten_by_a_caption(self):
        caption = block('Figure 1. A complete caption.', [30, 450, 280, 462],
                        **{'font': 'Helvetica-Bold', 'font size': 8})
        body = block('The body remains a separate paragraph.', [30, 438, 280, 450],
                     **{'font': 'Times-Roman', 'font size': 8})
        document = prepare_layout({'kids': [caption, body]})
        self.assertEqual(document['captions'][0]['content'], 'Fig. 1. A complete caption.')
        self.assertEqual(body['content'], 'The body remains a separate paragraph.')

    def test_receiving_a_packet_is_not_an_author_biography(self):
        region = [30, 100, 110, 200]
        node = block('PEER ONE received the packet from a neighboring peer.', [120, 400, 280, 500])
        self.assertEqual(_portrait_label(region, [node], 600), '')

    def test_degree_biography_can_identify_a_source_portrait(self):
        region = [30, 100, 110, 200]
        node = block('ALEX EXAMPLE received the B.S. degree in computer science.', [120, 400, 280, 500])
        self.assertEqual(_portrait_label(region, [node], 600), 'ALEX EXAMPLE')

    def test_native_bullet_and_reference_markers_are_not_duplicated(self):
        document = {'kids': [block('• The first requirement is retained.', [30, 400, 280, 420]),
                             block('• The second requirement is retained.', [30, 370, 280, 390], type='list item'),
                             block('2) The numbered requirement is retained.', [30, 340, 280, 360], type='list item'),
                             block('[12] A source reference.', [30, 310, 280, 330], type='list item')]}
        self.assertEqual([b.text for b in extract_blocks(document)],
                         ['- The first requirement is retained.', '- The second requirement is retained.',
                          '2) The numbered requirement is retained.', '[12] A source reference.'])


if __name__ == '__main__':
    unittest.main()
