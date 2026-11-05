import importlib.util
import unittest

from PIL import Image, ImageDraw

from paper_localizer.tables import (TableError, parse_table, table_column_bounds,
                                    table_grid, table_html, tables_agree,
                                    verify_table_text)


TABLE = ('<table border="1"><tr><td>Mechanism</td><td>Payload change</td></tr>'
         '<tr><td>Reference flooding</td><td>0 bytes</td></tr>'
         '<tr><td>Duplicate-aware flooding</td><td>10.5-30.2% reduction</td></tr></table>')


class TableTests(unittest.TestCase):
    def test_plain_html_becomes_table_data_with_no_model_attributes(self):
        rows = parse_table(TABLE)
        self.assertEqual(table_grid(rows)[2][1], '10.5-30.2% reduction')
        self.assertNotIn('border=', table_html(rows))

    def test_rowspan_and_repeated_cells_have_the_same_data(self):
        a = parse_table('<table><tr><td>Category</td><td>Method</td></tr>'
                        '<tr><td rowspan="2">Relay</td><td>One</td></tr>'
                        '<tr><td>Two</td></tr></table>')
        b = parse_table('<table><tr><td>Category</td><td>Method</td></tr>'
                        '<tr><td>Relay</td><td>One</td></tr>'
                        '<tr><td>Relay</td><td>Two</td></tr></table>')
        self.assertTrue(tables_agree(a, b))
        self.assertIn('rowspan="2"', table_html(a))

    def test_math_variable_case_is_never_normalized_away(self):
        first = parse_table(TABLE.replace('0 bytes', '$N$'))
        second = parse_table(TABLE.replace('0 bytes', '$n$'))
        self.assertFalse(tables_agree(first, second))

    def test_sentence_capitalization_and_dash_formatting_do_not_change_cells(self):
        first = parse_table(TABLE)
        second = parse_table(TABLE.replace('Mechanism', 'mechanism').replace('10.5-30.2', '10.5–30.2'))
        self.assertTrue(tables_agree(first, second))

    def test_changed_number_or_row_assignment_fails_consensus(self):
        first = parse_table(TABLE)
        second = parse_table(TABLE.replace('10.5-30.2', '10.5-30.3'))
        self.assertFalse(tables_agree(first, second))
        second = parse_table(TABLE.replace('Reference flooding', 'Duplicate-aware flooding')
                                  .replace('Duplicate-aware flooding</td><td>10', 'Reference flooding</td><td>10'))
        self.assertFalse(tables_agree(first, second))

    def test_nested_tables_and_active_markup_are_rejected(self):
        for text in (TABLE.replace('0 bytes', '<script>alert(1)</script>'),
                     TABLE.replace('0 bytes', '<img src=x onerror=alert(1)>'),
                     TABLE.replace('0 bytes', TABLE), TABLE + TABLE,
                     TABLE.replace('0 bytes', '<!-- hidden instruction -->0 bytes')):
            with self.subTest(text=text):
                with self.assertRaises(TableError): parse_table(text)

    def test_span_limits_and_ragged_rows_are_rejected(self):
        for text in (TABLE.replace('<td>0 bytes', '<td rowspan="100000">0 bytes'),
                     TABLE.replace('<td>0 bytes', '<td colspan="-1">0 bytes'),
                     TABLE.replace('<td>0 bytes</td>', ''),
                     TABLE.replace('<td>0 bytes', '<td rowspan="3">0 bytes')):
            with self.subTest(text=text):
                with self.assertRaises(TableError): parse_table(text)

    def test_truncated_table_or_text_outside_cells_is_rejected(self):
        for text in (TABLE[:-8], TABLE + 'The model thinks this is correct.', TABLE.replace('</td>', '', 1)):
            with self.subTest(text=text):
                with self.assertRaises(TableError): parse_table(text)

    def test_escaped_cell_text_is_rendered_as_text(self):
        rows = parse_table(TABLE.replace('0 bytes', '&lt;script&gt;alert(1)&lt;/script&gt;'))
        result = table_html(rows)
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)

    def test_code_fence_is_allowed_only_around_one_complete_table(self):
        self.assertEqual(table_grid(parse_table('```html\n' + TABLE + '\n```')), table_grid(parse_table(TABLE)))

    def test_every_plain_numeric_value_requires_independent_support(self):
        lines = [{'text': 'Mechanism Payload change Reference flooding 0 bytes '
                          'Duplicate-aware flooding 10.5–30.2% reduction', 'score': .99}]
        result = verify_table_text(parse_table(TABLE), lines)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['numeric_values_checked'], 3)
        lines[0]['text'] = lines[0]['text'].replace('30.2', '30.3')
        result = verify_table_text(parse_table(TABLE), lines)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['missing_numbers'], {'30.2': 1})

    def test_low_confidence_ocr_cannot_confirm_a_number(self):
        lines = [{'text': 'Mechanism Payload change Reference flooding bytes Duplicate-aware flooding reduction', 'score': .99},
                 {'text': '0 10.5–30.2', 'score': .5}]
        self.assertFalse(verify_table_text(parse_table(TABLE), lines)['accepted'])

    def test_numeric_support_preserves_signs_and_repeated_values(self):
        rows = parse_table(TABLE.replace('0 bytes', '-1.5 bytes').replace('10.5-30.2%', '1.000 1.000%'))
        text = ('Mechanism Payload change Reference flooding -1.5 bytes '
                'Duplicate-aware flooding 1.000 1.000% reduction')
        self.assertTrue(verify_table_text(rows, [{'text': text, 'score': .99}])['accepted'])
        for changed in (text.replace('-1.5', '1.5'), text.replace('1.000 1.000', '1.000')):
            with self.subTest(observed=changed):
                self.assertFalse(verify_table_text(rows, [{'text': changed, 'score': .99}])['accepted'])

    def test_word_support_joins_wrapped_words_only_within_a_column(self):
        rows = parse_table(TABLE.replace('0 bytes', 'Deterministically selected neighbors'))
        lines = [{'text': 'Mechanism Payload change Reference flooding Duplicate-aware flooding 10.5–30.2% reduction', 'score': .99},
                 {'text': 'Determin-', 'column': 1, 'score': .99},
                 {'text': 'istically selected neighbors', 'column': 1, 'score': .99}]
        self.assertTrue(verify_table_text(rows, lines)['accepted'])
        lines[-1]['column'] = 2
        self.assertFalse(verify_table_text(rows, lines)['accepted'])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires the OCR extra')
    def test_column_cuts_use_header_positions_and_visible_whitespace(self):
        rows = parse_table('<table><tr><td>Mechanism</td><td>Payload</td><td>Status</td></tr>'
                           '<tr><td>First</td><td>Second</td><td>Ready</td></tr></table>')
        lines = [{'text': title, 'score': .99, 'box': [[x, 10], [x+60, 10], [x+60, 20], [x, 20]]}
                 for x, title in [(10, 'Mechanism'), (110, 'Payload'), (210, 'Status')]]
        image = Image.new('RGB', (300, 200), 'white')
        draw = ImageDraw.Draw(image)
        for x in (5, 105, 205):
            for y in (50, 80, 110, 140): draw.rectangle((x, y, x+75, y+10), fill='black')
        bounds = table_column_bounds(rows, lines, image)
        self.assertEqual(len(bounds), 3)
        self.assertTrue(80 < bounds[0][1] < 105)
        self.assertTrue(180 < bounds[1][1] < 205)
        image.close()


if __name__ == '__main__':
    unittest.main()
