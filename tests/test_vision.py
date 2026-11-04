from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw, ImageFont

from paper_localizer.article import _md_html
from paper_localizer.core import _summary_prompt, _translate_chunk, extract_blocks, make_chunks
from paper_localizer.math_markup import mathml, valid_math
from paper_localizer.ocr import _math_line
from paper_localizer.vision import (_math_key, _prose_agrees, _prose_consistent,
    _formula_signature, _same_short_formula, refine_document, VisionOcrError)
from test_formulas import math_fixture


class VisionTests(unittest.TestCase):
    def test_typography_changes_are_checked_separately_from_ocr_confidence(self):
        self.assertTrue(_math_line('distance between 2i and 2i+1 from itself.'))
        self.assertTrue(_math_line('with probability 1 - 2−k.'))
        self.assertTrue(_math_line('a ⊕ b.'))
        self.assertFalse(_math_line('The node uses a random identifier.'))

    def test_consistency_preserves_exponents_but_ignores_array_alignment(self):
        from paper_localizer.vision import _semantic_latex
        self.assertEqual(_semantic_latex(r'|L|=1 6, |M|=3 2'),r'|L|=16, |M|=32')
        self.assertEqual(_semantic_latex(r'\mathrm {i f} p>q'),r'\mathrm{if} p>q')
        self.assertNotEqual(_math_key(r'(q/p)^{z}'), _math_key(r'(q/p)^{2}'))
        self.assertEqual(_math_key(r'\begin{array}{cc}1&\mathrm{if}\end{array}'),
                         _math_key(r'\begin{array}{cl}1&if\end{array}'))

    def test_prose_correction_cannot_silently_change_data(self):
        source = 'For nodes of distance between 2i and 2i+1, keep 20 contacts.'
        self.assertTrue(_prose_agrees(source, 'For nodes of distance between $2^{i}$ and $2^{i+1}$, keep 20 contacts.'))
        self.assertFalse(_prose_agrees(source, 'For nodes of distance between $2^{i}$ and $2^{i+1}$, keep 30 contacts.'))
        self.assertFalse(_prose_consistent('The bound is $2^{i}$.','The bound is $2^{2}$.'))

    def test_mathml_is_offline_balanced_and_sanitized(self):
        self.assertIn('<msup>', mathml('2^{i+1}'))
        self.assertIn('<math', _md_html('distance $2^{i+1}$ from itself.'))
        self.assertNotIn('javascript:', mathml(r'\href{javascript:alert(1)}{x}'))
        self.assertFalse(valid_math('$2^{i$'))
        self.assertFalse(valid_math('incomplete $2^i'))

    def test_models_receive_equation_meaning_without_asset_paths(self):
        image = '![Equation](figs/paper/equation.png)'
        prompt = _summary_prompt('Bound: '+image, (4,), 'English', {'figs/paper/equation.png':'2^{i}'})
        self.assertIn('2^{i}', prompt)
        self.assertNotIn('figs/', prompt)
        self.assertIn('unresolved equation', _summary_prompt(image,(4,),'English'))
        class Client:
            def chat(self, model, prompt, **kwargs):
                self.prompt = prompt
                return 'Bound [[MATH_0001]] and [[MATH_0002]].'
        client=Client()
        translated=_translate_chunk(client,'test',image+' and $2^{i}$.','English', {'figs/paper/equation.png':'2^{i+1}'})
        self.assertIn('$2^{i}$',translated)
        self.assertIn(image,translated)
        self.assertIn('2^{i+1}',client.prompt)
        self.assertIn('"[[MATH_0002]]": "2^{i}"',client.prompt)

    def test_disagreeing_real_source_crops_are_not_promoted_and_cache_tracks_digest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);pdf=root/'source.pdf';math_fixture(pdf)
            (root/'figs/test').mkdir(parents=True)
            Image.new('RGB',(50,30),'white').save(root/'figs/test/equation.png')
            doc={'kids':[{'type':'formula','page number':1,'content':'![Eq](figs/test/equation.png)',
                          'formula image':True,'bounding box':[80,60,110,75]}],
                 'formulas':[{'page number':1,'source':'figs/test/equation.png','bounding box':[80,60,110,75]}]}
            calls=[]
            def request(url,path,body=None,**kwargs):
                if path=='/api/tags':return {'models':[{'name':'glm-ocr:latest','digest':'sha256:model-a'}]}
                calls.append(body)
                return {'response':['$$x^{z}$$','$$x^{2}$$','$$x^{3}$$'][len(calls)-1],'done_reason':'stop'}
            refine_document(pdf,doc,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
            self.assertEqual(doc['math transcriptions'],{})
            self.assertFalse(doc['vision ocr']['regions'][0]['accepted'])
            self.assertEqual(doc['kids'][0]['content'],'![Eq](figs/test/equation.png)')
            self.assertEqual(len(calls),3)
            calls.clear()
            refine_document(pdf,doc,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
            self.assertEqual(calls,[])

    def test_repeated_formula_matching_does_not_ignore_a_changed_operator(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            font=ImageFont.load_default(size=50)
            for label,text in [('a','x+1'),('b','x+1'),('changed','x-1')]:
                image=Image.new('RGB',(200,80),'white');ImageDraw.Draw(image).text((10,10),text,font=font,fill='black')
                image.save(root/(label+'.png'));image.close()
            signatures=[_formula_signature(root/(label+'.png')) for label in ['a','b','changed']]
            try:
                self.assertTrue(_same_short_formula(signatures[0],signatures[1]))
                self.assertFalse(_same_short_formula(signatures[0],signatures[2]))
            finally:
                for signature in signatures:signature[0].close()

    def test_code_html_does_not_double_escape_operators_or_parse_dollar_signs(self):
        from paper_localizer.article import Unit, _unit_html, _unit_markdown
        source='```\n#include <math.h>\nif (x < y) return "$x$";\n```'
        unit=Unit(source,kind='code',pages=(7,))
        result=_unit_html(unit)
        self.assertIn('&lt;math.h&gt;',result)
        self.assertNotIn('&amp;lt;',result)
        self.assertNotIn('<math ',result)
        self.assertIn('$x$',_unit_markdown(unit))

    def test_first_page_funding_note_does_not_interrupt_main_sentence(self):
        document={'kids':[{'type':'paragraph','content':'Nodes receive lookup queries from precisely the same distribution of','page number':1},
                          {'type':'footnote','content':'Research was supported.','page number':1},
                          {'type':'paragraph','content':'nodes in their routing tables.','page number':2}]}
        blocks=extract_blocks(document)
        self.assertIn('distribution of nodes in their routing tables.',blocks[0].text)
        self.assertEqual(blocks[0].source_pages,(1,2))
        self.assertEqual(blocks[-2].text,'## Notes')
        self.assertEqual(make_chunks(blocks)[0].pages,(1,2))


if __name__=='__main__':unittest.main()
