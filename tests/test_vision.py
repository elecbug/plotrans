from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw, ImageFont

from paper_localizer.article import _md_html
from paper_localizer.core import _translate_chunk, extract_blocks, make_chunks
from paper_localizer.math_markup import mathml, valid_math
from paper_localizer.ocr import _math_line
from paper_localizer.vision import (_math_key, _prose_agrees, _prose_consistent,
    _formula_signature, _same_short_formula, refine_document, VisionOcrError)
from test_formulas import math_fixture
from test_structure import fixture_pdf


class VisionTests(unittest.TestCase):
    def test_snapshot_ocr_cannot_omit_native_prose_outside_its_original_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root/'source.pdf'
            preceding = 'We begin by defining conditions for miner stability.'
            heading = 'Proposition One describes the stable network topology.'
            fixture_pdf(pdf, [b'BT /F2 10 Tf 30 520 Td ('+preceding.encode()+b') Tj ET '
                              b'BT /F2 10 Tf 30 500 Td ('+heading.encode()+b') Tj ET'])
            source = 'figs/source/paragraph.png'
            (root/source).parent.mkdir(parents=True)
            Image.new('RGB',(500,80),'white').save(root/source)
            box = [29,495,400,530]
            document = {'kids':[{'type':'paragraph','page number':1,'font':'Times-Roman',
                                'content':heading,'bounding box':box,
                                'formula snapshot':f'![Source]({source})'}],
                        'formulas':[{'source':source,'page number':1,'bounding box':box,'block':True}]}
            def request(url,path,body=None,**kwargs):
                return ({'models':[{'name':'glm-ocr:latest','digest':'model'}]} if path=='/api/tags'
                        else {'response':heading,'done_reason':'stop'})
            refine_document(pdf,document,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
            region = document['vision ocr']['regions'][0]
            self.assertFalse(region['accepted'])
            self.assertFalse(region['native_crop_prose_preserved'])
            self.assertIn('complete source crop',region['reason'])
            self.assertIn('formula snapshot',document['kids'][0])
            self.assertEqual(document['vision sources'],[])

    def test_agreeing_model_responses_cannot_add_source_absolute_value_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root/'source.pdf'
            fixture_pdf(pdf,[b'BT /F1 12 Tf 30 500 Td (x+y) Tj ET'])
            source = 'figs/source/expression.png'
            (root/source).parent.mkdir(parents=True)
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(pdf) as pages:
                page = pages[0]
                w,h = page.get_size()
                bitmap = page.render(scale=2,crop=(29,495,w-80,h-515))
                bitmap.to_pil().save(root/source)
                bitmap.close()
            doc = {'kids':[{'type':'formula','page number':1,'content':f'![Eq]({source})',
                            'formula image':True,'bounding box':[29,495,80,515]}],
                   'formulas':[{'page number':1,'source':source,'bounding box':[29,495,80,515]}]}
            def request(url,path,body=None,**kwargs):
                return ({'models':[{'name':'glm-ocr:latest','digest':'model'}]} if path=='/api/tags' else
                        {'response':r'\left|x+y\right|','done_reason':'stop'})
            with patch('paper_localizer.vision.native_expression',return_value=None):
                refine_document(pdf,doc,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
            self.assertEqual(doc['math transcriptions'],{})
            detail = doc['vision ocr']['regions'][0]
            self.assertTrue(detail['native_operands_agree'])
            self.assertFalse(detail['native_symbols_agree'])
            self.assertFalse(detail['accepted'])
            self.assertIn('image retained',detail['reason'])

    def test_native_prose_preserves_accented_author_names(self):
        from paper_localizer.vision import _native_prose_preserved
        source = 'The authors Díaz and Erdős describe the complete protocol.'
        self.assertTrue(_native_prose_preserved(source,source))
        self.assertFalse(_native_prose_preserved(source,source.replace('Díaz','Níaz')))
        self.assertFalse(_native_prose_preserved(source,source.replace('Erdős','Erdos')))

    def equation_document(self,root):
        pdf = root/'source.pdf'
        math_fixture(pdf)
        source = 'figs/test/equation.png'
        (root/source).parent.mkdir(parents=True)
        Image.new('RGB',(50,30),'white').save(root/source)
        return pdf, {'kids':[{'type':'formula','page number':1,'content':f'![Eq]({source})',
                              'formula image':True,'bounding box':[80,60,110,75]}],
                     'formulas':[{'page number':1,'source':source,'bounding box':[80,60,110,75]}]}

    def test_generation_limit_reuses_unchanged_failure_then_retries_after_expiry(self):
        from paper_localizer.vision import _LIMIT_CACHE_SECONDS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf,original = self.equation_document(root)
            calls = []
            digest = 'original-model'
            def request(url,path,body=None,**kwargs):
                if path=='/api/tags':
                    return {'models':[{'name':'glm-ocr:latest','digest':digest}]}
                calls.append(body['options'])
                return {'response':'unfinished transcription','done_reason':'length','eval_count':4096}
            def run(now):
                doc = deepcopy(original)
                with patch('paper_localizer.vision.time.time',return_value=now):
                    refine_document(pdf,doc,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
                self.assertFalse(doc['vision ocr']['regions'][0]['accepted'])
                self.assertEqual(doc['kids'][0]['content'],original['kids'][0]['content'])
                return doc
            run(1000)
            self.assertEqual(len(calls),1)
            cached = run(1001)
            self.assertEqual(len(calls),1)
            self.assertIn('cached',cached['vision ocr']['regions'][0]['reason'])
            run(1000+_LIMIT_CACHE_SECONDS+1)
            self.assertEqual(len(calls),2)
            digest = 'updated-model'
            run(1002)
            self.assertEqual(len(calls),3)
            self.assertTrue(list((root/'.vision-cache').rglob('*.failed.json')))

    def test_network_and_empty_responses_remain_retryable(self):
        for failure in ['network','empty']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf,original = self.equation_document(root)
                calls = []
                def request(url,path,body=None,**kwargs):
                    if path=='/api/tags':
                        return {'models':[{'name':'glm-ocr:latest','digest':'model'}]}
                    calls.append(path)
                    if failure=='network':
                        raise VisionOcrError('Temporary connection error')
                    return {'response':'','done_reason':'stop'}
                for _ in range(2):
                    refine_document(pdf,deepcopy(original),root,model='glm-ocr:latest',
                                    url='http://localhost:11434',requester=request)
                self.assertEqual(len(calls),4 if failure=='network' else 2)
                self.assertFalse(list((root/'.vision-cache').rglob('*.failed.json')))

    def test_incomplete_region_response_preserves_source_and_records_failure(self):
        for response in ({"response": "$$x", "done_reason": "length"},
                         {"response": "", "done_reason": "stop"},
                         {"response": None, "done_reason": "stop"}):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); pdf = root / "source.pdf"; math_fixture(pdf)
                source = "figs/test/equation.png"
                (root / source).parent.mkdir(parents=True)
                Image.new("RGB", (50, 30), "white").save(root / source)
                document = {"kids": [{"type": "formula", "page number": 1,
                    "content": f"![Eq]({source})", "formula image": True,
                    "bounding box": [80, 60, 110, 75]}],
                    "formulas": [{"page number": 1, "source": source,
                                  "bounding box": [80, 60, 110, 75]}]}
                def request(url, path, body=None, **kwargs):
                    return ({"models": [{"name": "glm-ocr:latest", "digest": "model"}]}
                            if path == "/api/tags" else response)
                refine_document(pdf, document, root, model="glm-ocr:latest",
                                url="http://localhost:11434", requester=request)
                self.assertEqual(document["math transcriptions"], {})
                self.assertEqual(document["kids"][0]["content"], f"![Eq]({source})")
                self.assertFalse(document["vision ocr"]["regions"][0]["accepted"])
                self.assertIn("retained", document["vision ocr"]["regions"][0]["reason"])
                self.assertTrue((root / source).is_file())

    def test_table_captioned_code_listing_is_not_transcribed_as_a_data_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pdf=root/'source.pdf';math_fixture(pdf)
            document={'kids':[], 'figures':[{'page number':1,'source':'figs/test/code.png',
                      'asset kind':'table','source listing':True,'bounding box':[30,60,180,120]}]}
            calls=[]
            def request(url,path,body=None,**kwargs):
                calls.append(path)
                if path!='/api/tags':
                    self.fail('Code listing was sent to tabular OCR.')
                return {'models':[{'name':'glm-ocr:latest','digest':'model'}]}
            refine_document(pdf,document,root,model='glm-ocr:latest',url='http://localhost:11434',requester=request)
            self.assertEqual(calls,['/api/tags'])
            self.assertFalse(document['table transcriptions'])
            self.assertIn('code listing retained',document['vision ocr']['regions'][0]['reason'])

    def test_table_consensus_still_requires_independent_numeric_support(self):
        for verified in (True, False):
            with self.subTest(verified=verified), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); pdf = root / "source.pdf"; math_fixture(pdf)
                source = "figs/test/table.png"
                document = {"kids": [], "figures": [{"page number": 1, "source": source,
                    "asset kind": "table", "bounding box": [30, 60, 180, 120]}]}
                def request(url, path, body=None, **kwargs):
                    return ({"models": [{"name": "glm-ocr:latest", "digest": "model"}]}
                            if path == "/api/tags" else {"response":
                                "<table><tr><th>Method</th><th>Value</th></tr>"
                                "<tr><td>Example</td><td>12.5</td></tr></table>", "done_reason": "stop"})
                with patch("paper_localizer.vision._verify_table_image",
                           return_value={"accepted": verified, "numeric_values_checked": 1}) as verify:
                    refine_document(pdf, document, root, model="glm-ocr:latest",
                                    url="http://localhost:11434", requester=request)
                verify.assert_called_once()
                self.assertEqual(bool(document["table transcriptions"]), verified)
                self.assertEqual(document["vision ocr"]["regions"][0]["accepted"], verified)

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

    def test_generic_translation_receives_equation_meaning_without_asset_paths(self):
        image = '![Equation](figs/paper/equation.png)'
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
