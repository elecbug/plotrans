import tempfile
import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from paper_localizer.core import (Block, OllamaClient, ProcessingError, TRANSLATION_MODEL, _translate_chunk,
                                  _translation_context, extract_blocks, make_chunks, process_pdf, render_markdown)


DOCUMENT = {
    "title": "Sample Paper",
    "number of pages": 2,
    "kids": [
        {"type": "heading", "page number": 1, "heading level": 1, "content": "Introduction"},
        {"type": "paragraph", "page number": 1, "content": "The method improved accuracy."},
        {"type": "table", "page number": 2, "rows": [
            {"cells": [
                {"column number": 1, "kids": [{"content": "Model"}]},
                {"column number": 2, "kids": [{"content": "Score"}]},
            ]},
            {"cells": [
                {"column number": 1, "kids": [{"content": "A"}]},
                {"column number": 2, "kids": [{"content": "0.9"}]},
            ]},
        ]},
        {"type": "footer", "page number": 2, "kids": [{"content": "Footer"}]},
    ],
}


class FakeClient:
    def __init__(self):
        self.calls = []

    def check_models(self, *models):
        self.models = models

    def chat(self, model, prompt, *, translation=False):
        self.calls.append((model, translation, prompt))
        assert translation, "Only translation requests are supported."
        return "번역된 내용"


class CoreTests(unittest.TestCase):
    def test_translation_retries_missing_numeric_literals(self):
        class Client:
            def __init__(self):self.results=iter(['지연은 11ms입니다 [7].','지연은 10ms입니다 [7].'])
            def chat(self,*args,**kwargs):return next(self.results)
        self.assertEqual(_translate_chunk(Client(),TRANSLATION_MODEL,'The delay is 10 ms [7].','Korean'),
                         '지연은 10ms입니다 [7].')

    def test_translation_cannot_replace_a_citation_with_a_plain_number(self):
        class Client:
            def __init__(self):self.results=iter(['7개의 피어와 7개의 항목을 사용합니다.','7개의 피어를 사용합니다 [7].'])
            def chat(self,*args,**kwargs):return next(self.results)
        self.assertEqual(_translate_chunk(Client(),TRANSLATION_MODEL,'Use 7 peers [7].','Korean'),
                         '7개의 피어를 사용합니다 [7].')

    def test_numeric_validation_accepts_versions_coordinates_and_thousands_separators(self):
        class Client:
            def chat(self,*args,**kwargs):return 'v1.1 버전은 1,000개의 노드와 (8, 3) 좌표를 사용합니다.'
        self.assertIn('1,000',_translate_chunk(Client(),TRANSLATION_MODEL,
                      'Version v1.1 uses 1000 nodes and coordinate (8,3).','Korean'))

    def test_reference_scope_ends_at_the_next_section(self):
        chunks=make_chunks([Block(1,'heading','## 7. References'),
                            Block(1,'paragraph','[14] A paper. pp. 2142–2160.'),
                            Block(2,'heading','### Additional citations'),
                            Block(2,'paragraph','[15] Another paper.'),
                            Block(3,'heading','## Appendix A'),
                            Block(3,'paragraph','An appendix result.'),
                            Block(4,'heading','## Notes'),Block(4,'paragraph','A source note.')])
        self.assertTrue(chunks[0].reference_section)
        self.assertTrue(chunks[1].reference_section)
        self.assertTrue(all(not c.reference_section for c in chunks[2:]))

    def test_reference_pages_are_never_sent_to_the_translation_model(self):
        document={'kids':[{'type':'heading','heading level':2,'content':'References'},
                          {'type':'paragraph','content':'[14] A paper. pp. 2142–2160.'}]}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'paper.pdf';source.write_bytes(b'%PDF-test')
            client=FakeClient();result=process_pdf(source,Path(directory)/'paper.md',parser=lambda p:document,client=client)
            self.assertEqual(client.calls,[])
            self.assertIn('pp. 2142–2160.',result.read_text())
            self.assertIn('2142–2160',result.with_suffix('.html').read_text())
            audit=json.loads(result.with_suffix('.translation.json').read_text())
            self.assertEqual(audit['reference_policy'],'preserve-original')
            self.assertTrue(audit['reference_chunks'])

    def test_extraction_and_chunking_keep_pages_and_tables(self):
        blocks = extract_blocks(DOCUMENT)
        self.assertEqual([block.page for block in blocks], [1, 1, 2])
        self.assertIn("| Model | Score |", blocks[2].text)
        self.assertNotIn("Footer", " ".join(block.text for block in blocks))
        chunks = make_chunks(blocks, 200)
        self.assertEqual([chunk.pages for chunk in chunks], [(1,), (2,)])

    def test_pipeline_writes_grounded_aligned_markdown(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "paper.pdf"
            source.write_bytes(b"%PDF-fake test input")
            output = Path(directory) / "result.md"
            client = FakeClient()
            process_pdf(source, output, parser=lambda path: DOCUMENT, client=client)
            markdown = output.read_text(encoding="utf-8")
            self.assertNotIn("## Article summary", markdown)
            self.assertNotIn('summary-card',output.with_suffix('.html').read_text())
            self.assertEqual(client.models,(TRANSLATION_MODEL,))
            self.assertTrue(all(model==TRANSLATION_MODEL and translation for model,translation,_ in client.calls))
            self.assertIn("The method improved accuracy.", markdown)
            self.assertIn("번역된 내용", markdown)
            self.assertIn("<!-- p.1 -->", markdown)
            self.assertIn("<!-- p.2 -->", markdown)
            self.assertNotIn("p.1, 2", markdown)
            self.assertFalse(output.with_suffix(".progress.json").exists())
            self.assertEqual(client.models[0], "kaelri/hy-mt2:7b-q4_K_M")

    def test_generic_pdf_metadata_title_uses_first_heading(self):
        document = dict(DOCUMENT, title="paper.dvi")
        result = render_markdown(document, "paper.pdf", [], [], "Korean")
        self.assertTrue(result.startswith("# Introduction\n"))

    def test_ollama_http_payload(self):
        requests = []

        def fake_open(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/api/tags"):
                return BytesIO(b'{"models":[{"name":"translate"}]}')
            return BytesIO(b'{"message":{"content":"translated"}}')

        with patch("paper_localizer.core.urllib.request.urlopen", side_effect=fake_open):
            client = OllamaClient()
            client.check_models("translate")
            self.assertEqual(client.chat("translate", "source", translation=True), "translated")
        self.assertIn(b'"think": false', requests[1].data)
        self.assertIn(b'"num_predict": 4096', requests[1].data)

    def test_incomplete_generation_is_not_saved_as_a_translation(self):
        from paper_localizer.core import ProcessingError
        for translation in [False,True]:
            with self.subTest(translation=translation), patch.object(OllamaClient,'_request',return_value={
                    'done_reason':'length','message':{'content':'A plausible but incomplete answer'}}):
                with self.assertRaisesRegex(ProcessingError,'generation limit'):
                    OllamaClient().chat('model','source',translation=translation)

    def test_dense_mt2_uses_verified_native_tokens_instead_of_a_legacy_template(self):
        requests=[]
        def request(path,payload=None):
            requests.append((path,payload))
            if path=='/api/show':
                return {'model_info':{'general.architecture':'hunyuan-dense',
                        'tokenizer.ggml.tokens':['<|startoftext|>','<|extra_0|>','<|eos|>']}}
            return {'response':'각 노드는 라우팅 정보를 유지한다.','done_reason':'stop'}
        client=OllamaClient()
        with patch.object(client,'_request',side_effect=request):
            for _ in range(2):
                self.assertEqual(client.chat(TRANSLATION_MODEL,'Translate this.',translation=True),'각 노드는 라우팅 정보를 유지한다.')
        self.assertEqual([p for p,_ in requests],['/api/show','/api/generate','/api/generate'])
        payload=requests[1][1]
        self.assertTrue(payload['raw'])
        self.assertEqual(payload['prompt'],'<|startoftext|>Translate this.<|extra_0|>')
        self.assertNotIn('messages',payload)
        self.assertEqual(payload['options']['num_ctx'],8192)

    def test_dense_mt2_without_native_token_evidence_is_rejected(self):
        client=OllamaClient()
        with patch.object(client,'_request',return_value={'model_info':{
                'general.architecture':'hunyuan-dense','tokenizer.ggml.tokens':['<｜hy_User｜>']}}) as request:
            with self.assertRaisesRegex(ProcessingError,'tokens could not be verified'):
                client.chat(TRANSLATION_MODEL,'source',translation=True)
        self.assertEqual(request.call_count,1)

    def test_other_mt2_architectures_keep_their_own_chat_template(self):
        client=OllamaClient()
        with patch.object(client,'_request',side_effect=[{'model_info':{'general.architecture':'qwen2'}},
                {'message':{'content':'번역'}}]) as request:
            self.assertEqual(client.chat('example/hy-mt2:small','source',translation=True),'번역')
        self.assertEqual(request.call_args[0][0],'/api/chat')

    def test_translation_retries_replacement_characters_and_invented_math_markers(self):
        class Client:
            def __init__(self,first):self.values=iter([first,'정상 번역']);self.calls=0
            def chat(self,*args,**kwargs):self.calls+=1;return next(self.values)
        for first in ['손상\ufffd','잘못된 [[MATH_0001]]']:
            client=Client(first)
            self.assertEqual(_translate_chunk(client,'model','Source.','Korean'),'정상 번역')
            self.assertEqual(client.calls,2)

    def test_terminology_context_contains_source_context_without_asset_paths(self):
        doc={'title':'Peer-to-peer routing','kids':[{'type':'abstract','content':
             'Nodes leave the network. <img src="figs/paper/math.png" alt="Eq" class="inline-math"> $n^2$ hops.'}]}
        context=_translation_context(doc,'paper.pdf')
        self.assertIn('Peer-to-peer routing',context)
        self.assertNotIn('Nodes leave the network.',context)
        self.assertNotIn('figs/',context)
        self.assertNotIn('$n^2$',context)

    def test_network_context_is_terminology_without_abstract_claims(self):
        doc={'title':'Peer-to-peer routing','kids':[{'type':'abstract','content':
             'The method reduces delay by 99.9% and improves delivery.'}]}
        context=_translation_context(doc,'paper.pdf')
        self.assertIn('Churn refers to network nodes',context)
        self.assertNotIn('99.9',context)

    def test_table_translation_cannot_change_spans_or_move_numeric_cells(self):
        from paper_localizer.core import _translate_native_table
        from copy import deepcopy
        rows=[[{'text':'Metric','rowspan':1,'colspan':1,'header':True},
               {'text':'Result','rowspan':1,'colspan':2,'header':True}],
              [{'text':'Delivery','rowspan':1,'colspan':1,'header':False},
               {'text':'99.9%','rowspan':1,'colspan':1,'header':False},
               {'text':'$n^2$','rowspan':1,'colspan':1,'header':False}],
              [{'text':'Delivery','rowspan':1,'colspan':1,'header':False},
               {'text':'34×','rowspan':1,'colspan':1,'header':False},
               {'text':'GS(8,3)','rowspan':1,'colspan':1,'header':False}]]
        original=deepcopy(rows)
        class Client:
            def __init__(self):self.prompts=[]
            def chat(self,model,prompt,**kwargs):self.prompts.append(prompt);return '번역된 셀'
        client=Client();result=_translate_native_table(client,'model',rows,'Korean')
        self.assertEqual(rows,original)
        self.assertEqual(len(client.prompts),1)
        self.assertEqual([cell['text'] for cell in result[0]],['지표','결과'])
        for row,target in zip(rows,result):
            for cell,translated in zip(row,target):
                self.assertEqual({k:v for k,v in cell.items() if k!='text'},
                                 {k:v for k,v in translated.items() if k!='text'})
        self.assertEqual(result[1][1]['text'],'99.9%')
        self.assertEqual(result[1][2]['text'],'$n^2$')
        self.assertEqual(result[2][1]['text'],'34×')
        self.assertEqual(result[2][2]['text'],'GS(8,3)')

    def test_paper_context_and_equation_context_are_both_passed_to_translation(self):
        class Client:
            def chat(self,model,prompt,**kwargs):self.prompt=prompt;return '계산 [[MATH_0001]]'
        client=Client()
        _translate_chunk(client,'model','Calculation $n^2$.','Korean',context='Peer-to-peer networking')
        self.assertIn('Peer-to-peer networking',client.prompt)
        self.assertIn('"[[MATH_0001]]": "n^2"',client.prompt)
        self.assertTrue(client.prompt.endswith('Calculation [[MATH_0001]].'))
        self.assertLess(client.prompt.index('Local OCR equation context'),client.prompt.index('Translate the following text'))

    def test_column_identifiers_under_a_spanning_header_keep_their_names(self):
        from paper_localizer.core import _translate_native_table
        rows=[[{'text':'Metric','rowspan':2,'colspan':1,'header':True},
               {'text':'Protocol family','rowspan':1,'colspan':2,'header':True}],
              [{'text':'Pastry','rowspan':1,'colspan':1,'header':False},
               {'text':'Viceroy','rowspan':1,'colspan':1,'header':False}],
              [{'text':'Delivery','rowspan':1,'colspan':1,'header':False},
               {'text':'99.9%','rowspan':1,'colspan':1,'header':False},
               {'text':'99.8%','rowspan':1,'colspan':1,'header':False}]]
        class Client:
            def __init__(self):self.prompts=[]
            def chat(self,model,prompt,**kwargs):self.prompts.append(prompt);return '번역된 값'
        client=Client();target=_translate_native_table(client,TRANSLATION_MODEL,rows,'Korean')
        self.assertEqual(target[1],rows[1])
        self.assertEqual(len(client.prompts),2)
        self.assertEqual(target[0][0]['text'],'지표')
        self.assertFalse(any('Pastry' in p or 'Viceroy' in p for p in client.prompts))

    def test_dedicated_mt2_receives_no_free_form_context_or_marker_glossary(self):
        class Client:
            def chat(self,model,prompt,**kwargs):self.prompt=prompt;return '계산 [[MATH_0001]]'
        client=Client()
        _translate_chunk(client,TRANSLATION_MODEL,'Calculation $n^2$.','Korean',context='Paper title: Context to omit')
        self.assertNotIn('Context to omit',client.prompt)
        self.assertNotIn('Local OCR equation context',client.prompt)
        self.assertNotIn('Required unchanged placeholders',client.prompt)
        self.assertEqual(client.prompt.count('[[MATH_0001]]'),1)
        self.assertTrue(client.prompt.endswith('Calculation [[MATH_0001]].'))

    def test_failed_translation_retains_source_and_exports_a_location_report(self):
        class Client(FakeClient):
            def chat(self,model,prompt,*,translation=False):
                if translation:return '손상\ufffd'
                return super().chat(model,prompt,translation=translation)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'paper.pdf';source.write_bytes(b'%PDF-test')
            result=process_pdf(source,Path(directory)/'paper.md',parser=lambda path:DOCUMENT,client=Client())
            report=json.loads(result.with_suffix('.translation.json').read_text())
            self.assertEqual(report['issues'][0]['kind'],'excerpt')
            self.assertEqual(report['issues'][0]['pages'],[1])
            self.assertIn('The method improved accuracy.',result.read_text())
            self.assertIn('retained 1 source excerpt(s)',result.with_suffix('.html').read_text())
            self.assertNotIn('\ufffd',result.read_text())
            from zipfile import ZipFile
            with ZipFile(result.with_suffix('.zip')) as archive:
                self.assertIn('paper.translation.json',archive.namelist())

    def test_invalid_table_cell_retains_only_that_cell(self):
        from paper_localizer.core import _translate_native_table
        class Client:
            def chat(self,model,prompt,**kwargs):
                return '손상\ufffd' if 'Bad value' in prompt else '정상 값'
        rows=[[{'text':text,'rowspan':1,'colspan':1,'header':False} for text in ['Bad value','Good value','3.14']]]
        failures=[]
        result=_translate_native_table(Client(),TRANSLATION_MODEL,rows,'Korean',failures=failures)
        self.assertEqual([c['text'] for c in result[0]],['Bad value','정상 값','3.14'])
        self.assertEqual((failures[0]['kind'],failures[0]['row'],failures[0]['cell']),('table-cell',1,1))

    def test_connection_failures_do_not_become_retained_source_results(self):
        class Client(FakeClient):
            def chat(self,*args,**kwargs):raise ProcessingError('Connection failed')
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'paper.pdf';source.write_bytes(b'%PDF-test')
            output=Path(directory)/'paper.md'
            with self.assertRaisesRegex(ProcessingError,'Connection failed'):
                process_pdf(source,output,parser=lambda path:DOCUMENT,client=Client())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.translation.json').exists())

    def test_split_tuple_delimiters_cannot_be_dropped_individually(self):
        class Client:
            def chat(self,model,prompt,**kwargs):self.prompt=prompt;return '[[MATH_0001]] 쌍을 저장한다.'
        client=Client();source=r'Store $\langle$ key, value $\rangle$ pairs.'
        result=_translate_chunk(client,'model',source,'Korean')
        self.assertEqual(result,r'$\langle$ key, value $\rangle$ 쌍을 저장한다.')
        self.assertTrue(client.prompt.endswith('Store [[MATH_0001]] pairs.'))
        self.assertNotIn('[[MATH_0002]]',client.prompt)

    def test_checkpoint_resumes_only_with_the_same_model_digest(self):
        for version in ['old','new']:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'paper.pdf';source.write_bytes(b'%PDF-test')
                output=Path(directory)/'paper.md'
                from copy import deepcopy
                document=deepcopy(DOCUMENT)
                document['kids'][2]['rows'][1]['cells'][0]['kids'][0]['content']='Sample method'
                class Interrupted(FakeClient):
                    def chat(self,model,prompt,*,translation=False):
                        if self.calls:raise ProcessingError('interrupted')
                        return super().chat(model,prompt,translation=translation)
                first=Interrupted();first.model_digests={TRANSLATION_MODEL:'old'}
                with self.assertRaisesRegex(ProcessingError,'interrupted'):
                    process_pdf(source,output,parser=lambda path:document,client=first)
                self.assertTrue(output.with_suffix('.progress.json').exists())
                second=FakeClient();second.model_digests={TRANSLATION_MODEL:version}
                process_pdf(source,output,parser=lambda path:document,client=second)
                repeated=any(translation and 'The method improved accuracy.' in prompt
                             for _,translation,prompt in second.calls)
                self.assertEqual(repeated,version=='new')
                self.assertFalse(output.with_suffix('.progress.json').exists())


if __name__ == "__main__":
    unittest.main()
