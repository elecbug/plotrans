"""Source-checked semantic regression cases for the user's local example PDFs.

These targeted cases are not an estimate of whole-corpus transcription accuracy.
Run audit_corpus.py first; no model calls are made by this evaluator.
"""

import argparse
from html import escape
import json
import math
from pathlib import Path
import re

from markdown_it import MarkdownIt
from paper_localizer.core import extract_blocks, make_chunks
from paper_localizer.figures import _nodes
from paper_localizer.vision import _math_key
from paper_localizer.markup import FIGURE_LINK,image_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    documents={name:json.loads(next(root.glob('*'+name+'*.audit.json')).read_text('utf-8'))
               for name in ['kademlia','bitcoin','pastry']}
    checks=[]
    def check(label, condition, source):
        checks.append({'case':label,'passed':bool(condition),'source':source})
    nodes=lambda name:list(_nodes(documents[name]))
    content=lambda name:'\n'.join(n.get('content','') for n in nodes(name) if not n.get('suppressed') and not n.get('formula snapshot'))
    k=content('kademlia');b=content('bitcoin')
    check('Kademlia bucket exponent i',re.search(r'2\s*\^\s*\{\s*i\s*\}',k),'Kademlia p.4/p.10')
    check('Kademlia bucket exponent i+1',re.search(r'2\s*\^\s*\{\s*i\s*\+\s*1\s*\}',k),'Kademlia p.4/p.10')
    check('Kademlia recovery probability',re.search(r'1\s*[-−]\s*2\s*\^\s*\{\s*[-−]\s*k\s*\}',k),'Kademlia p.11')
    check('XOR definition reaches text models',any('bitwise exclusive' in n.get('content','') and '\\oplus' in n.get('content','') and not n.get('formula snapshot') for n in nodes('kademlia')),'Kademlia p.4')
    check('Section 4 retained',any(n.get('type')=='heading' and re.match(r'4\s+Implementation notes',n.get('content','')) for n in nodes('kademlia')),'Kademlia p.11')
    check('Figure axis labels owned by figure',not any(n.get('content') in {'Space of 160-bit numbers','11...11','00...00'} and not n.get('suppressed') for n in nodes('kademlia')),'Kademlia p.3')
    blocks=extract_blocks(documents['kademlia']);texts=[block.text for block in blocks]
    check('Funding note moved out of main sentence','## Notes' in texts and next(i for i,s in enumerate(texts) if 'partially supported' in s)>texts.index('## Notes'),'Kademlia p.1→p.2')
    check('Page provenance follows paragraph spans',all(len(chunk.pages)==1 or '\n\n' not in chunk.source for doc in documents.values() for chunk in make_chunks(extract_blocks(doc))),'All three papers; only actual page continuations share a range')
    blocks7=[a for a in documents['bitcoin']['formulas'] if a['page number']==7 and a.get('block')]
    check('Three atomic display equations on p.7',len(blocks7)==3,'Bitcoin p.7; crops visually compared with source')
    code=next((n['content'] for n in nodes('bitcoin') if n.get('type')=='code' and 'AttackerSuccessProbability' in n.get('content','')),'')
    check('C listing remains text',code.startswith('```') and '<img' not in code and 'poisson *= lambda / i;' in code and 'sum -= poisson * (1 - pow(q / p, z - k));' in code,'Bitcoin p.7')
    equations=documents['bitcoin'].get('math transcriptions',{})
    # Crop numbering changes when complete display equations are emitted before
    # inline notation. Identify the source display on page 6 by its role.
    displays={asset['source'] for asset in documents['bitcoin'].get('formulas',[])
              if asset['page number']==6 and asset.get('block')}
    piecewise=next((text for source,text in equations.items() if source in displays),'')
    check('Piecewise exponent is z',re.search(r'\^\s*\{\s*z\s*\}',piecewise) and not re.search(r'\^\s*\{\s*2\s*\}',piecewise),'Bitcoin p.6')
    caption=next(n for n in nodes('pastry') if n.get('type')=='caption' and n.get('caption label')=='Fig. 7')
    check('Fig.7 caption retains three parameter images',caption.get('content','').count('<img')==3 and '5,000 nodes' in caption.get('content',''),'Pastry p.16')
    parameter_text=documents['pastry'].get('math transcriptions',{})
    parameter_sources=[image_reference(match).source for match in FIGURE_LINK.finditer(caption.get('content',''))]
    # Image ordinals change when phantom glyphs or unrelated earlier crops
    # disappear. The caption's three owned regions are the stable evidence.
    for index,(letter,value) in enumerate([('b','4'),('L','16'),('M','32')]):
        text=parameter_text.get(parameter_sources[index],'') if index<len(parameter_sources) else ''
        check(f'Fig.7 {letter}={value} reaches text models',letter in text and value in re.sub(r'\s+','',text),'Pastry p.16')
    probability_rows=[]
    for q,z,expected in [(0.1,1,.2045873),(0.1,6,.0002428),(0.1,10,.0000012),(0.3,5,.1773523),(0.3,20,.0024804),(0.3,50,.0000006)]:
        p=1-q;lam=z*q/p
        actual=1-math.fsum(math.exp(-lam)*lam**i/math.factorial(i)*(1-(q/p)**(z-i)) for i in range(z+1))
        probability_rows.append({'q':q,'z':z,'source':f'{expected:.7f}','recomputed':f'{actual:.7f}'})
        check(f'Probability q={q}, z={z}',f'{actual:.7f}'==f'{expected:.7f}' and f'{expected:.7f}' in b,'Bitcoin p.8; independently recomputed from p.7')
    result={'scope':'Targeted semantic regression checks, not overall accuracy','checks':checks,'probabilities':probability_rows,
            'models':{name: {key:value for key,value in doc.get('vision ocr',{}).items() if key!='regions'} for name,doc in documents.items()},
            'region_counts':{name:{'checked':len(doc.get('vision ocr',{}).get('regions',[])),
                'accepted':sum(r['accepted'] for r in doc.get('vision ocr',{}).get('regions',[])),
                'retained_source':sum(not r['accepted'] for r in doc.get('vision ocr',{}).get('regions',[]))} for name,doc in documents.items()}}
    (root/'semantic-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# PLoTrans semantic regression review','','These source-checked examples evaluate specific failures found during qualitative reading. They do not measure whole-corpus accuracy. Translation and summarization quality require a separate HY-MT2/Qwen inference review.','', '| Check | Result | Original PDF |','| --- | --- | --- |']
    lines += [f"| {row['case']} | {'PASS' if row['passed'] else 'REVIEW REQUIRED'} | {row['source']} |" for row in checks]
    lines += ['', '## OCR consistency checks', '', 'Accepted means that the configured checks passed. It does not mean that a human verified every symbol.', '', '| Paper | Checked regions | Accepted | Source retained |','| --- | ---: | ---: | ---: |']
    lines += [f"| {name} | {row['checked']} | {row['accepted']} | {row['retained_source']} |" for name,row in result['region_counts'].items()]
    lines += ['', '## Independent numerical spot checks', '', '| q | z | Original / extraction | Recomputed |', '| ---: | ---: | ---: | ---: |']
    lines += [f"| {r['q']} | {r['z']} | {r['source']} | {r['recomputed']} |" for r in probability_rows]
    text='\n'.join(lines)+'\n';(root/'semantic-review.md').write_text(text,encoding='utf-8')
    html='<html lang="en"><meta charset="utf-8"><title>PLoTrans semantic review</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#172f43}table{border-collapse:collapse;width:100%}td,th{border:1px solid #b9c7d1;padding:9px;text-align:left}th{background:#edf2f6}</style>'+MarkdownIt().enable('table').render(text)+'</html>'
    (root/'semantic-review.html').write_text(html,encoding='utf-8')
    print(json.dumps({'passed':sum(row['passed'] for row in checks),'total':len(checks),'review_required':[r['case'] for r in checks if not r['passed']]}))


if __name__=='__main__':main()
