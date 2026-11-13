"""Author-sourced fixed-seed cohorts; gold fields never enter the model view."""
from pathlib import Path
from collections import Counter
import hashlib,json,random
import numpy as np
import pandas as pd
from .protocol import normalize_answer
ROOT=Path(__file__).resolve().parents[2];PILOT=ROOT/'experiments/llm_extension/pilot'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_hotpot(row):
    context=row['context'];parts=[]
    for title,sentences in zip(context['title'],context['sentences']):
        parts.append(str(title)+'\n'+''.join(str(s) for s in sentences))
    return 'Answer the question using the supplied ten distractor-setting paragraphs.\nQuestion: '+row['question']+'\n\nContext:\n\n'+'\n\n'.join(f'[{i+1}] {v}' for i,v in enumerate(parts))

def model_view(row):
    return {'question':row['model_input'],'answer_kind':row['answer_kind'],'item_id':row['item_id'],'dataset':row['dataset']}

def answer_f1(prediction,gold):
    p=normalize_answer(prediction,'text');g=normalize_answer(gold,'text')
    if p is None or g is None:return 0.
    if p in {'yes','no','noanswer'} or g in {'yes','no','noanswer'}:return float(p==g)
    pt=p.split();gt=g.split();common=sum((Counter(pt)&Counter(gt)).values())
    if not common:return 0.
    precision=common/len(pt);recall=common/len(gt);return 2*precision*recall/(precision+recall)

def build_cohort(output_dir=PILOT,seed=20261113):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    gsm_path=ROOT/'experiments/llm_extension/test.jsonl';hot_path=PILOT/'source_data/hotpotqa_validation.parquet';bbh_path=PILOT/'source_data/bbh_logical_deduction_three_objects.json'
    gsm=[json.loads(s) for s in gsm_path.read_text().splitlines() if s.strip()];hot=pd.read_parquet(hot_path).to_dict('records');bbh=json.loads(bbh_path.read_text())['examples']
    specs=[('gsm8k',gsm,gsm_path,'numeric','official_test'),('hotpotqa',hot,hot_path,'text','distractor_validation'),('bbh',bbh,bbh_path,'choice','logical_deduction_three_objects')];rows=[];sources={}
    for offset,(name,data,path,kind,split) in enumerate(specs):
        indices=random.Random(seed+offset).sample(range(len(data)),50);source_hash=digest(path);sources[name]={'path':str(path.relative_to(ROOT)),'sha256':source_hash,'source_rows':len(data),'selected_rows':50,'seed':seed+offset,'indices':indices,'split_or_task':split}
        for ordinal,index in enumerate(indices):
            raw=data[index]
            if name=='gsm8k':
                question=raw['question'];gold=raw['answer'].rsplit('####',1)[-1].strip();source_id=str(index)
            elif name=='hotpotqa':question=render_hotpot(raw);gold=raw['answer'];source_id=str(raw['id'])
            else:question=raw['input'];gold=raw['target'];source_id=str(index)
            canonical=normalize_answer(gold,kind)
            if canonical is None:raise ValueError(f'Invalid gold normalization {name} {index}')
            rows.append({'dataset':name,'item_id':f'{name}:{split}:{source_id}','source_index':index,'source_id':source_id,'sample_ordinal':ordinal,'sampling_seed':seed+offset,'answer_kind':kind,'model_input':question,'gold_answer':canonical,'gold_original':gold,'source_sha256':source_hash})
    if len(rows)!=150 or len({x['item_id'] for x in rows})!=150:raise ValueError('Pilot must be exactly150 unique rows')
    payload=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows);cohort_path=out/'cohort.jsonl'
    if cohort_path.exists() and cohort_path.read_text()!=payload:raise RuntimeError('Refusing to replace a frozen cohort')
    cohort_path.write_text(payload);manifest={'scope':'pilot_only','examples_per_dataset':50,'total_examples':150,'base_seed':seed,'cohort_sha256':hashlib.sha256(payload.encode()).hexdigest(),'sources':sources,'notes':['Exact random sample without replacement, no filtering by answer or difficulty','GSM8K test sample; HotpotQA distractor validation sample; BBH one task only','Reference answers and supporting_fact labels excluded from model_view; supplied QA passages naturally contain answer evidence','Provider sampling seed is not set because requested gpt5mini API does not expose a verified supported seed; only cohort/order/controller randomness is fixed']};(out/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows

if __name__=='__main__':
    rows=build_cohort();print('Frozen150 examples:50 each GSM8K,HotpotQA,BBH logical deduction')
