"""Prepare, validate and execute exactly150 approved three-dataset MAS pilot items."""
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,os,random,time
from datetime import datetime,timezone
from .datasets import build_cohort,model_view,answer_f1
from .protocol import ProtocolConfig,METHODS,build_initial_payload,build_refinement_payload,build_adjudication_payload,parse_candidate,choose_edges,communication_record
from .transport import AzureResponsesTransport,TransportError
from .ledger import Ledger,BudgetExceeded
ROOT=Path(__file__).resolve().parents[2];PILOT=ROOT/'experiments/llm_extension/pilot';BASE='https://tiwarishrayansh-0806-resource.services.ai.azure.com/openai/v1/'

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(value):return hashlib.sha256(value.encode()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def code_hash():
    # Read-only analysis/report changes cannot alter the frozen inference protocol.
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')) if p.name not in {'analysis.py','report.py'})).hexdigest()
def input_bound(payload):
    return len(json.dumps(payload['input'],ensure_ascii=False).encode())+len(payload.get('instructions','').encode())+512

def prepare():
    rows=build_cohort();source=code_hash();manifest_path=PILOT/'run_manifest.json'
    if (PILOT/'budget.sqlite3').exists():
        old=json.loads(manifest_path.read_text())
        if old['source_hash']!=source:raise RuntimeError('Cannot change source after starting pilot')
    bounds=[];configs={}
    for row in rows:
        config=ProtocolConfig(answer_kind=row['answer_kind']);configs[row['dataset']]=config.to_dict();q=model_view(row)['question'];answer={'numeric':'123456789','text':'X'*200,'choice':'A'}[row['answer_kind']];candidates=[{'answer':answer,'summary':'s'*600,'valid':True} for _ in range(3)]
        for method in METHODS:
            edges=choose_edges(method,candidates,row['item_id'],config)['edges'];payloads=[build_initial_payload(q,j,config) for j in range(3)]+[build_refinement_payload(q,j,candidates,edges,config) for j in range(3)]+[build_adjudication_payload(q,candidates,config)]
            bounds.extend(input_bound(p) for p in payloads)
    max_bound=max(bounds)
    if max_bound>32768:raise ValueError('Frozen task prompts exceed32768-byte conservative input gate')
    data={'scope':'pilot_only','dataset_counts':{d:50 for d in ['gsm8k','hotpotqa','bbh']},'examples':150,'methods':list(METHODS),'agents':3,'logical_calls_per_method_item':7,'logical_calls':5250,'provider':{'base_url':BASE,'deployment':'gpt-5-mini','verified_model_version':'2025-08-07','sku':'GlobalStandard','region':'koreacentral'},'cohort_sha256':json.loads((PILOT/'cohort_manifest.json').read_text())['cohort_sha256'],'source_hash':source,'protocols':configs,'seed':20261113,'provider_seed_supported':False,'max_output_tokens':4096,'max_input_token_bound':32768,'max_http_attempts':15750,'rpm':200,'tpm':200000,'prices_usd_per_million':{'input':.25,'output':2},'expected_tokens_per_call':{'input':2000,'output_including_reasoning':800},'expected_cost_usd':11.025,'proposed_cap_usd':25,'prompt_byte_bound_scenario':{'min':min(bounds),'max':max_bound,'sum_for_5250_calls':sum(bounds),'note':'Offline bound scenario uses200-character text answers and600-character summaries. Real responses are checked before every call; this is not an exact measured token workload or unconditional worst-case bound.'},'cost_ceiling_one_attempt_at_global_bounds_usd':5250*(32768*.25+4096*2)/1e6,'scientific_status':'Exploratory150-example pilot; no physicalphase mapping qualification claimed; one real API realization per logical call','robustness_conditions':'None in this pilot. Dropout/corruption/delay sweeps require a separate scope and approval.'}
    data['manifest_sha256']=sha(canonical(data));atomic(manifest_path,data)
    atomic(PILOT/'offline_validation.json',{'passed':True,'model_requests_sent':0,'items':150,'payloads_built':len(bounds),'maximum_prompt_byte_bound':max_bound,'frozen_seed':20261113,'dataset_gold_isolation':'build payload accepts model_view question only; gold is read only in local scorer','not_results':'Synthetic candidate summaries used only to validate payload construction; no generated answers or accuracy estimates.'})
    return data

def load_approval(path,manifest):
    approved=json.loads(Path(path).read_text())
    if not approved.get('approved') or approved.get('approved_by')!='user' or approved.get('manifest_sha256')!=manifest['manifest_sha256']:
        raise ValueError('An explicit user approval bound to this frozen run manifest is required')
    if not isinstance(approved.get('max_usd'),(int,float)) or not 0<approved['max_usd']<=1000:
        raise ValueError('A finite explicit spending cap is required')
    return approved

def key_loader():
    value=os.environ.get('AZURE_OPENAI_API_KEY')
    if value:return value.strip()
    p=Path('/tmp/stochastic-pll-azure-key')
    if p.stat().st_mode&0o077:raise ValueError('External credential file must be mode0600')
    return p.read_text().strip()

def step_call(payload,logical_id,path,transport):
    fingerprint=sha(canonical(payload))
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['payload_hash']!=fingerprint:raise RuntimeError('Resume payload hash mismatch')
        return saved['result']
    try:
        result=transport.request(payload['input'],instructions=payload['instructions'],logical_id=logical_id,reasoning_effort=payload['reasoning']['effort'],input_token_bound=input_bound(payload))
    except BudgetExceeded:raise
    except TransportError as error:
        result=error.result or {};result.update(outcome=error.outcome,error_type=type(error).__name__,text=result.get('text',''))
    except ValueError as error:
        result={'outcome':'input_validation_failed','error_type':type(error).__name__,'error':str(error),'text':'','usage':None,'cost_usd':0.,'attempt_count':0,'latency_seconds':0.}
    atomic(path,{'payload_hash':fingerprint,'logical_id':logical_id,'result':result});return result

def run_condition(row,method,transport,output_dir=PILOT/'artifacts',source=None):
    source=source or code_hash();config=ProtocolConfig(answer_kind=row['answer_kind']);cid=f"{row['dataset']}__{row['source_index']}__{method}";folder=Path(output_dir)/'conditions'/cid;folder.mkdir(parents=True,exist_ok=True);resultpath=folder/'result.json';fingerprint=sha(canonical({'source':source,'row':row,'method':method,'config':config.to_dict()}))
    if resultpath.exists():
        result=json.loads(resultpath.read_text())
        if result['fingerprint']!=fingerprint:raise RuntimeError('Condition checkpoint changed')
        return result
    q=model_view(row)['question'];started=time.monotonic();calls=[];errors=[]
    def call(stage,index,payload):
        lid=f"{row['item_id']}|{method}|{stage}:{index}";raw=step_call(payload,lid,folder/f'{stage}_{index}.json',transport);calls.append((lid,raw));candidate=parse_candidate(raw.get('text',''),config.answer_kind) if raw.get('outcome')=='completed' else {'answer':None,'summary':'','valid':False}
        if raw.get('outcome')!='completed' or not candidate['valid']:errors.append({'stage':stage,'agent':index,'outcome':raw.get('outcome'),'valid_json_candidate':candidate['valid']})
        return candidate
    initial=[call('initial',i,build_initial_payload(q,i,config)) for i in range(3)];control=choose_edges(method,initial,row['item_id'],config);edges=control['edges'];communication=communication_record(initial,edges,config);refined=[call('refinement',i,build_refinement_payload(q,i,initial,edges,config)) for i in range(3)];final=call('adjudication',0,build_adjudication_payload(q,refined,config));correct=bool(final['valid'] and final['answer']==row['gold_answer'])
    usage={'input_tokens':0,'output_tokens':0,'reasoning_tokens':0};cost=0.;attempts=0;unknown=0;latency=0.
    for lid,raw in calls:
        ledger=getattr(transport,'ledger',None);history=ledger.logical_attempts(lid) if ledger else []
        if history:
            for attempt in history:
                attempts+=1;used=json.loads(attempt['usage_json']) if attempt['usage_json'] else None;cost+=(attempt['actual_nusd'] if attempt['actual_nusd'] is not None else attempt['reserved_nusd'])/1e9;unknown+=used is None
                if used:
                    for field in ['input_tokens','output_tokens']:usage[field]+=used.get(field,0)
                    usage['reasoning_tokens']+=used.get('output_tokens_details',{}).get('reasoning_tokens',0)
                ap=Path(transport.output_dir)/(attempt['attempt_id']+'.attempt.json')
                if ap.exists():latency+=json.loads(ap.read_text()).get('latency_seconds',0.)
        else:
            attempts+=raw.get('attempt_count',0);cost+=raw.get('cost_usd',0.);latency+=raw.get('latency_seconds',0.);used=raw.get('usage') or {};unknown+=not bool(used)
            for field in ['input_tokens','output_tokens']:usage[field]+=used.get(field,0)
            usage['reasoning_tokens']+=used.get('output_tokens_details',{}).get('reasoning_tokens',0)
    result={'dataset':row['dataset'],'item_id':row['item_id'],'source_index':row['source_index'],'method':method,'answer_kind':row['answer_kind'],'gold_answer':row['gold_answer'],'final_answer':final['answer'],'correct':correct,'f1':answer_f1(final['answer'],row['gold_answer']) if row['dataset']=='hotpotqa' and final['valid'] else float(correct),'completed':len(errors)==0,'final_valid':final['valid'],'call_successes':sum(r.get('outcome')=='completed' for _,r in calls),'n_logical_calls':7,'attempts':attempts,**usage,'usage_unknown_attempts':unknown,'cost_usd':cost,'latency_seconds':latency,'wall_seconds':time.monotonic()-started,'communication':communication,'initial_disagreement':control['disagreement'],'controller':control,'initial_answers':[c['answer'] for c in initial],'refined_answers':[c['answer'] for c in refined],'source_hash':source,'config_hash':config.fingerprint,'fingerprint':fingerprint,'errors':errors};atomic(resultpath,result);return result

def execute(approval_path,workers=12):
    manifest=json.loads((PILOT/'run_manifest.json').read_text());approval=load_approval(approval_path,manifest)
    if manifest['source_hash']!=code_hash():raise RuntimeError('Code changed after preparation; repeatprepare and approval')
    cohort_path=PILOT/'cohort.jsonl'
    if hashlib.sha256(cohort_path.read_bytes()).hexdigest()!=manifest['cohort_sha256']:raise RuntimeError('Frozen cohort hash mismatch')
    rows=[json.loads(x) for x in cohort_path.read_text().splitlines()];ledger=Ledger(PILOT/'budget.sqlite3',manifest['manifest_sha256'],approval['max_usd'],manifest['max_http_attempts'],prices=manifest['prices_usd_per_million'],max_logical_calls=5250);transport=AzureResponsesTransport(BASE,'gpt-5-mini',key_loader,ledger,PILOT/'artifacts/http',max_input_token_bound=32768,rpm=manifest['rpm'],tpm=manifest['tpm'],timeout=180)
    tasks=[(row,method) for row in rows for method in METHODS];random.Random(20261113).shuffle(tasks);started=time.monotonic();done=0;stopped=None
    atomic(PILOT/'run_status.json',{'status':'running','expected_conditions':750,'completed_conditions':0,'inference_calls':ledger.summary()['logical_calls'],'budget':ledger.summary()})
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run_condition,row,method,transport,PILOT/'artifacts',manifest['source_hash']):(row,method) for row,method in tasks}
            for future in as_completed(futures):
                try:result=future.result()
                except BudgetExceeded as error:
                    stopped=str(error)
                    for other in futures:other.cancel()
                    break
                done+=1;state={'status':'running','expected_conditions':750,'completed_conditions':done,'inference_calls':ledger.summary()['logical_calls'],'budget':ledger.summary(),'wall_seconds':time.monotonic()-started};atomic(PILOT/'run_status.json',state)
                print(f"{done}/750 {result['dataset']} {result['method']} source_index={result['source_index']} valid={result['final_valid']} cost_total=${state['budget']['charged_usd']:.4f}",flush=True)
    finally:
        complete=len(list((PILOT/'artifacts/conditions').glob('*/result.json')));status='complete' if complete==750 else 'stopped';state={'status':status,'completed_conditions':complete,'expected_conditions':750,'expected_examples':150,'logical_calls':ledger.summary()['logical_calls'],'budget':ledger.summary(),'wall_seconds':time.monotonic()-started,'stop_reason':stopped};atomic(PILOT/'run_status.json',state);print(canonical(state),flush=True)
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--approval');parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    elif not args.approval:raise SystemExit('Use --approval with explicit user authorization JSON')
    else:execute(args.approval,args.workers)
if __name__=='__main__':main()
