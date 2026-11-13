from pathlib import Path
import importlib,json

def data_module():
    assert Path('src/llm_mas/datasets.py').exists(),'Real dataset cohort loader required'
    return importlib.import_module('src.llm_mas.datasets')

def test_exactly_fifty_each_fixed_seed_and_provenance(tmp_path):
    m=data_module();a=m.build_cohort(tmp_path/'a',seed=20261113);b=m.build_cohort(tmp_path/'b',seed=20261113)
    assert len(a)==150 and a==b
    assert {k:sum(x['dataset']==k for x in a) for k in ['gsm8k','hotpotqa','bbh']}=={'gsm8k':50,'hotpotqa':50,'bbh':50}
    assert len({x['item_id'] for x in a})==150
    assert all(x['source_sha256'] and x['gold_answer'] is not None for x in a)

def test_question_renderer_excludes_gold_metadata():
    m=data_module();row={'question':'Who won?','answer':'PRIVATE_GOLD_SENTINEL','supporting_facts':{'title':['GOLD_SUPPORT']},'context':{'title':['Passage'],'sentences':[['Visible context.']]}}
    rendered=m.render_hotpot(row)
    assert 'Visible context.' in rendered
    assert 'PRIVATE_GOLD_SENTINEL' not in rendered and 'GOLD_SUPPORT' not in rendered

def test_model_view_contains_only_public_task():
    m=data_module();view=m.model_view({'model_input':'Public question','gold_answer':'SECRET','answer_kind':'numeric','item_id':'x','dataset':'gsm8k'})
    assert view=={'question':'Public question','answer_kind':'numeric','item_id':'x','dataset':'gsm8k'}

def test_hotpot_f1_special_answers():
    m=data_module();assert m.answer_f1('yes','no')==0
    assert m.answer_f1('The Eiffel Tower','Eiffel Tower')==1

def test_runner_executes_seven_and_resume_zero_calls(tmp_path):
    assert Path('src/llm_mas/pilot.py').exists(),'Budget-gated pilot runner required'
    m=importlib.import_module('src.llm_mas.pilot')
    class FakeTransport:
        calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            return {'text':'{"answer":"2","summary":"A short test-only summary."}','outcome':'completed','usage':{'input_tokens':100,'output_tokens':40},'usage_known':True,'cost_usd':.000105,'latency_seconds':.01,'attempt_count':1,'model':'gpt-5-mini','status':'completed'}
    transport=FakeTransport();row={'dataset':'gsm8k','item_id':'gsm8k:test:fixture','source_index':0,'answer_kind':'numeric','model_input':'What is one plus one?','gold_answer':'2','source_sha256':'fixture'}
    first=m.run_condition(row,'full',transport,tmp_path,'fixture-code-hash')
    assert transport.calls==7 and first['correct'] and first['communication']['directed_messages']==6
    assert m.run_condition(row,'full',transport,tmp_path,'fixture-code-hash')==first
    assert transport.calls==7

def test_execution_refuses_without_approval():
    assert Path('src/llm_mas/pilot.py').exists(),'Budget-gated pilot runner required'
    m=importlib.import_module('src.llm_mas.pilot')
    import pytest
    with pytest.raises((FileNotFoundError,ValueError)):
        m.load_approval(Path('/tmp/nonexistent-pll-approval-for-test.json'),{'manifest_sha256':'fixture'})
