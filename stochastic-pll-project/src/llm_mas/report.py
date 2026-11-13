"""Build an offline pilot supplement and incorporate it into the complete report."""
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .pilot import PILOT, ROOT, atomic
from .protocol import METHODS

LABELS = {'independent': 'Independent', 'full': 'Full communication', 'fixed_sparse': 'Fixed ring',
          'disagreement_triggered': 'Disagreement trigger', 'stochastic_controller': 'Stochastic controller'}
DATASETS = {'gsm8k': 'GSM8K', 'hotpotqa': 'HotpotQA', 'bbh': 'BBH logical deduction (three objects)'}


def esc(value):
    return html.escape(str(value))


def pct(value):
    return f'{100 * value:.1f}%'


def interval(values):
    return f'[{100 * values[0]:.1f}, {100 * values[1]:.1f}]'


def table(headers, rows, caption=''):
    return ('<div class="mas-scroll"><table><caption>' + esc(caption) + '</caption><thead><tr>'
            + ''.join('<th scope="col">' + esc(h) + '</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(v) + '</td>' for v in r) + '</tr>' for r in rows)
            + '</tbody></table></div>')


def figures(data):
    output = PILOT / 'analysis/figures'
    output.mkdir(exist_ok=True)
    paths = {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    for dataset in DATASETS:
        rows = [r for r in data['summary'] if r['dataset'] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={'width_ratios': [1.1, 1]})
        names = [LABELS[r['method']] for r in rows]
        y = np.arange(len(rows))
        accuracy = np.array([r['accuracy'] * 100 for r in rows])
        lower = np.maximum(0, accuracy - np.array([r['accuracy_ci'][0] * 100 for r in rows]))
        upper = np.maximum(0, np.array([r['accuracy_ci'][1] * 100 for r in rows]) - accuracy)
        axes[0].errorbar(accuracy, y, xerr=[lower, upper], fmt='o', color='#174f63', capsize=4, markersize=6)
        axes[0].set_yticks(y, names)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 103)
        axes[0].set_xlabel('Exact-match accuracy (%) with Wilson 95% interval')
        axes[0].set_ylabel('Communication method')
        axes[0].set_title(DATASETS[dataset] + ': accuracy on 50 questions', loc='left')
        axes[0].grid(axis='x', alpha=.2)
        traffic = [r['directed_messages'] / r['n'] for r in rows]
        axes[1].barh(y, traffic, color='#174f63', height=.5)
        axes[1].set_yticks(y, names)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 6.5)
        axes[1].set_xlabel('Mean directed peer messages per question (count)')
        axes[1].set_ylabel('Communication method')
        axes[1].set_title('Actual delivered traffic in the frozen round', loc='left')
        for i, value in enumerate(traffic):
            axes[1].text(value + .07, i, f'{value:.2f}', va='center')
        fig.tight_layout()
        path = output / f'{dataset}_accuracy_traffic.svg'
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths[dataset] = path
    return paths


def evidence_archive():
    """Compress completed raw evidence; credentials and the report itself are excluded."""
    with sqlite3.connect(f'file:{PILOT / "budget.sqlite3"}?mode=ro', uri=True) as source:
        with sqlite3.connect(':memory:') as snapshot:
            source.backup(snapshot)
            (PILOT / 'analysis/budget_snapshot.sqlite3').write_bytes(snapshot.serialize())
    files = [p for p in (PILOT / 'artifacts').rglob('*') if p.is_file()]
    files += [p for p in (PILOT / 'analysis').rglob('*') if p.is_file() and p.suffix not in {'.zip', '.html'}
              and p.name not in {'report_generation.json', 'complete_report_integration.json'}]
    files += [p for p in PILOT.glob('*.json') if p.is_file() and p.name != 'finalization.json']
    files += [PILOT / 'cohort.jsonl', PILOT / 'README.md', ROOT / 'experiments/llm_extension/azure_pricing_sources.json']
    files += [PILOT / 'source_data/manifest.json', ROOT / 'experiments/llm_extension/dataset_manifest.json',
              ROOT / 'experiments/llm_extension/test.jsonl', PILOT / 'source_data/hotpotqa_validation.parquet',
              PILOT / 'source_data/bbh_logical_deduction_three_objects.json', ROOT / 'requirements_llm_pilot.txt']
    files += [p for p in (PILOT / 'tokenizer_cache').glob('*') if p.is_file()]
    files += list((ROOT / 'src/llm_mas').glob('*.py'))
    files += list((ROOT / 'tests').glob('test_mas*.py'))
    files += [p for p in (PILOT / 'logs').glob('*') if p.is_file() and p.name in {
        'preflight.txt', 'preflight.xml', 'extended_preflight.txt', 'execution.log',
        'progress_history.jsonl', 'browser_verify.cjs', 'complete_report_browser_verify.cjs',
        'final_tests.txt', 'final_tests.xml'}]
    files = sorted(set(p for p in files if p.exists()))
    secret_path = Path('/tmp/stochastic-pll-azure-key')
    secret = secret_path.read_bytes().strip() if secret_path.exists() else None
    stream = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            content = path.read_bytes()
            if secret and secret in content:
                raise RuntimeError('Credential material detected; evidence packaging stopped')
            name = str(path.relative_to(ROOT))
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        snapshot_bytes = (PILOT / 'analysis/budget_snapshot.sqlite3').read_bytes()
        ledger_name = str((PILOT / 'budget.sqlite3').relative_to(ROOT))
        hashes[ledger_name] = hashlib.sha256(snapshot_bytes).hexdigest()
        archive.writestr(ledger_name, snapshot_bytes)
        archive.writestr('SHA256.json', json.dumps(hashes, indent=2))
    content = stream.getvalue()
    path = PILOT / 'analysis/pilot_raw_evidence.zip'
    path.write_bytes(content)
    return content, {'sha256': hashlib.sha256(content).hexdigest(), 'files': len(hashes) + 1,
                     'checksum_entries': len(hashes), 'bytes': len(content)}


CSS = '''
.mas-pilot { --mas-bg:#fff; --mas-fg:#162a32; --mas-muted:#475c64; --mas-line:#b5c3c8; --mas-accent:#174f63; --mas-panel:#eef3f5; color:var(--mas-fg); background:var(--mas-bg); font:16px/1.65 system-ui,sans-serif; padding:32px; overflow-wrap:anywhere; }
html[data-theme="dark"] .mas-pilot:not([data-mas-theme="light"]), .mas-pilot[data-mas-theme="dark"] { --mas-bg:#142128; --mas-fg:#edf3f5; --mas-muted:#bdcbd1; --mas-line:#5f7782; --mas-accent:#98d5e8; --mas-panel:#20343e; color-scheme:dark; }
.mas-pilot * { box-sizing:border-box; }
.mas-pilot .mas-wrap { max-width:1160px; margin:auto; }
.mas-pilot h2 { font:600 2rem/1.2 Georgia,serif; margin:0 0 20px; color:var(--mas-fg); }
.mas-pilot h3 { font-size:1.25rem; margin:40px 0 12px; color:var(--mas-fg); }
.mas-pilot h4 { font-size:1.05rem; margin:22px 0 8px; }
.mas-pilot p { max-width:88ch; }
.mas-pilot a { color:var(--mas-accent); text-decoration:underline; }
.mas-pilot button,.mas-pilot select { font:inherit; padding:9px 12px; border:1px solid var(--mas-line); border-radius:3px; color:var(--mas-fg); background:var(--mas-bg); min-height:44px; }
.mas-pilot :focus-visible { outline:3px solid var(--mas-accent); outline-offset:3px; }
.mas-pilot .mas-controls { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin:24px 0; }
.mas-pilot .mas-controls label { display:flex; gap:8px; align-items:center; flex-wrap:wrap; max-width:100%; min-width:0; }
.mas-pilot .mas-controls select { max-width:100%; min-width:0; }
.mas-pilot .mas-notice { border-left:4px solid var(--mas-accent); background:var(--mas-panel); padding:14px 20px; margin:22px 0; }
.mas-pilot .mas-scroll { overflow-x:auto; max-width:100%; margin:20px 0; }
.mas-pilot table { border-collapse:collapse; width:100%; color:var(--mas-fg); font-size:14px; }
.mas-pilot th { text-align:left; font-weight:650; background:var(--mas-panel); }
.mas-pilot td,.mas-pilot th { padding:10px 12px; border-bottom:1px solid var(--mas-line); vertical-align:top; }
.mas-pilot caption { text-align:left; font-weight:600; padding:0 0 10px; }
.mas-pilot figure { margin:28px 0; }
.mas-pilot figure svg { display:block; width:100%; height:auto; background:white; }
.mas-pilot figcaption { font-size:14px; color:var(--mas-muted); margin-top:8px; }
.mas-pilot details { border-bottom:1px solid var(--mas-line); padding:14px 0; }
.mas-pilot summary { cursor:pointer; font-weight:600; min-height:32px; }
.mas-pilot pre { white-space:pre-wrap; word-break:break-word; max-width:100%; font-size:13px; background:var(--mas-panel); padding:14px; color:var(--mas-fg); }
.mas-pilot code { font-family:ui-monospace,monospace; }
.mas-pilot #mas-network { width:100%; max-width:680px; height:auto; display:block; margin:16px 0; }
.mas-pilot #mas-network text { fill:var(--mas-fg); font:14px system-ui,sans-serif; }
.mas-pilot [hidden] { display:none!important; }
.mas-pilot .mas-tiny { font-size:13px; color:var(--mas-muted); }
@media(max-width:600px) { .mas-pilot {padding:22px 12px;} .mas-pilot h2 {font-size:1.6rem;} .mas-pilot td,.mas-pilot th {padding:8px;} }
@media print { .mas-pilot .mas-controls {display:none;} .mas-pilot details:not([open]) {display:none;} .mas-pilot {font-size:11pt;} }
'''


JS = '''
(() => {
 const root=document.getElementById('llm-pilot');
 const select=root.querySelector('#mas-dataset');
 const replay=JSON.parse(document.getElementById('mas-replay-data').textContent);
 const itemSelect=root.querySelector('#mas-replay-item'), methodSelect=root.querySelector('#mas-replay-method');
 const ns='http://www.w3.org/2000/svg';
 function drawReplay(){
   const item=replay.find(r=>r.item_id===itemSelect.value); if(!item)return;
   const record=item.methods[methodSelect.value]; if(!record)return;
   const svg=root.querySelector('#mas-network'); svg.replaceChildren();
   const element=(tag,attributes,text)=>{const e=document.createElementNS(ns,tag); Object.entries(attributes).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;};
   const defs=element('defs',{}), marker=element('marker',{id:'mas-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});
   marker.append(element('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'var(--mas-accent)'}));defs.append(marker);svg.append(defs);
   const points=[[100,190],[500,190],[300,55]];
   record.edges.forEach(([a,b])=>{const [x1,y1]=points[a],[x2,y2]=points[b],length=Math.hypot(x2-x1,y2-y1),dx=(x2-x1)/length,dy=(y2-y1)/length;
     svg.append(element('path',{d:`M${x1+dx*30},${y1+dy*30} Q${(x1+x2)/2-dy*28},${(y1+y2)/2+dx*28} ${x2-dx*34},${y2-dy*34}`,fill:'none',stroke:'var(--mas-accent)','stroke-width':2,'marker-end':'url(#mas-arrow)','data-peer-edge':'true'}));});
   points.forEach(([x,y],i)=>{svg.append(element('circle',{cx:x,cy:y,r:27,fill:'var(--mas-panel)',stroke:'var(--mas-accent)','stroke-width':2}));svg.append(element('text',{x,y:y+5,'text-anchor':'middle'},String(i)));svg.append(element('text',{x,y:y+49,'text-anchor':'middle'},'Agent '+i));});
   root.querySelector('#mas-replay-readout').textContent=`${item.item_id}; reference: ${item.gold}. Initial answers: ${record.initial.map(x=>x===null?'invalid':x).join(' | ')}. D=${record.disagreement.toFixed(3)}; directed messages=${record.edges.length}; final=${record.final}; ${record.correct?'correct':'incorrect'}.`;
   root.querySelector('#mas-replay-states').textContent=record.states.length?JSON.stringify(record.states,null,2):'This method has no stochastic gate states.';
 }
 function updateMethods(){const item=replay.find(r=>r.item_id===itemSelect.value);methodSelect.replaceChildren();if(!item)return;Object.entries(item.methods).forEach(([value,r])=>{const option=document.createElement('option');option.value=value;option.textContent=r.label;methodSelect.append(option);});drawReplay();}
 function updateItems(){const previous=itemSelect.value;itemSelect.replaceChildren();replay.filter(r=>select.value==='all'||r.dataset===select.value).forEach(r=>{const option=document.createElement('option');option.value=r.item_id;option.textContent=r.item_id;itemSelect.append(option);});if([...itemSelect.options].some(o=>o.value===previous))itemSelect.value=previous;updateMethods();}
 itemSelect.addEventListener('change',updateMethods);methodSelect.addEventListener('change',drawReplay);updateItems();
 select.addEventListener('change',()=>{
   root.querySelectorAll('[data-mas-dataset]').forEach(el=>el.hidden=select.value!=='all'&&el.dataset.masDataset!==select.value);
   root.querySelector('#mas-selection').textContent=select.value==='all'?'Showing all three datasets.':'Showing '+select.options[select.selectedIndex].text+'.';
   updateItems();
 });
 const themeButton=root.querySelector('#mas-theme');
 const isDark=()=>root.dataset.masTheme?root.dataset.masTheme==='dark':document.documentElement.dataset.theme==='dark';
 const themeReadout=()=>{themeButton.setAttribute('aria-pressed',String(isDark()));themeButton.textContent=isDark()?'Light theme':'Dark theme';};
 themeButton.addEventListener('click',()=>{root.dataset.masTheme=isDark()?'light':'dark';themeReadout();});
 new MutationObserver(()=>{delete root.dataset.masTheme;themeReadout();}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});themeReadout();
 root.querySelector('#mas-evidence').addEventListener('click',()=>{
   const b64=document.getElementById('mas-archive').textContent.trim();
   const binary=atob(b64), bytes=new Uint8Array(binary.length);
   for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
   const url=URL.createObjectURL(new Blob([bytes],{type:'application/zip'}));
   const a=document.createElement('a'); a.href=url; a.download='pilot_raw_evidence.zip'; a.click();
   setTimeout(()=>URL.revokeObjectURL(url),10000);
   root.querySelector('#mas-download-state').textContent='Evidence archive prepared for download.';
 });
})();
'''


def build_section(data, paths, archive, archive_meta, cohort, records):
    integrity = data['integrity']
    completed = data['status'] == 'complete' and integrity['passed']
    status = 'Completed exploratory pilot' if completed else 'Partial pilot snapshot: inference is still incomplete'
    output = ['<style>' + CSS + '</style><section class="mas-pilot" id="llm-pilot"><div class="mas-wrap">',
              '<h2>K. Three-dataset LLM multi-agent pilot</h2>',
              '<p><strong>' + esc(status) + '.</strong> Exactly 50 GSM8K questions, 50 HotpotQA questions and 50 questions from one BBH logical-deduction task. Five communication methods use the same seven-call architecture for every question.</p>',
              '<div class="mas-notice"><strong>Scope and interpretation.</strong> The user replaced the earlier full-benchmark proposal with this 150-question pilot and approved a US$25 cap. This is an exploratory comparison of communication policies. The stochastic gate is an OU-style feedback variable; these results do not validate a physical PLL interpretation or a language-model phase representation.</div>',
              f'<p>{integrity["observed_conditions"]}/750 method/question records; {integrity["logical_calls"]:,}/5,250 logical calls; {integrity["http_attempts"]:,} physical HTTP attempts. Conservative list-price usage: <strong>US${integrity["conservative_cost_usd"]:.6f}</strong>, including US${integrity["uncertain_reservation_usd"]:.6f} reserved for missing-usage outcomes. These are token-based estimates, not a reconciled Azure invoice.</p>',
              '<div class="mas-controls"><label for="mas-dataset">Dataset <select id="mas-dataset"><option value="all">All three datasets</option><option value="gsm8k">GSM8K</option><option value="hotpotqa">HotpotQA</option><option value="bbh">BBH logical deduction</option></select></label><button type="button" id="mas-theme" aria-pressed="false">Dark theme</button><button type="button" id="mas-evidence">Download raw evidence ZIP</button></div>',
              '<p id="mas-selection" class="mas-tiny" role="status">Showing all three datasets.</p><p id="mas-download-state" class="mas-tiny" role="status"></p>',
              '<h3>Accuracy and uncertainty</h3><p>Every accuracy denominator contains all 50 frozen questions, including failed or invalid final outputs. Wilson intervals describe question-level uncertainty. Intermediate failures are reported separately from final-answer correctness. An item can have a correct final answer while its complete seven-step protocol contains an error.</p>']
    for dataset in DATASETS:
        rows = [r for r in data['summary'] if r['dataset'] == dataset]
        output.append('<div data-mas-dataset="' + dataset + '"><h4>' + esc(DATASETS[dataset]) + '</h4>')
        output.append(table(['Method', 'Observed / 50', 'Correct / 50', 'EM', '95% CI (%)', 'Answer F1 [95% CI]', 'All 7 steps valid'],
            [[esc(LABELS[r['method']]), f'{r["observed"]}/50', f'{r["correct"]}/50', pct(r['accuracy']), interval(r['accuracy_ci']),
              f'{r["f1"]:.3f} [{r["f1_ci"][0]:.3f}, {r["f1_ci"][1]:.3f}]' if dataset == 'hotpotqa' else 'Not applicable', f'{r["protocol_complete"]}/50'] for r in rows],
            'Final-answer results; F1 is computed only for HotpotQA.'))
        svg = paths[dataset].read_text(); svg = svg[svg.index('<svg'):]
        output.append('<figure role="img" aria-label="' + esc(DATASETS[dataset]) + ' exact-match accuracy with confidence intervals and mean peer-message traffic">' + svg
            + '<figcaption>Source: frozen pilot records. Left: exact-match accuracy with Wilson 95% intervals, n = 50 per method. Right: actual directed messages delivered between the three agents in one frozen round. Independent agents still use a common adjudicator.</figcaption></figure></div>')
    output.append('<h3>Paired comparisons against independent agents</h3><p>Accuracy differences use the same questions in both methods. The 95% percentile intervals use 20,000 question-paired bootstrap resamples. McNemar tests use exact two-sided binomial probabilities on discordant answers. Holm adjustment covers all 12 method/dataset comparisons; these exploratory intervals are not simultaneous intervals. Independently generated drafts mean provider randomness is not coupled across methods.</p>')
    for dataset in DATASETS:
        rows = [r for r in data['paired_comparisons'] if r['dataset'] == dataset]
        output.append('<div data-mas-dataset="' + dataset + '">' + table(
            ['Method vs independent', 'Difference (pp)', 'Bootstrap 95% CI (pp)', 'Wins / losses', 'Exact p', 'Holm p'],
            [[esc(LABELS[r['method']]), f'{100*r["delta"]:+.1f}', interval(r['bootstrap_ci']),
              f'{r["wins"]} / {r["losses"]}', f'{r["mcnemar_p"]:.4g}', f'{r["holm_p"]:.4g}'] for r in rows], DATASETS[dataset]) + '</div>')
    significant = [r for r in data['paired_comparisons'] if r['holm_p'] < .05]
    interpretation = ('No comparison passes the 0.05 Holm-adjusted threshold. The pilot does not establish an accuracy advantage for a communication policy.' if not significant else f'{len(significant)} comparison(s) pass the 0.05 Holm-adjusted threshold; direction and intervals are shown above. The limited tasks and single model realization restrict generalization.') if completed else 'Inference is incomplete. These fixed-denominator values are execution snapshots and are not interpreted as scientific comparisons.'
    output.append('<p><strong>Statistical reading.</strong> ' + interpretation + '</p>')
    output.append('<h3>Communication, usage and latency</h3><p>Peer-content tokens count each delivered serialized message using tiktoken 0.12.0/o200k_base. They exclude the repeated question, own initial response, API framing and the common adjudication. They are a transparent traffic measure, not a provider-billed token attribution. Provider output usage includes hidden reasoning tokens.</p>')
    for dataset in DATASETS:
        rows = [r for r in data['summary'] if r['dataset'] == dataset]
        output.append('<div data-mas-dataset="' + dataset + '">' + table(
            ['Method', 'Messages', 'Peer-content tokens', 'Input tokens', 'Output tokens', 'Reasoning tokens', 'Cost (US$)', 'Median HTTP sum / item (s)'],
            [[esc(LABELS[r['method']]), f'{r["directed_messages"]:,}', f'{r["peer_content_tokens"]:,}' if r['peer_content_tokens'] is not None else 'Unmeasured',
              f'{r["input_tokens"]:,}', f'{r["output_tokens"]:,}', f'{r["reasoning_tokens"]:,}', f'{r["cost_usd"]:.6f}',
              f'{r["median_http_seconds_per_condition"]:.2f}' if r['median_http_seconds_per_condition'] is not None else 'Unmeasured'] for r in rows], DATASETS[dataset] + ': totals across 50 questions, except median latency.') + '</div>')
    output.append('<p>HTTP latency is the sum of recorded attempt elapsed times for the seven sequential calls, including any retries. It excludes rate-limiter waiting. Recorded condition wall times include shared-queue waiting and are not isolated method-speed measurements. The runner uses 12 workers with conservative planning limits of 200 requests/minute and 200,000 bounded tokens/minute; these settings are not verified account quotas. Repeated question/context text in all seven calls and the common adjudication contribute input tokens even when no peer messages are sent.</p>')
    output.append('<h3>Frozen protocol and mathematical boundary</h3>')
    output.append(table(['Stage', 'Calls', 'Information available'], [
        ['Independent drafts', '3', 'Question only; three separate calls within every method.'],
        ['Frozen communication round', '0', 'Route initial answer/summary messages; later refinements cannot affect another agent in the same round.'],
        ['Refinements', '3', 'Question, own initial response and routed peer messages.'],
        ['Common adjudication', '1', 'Question and the three refined responses.']], 'Seven logical calls per question/method; no cross-method generated-response cache.'))
    output.append('<p>Full communication sends all six directed edges. The fixed sparse method uses 0→1, 1→2, 2→0. The disagreement trigger sends all six edges when the fraction D of disagreeing initial-answer pairs exceeds 0.5. Missing/invalid answers count as disagreement.</p><pre>D = (1/3) Σ over the three unordered agent pairs 1[answers disagree or are missing]\nx(next) = clip(x + 1.2(D − x)·0.1 + 0.35·√0.1·Z, −2, 2), Z ~ N(0,1)\np(sender) = sigmoid(−1.25 + 2D + 0.8x(next))\nSend each outgoing directed edge if its recorded U ~ Uniform(0,1) is below p(sender).</pre><p>Three numerical substeps update the gate once per sender, while all initial agent messages remain frozen. The OU-style gate does not have a phase, periodicity or a demonstrated predictive advantage beyond ordinary disagreement. Its parameters were fixed before inference, without tuning on these 150 labels.</p>')
    output.append(table(['Dataset', 'Source / split', 'Population', 'Sample', 'Sampling seed'], [
        ['GSM8K', '<a href="https://github.com/openai/grade-school-math">Authors’ official test JSONL</a>', '1,319', '50 without replacement', '20261113'],
        ['HotpotQA', '<a href="https://huggingface.co/datasets/hotpotqa/hotpot_qa">Distractor validation, maintained distribution</a>', '7,405', '50 without replacement', '20261114'],
        ['BBH', '<a href="https://github.com/suzgunmirac/BIG-Bench-Hard">logical_deduction_three_objects</a>', '250', '50 without replacement', '20261115']], 'Three real source datasets; source bytes and all chosen indices are preserved.'))
    output.append('<p>HotpotQA receives all ten supplied context paragraphs, without supporting-fact annotations or reference-answer fields. Gold labels enter only the local scorer. GSM8K uses exact rational-number normalization; HotpotQA uses conventional punctuation/article/whitespace-normalized EM and token F1; BBH uses the final A/B/C option.</p><p>Model: Azure gpt-5-mini, verified deployment version 2025-08-07, GlobalStandard, Korea Central. Every request uses low reasoning effort, max_output_tokens = 4096, store = false, with no tools, temperature/top_p or unsupported deterministic API seed. Only sampling, task order and controller random streams are fixed. A brief external summary is capped at 600 characters before delivery to peers.</p>')
    output.append('<h3>Operational integrity and failure accounting</h3>')
    output.append(table(['Check', 'Observed evidence'], [
        ['Cohort checksum', '<code>' + data['cohort_sha256'] + '</code>'],
        ['Audit', 'PASS' if integrity['passed'] else esc('; '.join(integrity['issues']))],
        ['Reconstructed actual payloads', str(integrity['reconstructed_request_payloads'])],
        ['Raw responses', str(integrity['raw_responses'])],
        ['Provider model identities', esc(json.dumps(integrity.get('provider_model_versions', {})))],
        ['Physical outcomes', esc(json.dumps(integrity['attempt_outcomes'], sort_keys=True))],
        ['Cached input tokens reported by provider', str(integrity['usage'].get('cached_input_tokens', 0))],
        ['Cap check', 'Within approved US$25 cap' if integrity['within_approved_cap'] else 'Cap violation'],
        ['Evidence archive', f'{archive_meta["files"]:,} files; {archive_meta["bytes"]:,} compressed bytes; SHA256 <code>{archive_meta["sha256"]}</code>']], 'Reconstructed routing, prompts, final scoring and raw/ledger token usage are checked from saved artifacts.'))
    output.append('<p>Every HTTP attempt reserves its conservative maximum cost before dispatch. Unknown-usage outcomes retain that reservation. Only HTTP 429/500/502/503/504 can retry, at most three attempts. Content filters, incomplete output, parse failures and ambiguous timeouts are not repaired or silently resent. Errors remain in the denominator and evidence. Public input/output rates are US$0.25/US$2.00 per million tokens; cached discounts are ignored in the cap ledger. The US$11.025 planning estimate assumed 2,000 input and 800 output tokens per call, and was not a measured result.</p>')
    error_rows = [[esc(r['dataset']), esc(LABELS[r['method']]), esc(r['item_id']), esc(json.dumps(r['errors']))] for r in records if r['errors']]
    output.append(table(['Dataset', 'Method', 'Item', 'Recorded errors'], error_rows, 'All conditions with protocol errors.') if error_rows else '<p>All saved method/question conditions have seven valid completed stages; no protocol errors were recorded.</p>')
    replay = []
    for item in cohort:
        available = [r for r in records if r['item_id'] == item['item_id']]
        if available:
            replay.append({'dataset': item['dataset'], 'item_id': item['item_id'], 'gold': item['gold_answer'],
                'methods': {r['method']: {'label': LABELS[r['method']], 'initial': r['initial_answers'],
                    'edges': r['controller']['edges'], 'states': r['controller']['states'],
                    'disagreement': r['initial_disagreement'], 'final': r['final_answer'], 'correct': r['correct']} for r in available}})
    output.append('<h3>Replay actual communication decisions</h3><p>Choose a question and a completed method to see its recorded directed edges. Node numbers are agent identifiers; an arrow shows one delivered peer message. The readout uses saved answers and controller states. This viewer makes no network calls.</p><div class="mas-controls"><label for="mas-replay-item">Question <select id="mas-replay-item" style="max-width:100%"></select></label><label for="mas-replay-method">Method <select id="mas-replay-method"></select></label></div><svg id="mas-network" viewBox="0 0 600 270" role="img" aria-label="Recorded three-agent communication graph" aria-describedby="mas-replay-readout"></svg><p id="mas-replay-readout" role="status"></p><details><summary>Recorded stochastic substeps for the selected run</summary><pre id="mas-replay-states"></pre></details>')
    output.append('<h3>Question-level evidence</h3><p>Open a question to inspect its exact model input, local reference label and every method’s final answer. The embedded ZIP contains all saved request/response/attempt files, checkpoints, source code, manifests, selected cohort and analysis. Hidden internal reasoning is not available from the provider and is not reconstructed.</p>')
    lookup = {(r['item_id'], r['method']): r for r in records}
    for row in cohort:
        output.append('<details data-mas-dataset="' + row['dataset'] + '"><summary>' + esc(DATASETS[row['dataset']] + ' · source index ' + str(row['source_index']) + ' · ' + row['item_id']) + '</summary><p><strong>Reference answer:</strong> ' + esc(row['gold_answer']) + '</p><pre>' + esc(row['model_input']) + '</pre>')
        answer_rows = []
        for method in METHODS:
            r = lookup.get((row['item_id'], method))
            answer_rows.append([esc(LABELS[method]), esc(r['final_answer']) if r else 'Missing',
                               'Correct' if r and r['correct'] else 'Incorrect / missing',
                               esc(', '.join(str(x) for x in r['initial_answers'])) if r else 'Missing',
                               str(r['communication']['directed_messages']) if r else '0'])
        output.append(table(['Method', 'Final answer', 'Scored outcome', 'Initial answers', 'Peer messages'], answer_rows) + '</details>')
    output.append('<h3>Limitations and next decisions</h3><ul>' + ''.join('<li>' + esc(s) + '</li>' for s in data['limitations']) + '</ul><p>Any larger study should first state the desired accuracy/traffic tradeoff, repeat provider realizations, include a broader reasoning suite and evaluate dropout/corruption separately. A physical-phase claim requires its own defined representation and held-out incremental predictive validation. Those experiments were not part of this pilot.</p>')
    output.append('<details><summary>Complete protocol source</summary><pre>' + esc((ROOT / 'src/llm_mas/protocol.py').read_text()) + '</pre></details><details><summary>Complete stochastic gate source</summary><pre>' + esc((ROOT / 'src/llm_mas/controller.py').read_text()) + '</pre></details>')
    output.append('<h3>Implementation and reproduction</h3><p>The dataset module freezes question identities before inference and exposes a label-free model view. The protocol module builds stage-specific payloads without accepting reference labels. The pilot runner gives every method/question/stage its own logical identity, checkpoints every output and resumes saved outputs without generating replacement answers. The transport writes each request and attempt record after a durable budget reservation, then settles only from provider-reported usage. A retry is a separate physical attempt with its own reservation.</p><p>The analysis independently reconstructs all seven payloads, replays routing and locally rescores final answers, then reconciles raw response usage with the attempt ledger. Report generation reads only these artifacts. Its plotting, token counting and evidence packaging make no model calls. The archive includes the original source populations used for sampling, the selected150question cohort, tokenizer vocabulary cache, pinned Python dependencies, all checkpoints, and a consistent budget database. Source population copies do not represent additional model-evaluated questions. Repeating the inference command with completed checkpoints reads saved results; a fresh inference study requires a separate namespace and explicit scope. Installing Python dependencies is the only environment-setup step; subsequent saved-output analysis uses local evidence.</p><pre>python -m pip install -r requirements_llm_pilot.txt\npython -m pytest tests/test_mas_protocol.py tests/test_mas_transport.py tests/test_mas_pilot.py tests/test_mas_analysis.py -q\npython -m src.llm_mas.pilot run --approval experiments/llm_extension/pilot/approval_record.json --workers 12\npython -m src.llm_mas.analysis\nMPLCONFIGDIR=/tmp/stochastic-mpl-cache python -m src.llm_mas.report</pre>')
    for name in ['datasets.py', 'ledger.py', 'transport.py', 'pilot.py', 'analysis.py', 'report.py']:
        output.append('<details><summary>Complete ' + esc(name) + ' source</summary><pre>' + esc((ROOT / 'src/llm_mas' / name).read_text()) + '</pre></details>')
    output.append('<details><summary>Frozen manifest and approval record</summary><pre>' + esc((PILOT / 'run_manifest.json').read_text()) + '\n' + esc((PILOT / 'approval_record.json').read_text()) + '</pre></details>')
    output.append('<p class="mas-tiny">Generated from actual artifacts at ' + esc(data['created_utc']) + '. Design: a research supplement for university readers; paper typography, restrained blue accent, wide comparison tables and responsive figures serve evidence inspection. Energy 1 / rhythm 2 / motion 1. Download and theme/filter controls work offline.</p>')
    replay_json = json.dumps(replay, ensure_ascii=False).replace('<', '\\u003c')
    output.append('</div></section><script type="application/json" id="mas-replay-data">' + replay_json + '</script><script type="application/octet-stream" id="mas-archive">' + base64.b64encode(archive).decode() + '</script><script>' + JS + '</script>')
    return ''.join(output)


def build(integrate=False):
    data = json.loads((PILOT / 'analysis/results.json').read_text())
    paths = figures(data)
    cohort = [json.loads(s) for s in (PILOT / 'cohort.jsonl').read_text().splitlines()]
    records = [json.loads(s) for s in (PILOT / 'analysis/all_condition_results.jsonl').read_text().splitlines()]
    archive, archive_meta = evidence_archive()
    section = build_section(data, paths, archive, archive_meta, cohort, records)
    standalone = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-dataset LLM MAS pilot</title><style>body{margin:0}</style></head><body>' + section + '</body></html>'
    target = ROOT / 'report/llm_mas_pilot_report.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(standalone)
    if integrate:
        if data['status'] != 'complete' or not data['integrity']['passed']:
            raise RuntimeError('Only a completed, audited pilot can update the complete research report')
        original = ROOT / 'report/stochastic_pll_complete_report.html'
        backup = ROOT / 'backups/stochastic_pll_complete_report_before_llm_pilot.html'
        if not backup.exists():
            backup.write_bytes(original.read_bytes())
        text = backup.read_text()
        # Retain the exact original local-study document and identify its inference cutoff.
        notice = '<aside style="padding:16px;border:2px solid #174f63;margin:20px"><strong>LLM pilot update:</strong> The local-study sections below retain their original pre-inference cutoff. A subsequently approved 150-question pilot is now completed and audited in <a href="#llm-pilot">Section K: LLM MAS pilot</a>. Earlier statements about no paid inference describe the historical local-study cutoff; they do not describe this update.</aside>'
        main_start = text.index('>', text.index('<main')) + 1
        text = text[:main_start] + notice + text[main_start:]
        text = text.replace('</nav>', '<a href="#llm-pilot"><span>K</span> LLM pilot results</a></nav>', 1)
        status_phrase = 'All 10 sections available'
        script_position = text.rfind(status_phrase)
        if script_position >= 0:
            text = text[:script_position] + text[script_position:].replace(status_phrase, 'All 11 sections available', 1)
        status_element = 'id="search-status"'
        status_position = text.find(status_element)
        if status_position >= 0:
            close = text.find('</', status_position)
            text = text[:status_position] + text[status_position:close].replace(status_phrase, 'All 11 sections available') + text[close:]
        text = text.replace('</main>', section + '</main>', 1)
        original.write_text(text)
        atomic(PILOT / 'analysis/complete_report_integration.json', {'original_backup': str(backup.relative_to(ROOT)),
               'original_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(),
               'updated_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
               'historical_sections_preserved': True, 'new_section': 'llm-pilot'})
    atomic(PILOT / 'analysis/report_generation.json', {'standalone': str(target.relative_to(ROOT)),
           'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
           'integrated': integrate, 'archive': archive_meta, 'external_runtime_dependencies': 0})
    print(json.dumps({'report': str(target), 'bytes': target.stat().st_size, 'integrated': integrate}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrate', action='store_true')
    build(integrate=parser.parse_args().integrate)
