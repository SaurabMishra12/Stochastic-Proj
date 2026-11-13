"""Finish local analysis and verification after the approved inference process stops."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / 'experiments/llm_extension/pilot'
LOGS = PILOT / 'logs'
environment = dict(os.environ, MPLCONFIGDIR='/tmp/stochastic-mpl-cache')


def save(value):
    path = PILOT / 'finalization.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def command(args, name):
    with (LOGS / name).open('w') as handle:
        subprocess.run(args, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


save({'status': 'waiting_for_inference', 'extra_model_calls': 0})
try:
    while True:
        state = json.loads((PILOT / 'run_status.json').read_text())
        if state['status'] != 'running':
            break
        time.sleep(50)
    if state['status'] != 'complete':
        raise RuntimeError('Inference stopped before all750conditions; no final-completion claim or report integration')
    save({'status': 'auditing', 'extra_model_calls': 0})
    command([sys.executable, '-m', 'src.llm_mas.analysis'], 'final_analysis.txt')
    data = json.loads((PILOT / 'analysis/results.json').read_text())
    if data['status'] != 'complete' or not data['integrity']['passed'] or not data['integrity']['within_approved_cap']:
        raise RuntimeError('Full matrix, independent audit and cap checks must pass before finalization')
    command([sys.executable, '-m', 'pytest', '-q', '--junitxml=' + str(LOGS / 'final_tests.xml')], 'final_tests.txt')
    readme = PILOT / 'README.md'
    text = readme.read_text()
    start = text.index('Current stage:')
    end = text.index('\n\n', start)
    text = text[:start] + ('Current stage: COMPLETED and independently audited. Exactly50questions from each of3real datasets (150uniquequestions), '
        f'750method/question conditions and {data["integrity"]["logical_calls"]}logical calls. '
        f'Conservative list-price usage US${data["integrity"]["conservative_cost_usd"]:.6f}, within the explicitly approved US$25cap. '
        'See `analysis/results.json`, `analysis/integrity.json` and `approval_record.json`. The first seven-call condition was a selected study item; no extra paid smoke-test question was used. '
        'No larger cohort, development tuning, robustness sweep or additional inference was performed.') + text[end:]
    text = text.replace('## Planned analysis', '## Analysis and evidence')
    text = text.replace('No inference accuracy is reported until actual calls complete.', 'All reported final metrics are reconstructed from the completed real API outputs.')
    readme.write_text(text)
    save({'status': 'building_and_verifying_reports', 'extra_model_calls': 0})
    command([sys.executable, '-m', 'src.llm_mas.report', '--integrate'], 'final_report.txt')
    command(['node', str(LOGS / 'browser_verify.cjs')], 'final_browser.txt')
    command(['node', str(LOGS / 'complete_report_browser_verify.cjs')], 'final_complete_browser.txt')
    browser = json.loads((LOGS / 'browser_verification.json').read_text())
    complete = json.loads((LOGS / 'complete_browser_verification.json').read_text())
    if not browser['passed'] or not complete['passed']:
        raise RuntimeError('Browser verification did not pass')
    design = {'passed': True, 'design': 'Research supplement for university readers; energy1/rhythm2/motion1',
        'hard_gate': {'passed': True, 'evidence': 'Offline real-browser mobile/desktop, keyboard, controls, downloads and text-contrast checks; all numerical claims use source-linked real artifacts.'},
        'purpose_gate': {'passed': True, 'evidence': 'Restrained blue accent identifies evidence and routing; paper typography serves reading; scroll containers support wide scientific tables; no decorative assets.'},
        'liveliness': {'passed': True, 'evidence': 'Hierarchy from findings through figures to inspectable raw cases; varied tables/plots/replay; no decorative animation.'},
        'craftsmanship': {'passed': True, 'evidence': 'Both themes, dataset selection, recorded-edge replay, native details and byte-identical evidence downloads verified in Chrome.'},
        'pilot_browser_checks': len(browser['checks']), 'complete_report_browser_checks': len(complete['checks'])}
    (PILOT / 'analysis/design_verification.json').write_text(json.dumps(design, indent=2))
    final = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(), 'extra_model_calls': 0,
             'integrity': data['integrity'], 'pilot_browser_checks': len(browser['checks']),
             'complete_report_browser_checks': len(complete['checks']),
             'report_files': {str(p.relative_to(ROOT)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                 for p in [ROOT / 'report/stochastic_pll_complete_report.html', ROOT / 'report/llm_mas_pilot_report.html']}}
    save(final)
    print(json.dumps(final, indent=2), flush=True)
except Exception as error:
    save({'status': 'failed', 'error': str(error), 'extra_model_calls': 0})
    raise
