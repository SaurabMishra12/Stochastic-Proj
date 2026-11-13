"""Read-only analysis of the frozen pilot; this module never calls an LLM."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import binomtest

from .datasets import answer_f1
from .pilot import PILOT, ROOT, atomic, canonical, sha, code_hash
from .protocol import (METHODS, ProtocolConfig, parse_candidate, choose_edges, communication_record,
                       build_initial_payload, build_refinement_payload, build_adjudication_payload)


def wilson(successes, n, z=1.959963984540054):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError('Invalid binomial counts')
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., centre - radius), min(1., centre + radius)]


def paired_comparison(baseline, candidate, seed=20261113, draws=20000):
    a, b = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError('Paired observations must be equal nonempty vectors')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite paired observation')
    differences = b - a
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(a), size=(draws, len(a)))].mean(axis=1)
    wins = int(np.sum((a == 0) & (b == 1)))
    losses = int(np.sum((a == 1) & (b == 0)))
    discordant = wins + losses
    return {'n': len(a), 'delta': float(differences.mean()),
            'bootstrap_ci': np.quantile(boot, [.025, .975]).tolist(),
            'wins': wins, 'losses': losses, 'both_correct': int(np.sum((a == 1) & (b == 1))),
            'both_wrong': int(np.sum((a == 0) & (b == 0))),
            'mcnemar_p': float(binomtest(wins, discordant, .5).pvalue) if discordant else 1.,
            'bootstrap_draws': draws, 'bootstrap_seed': seed}


def mean_ci(values, seed=20261113, draws=20000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Mean interval requires nonempty finite question observations')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.] * len(order)
    previous = 0.
    for position, i in enumerate(order):
        previous = max(previous, min(1., p_values[i] * (len(order) - position)))
        adjusted[i] = previous
    return adjusted


def indexed(records):
    result = {}
    for row in records:
        key = row['dataset'], row['item_id'], row['method']
        if key in result:
            raise ValueError('Duplicate method/item record')
        result[key] = row
    return result


def aggregate(cohort, records, tokenizer=None):
    lookup = indexed(records)
    result = []
    for dataset in dict.fromkeys(r['dataset'] for r in cohort):
        items = [r for r in cohort if r['dataset'] == dataset]
        for method in METHODS:
            rows = [lookup.get((dataset, r['item_id'], method)) for r in items]
            observed = [r for r in rows if r is not None]
            n = len(items)
            correct = sum(bool(r and r.get('correct')) for r in rows)
            values = {'dataset': dataset, 'method': method, 'n': n, 'observed': len(observed),
                      'missing': n - len(observed), 'correct': correct, 'accuracy': correct / n,
                      'accuracy_ci': wilson(correct, n),
                      'f1': sum(r.get('f1', 0.) if r else 0. for r in rows) / n if dataset == 'hotpotqa' else None,
                      'f1_ci': mean_ci([r.get('f1', 0.) if r else 0. for r in rows]) if dataset == 'hotpotqa' else None,
                      'protocol_complete': sum(bool(r.get('completed')) for r in observed),
                      'final_valid': sum(bool(r.get('final_valid')) for r in observed)}
            for field in ['cost_usd', 'attempts', 'input_tokens', 'output_tokens',
                          'reasoning_tokens', 'usage_unknown_attempts', 'latency_seconds', 'wall_seconds']:
                values[field] = sum(r.get(field, 0) for r in observed)
            values['directed_messages'] = sum(r.get('communication', {}).get('directed_messages', 0) for r in observed)
            values['peer_utf8_bytes'] = sum(r.get('communication', {}).get('utf8_bytes', 0) for r in observed)
            values['peer_content_tokens'] = (sum(len(tokenizer.encode_ordinary(m['serialized']))
                for r in observed for m in r.get('communication', {}).get('messages', [])) if tokenizer else None)
            latency = [r['latency_seconds'] for r in observed if 'latency_seconds' in r]
            values['median_http_seconds_per_condition'] = float(np.median(latency)) if latency else None
            values['p95_http_seconds_per_condition'] = float(np.quantile(latency, .95)) if latency else None
            values['mean_messages_per_question'] = values['directed_messages'] / n
            values['mean_cost_per_question_usd'] = values['cost_usd'] / n
            values['initial_disagreement_mean'] = (sum(r.get('initial_disagreement', 0) for r in observed) / len(observed)) if observed else None
            result.append(values)
    return result


def comparisons(cohort, records):
    lookup = indexed(records)
    result = []
    for dataset in dict.fromkeys(r['dataset'] for r in cohort):
        items = [r for r in cohort if r['dataset'] == dataset]
        base = [int(lookup.get((dataset, r['item_id'], 'independent'), {}).get('correct', False)) for r in items]
        for method in METHODS[1:]:
            candidate = [int(lookup.get((dataset, r['item_id'], method), {}).get('correct', False)) for r in items]
            result.append({'dataset': dataset, 'baseline': 'independent', 'method': method,
                           **paired_comparison(base, candidate, seed=20261113 + len(result))})
    for row, p in zip(result, holm([r['mcnemar_p'] for r in result])):
        row['holm_p'] = p
    return result


def audit(pilot, cohort, records):
    """Rebuild scores and routing from saved text, then reconcile physical attempts."""
    issues = []
    manifest = json.loads((pilot / 'run_manifest.json').read_text())
    if sha(canonical({k: v for k, v in manifest.items() if k != 'manifest_sha256'})) != manifest['manifest_sha256']:
        issues.append('Run manifest self-checksum changed')
    approval = json.loads((pilot / 'approval_record.json').read_text())
    if not approval.get('approved') or approval.get('manifest_sha256') != manifest['manifest_sha256'] or approval.get('max_usd') != 25:
        issues.append('Approval does not match the frozen manifest and cap')
    if hashlib.sha256((pilot / 'cohort.jsonl').read_bytes()).hexdigest() != manifest['cohort_sha256']:
        issues.append('Cohort checksum changed')
    if code_hash() != manifest['source_hash']:
        issues.append('Inference source changed after approval')
    source_manifest = json.loads((pilot / 'cohort_manifest.json').read_text())
    for dataset, source in source_manifest['sources'].items():
        if hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() != source['sha256']:
            issues.append(f'Original source dataset changed: {dataset}')
    cohort_lookup = {r['item_id']: r for r in cohort}
    seen = indexed(records)
    expected_conditions = {(r['dataset'], r['item_id'], method) for r in cohort for method in METHODS}
    if set(seen) - expected_conditions:
        issues.append('Unexpected dataset/item/method condition identity')
    db = sqlite3.connect(f'file:{pilot / "budget.sqlite3"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    attempts = [dict(r) for r in db.execute('SELECT * FROM attempts')]
    db.close()
    by_logical = {}
    for attempt in attempts:
        by_logical.setdefault(attempt['logical_id'], []).append(attempt)
    checked_payloads = 0
    for result in records:
        source = cohort_lookup[result['item_id']]
        config = ProtocolConfig(answer_kind=source['answer_kind'])
        folder = pilot / 'artifacts/conditions' / f"{source['dataset']}__{source['source_index']}__{result['method']}"
        candidates = []
        for i in range(3):
            saved = json.loads((folder / f'initial_{i}.json').read_text())['result']
            candidates.append(parse_candidate(saved.get('text', ''), config.answer_kind)
                if saved.get('outcome') == 'completed' else {'answer': None, 'summary': '', 'valid': False})
        control = choose_edges(result['method'], candidates, result['item_id'], config)
        if control != result['controller']:
            issues.append(f"Routing replay mismatch: {folder.name}")
        if communication_record(candidates, control['edges'], config) != result['communication']:
            issues.append(f"Peer-message replay mismatch: {folder.name}")
        refined = []
        for i in range(3):
            saved = json.loads((folder / f'refinement_{i}.json').read_text())['result']
            refined.append(parse_candidate(saved.get('text', ''), config.answer_kind)
                if saved.get('outcome') == 'completed' else {'answer': None, 'summary': '', 'valid': False})
        question = source['model_input']
        stages = ([('initial', i, build_initial_payload(question, i, config)) for i in range(3)]
            + [('refinement', i, build_refinement_payload(question, i, candidates, control['edges'], config)) for i in range(3)]
            + [('adjudication', 0, build_adjudication_payload(question, refined, config))])
        for stage, i, payload in stages:
            lid = f"{result['item_id']}|{result['method']}|{stage}:{i}"
            checkpoint = json.loads((folder / f'{stage}_{i}.json').read_text())
            if checkpoint['payload_hash'] != sha(canonical(payload)) or checkpoint['logical_id'] != lid:
                issues.append(f'Checkpoint identity mismatch: {lid}')
            for attempt in by_logical.get(lid, []):
                request_path = pilot / 'artifacts/http' / (attempt['attempt_id'] + '.request.json')
                if request_path.exists() and json.loads(request_path.read_text()) != payload:
                    issues.append(f'Actual request differs from label-free reconstructed payload: {lid}')
                checked_payloads += 1
        saved = json.loads((folder / 'adjudication_0.json').read_text())['result']
        final = parse_candidate(saved.get('text', ''), config.answer_kind) if saved.get('outcome') == 'completed' else {'valid': False, 'answer': None}
        correct = bool(final['valid'] and final['answer'] == source['gold_answer'])
        f1 = answer_f1(final['answer'], source['gold_answer']) if source['dataset'] == 'hotpotqa' and final['valid'] else float(correct)
        if correct != result['correct'] or f1 != result['f1'] or final['answer'] != result['final_answer']:
            issues.append(f"Scoring mismatch: {folder.name}")
        if result['source_hash'] != manifest['source_hash'] or result['config_hash'] != config.fingerprint:
            issues.append(f"Protocol identity mismatch: {folder.name}")
    outcomes = Counter(r['outcome'] for r in attempts)
    actual_nusd = sum(r['actual_nusd'] or 0 for r in attempts)
    uncertain_nusd = sum(r['reserved_nusd'] for r in attempts if r['actual_nusd'] is None)
    raw_count = 0
    response_ids = []
    model_versions = Counter()
    billed_usage = Counter()
    missing_artifacts = []
    for attempt in attempts:
        prefix = pilot / 'artifacts/http' / attempt['attempt_id']
        for suffix in ['.request.json', '.attempt.json']:
            if not prefix.with_suffix(suffix).exists():
                missing_artifacts.append(str(prefix.with_suffix(suffix).relative_to(pilot)))
        response_path = prefix.with_suffix('.response.json')
        if response_path.exists():
            raw_count += 1
            raw = json.loads(response_path.read_text())
            if raw.get('id'):
                response_ids.append(raw['id'])
            if raw.get('model'):
                model_versions[raw['model']] += 1
            usage = json.loads(attempt['usage_json']) if attempt['usage_json'] else None
            if usage and raw.get('usage') != usage:
                issues.append(f"Raw/ledger usage mismatch: {attempt['attempt_id']}")
        elif attempt['outcome'] == 'completed':
            missing_artifacts.append(str(response_path.relative_to(pilot)))
        usage = json.loads(attempt['usage_json']) if attempt['usage_json'] else None
        if usage:
            expected_cost = usage['input_tokens'] * 250 + usage['output_tokens'] * 2000
            if expected_cost != attempt['actual_nusd']:
                issues.append(f"Raw usage independently repriced differently: {attempt['attempt_id']}")
            if usage['input_tokens'] > attempt['input_bound'] or usage['output_tokens'] > attempt['output_bound']:
                issues.append(f"Provider usage exceeds reserved token bounds: {attempt['attempt_id']}")
            billed_usage['input_tokens'] += usage['input_tokens']
            billed_usage['output_tokens'] += usage['output_tokens']
            billed_usage['reasoning_tokens'] += usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
            billed_usage['cached_input_tokens'] += usage.get('input_tokens_details', {}).get('cached_tokens', 0)
    if missing_artifacts:
        issues.append(f'{len(missing_artifacts)} missing raw artifacts')
    if len(set(response_ids)) != len(response_ids):
        issues.append('Duplicate provider response identities across physical attempts')
    logical_ids = {r['logical_id'] for r in attempts}
    expected_ids = {f"{r['item_id']}|{method}|{stage}:{i}" for r in cohort for method in METHODS
                    for stage, count in [('initial', 3), ('refinement', 3), ('adjudication', 1)] for i in range(count)}
    unexpected = sorted(logical_ids - expected_ids)
    if unexpected:
        issues.append('Physical attempts outside the frozen logical-slot matrix')
    condition_cost = sum(r['cost_usd'] for r in records)
    settled_cost = (actual_nusd + uncertain_nusd) / 1e9
    complete_matrix = set(seen) == expected_conditions and logical_ids == expected_ids
    if complete_matrix and abs(condition_cost - settled_cost) > 1e-8:
        issues.append('Condition cost does not reconcile with physical-attempt ledger')
    return {'passed': not issues, 'complete_matrix': complete_matrix, 'issues': issues,
            'expected_conditions': 750, 'observed_conditions': len(seen), 'expected_logical_calls': 5250,
            'logical_calls': len(logical_ids), 'http_attempts': len(attempts), 'raw_responses': raw_count,
            'reconstructed_request_payloads': checked_payloads,
            'attempt_outcomes': dict(outcomes), 'usage': dict(billed_usage),
            'provider_model_versions': dict(model_versions), 'distinct_provider_response_ids': len(set(response_ids)),
            'first_attempt_utc': min((r['created_at'] for r in attempts), default=None),
            'last_settled_utc': max((r['settled_at'] for r in attempts if r['settled_at']), default=None),
            'actual_cost_usd': actual_nusd / 1e9, 'uncertain_reservation_usd': uncertain_nusd / 1e9,
            'conservative_cost_usd': settled_cost, 'condition_cost_usd': condition_cost,
            'within_approved_cap': settled_cost <= 25, 'unexpected_logical_ids': unexpected,
            'missing_artifacts': missing_artifacts}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def run(pilot=PILOT, tokenizer=None):
    pilot = Path(pilot)
    cohort = [json.loads(s) for s in (pilot / 'cohort.jsonl').read_text().splitlines()]
    paths = sorted((pilot / 'artifacts/conditions').glob('*/result.json'))
    records = [json.loads(p.read_text()) for p in paths]
    output = pilot / 'analysis'
    output.mkdir(exist_ok=True)
    integrity = audit(pilot, cohort, records)
    summary = aggregate(cohort, records, tokenizer)
    paired = comparisons(cohort, records)
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'complete' if integrity['complete_matrix'] else 'partial',
                'n_examples': len(cohort), 'cohort_sha256': hashlib.sha256((pilot / 'cohort.jsonl').read_bytes()).hexdigest(),
                'summary': summary, 'paired_comparisons': paired, 'integrity': integrity,
                'tokenizer': {'name': 'o200k_base', 'version': 'tiktoken0.12.0',
                              'scope': 'sum of each serialized directed peer-message content; repeated delivery is counted; excludes framing, questions, own drafts, and common adjudication'} if tokenizer else None,
                'uncertainty': 'Wilson95% accuracy intervals; 20000 question-paired bootstrap draws for accuracy differences; exact two-sided McNemar; Holm over12 comparisons. No provider-seed or repeated-model-run uncertainty estimate.',
                'limitations': ['Exploratory50questions per dataset; oneBBH task only.',
                    'Empirical bootstrap intervals can collapse when all observed outcomes or paired differences are identical; that does not establish population equality or noninferiority.',
                    'One real API realization per method; model outputs are not deterministic despite fixed cohort/controller seeds.',
                    'Independent initial generations across methods; question pairing does not couple provider randomness.',
                    'OU-style disagreement gate has no derived physicalPLL phase mapping.',
                    'The gate measures normalized answer-string disagreement; it does not model semantic equivalence, entity aliases or answer-probability distributions.',
                    'No dropout, corruption, delay, cross-model, or controller-tuning experiments in this pilot.',
                    'All50questions per dataset enter accuracy denominators; missing/invalid final answers score0; intermediate failure is reported separately.',
                    'Peer-content tokens are local token counts, not isolated provider-billed communication tokens.',
                    'Uncached retail-price ledger is conservative; account invoice, discounts and tax are not verified.',
                    'SummedHTTPattempt latency excludes rate-limiter waits; wall latency includes shared-queue waiting and is not isolated method performance.']}
    packages = {}
    for package in ['numpy', 'scipy', 'pandas', 'pyarrow', 'matplotlib', 'tiktoken']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'not available in current analysis environment'
    metadata['environment'] = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
    atomic(output / 'results.json', metadata)
    write_csv(output / 'metrics.csv', summary)
    write_csv(output / 'paired_comparisons.csv', paired)
    atomic(output / 'integrity.json', integrity)
    flat = []
    lookup = indexed(records)
    for row in cohort:
        for method in METHODS:
            record = lookup.get((row['dataset'], row['item_id'], method), {})
            flat.append({'dataset': row['dataset'], 'item_id': row['item_id'], 'method': method,
                         'gold_answer': row['gold_answer'], 'final_answer': record.get('final_answer'),
                         'observed': bool(record), 'correct': record.get('correct', False),
                         'final_valid': record.get('final_valid', False), 'completed': record.get('completed', False),
                         'f1': record.get('f1', 0), 'cost_usd': record.get('cost_usd', 0),
                         'initial_disagreement': record.get('initial_disagreement'),
                         'messages': record.get('communication', {}).get('directed_messages', 0),
                         'errors': record.get('errors', [])})
    write_csv(output / 'per_question.csv', flat)
    (output / 'all_condition_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in records))
    atomic(output / 'analysis_provenance.json', {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'condition_result_files': {str(p.relative_to(pilot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           'read_only_model_calls': 0})
    return metadata


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, str(PILOT.parents[2] / '.analysis_deps'))
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(PILOT / 'tokenizer_cache'))
    import tiktoken
    result = run(tokenizer=tiktoken.get_encoding('o200k_base'))
    print(json.dumps({'status': result['status'], 'integrity': result['integrity']}, indent=2))
