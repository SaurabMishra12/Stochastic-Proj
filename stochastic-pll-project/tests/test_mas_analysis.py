import importlib

import pytest


def module():
    return importlib.import_module('src.llm_mas.analysis')


def test_wilson_boundaries_and_exact_paired_test():
    m = module()
    low, high = m.wilson(0, 50)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(0.0713476, rel=1e-5)
    assert m.paired_comparison([0] * 50, [1] * 6 + [0] * 44)['mcnemar_p'] == pytest.approx(0.03125)


def test_pairing_requires_equal_nonempty_samples():
    m = module()
    with pytest.raises(ValueError):
        m.paired_comparison([0, 1], [1])
    with pytest.raises(ValueError):
        m.paired_comparison([], [])


def test_holm_keeps_original_order_and_is_monotone():
    m = module()
    assert m.holm([0.04, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])


def test_missing_rows_stay_in_fixed_denominator():
    m = module()
    cohort = [{'dataset': 'gsm8k', 'item_id': str(i)} for i in range(50)]
    records = [{'dataset': 'gsm8k', 'item_id': '0', 'method': 'independent', 'correct': True,
                'completed': True, 'final_valid': True, 'f1': 1,
                'communication': {'messages': [], 'directed_messages': 0}, 'cost_usd': 0.01}]
    rows = m.aggregate(cohort, records)
    base = next(r for r in rows if r['method'] == 'independent')
    assert base['n'] == 50 and base['observed'] == 1 and base['missing'] == 49
    assert base['correct'] == 1 and base['accuracy'] == 0.02
    assert len(rows) == 5


def test_duplicate_condition_rejected():
    m = module()
    row = {'dataset': 'gsm8k', 'item_id': '0', 'method': 'independent', 'correct': True}
    with pytest.raises(ValueError):
        m.aggregate([{'dataset': 'gsm8k', 'item_id': '0'}], [row, row])


def test_bootstrap_is_paired_and_reproducible():
    m = module()
    a = [0, 1, 0, 1] * 12 + [0, 1]
    r = m.paired_comparison(a, a, seed=20261113)
    assert r == m.paired_comparison(a, a, seed=20261113)
    assert r['delta'] == 0 and r['bootstrap_ci'] == [0, 0] and r['mcnemar_p'] == 1


def test_f1_question_bootstrap_constant_boundaries():
    m = module()
    assert m.mean_ci([1.] * 50) == [1., 1.]
    assert m.mean_ci([0.] * 50) == [0., 0.]
