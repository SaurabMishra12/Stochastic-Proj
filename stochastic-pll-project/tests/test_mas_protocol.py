"""Offline checks for message isolation, exact scoring and stochastic routing."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
import json
from pathlib import Path

import pytest


def protocol():
    path = Path(__file__).parents[1] / 'src' / 'llm_mas' / 'protocol.py'
    assert path.exists(), 'The offline MAS protocol has not been implemented.'
    return importlib.import_module('src.llm_mas.protocol')


@pytest.mark.parametrize(('raw', 'expected'), [
    ('42', '42'), ('$1,234.50', '2469/2'), ('-£1,200', '-1200'),
    ('3/6', '1/2'), ('1.25e2', '125'), ('+0.50', '1/2'),
    ('-0', '0'), ('  1,000  ', '1000'), ('1.5/3', '1/2'),
])
def test_numeric_answers_are_exact_and_currency_and_fractions_compare(raw, expected):
    assert protocol().normalize_numeric(raw) == expected


@pytest.mark.parametrize('raw', [
    '', 'The answer is 42', '42 or 43', '12 apples', '1,2', '1/0',
    'NaN', 'Infinity', '1e999999', 'True', '42%', '--2', '$$2', '2/3/4',
])
def test_numeric_normalization_rejects_ambiguous_or_malformed_values(raw):
    assert protocol().normalize_numeric(raw) is None


def test_candidate_parser_accepts_only_one_unambiguous_json_object():
    p = protocol()
    assert p.parse_candidate('{"answer":"$1,200.00","summary":"Twelve hundreds."}') == {
        'answer': '1200', 'summary': 'Twelve hundreds.', 'valid': True}
    assert p.parse_candidate('```json\n{"answer":0.1,"summary":"A tenth."}\n```')['answer'] == '1/10'
    invalid = [
        '', 'Answer: 42', '{"answer":"42"}', '{"answer":true,"summary":"yes"}',
        '{"answer":"41","answer":"42","summary":"duplicate"}',
        '{"answer":NaN,"summary":"nan"}',
        '{"answer":"42","summary":"one"} {"answer":"43","summary":"two"}',
        'Here is JSON: {"answer":"42","summary":"prose wrapper"}',
        '{"answer":"42 or 43","summary":"ambiguous"}',
    ]
    for text in invalid:
        assert p.parse_candidate(text) == {'answer': None, 'summary': '', 'valid': False}


def test_qa_and_choice_parsing_use_exact_dataset_specific_normalization():
    p = protocol()
    assert p.normalize_answer(' The Eiffel Tower! ', 'text') == 'eiffel tower'
    assert p.normalize_answer('A New-York building', 'text') == 'newyork building'
    assert p.normalize_answer('yes', 'text') == 'yes'
    assert p.normalize_answer('the', 'text') is None
    assert p.normalize_answer('(b)', 'choice') == 'B'
    assert p.normalize_answer('A or B', 'choice') is None
    assert p.normalize_answer('D', 'choice') is None
    assert p.parse_candidate('{"answer":"The Eiffel Tower.","summary":"The landmark."}', 'text')['answer'] == 'eiffel tower'
    assert p.parse_candidate('{"answer":"(C)","summary":"Third option."}', 'choice')['answer'] == 'C'
    for kind in ('numeric', 'text', 'choice'):
        c = p.ProtocolConfig(answer_kind=kind)
        initial = p.build_initial_payload('QUESTION', 0, c)
        assert kind in initial['instructions']
    with pytest.raises(ValueError):
        p.ProtocolConfig(answer_kind='anything')


def candidates():
    return [
        {'answer': '10', 'summary': 'INITIAL_ZERO', 'valid': True},
        {'answer': '10', 'summary': 'INITIAL_ONE', 'valid': True},
        {'answer': '11', 'summary': 'INITIAL_TWO', 'valid': True},
    ]


def test_five_methods_keep_directed_topologies_and_conservative_disagreement():
    p = protocol()
    c = p.ProtocolConfig()
    all_edges = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    assert p.choose_edges('independent', candidates(), 'i', c)['edges'] == []
    assert p.choose_edges('full', candidates(), 'i', c)['edges'] == all_edges
    assert p.choose_edges('fixed_sparse', candidates(), 'i', c)['edges'] == [[0, 1], [1, 2], [2, 0]]
    triggered = p.choose_edges('disagreement_triggered', candidates(), 'i', c)
    assert triggered['disagreement'] == pytest.approx(2 / 3)
    assert triggered['edges'] == all_edges
    equal = [{'answer': '10', 'summary': 'ok', 'valid': True}] * 3
    assert p.choose_edges('disagreement_triggered', equal, 'i', c)['edges'] == []
    missing = [{'answer': None, 'summary': '', 'valid': False}] * 3
    assert p.choose_edges('disagreement_triggered', missing, 'i', c)['disagreement'] == 1
    assert p.choose_edges('disagreement_triggered', missing, 'i', c)['edges'] == all_edges


def test_controller_records_replayable_non_degenerate_independent_streams():
    p = protocol()
    c = p.ProtocolConfig()
    first = p.choose_edges('stochastic_controller', candidates(), 'i', c)
    assert first == p.choose_edges('stochastic_controller', candidates(), 'i', c)
    assert first['controller_seed'] != p.choose_edges('stochastic_controller', candidates(), 'j', c)['controller_seed']
    assert first['controller_seed'] != p.choose_edges('stochastic_controller', candidates(), 'i', replace(c, replicate_id=1))['controller_seed']
    assert len(first['states']) == 3
    assert len(first['edge_probabilities']) == len(first['uniform_draws']) == 6
    assert all(0 < row['probability'] < 1 for row in first['edge_probabilities'])
    assert all(0 <= row['draw'] < 1 for row in first['uniform_draws'])
    accepted = [row['edge'] for row, prob in zip(first['uniform_draws'], first['edge_probabilities'])
                if row['draw'] < prob['probability']]
    assert first['edges'] == accepted
    assert all(-2 <= row['state'] <= 2 for row in first['states'])
    sent_counts = {len(p.choose_edges('stochastic_controller', candidates(), str(i), c)['edges'])
                   for i in range(60)}
    assert len(sent_counts) > 1
    assert min(sent_counts) < 6 and max(sent_counts) > 0


def test_frozen_refinement_round_receives_only_selected_initial_peers_and_private_draft():
    p = protocol()
    c = p.ProtocolConfig()
    initial = candidates()
    snapshot = deepcopy(initial)
    payloads = [p.build_refinement_payload('QUESTION_SENTINEL', agent, initial,
                                        [[0, 1], [1, 2], [2, 0]], c)
                for agent in range(3)]
    body = [json.loads(payload['input'][0]['content']) for payload in payloads]
    assert body[0]['own_initial']['summary'] == 'INITIAL_ZERO'
    assert body[0]['received_messages'][0]['sender'] == 2
    assert body[0]['received_messages'][0]['summary'] == 'INITIAL_TWO'
    assert 'INITIAL_ONE' not in json.dumps(body[0])
    assert body[1]['received_messages'][0]['sender'] == 0
    assert body[2]['received_messages'][0]['sender'] == 1
    assert initial == snapshot
    initial[0]['summary'] = 'REFINED_SENTINEL'
    assert 'REFINED_SENTINEL' not in json.dumps(payloads)


def test_same_stage_instructions_and_settings_and_seven_call_structure():
    p = protocol()
    c = p.ProtocolConfig()
    question = 'QUESTION_ONLY_SENTINEL'
    own = candidates()
    initial = [p.build_initial_payload(question, i, c) for i in range(3)]
    adjudication = p.build_adjudication_payload(question, own, c)
    assert len(initial) + 3 + 1 == 7
    for method in p.METHODS:
        edges = p.choose_edges(method, own, 'i', c)['edges']
        refine = [p.build_refinement_payload(question, i, own, edges, c) for i in range(3)]
        for payload in initial + refine + [adjudication]:
            assert payload['model'] == c.model
            assert payload['max_output_tokens'] == 4096
            assert payload['reasoning'] == {'effort': 'low'}
            assert 'temperature' not in payload and 'seed' not in payload
            assert 'GOLD_SENTINEL' not in json.dumps(payload)
            assert question in payload['input'][0]['content']
        assert len({x['instructions'] for x in refine}) == 1
        assert refine[0]['instructions'] == p.build_refinement_payload(question, 0, own, [], c)['instructions']
    assert len({x['instructions'] for x in initial}) == 1
    assert json.loads(adjudication['input'][0]['content'])['refined_messages'][2]['sender'] == 2
    with_gold = deepcopy(own)
    with_gold[0]['gold_answer'] = 'GOLD_SENTINEL'
    assert 'GOLD_SENTINEL' not in json.dumps(p.build_refinement_payload(question, 1, with_gold, [[0, 1]], c))


def test_communication_content_is_capped_missing_explicit_and_counts_actual_bytes():
    p = protocol()
    c = p.ProtocolConfig()
    draft = candidates()
    draft[0]['summary'] = 'é' * 1000
    draft[1] = {'answer': None, 'summary': '', 'valid': False}
    record = p.communication_record(draft, [[0, 1], [1, 2], [2, 0]], c)
    assert record['directed_messages'] == 3
    assert len(record['messages'][0]['content']['summary']) == 600
    assert record['messages'][1]['content']['missing_response'] is True
    assert record['utf8_bytes'] == sum(len(x['serialized'].encode('utf-8')) for x in record['messages'])
    assert record['characters'] == sum(len(x['serialized']) for x in record['messages'])
    assert record['utf8_bytes'] > record['characters']
    assert 'QUESTION_ONLY_SENTINEL' not in json.dumps(record)


def test_config_is_immutable_and_fingerprint_changes_when_design_changes():
    p = protocol()
    config = p.ProtocolConfig()
    with pytest.raises(FrozenInstanceError):
        config.disagreement_threshold = 0
    assert len(config.fingerprint) == 64
    assert config.fingerprint == p.ProtocolConfig().fingerprint
    assert config.fingerprint != replace(config, disagreement_threshold=0.25).fingerprint
    for changes in ({'summary_max_chars': 601}, {'max_output_tokens': 0},
                    {'disagreement_threshold': -1}, {'replicate_id': -1}):
        with pytest.raises(ValueError):
            p.ProtocolConfig(**changes)


def test_invalid_agent_count_method_or_edge_fails_closed():
    p = protocol()
    c = p.ProtocolConfig()
    with pytest.raises(ValueError):
        p.choose_edges('unknown', candidates(), 'i', c)
    with pytest.raises(ValueError):
        p.choose_edges('full', candidates()[:2], 'i', c)
    for edges in ([[0, 0]], [[0, 3]], [[0, 1], [0, 1]]):
        with pytest.raises(ValueError):
            p.build_refinement_payload('question', 0, candidates(), edges, c)
