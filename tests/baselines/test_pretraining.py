import math

import pytest

from src.evaluation.comparison import paired_user_bootstrap, summarize_seeds
from src.evaluation.pretraining import evaluate_pretraining_predictions


def evaluate(components):
    return evaluate_pretraining_predictions(components, vocabulary_policy='v1',
                                            candidate_policy='full', k=2)['components']


def test_masked_distribution_and_special_exclusion():
    report = evaluate({'masked': {'kind': 'categorical', 'targets': [0, 1, 0],
                                      'valid_mask': [True, True, True],
                                      'masked_mask': [True, True, True],
                                      'special_mask': [False, False, True],
                                      'probabilities': [[0.8, 0.2], [0.6, 0.4], [0, 0]]}})['masked']
    assert report['count'] == 2
    assert report['cross_entropy'] == pytest.approx(-math.log(0.32) / 2)
    assert report['accuracy'] == 0.5
    assert report['recall_at_k'] == 1
    assert report['mrr'] == 0.75


def test_numeric_invalid_mask_and_empty():
    result = evaluate({'next_value': {'kind': 'numeric', 'targets': [1, 3, 0],
                                         'predictions': [2, 5, float('nan')],
                                         'valid_mask': [True, True, False]}})['next_value']
    assert result['mse'] == 2.5
    result = evaluate({'next_key': {'kind': 'categorical', 'targets': [0], 'valid_mask': [False]}})
    assert result['next_key']['cross_entropy'] is None


def test_actual_joint_probability_and_unavailable_ranking():
    result = evaluate({'next_key': {'kind': 'categorical', 'targets': [15], 'valid_mask': [True],
                                       'target_probabilities': [0.12]}})['next_key']
    assert result['cross_entropy'] == pytest.approx(-math.log(0.12))
    assert result['accuracy'] is None
    assert result['status'] == 'partially_unavailable'
    with pytest.raises(ValueError, match='normalized'):
        evaluate({'next_key': {'kind': 'categorical', 'targets': [0], 'valid_mask': [True],
                                  'probabilities': [[0.5, 0.7]]}})


def test_comparison():
    summary = summarize_seeds([{'ndcg': 0.2}, {'ndcg': 0.3}, {'ndcg': 0.4}])
    assert summary['ndcg']['mean'] == pytest.approx(0.3)
    assert summary['ndcg']['std'] == pytest.approx(0.1)
    result = paired_user_bootstrap({'u': [0.5, 0.8]}, {'u': [0.4, 0.7]}, repetitions=20)
    assert result['ci_low'] == pytest.approx(0.1)
    assert result['ci_high'] == pytest.approx(0.1)
