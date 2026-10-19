import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Interaction,
    Label,
    Query,
)
from src.baselines.runner import run
from src.baselines.sequence import SequenceClassifier, sequence_samples


@pytest.fixture
def food():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    queries, candidates, labels = [], [], []
    for split, day in [('train', 1), ('validation', 3), ('test', 5)]:
        for i in range(2):
            request = f'{split}-{i}'
            cutoff = start + timedelta(days=day, hours=i)
            queries.append(Query(request, f'u{i}', cutoff, split))
            for item in ['a', 'b']:
                candidates.append(Candidate(request, item))
                labels.append(Label(request, item, int(item == 'a'), cutoff + timedelta(minutes=1)))
    history = (Interaction('e', 'u0', 'a', 'order', start, start),)
    return BaselineDataset(DatasetMetadata('fixture', 'fixture-v1', 'food_recommendation',
                                          'food', 'UTC', 'fixed', 'explicit'),
                           tuple(queries), tuple(candidates), tuple(labels), history)


def cutoff(dataset, split):
    return min(q.cutoff_time for q in dataset.queries if q.split == split)


def checkpoint(model, tmp_path):
    vocabulary = tmp_path / 'vocabulary.json'
    model.artifacts.save(vocabulary)
    path = tmp_path / 'fixture.pt'
    torch.save({'format_version': 1, 'fixture_only': True,
                    'vocabulary_file': 'vocabulary.json',
                    'vocabulary_sha256': hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
                    'model_config': asdict(model.model_config),
                    'model_state_dict': model.backbone.state_dict()}, path)
    return str(path)


def test_scratch_reload_and_future_invariance(food, tmp_path):
    torch.set_num_threads(1)
    model = SequenceClassifier(epochs=2, fixture_only=True).fit(
        food, training_cutoff=cutoff(food, 'validation'))
    predictions = model.predict(food, 'test', 'run', frozen_cutoff=cutoff(food, 'test'))
    model.save(tmp_path / 'model')
    restored = SequenceClassifier.load(tmp_path / 'model')
    actual = restored.predict(food, 'test', 'run', frozen_cutoff=cutoff(food, 'test'))
    np.testing.assert_allclose(predictions['score'], actual['score'], atol=1e-6)
    future = Interaction('future', 'u0', 'b', 'order', cutoff(food, 'test') + timedelta(days=1),
                         cutoff(food, 'test') + timedelta(days=1))
    changed = replace(food, history=food.history + (future,))
    assert sequence_samples(food, 'test', cutoff(food, 'test')) == sequence_samples(
        changed, 'test', cutoff(food, 'test'))
    np.testing.assert_allclose(predictions['score'], model.predict(
        changed, 'test', 'run', frozen_cutoff=cutoff(food, 'test'))['score'])
    with pytest.raises(ContractError, match='B2'):
        SequenceClassifier(checkpoint_path='wrong')


@pytest.mark.parametrize('kind', ['B3', 'B4'])
def test_freeze_or_finetune(food, tmp_path, kind):
    torch.set_num_threads(1)
    scratch = SequenceClassifier(epochs=1, fixture_only=True).fit(
        food, training_cutoff=cutoff(food, 'validation'))
    path = checkpoint(scratch, tmp_path)
    before = {k: v.clone() for k, v in scratch.backbone.state_dict().items()}
    model = SequenceClassifier(model_kind=kind, epochs=2, checkpoint_path=path,
                               fixture_only=True).fit(food, training_cutoff=cutoff(food, 'validation'))
    changed = any(not torch.equal(before[k], v) for k, v in model.backbone.state_dict().items())
    assert changed == (kind == 'B4')
    model.save(tmp_path / kind)
    restored = SequenceClassifier.load(tmp_path / kind)
    expected = model.predict(food, 'test', 'r', frozen_cutoff=cutoff(food, 'test'))
    actual = restored.predict(food, 'test', 'r', frozen_cutoff=cutoff(food, 'test'))
    np.testing.assert_allclose(expected['score'], actual['score'], atol=1e-6)
    if kind == 'B3':
        ads = replace(food, metadata=replace(food.metadata, task='ad_optimization', item_kind='ad'))
        ad_model = SequenceClassifier(model_kind='B3', epochs=1, checkpoint_path=path,
                                      fixture_only=True).fit(ads, training_cutoff=cutoff(ads, 'validation'))
        scores = ad_model.predict(ads, 'test', 'r', frozen_cutoff=cutoff(ads, 'test'))['score'].to_pylist()
        assert all(0 <= score <= 1 for score in scores)


def test_profile_and_checkpoint_validation(food, tmp_path):
    with pytest.raises(ContractError, match='profile'):
        sequence_samples(food, 'train', profiles={'train-0': {
            'as_of': cutoff(food, 'test').isoformat(),
            'available_at': cutoff(food, 'test').isoformat(), 'values': {}}})
    with pytest.raises(ContractError, match='require checkpoint'):
        SequenceClassifier(model_kind='B3').fit(food, training_cutoff=cutoff(food, 'validation'))


@pytest.mark.parametrize('kind', ['B2', 'B3', 'B4'])
def test_sequence_runner(food, tmp_path, kind):
    dataset = asdict(food)
    (tmp_path / 'data.json').write_text(json.dumps(dataset, default=lambda v: v.isoformat()),
                                      encoding='utf-8')
    parameters = {'epochs': 1, 'fixture_only': True}
    if kind != 'B2':
        scratch = SequenceClassifier(epochs=1, fixture_only=True).fit(
            food, training_cutoff=cutoff(food, 'validation'))
        parameters['checkpoint_path'] = checkpoint(scratch, tmp_path)
    config = {'dataset_path': 'data.json',
              'artifact_dir': str(Path(__file__).resolve().parents[2] / 'experiments'),
              'model_kind': kind, 'split': 'test',
              'fit_cutoff': cutoff(food, 'validation').isoformat(),
              'prediction_cutoff': cutoff(food, 'test').isoformat(),
              'parameters': parameters, 'ks': [10, 20], 'zero_positive_policy': 'include'}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    assert run(path, validate_only=True)['status'] == 'passed'
    result = run(path)
    manifest = json.loads((Path(result['artifact_dir']) / 'run_manifest.json').read_text())
    assert manifest['model_kind'] == kind
    assert manifest['resources']['parameter_count'] > 0
    assert manifest['fixture_only'] is True
    assert 'ndcg_at_10' in result['metrics']


def test_corrupt_checkpoint_rejected(food, tmp_path):
    scratch = SequenceClassifier(epochs=1).fit(food, training_cutoff=cutoff(food, 'validation'))
    path = checkpoint(scratch, tmp_path)
    (tmp_path / 'vocabulary.json').write_text('{}')
    with pytest.raises(ContractError, match='checksum'):
        SequenceClassifier(model_kind='B3', fixture_only=True, checkpoint_path=path).fit(
            food, training_cutoff=cutoff(food, 'validation'))
