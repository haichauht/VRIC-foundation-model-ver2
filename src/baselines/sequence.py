"""Sequence baselines using only the public foundation tokenizer and embedding API."""

import copy
import ctypes
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import torch
from torch import nn

from src.foundation_input.dataset import DatasetConfig, encode_sample, foundation_collate
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.types import SPECIAL_TOKEN_IDS, CanonicalSample, EventRecord
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary_from_samples,
)
from src.models.encoder import FoundationModel, FoundationModelConfig

from ..evaluation.metrics import advertising_metrics, ranking_metrics
from ..evaluation.predictions import PREDICTION_SCHEMA, validate_predictions
from .contracts import ContractError, eligible_history


def process_peak_memory():
    """Process high-water RSS, including native tensors (not per-run allocation)."""
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def tokenizer_from_artifacts(artifacts):
    contract = artifacts.tokenizer_contract
    if not contract:
        raise ContractError("checkpoint vocabulary missing tokenizer contract")
    return TokenizerConfig(
        context_allowlist={k: tuple(v) for k, v in contract['context_allowlist'].items()},
        numeric_fields=frozenset(tuple(v) for v in contract['numeric_fields']),
        max_text_chars=contract['max_text_chars'],
        include_action_content=contract['include_action_content'],
        time_gap_boundaries_seconds=tuple(contract['time_gap_boundaries_seconds']),
    )


def sequence_samples(dataset, split, frozen_cutoff=None, profiles=None):
    """Profiles are request keyed, with explicit as_of/available_at timestamps."""
    dataset.validate()
    samples = []
    for query in dataset.queries:
        if query.split != split:
            continue
        cutoff = frozen_cutoff or query.cutoff_time
        if cutoff > query.cutoff_time or (split == 'train' and frozen_cutoff is not None):
            raise ContractError("invalid history boundary")
        profile = (profiles or {}).get(query.request_id)
        values = {}
        if profile:
            for field in ('as_of', 'available_at'):
                timestamp = datetime.fromisoformat(profile[field].replace('Z', '+00:00'))
                if timestamp.tzinfo is None or timestamp >= cutoff:
                    raise ContractError(f"{query.request_id}: profile {field} violates cutoff")
            values = dict(profile['values'])
        events = []
        history = sorted(eligible_history(dataset.history, cutoff),
                         key=lambda event: (event.event_time, event.event_id))
        for event in history:
            if event.user_id != query.user_id:
                continue
            events.append(EventRecord(
                event.event_id, event.event_id, len(events), 0, 1, event.event_time,
                'second', event.available_at, False, None, event.event_type,
                {}, {'entity_id': event.entity_id}, event.user_id,
            ))
        samples.append(CanonicalSample(
            query.request_id, dataset.metadata.dataset_name, split, query.user_id,
            query.cutoff_time, False, True, False, False, values, tuple(events),
        ))
    if not samples:
        raise ContractError(f"missing {split} queries")
    return samples


class SequenceClassifier:
    """B2 scratch, B3 frozen, B4 fine-tuned; one independently fitted task per instance."""

    def __init__(self, model_kind='B2', seed=42, epochs=10, learning_rate=0.001,
                 patience=3, d_model=32, num_heads=4, num_layers=1,
                 max_tokens=128, checkpoint_path=None, fixture_only=False):
        if model_kind not in {'B2', 'B3', 'B4'}:
            raise ContractError('unknown sequence model kind')
        if epochs <= 0 or patience <= 0 or learning_rate <= 0 or max_tokens < 4:
            raise ContractError('invalid sequence training parameters')
        if model_kind == 'B2' and checkpoint_path is not None:
            raise ContractError('B2 must not load pretrained weights')
        self.config = {"model_kind": model_kind, "seed": seed, "epochs": epochs,
                           "learning_rate": learning_rate, "patience": patience, "d_model": d_model,
                           "num_heads": num_heads, "num_layers": num_layers, "max_tokens": max_tokens,
                           "checkpoint_path": checkpoint_path, "fixture_only": fixture_only}
        self.backbone = None
        self.training_log = []

    def _batch(self, sample):
        return foundation_collate([encode_sample(
            sample, self.artifacts, self.tokenizer,
            DatasetConfig(max_tokens=self.config['max_tokens']),
        )])

    def _initialize(self, dataset, samples):
        torch.manual_seed(self.config['seed'])
        kind = self.config['model_kind']
        if kind == 'B2':
            self.tokenizer = TokenizerConfig()
            self.artifacts = fit_vocabulary_from_samples(
                samples, VocabularyConfig(dataset.metadata.dataset_name, text_hash_buckets=64),
                self.tokenizer,
            )
            # Register routing schema even when this training sample has no numeric
            # fields or events. IDs are persisted in the artifact, not constants.
            for mapping, name in ((self.artifacts.modality_to_id, 'numeric'),
                                  (self.artifacts.source_to_id, 'sequence')):
                if name not in mapping:
                    mapping[name] = max([*SPECIAL_TOKEN_IDS.values(), *mapping.values()]) + 1
            dimensions = self.artifacts.model_cardinalities()
            dimensions.pop('text_hash_buckets')
            self.model_config = FoundationModelConfig(
                **dimensions, d_model=self.config['d_model'],
                num_heads=self.config['num_heads'], num_layers=self.config['num_layers'],
                max_length=self.config['max_tokens'], num_user_buckets=128,
                categorical_pair_buckets=257,
                sequence_source_id=self.artifacts.source_to_id['sequence'],
                numeric_modality_id=self.artifacts.modality_to_id['numeric'],
            )
            self.backbone = FoundationModel(self.model_config)
            self.foundation = None
        else:
            path = self.config['checkpoint_path']
            if not path:
                raise ContractError('B3/B4 require checkpoint_path')
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            if bundle.get('format_version') != 1 or not bundle.get('fixture_only'):
                raise ContractError('production requires foundation-team public checkpoint loader')
            if not self.config['fixture_only']:
                raise ContractError('fixture checkpoint requires fixture_only=true')
            vocab_path = Path(path).parent / bundle['vocabulary_file']
            if not vocab_path.resolve().is_relative_to(Path(path).parent.resolve()):
                raise ContractError('checkpoint vocabulary path escapes bundle directory')
            if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != bundle['vocabulary_sha256']:
                raise ContractError('checkpoint vocabulary checksum mismatch')
            self.artifacts = VocabularyArtifacts.load(vocab_path)
            self.tokenizer = tokenizer_from_artifacts(self.artifacts)
            self.model_config = FoundationModelConfig(**bundle['model_config'])
            if self.artifacts.dataset_name != dataset.metadata.dataset_name:
                raise ContractError('checkpoint namespace mismatch; explicit mapping required')
            cardinalities = self.artifacts.model_cardinalities()
            for field in ('value_vocab_size', 'key_vocab_size', 'modality_vocab_size',
                          'source_vocab_size', 'time_gap_vocab_size'):
                if getattr(self.model_config, field) != cardinalities[field]:
                    raise ContractError('checkpoint vocabulary/model mismatch')
            if (self.model_config.sequence_source_id != self.artifacts.source_to_id['sequence']
                    or self.model_config.numeric_modality_id != self.artifacts.modality_to_id['numeric']):
                raise ContractError('checkpoint routing mismatch')
            self.backbone = FoundationModel(self.model_config)
            self.backbone.load_state_dict(bundle['model_state_dict'], strict=True)
            self.foundation = {'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               'fixture_only': True}
        self.items = {item: i + 1 for i, item in enumerate(sorted({
            c.candidate_id for c in dataset.candidates
            if any(q.request_id == c.request_id and q.split == 'train' for q in dataset.queries)
        }))}
        self.candidates = nn.Embedding(len(self.items) + 1, self.model_config.d_model)
        self.head = nn.Sequential(nn.Linear(3 * self.model_config.d_model, self.model_config.d_model),
                                  nn.ReLU(), nn.Linear(self.model_config.d_model, 1))
        self.backbone.requires_grad_(kind != 'B3')

    def _logits(self, sample, candidates):
        if self.config['model_kind'] == 'B3':
            self.backbone.eval()
            with torch.no_grad():
                embeddings = self.backbone.embed_users(self._batch(sample))
        else:
            embeddings = self.backbone.embed_users(self._batch(sample))
        user = torch.cat([embeddings['long_term'], embeddings['short_term']], dim=-1)
        ids = torch.tensor([self.items.get(c.candidate_id, 0) for c in candidates])
        return self.head(torch.cat([user.expand(len(ids), -1), self.candidates(ids)], dim=-1)).flatten()

    def _state(self):
        return {name: copy.deepcopy(getattr(self, name).state_dict())
                for name in ('backbone', 'candidates', 'head')}

    def fit(self, dataset, *, training_cutoff, profiles=None):
        train = sequence_samples(dataset, 'train', profiles=profiles)
        valid = sequence_samples(dataset, 'validation', training_cutoff, profiles)
        if any(s.cutoff_time >= training_cutoff for s in train):
            raise ContractError('train must precede training cutoff')
        tests = [q.cutoff_time for q in dataset.queries if q.split == 'test']
        if tests and max(s.cutoff_time for s in valid) >= min(tests):
            raise ContractError('validation must precede test')
        valid_ids = {s.sample_id for s in valid}
        if tests and any(l.available_at >= min(tests) for l in dataset.labels
                         if l.request_id in valid_ids):
            raise ContractError('validation label unavailable before test boundary')
        train_ids = {s.sample_id for s in train}
        if any(l.available_at >= training_cutoff for l in dataset.labels if l.request_id in train_ids):
            raise ContractError('train label unavailable at training cutoff')
        if dataset.metadata.task != 'food_recommendation' and self.config['model_kind'] != 'B3':
            raise ContractError('B2/B4 currently support Food only')
        self.identity = asdict(dataset.metadata)
        self.training_cutoff = training_cutoff
        self.training_log = []
        self._initialize(dataset, train)
        # Validate tokenizer/model compatibility before the first optimizer step.
        self.backbone.eval()
        with torch.no_grad():
            for sample in train + valid:
                self.backbone.embed_users(self._batch(sample))
        labels = {(l.request_id, l.candidate_id): l.target for l in dataset.labels}
        if {l.target for l in dataset.labels if l.request_id in train_ids} != {0, 1}:
            raise ContractError('training requires both classes')
        groups = {q.request_id: [c for c in dataset.candidates if c.request_id == q.request_id]
                  for q in dataset.queries}
        parameters = [{'params': list(self.head.parameters()) + list(self.candidates.parameters())}]
        if self.config['model_kind'] != 'B3':
            parameters.append({'params': self.backbone.parameters(),
                               'lr': self.config['learning_rate'] *
                               (0.1 if self.config['model_kind'] == 'B4' else 1)})
        optimizer = torch.optim.Adam(parameters, lr=self.config['learning_rate'])
        best, stale = float('-inf'), 0
        started = time.perf_counter()
        for epoch in range(self.config['epochs']):
            self.backbone.train(self.config['model_kind'] != 'B3')
            self.head.train()
            total = 0.0
            for sample in train:
                candidates = groups[sample.sample_id]
                target = torch.tensor([labels[(sample.sample_id, c.candidate_id)]
                                       for c in candidates], dtype=torch.float32)
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self._logits(sample, candidates), target)
                if not torch.isfinite(loss):
                    raise ContractError('non-finite training loss')
                loss.backward()
                optimizer.step()
                total += loss.item()
            self.backbone.eval()
            self.head.eval()
            scores, targets, rankings = [], [], []
            with torch.no_grad():
                for sample in valid:
                    candidates = groups[sample.sample_id]
                    probabilities = self._logits(sample, candidates).sigmoid().tolist()
                    rows = [(c.candidate_id, labels[(sample.sample_id, c.candidate_id)], p)
                            for c, p in zip(candidates, probabilities)]
                    rankings.append(ranking_metrics(rows)['ndcg_at_10'])
                    scores.extend(probabilities)
                    targets.extend(r[1] for r in rows)
            metric = (sum(rankings) / len(rankings) if dataset.metadata.task == 'food_recommendation'
                      else -advertising_metrics(targets, scores)['log_loss'])
            self.training_log.append({'epoch': epoch, 'loss': total / len(train),
                                      'validation_selection_score': metric})
            if metric > best:
                best, stale, state = metric, 0, self._state()
            else:
                stale += 1
            if stale >= self.config['patience']:
                break
        for name, weights in state.items():
            getattr(self, name).load_state_dict(weights)
        self.resources = {'training_seconds': time.perf_counter() - started,
                          'parameter_count': sum(p.numel() for module in
                                                 (self.backbone, self.head, self.candidates)
                                                 for p in module.parameters()),
                          'device': 'cpu', 'batch_size': 1,
                          'peak_memory_bytes': process_peak_memory(),
                          'memory_scope': 'process_lifetime_high_water_rss'}
        return self

    def predict(self, dataset, split, run_id, *, frozen_cutoff, profiles=None):
        if self.backbone is None:
            raise ContractError('unfitted sequence model')
        if any(getattr(dataset.metadata, k) != self.identity[k]
               for k in ('dataset_name', 'task', 'item_kind')):
            raise ContractError('prediction namespace/task mismatch')
        if frozen_cutoff < self.training_cutoff:
            raise ContractError('prediction cutoff precedes model fit')
        self.backbone.eval()
        self.head.eval()
        rows = []
        started = time.perf_counter()
        with torch.no_grad():
            for sample in sequence_samples(dataset, split, frozen_cutoff, profiles):
                candidates = [c for c in dataset.candidates if c.request_id == sample.sample_id]
                probabilities = self._logits(sample, candidates).sigmoid().tolist()
                for candidate, score in zip(candidates, probabilities):
                    rows.append({"schema_version": 1, "task": dataset.metadata.task, "split": split,
                                     "data_version": dataset.metadata.data_version, "run_id": run_id,
                                     "request_id": sample.sample_id, "user_id": sample.user_id,
                                     "cutoff_timestamp": int(sample.cutoff_time.timestamp()),
                                     "candidate_id": candidate.candidate_id, "score": score})
        self.resources['inference_seconds'] = time.perf_counter() - started
        self.resources['inference_seconds_per_request'] = self.resources['inference_seconds'] / len({r['request_id'] for r in rows})
        rows.sort(key=lambda r: (r['request_id'], -r['score'], r['candidate_id']))
        table = pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)
        validate_predictions(table, dataset, split)
        return table

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.artifacts.save(path / 'vocabulary.json')
        torch.save(self._state(), path / 'weights.pt')
        metadata = {"format_version": 1, "config": self.config, "identity": self.identity,
                        "model_config": asdict(self.model_config), "items": self.items,
                        "training_cutoff": self.training_cutoff.isoformat(), "foundation": self.foundation,
                        "resources": self.resources, "training_log": self.training_log,
                        "vocabulary_sha256": hashlib.sha256((path / 'vocabulary.json').read_bytes()).hexdigest(),
                        "weights_sha256": hashlib.sha256((path / 'weights.pt').read_bytes()).hexdigest()}
        (path / 'metadata.json').write_text(json.dumps(metadata, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if meta['format_version'] != 1:
            raise ContractError('unsupported sequence artifact')
        for filename, key in [('vocabulary.json', 'vocabulary_sha256'), ('weights.pt', 'weights_sha256')]:
            if hashlib.sha256((path / filename).read_bytes()).hexdigest() != meta[key]:
                raise ContractError('sequence artifact checksum mismatch')
        model = cls(**meta['config'])
        model.artifacts = VocabularyArtifacts.load(path / 'vocabulary.json')
        model.tokenizer = tokenizer_from_artifacts(model.artifacts)
        model.model_config = FoundationModelConfig(**meta['model_config'])
        model.backbone = FoundationModel(model.model_config)
        model.items = meta['items']
        model.candidates = nn.Embedding(len(model.items) + 1, model.model_config.d_model)
        model.head = nn.Sequential(nn.Linear(3 * model.model_config.d_model, model.model_config.d_model),
                                   nn.ReLU(), nn.Linear(model.model_config.d_model, 1))
        for name, weights in torch.load(path / 'weights.pt', weights_only=True).items():
            getattr(model, name).load_state_dict(weights)
        model.identity = meta['identity']
        model.training_cutoff = datetime.fromisoformat(meta['training_cutoff'])
        model.foundation = meta['foundation']
        model.resources = meta['resources']
        model.training_log = meta['training_log']
        model.backbone.requires_grad_(model.config['model_kind'] != 'B3')
        return model
