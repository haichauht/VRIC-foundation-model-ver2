"""Head-independent evaluation: producers supply actual normalized distributions.

For hierarchical heads probabilities must be joint group/local probabilities,
not a flat softmax applied to unrelated head logits. No sampler is added here.
"""

import math

import numpy as np


def evaluate_pretraining_predictions(components, *, vocabulary_policy, candidate_policy, k=10):
    """Evaluate named masked/next_key/next_value components.

    Each component declares kind (categorical/numeric), targets, valid_mask.
    Masked tasks additionally declare masked_mask and special_mask. Categorical
    input is normalized probabilities [N,C], optionally candidate_ids [C]; for
    huge heads use target_probabilities and ranked_ids [N,K] instead. Missing
    distribution/ranking yields unavailable metrics, never fabricated scores.
    """
    if not vocabulary_policy or not candidate_policy or k <= 0:
        raise ValueError('explicit vocabulary/candidate policies and positive k required')
    report = {'vocabulary_policy': vocabulary_policy, 'candidate_policy': candidate_policy,
              'components': {}}
    for name, component in components.items():
        if name not in {'masked', 'next_key', 'next_value'}:
            raise ValueError('unknown objective component')
        target = np.asarray(component['targets'])
        mask = np.asarray(component['valid_mask'])
        if target.ndim != 1 or mask.shape != target.shape or mask.dtype != bool:
            raise ValueError('targets/valid_mask shape or dtype mismatch')
        if name == 'masked':
            for field in ('masked_mask', 'special_mask'):
                flag = np.asarray(component[field])
                if flag.shape != mask.shape or flag.dtype != bool:
                    raise ValueError('masked/special mask mismatch')
                mask = mask & (flag if field == 'masked_mask' else ~flag)
        count = int(mask.sum())
        result = {'count': count, 'status': 'defined' if count else 'undefined'}
        if component['kind'] == 'numeric':
            prediction = np.asarray(component['predictions'], dtype=float)
            if prediction.shape != target.shape or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
                raise ValueError('invalid numeric prediction/target')
            mse = float(np.mean((prediction[mask] - target[mask]) ** 2)) if count else None
            result.update(mse=mse, rmse=math.sqrt(mse) if mse is not None else None)
        elif component['kind'] == 'categorical':
            if not np.issubdtype(target.dtype, np.integer):
                raise ValueError('categorical targets must be integer IDs')
            probability = None
            ranking = None
            if 'probabilities' in component:
                probs = np.asarray(component['probabilities'], dtype=float)
                if probs.ndim != 2 or probs.shape[0] != len(target) or probs.shape[1] == 0:
                    raise ValueError('probability shape mismatch')
                ids = np.asarray(component.get('candidate_ids', list(range(probs.shape[1]))))
                if ids.shape != (probs.shape[1],) or len(set(ids.tolist())) != len(ids):
                    raise ValueError('invalid candidate IDs')
                valid = probs[mask]
                if not np.isfinite(valid).all() or (valid < 0).any() or not np.allclose(valid.sum(axis=1), 1, atol=1e-6):
                    raise ValueError('probabilities must be normalized actual head distribution')
                index = {item: i for i, item in enumerate(ids.tolist())}
                if any(t not in index for t in target[mask]):
                    raise ValueError('target outside candidate distribution')
                probability = np.array([p[index[t]] for p, t in zip(valid, target[mask])])
                ranking = ids[np.argsort(-valid, axis=1, kind='stable')]
            else:
                if 'target_probabilities' in component:
                    values = np.asarray(component['target_probabilities'], dtype=float)
                    if values.shape != target.shape:
                        raise ValueError('target probability shape mismatch')
                    probability = values[mask]
                if 'ranked_ids' in component:
                    values = np.asarray(component['ranked_ids'])
                    if values.ndim != 2 or values.shape[0] != len(target):
                        raise ValueError('ranking shape mismatch')
                    ranking = values[mask]
                    if ranking.shape[1] < k:
                        raise ValueError('ranked_ids must cover requested K')
            if probability is not None and (not np.isfinite(probability).all() or (probability <= 0).any() or (probability > 1).any()):
                raise ValueError('target probability must be finite and in (0,1]')
            ce = float(-np.log(probability).mean()) if count and probability is not None else None
            result.update(cross_entropy=ce, perplexity=math.exp(ce) if ce is not None and ce < 709 else None,
                          accuracy=None, recall_at_k=None, mrr=None)
            if count and ranking is not None:
                hits = ranking == target[mask, None]
                result.update(accuracy=float(hits[:, 0].mean()),
                              recall_at_k=float(hits[:, :k].any(axis=1).mean()))
                # Truncated rankings cannot determine full-vocabulary MRR.
                if 'probabilities' in component or component.get('ranking_complete', False):
                    result['mrr'] = float(sum(1 / (np.flatnonzero(row)[0] + 1) if row.any() else 0
                                              for row in hits) / count)
            if count and (probability is None or ranking is None):
                result['status'] = 'partially_unavailable'
        else:
            raise ValueError('unknown prediction kind')
        report['components'][name] = result
    return report
