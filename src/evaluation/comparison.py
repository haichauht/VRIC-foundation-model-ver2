"""Seed summaries and paired user-cluster bootstrap for experiment reports."""

import numpy as np


def summarize_seeds(metrics_by_seed):
    if len(metrics_by_seed) < 3:
        raise ValueError('benchmark summary requires at least three seeds')
    keys = set(metrics_by_seed[0])
    if any(set(row) != keys for row in metrics_by_seed):
        raise ValueError('metric names differ between seeds')
    output = {}
    for name in sorted(keys):
        values = [row[name] for row in metrics_by_seed]
        if any(v is None or not np.isfinite(v) for v in values):
            output[name] = {'mean': None, 'std': None, 'status': 'undefined'}
        else:
            output[name] = {'mean': float(np.mean(values)),
                            'std': float(np.std(values, ddof=1)), 'status': 'defined'}
    return output


def paired_user_bootstrap(left, right, *, seed=42, repetitions=2000):
    """Inputs map user_id to matching per-request metric sequences (same order)."""
    if set(left) != set(right) or not left or repetitions < 1:
        raise ValueError('paired nonempty users and positive repetitions required')
    users = sorted(left)
    differences = []
    for user in users:
        a, b = np.asarray(left[user], dtype=float), np.asarray(right[user], dtype=float)
        if a.ndim != 1 or not len(a) or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('paired finite request metrics required')
        differences.append(a - b)
    rng = np.random.default_rng(seed)
    samples = [float(np.concatenate([differences[i] for i in rng.integers(0, len(users), len(users))]).mean())
               for _ in range(repetitions)]
    return {'delta': float(np.concatenate(differences).mean()),
            'ci_low': float(np.quantile(samples, 0.025)),
            'ci_high': float(np.quantile(samples, 0.975)),
            'cluster_count': len(users), 'seed': seed, 'repetitions': repetitions}
