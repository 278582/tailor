"""Random target-pool alignment; no encoding or distance-space changes."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
MODE = "target_pool_mc"


def plan_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def sampling_plan(train_rows, pool_rows, repeats=10, seed=20260420, path=None):
    if not 0 < pool_rows <= train_rows or repeats < 1:
        raise ValueError('Require 0 < reference rows <= train rows and repeats >= 1')
    if path and Path(path).exists():
        with np.load(path, allow_pickle=False) as z:
            if (int(z['train_rows']), int(z['pool_rows']), int(z['repeats']), int(z['seed'])) != (train_rows, pool_rows, repeats, seed):
                raise ValueError('Sampling plan metadata does not match current request')
            indices = z['indices'].copy()
        if indices.dtype.kind not in 'iu' or indices.shape != (repeats, pool_rows):
            raise ValueError('Invalid sampling-plan shape/dtype')
        if np.any(indices < 0) or np.any(indices >= train_rows) or any(len(np.unique(row)) != pool_rows for row in indices):
            raise ValueError('Sampling plan must contain distinct in-range row IDs per round')
        return indices.astype(np.int64)
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(train_rows, size=pool_rows, replace=False) for _ in range(repeats)])
    if path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive file creation prevents silently replacing a shared plan.
        with p.open('xb') as f:
            np.savez_compressed(f, indices=indices, train_rows=train_rows, pool_rows=pool_rows, repeats=repeats, seed=seed)
    return indices


def tie_half(train_dist, reference_dist):
    a, b = np.asarray(train_dist), np.asarray(reference_dist)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Distances must be finite and equally shaped')
    return (a < b).astype(float) + .5 * (a == b)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return {'value': None, 'status': 'both_zero' if numerator == 0 else 'positive_over_zero',
                'numerator': float(numerator), 'denominator': 0.0}
    return {'value': float(numerator / denominator), 'status': 'finite',
            'numerator': float(numerator), 'denominator': float(denominator)}


def signals(query, train, reference, *, repeats=10, seed=20260420, plan_path=None, indices=None,
            min_l1_fn=None):
    # Imported lazily to keep the legacy implementation independent of this mode.
    from .aligned_dcr import min_l1_distances
    if len(query) < 1 or len(reference) < 1:
        raise ValueError('Nonempty query and reference are required')
    if indices is None:
        indices = sampling_plan(len(train), len(reference), repeats, seed, plan_path)
    else:
        expected = sampling_plan(len(train), len(reference), repeats, seed, plan_path)
        if not np.array_equal(indices, expected):
            raise ValueError('Explicit indices differ from the shared sampling plan')
    distance = min_l1_fn or min_l1_distances
    subsets = [train[row] for row in indices]
    # Existing GPU kernels cache reference tensors by numpy object identity.
    # Keep backing reference arrays alive for the callable's lifetime so an
    # ephemeral sampled array ID cannot be reused for a different target pool.
    if min_l1_fn is not None and hasattr(distance, "__dict__"):
        retained = getattr(distance, "_target_pool_reference_arrays", {})
        for matrix in [reference, train, *subsets]:
            retained[id(matrix)] = matrix
        distance._target_pool_reference_arrays = retained
    dref = np.asarray(distance(query, reference), dtype=float)
    rounds = []; distances = []; ratios = []; tail_ratios = []
    for subset in subsets:
        dtrain = np.asarray(distance(query, subset), dtype=float)
        rounds.append(tie_half(dtrain, dref)); distances.append(dtrain)
        ratios.append(safe_ratio(np.median(dtrain), np.median(dref)))
        tail_ratios.append(safe_ratio(np.quantile(dtrain, .05), np.quantile(dref, .05)))
    round_q = np.stack(rounds); dtrain = np.stack(distances)
    rates = round_q.mean(axis=1); q = round_q.mean(axis=0)
    return {'aligned_closer_rate': q, 'dcr': float(rates.mean()), 'dcr_n': len(reference),
            'query_rows': len(query), 'dcr_repeats': repeats, 'dcr_requested_repeats': repeats,
            'dcr_align_mode': MODE, 'dcr_compute_cap': 0, 'dcr_cap': 0,
            'dcr_aligned': True, 'dcr_seed': seed, 'dcr_repeat_rates': rates.tolist(),
            'dcr_std': float(np.std(rates, ddof=1)) if repeats > 1 else 0.,
            'dcr_real': dtrain.mean(axis=0), 'dcr_test': dref,
            'margin': dref - dtrain.mean(axis=0),
            'is_real_closer': q >= .5,  # Candidate partitioning only; never aggregate this flag.
            'train_indices': indices, 'sampling_plan_sha256': plan_hash(indices),
            'round_q': round_q, 'train_distances': dtrain, 'reference_distances': dref,
            'median_ratios': ratios, 'p05_ratios': tail_ratios,
            'zero_reference_distances': int(np.sum(dref == 0)),
            'zero_train_distances_per_round': (dtrain == 0).sum(axis=1).tolist(),
            'tie_counts_per_round': (dtrain == dref).sum(axis=1).tolist(),
            'tie_rule': 'exact equality, half weight; no tolerance',
            'share_definition': 'mean per-round indicators, then mean rounds; all query rows'}


def evaluate(query, train, reference, **kwargs):
    value = signals(query, train, reference, **kwargs)
    extras = {k: value[k] for k in ('dcr_real', 'dcr_test', 'dcr_n', 'dcr_repeats',
        'dcr_requested_repeats', 'dcr_align_mode', 'dcr_compute_cap', 'dcr_aligned',
        'dcr_repeat_rates', 'dcr_std')}
    extras['dcr_target_pool_protocol'] = {k: value[k] for k in ('query_rows', 'sampling_plan_sha256',
        'median_ratios', 'p05_ratios', 'zero_reference_distances', 'zero_train_distances_per_round',
        'tie_counts_per_round', 'tie_rule', 'share_definition')}
    extras['aligned_dcr_seconds'] = 0.0
    extras['aligned_dcr_timed_repeats'] = value['dcr_repeats']
    return {'dcr': value['dcr']}, extras
