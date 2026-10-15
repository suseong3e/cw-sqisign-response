#!/usr/bin/env python3
"""Freeze target-only M candidates from source-ordered product/addition pairs."""
import hashlib
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't09-m-candidates.json'


def fixed_correlations(labels, traces, rows, index, points):
    subset = [rows[i] for i in index]
    wave = np.asarray(traces[np.ix_(index, points)], dtype=float)
    return np.diag(PRODUCT.CONTROL.grid(PRODUCT.residual(labels, subset),
                                        PRODUCT.residual(wave, subset)))


def usable(groups):
    return [(index, group) for index, group in enumerate(groups)
            if group.get('status') is None and group.get('split_agreement')]


def source_ordered(m00_group, m10_group):
    return (m00_group['product_anchor'] < m00_group['anchor'] <
            m10_group['product_anchor'] < m10_group['anchor'])


def full_matrix_candidates(pairs):
    """Return each distinct split-valid full matrix in the observed source order."""
    candidates, seen = [], set()
    for pair in pairs:
        for branch in pair['m10_candidates']:
            matrix = pair['matrix'] | {'m10': branch['m10']}
            identity = tuple(matrix[name] for name in ('m00', 'm01', 'm10', 'm11'))
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append({'matrix': matrix, 'source_order_pois': pair['source_order_pois'],
                               'validation_scores': pair['validation_scores'],
                               'm10_validation': branch})
    return candidates


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert not manifest['target_truth_loaded'] and manifest['known_key_trace_count'] == 0
    trace_path, schedule_path = DATA / 'traces.npy', DATA / 'public_schedule.jsonl'
    assert PRODUCT.rank.digest(trace_path) == manifest['target_trace_sha256']
    assert PRODUCT.rank.digest(schedule_path) == manifest['target_public_schedule_sha256']
    upstream = {stage: json.loads((HERE / f't{stage:02d}-{name}.json').read_text())
                for stage, name in ((5, 'm-product-groups'), (6, 'm-add-anchors'),
                                    (7, 'm00-groups'), (8, 'm10-groups'))}
    m00_groups = usable(upstream[7]['groups'])
    m10_groups = usable(upstream[8]['groups'])
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    traces = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    challenges = PRODUCT.rank.words(rows)
    validation = np.array([i for i, row in enumerate(rows) if row['block'] % 2 == 1])
    splits = {name: np.array([i for i, row in enumerate(rows) if row['block'] % 2 == parity])
              for name, parity in [('discovery', 0), ('validation', 1)]}
    pairs, rejected = [], []
    for m00_index, m00_group in m00_groups:
        m00, m01 = (int(m00_group['split_candidates']['discovery'], 16),
                    int(m00_group['product_candidate'], 16))
        m01_points = [m00_group['product_anchor'] + PRODUCT.GAP*row for row in range(6)]
        if m01_points[-1] >= traces.shape[1]:
            rejected.append({'reason': 'm01_product_points_truncated', 'm00_anchor': m00_group['anchor']})
            continue
        m01_labels = PRODUCT.rank.product_labels(m01, challenges[validation], PRODUCT.limb)[0].T
        m01_score = min(abs(value) for value in fixed_correlations(m01_labels, traces, rows, validation, m01_points))
        _, m01_state = PRODUCT.rank.product_labels(m01, challenges, PRODUCT.limb)
        m00_labels = PRODUCT.rank.add_labels(m00, m01_state[validation], PRODUCT.limb)[0]
        m00_score = min(abs(value) for value in fixed_correlations(m00_labels, traces, rows, validation,
                                                                    m00_group['points']))
        for m10_index, m10_group in m10_groups:
            if not source_ordered(m00_group, m10_group):
                rejected.append({'reason': 'source_order_violation',
                                 'm01_product_anchor': m00_group['product_anchor'],
                                 'm00_add_anchor': m00_group['anchor'],
                                 'm11_product_anchor': m10_group['product_anchor'],
                                 'm10_add_anchor': m10_group['anchor']})
                continue
            m11 = int(m10_group['product_candidate'], 16)
            m11_points = [m10_group['product_anchor'] + PRODUCT.GAP*row for row in range(6)]
            if m11_points[-1] >= traces.shape[1]:
                rejected.append({'reason': 'm11_product_points_truncated', 'm10_anchor': m10_group['anchor']})
                continue
            _, m11_state = PRODUCT.rank.product_labels(m11, challenges, PRODUCT.limb)
            m11_labels = PRODUCT.rank.product_labels(m11, challenges[validation], PRODUCT.limb)[0].T
            m11_score = min(abs(value) for value in fixed_correlations(m11_labels, traces, rows, validation, m11_points))
            branches = []
            m10_values = {int(m10_group['split_candidates'][name], 16) for name in splits}
            for m10 in sorted(m10_values):
                m10_labels = PRODUCT.rank.add_labels(m10, m11_state[validation], PRODUCT.limb)[0]
                m10_score = min(abs(value) for value in fixed_correlations(m10_labels, traces, rows, validation,
                                                                            m10_group['points']))
                branch = {'m10': hex(m10), 'first_word': hex(m10 & 0xffffffff),
                          'split_agreement': len(m10_values) == 1,
                          'determinant_odd': bool((m00*m11 - m01*m10) & 1),
                          'min_validation_abs_r': m10_score}
                if branch['split_agreement'] and branch['determinant_odd']:
                    branches.append(branch)
                else:
                    rejected.append({'reason': 'm10_split_or_determinant', 'branch': branch,
                                     'm00_anchor': m00_group['anchor'], 'm10_anchor': m10_group['anchor']})
            if branches:
                pairs.append({'matrix': {'m00': hex(m00), 'm01': hex(m01), 'm11': hex(m11)},
                              'm10_candidates': branches,
                              'm00_group_index': m00_index, 'm10_group_index': m10_index,
                              'source_order_pois': {'m01': m01_points, 'm00': m00_group['points'],
                                                    'm11': m11_points, 'm10': m10_group['points']},
                              'validation_scores': {'m01': m01_score, 'm00': m00_score,
                                                    'm11': m11_score,
                                                    'm10': max(item['min_validation_abs_r'] for item in branches)}})
    full_candidates = full_matrix_candidates(pairs)
    ordered_pairs = sorted(pairs, key=lambda pair: (-min(pair['validation_scores'].values()),
                                                     pair['matrix']['m00'], pair['matrix']['m01'],
                                                     pair['matrix']['m11']))
    best = ordered_pairs[0] if ordered_pairs else None
    matrix = (best['matrix'] | {'m10_candidates': [branch['m10'] for branch in best['m10_candidates']]}
              if best else None)
    report = {'schema': 'r3-target-only-m-candidates-v1',
              'status': ('analysis_incomplete' if any(item.get('status') == 'analysis_incomplete'
                                                      for item in upstream.values()) else
                         'no_candidates' if not full_candidates else
                         'one_candidate' if len(full_candidates) == 1 else
                         'two_candidates' if len(full_candidates) == 2 else 'multiple_candidates'),
              'matrix_scope': 'top_ranked_pair_compatibility_view',
              'matrix': matrix, 'source_order_pois': best['source_order_pois'] if best else None,
              'selected_pair_validation_scores': best['validation_scores'] if best else None,
              'full_matrix_candidates': full_candidates,
              'all_group_pairs': pairs, 'rejected_branches': rejected,
              'computation_counts': {'usable_m00_groups': len(m00_groups),
                                     'usable_m10_groups': len(m10_groups),
                                     'valid_group_pairs': len(pairs),
                                     'full_matrix_candidates': len(full_candidates),
                                     'rejected_branches': len(rejected)},
              'upstream_status': {str(stage): item.get('status') for stage, item in upstream.items()},
              'evidence_contract': {'evidence_origin': 'measured', 'evidence_purpose': 'diagnostic',
                                    'blinding': 'blinded', 'blinding_detail': 'target truth unavailable; rules developed on historical keys',
                                    'measurement_scope': 'isolated_primitive',
                                    'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_scope': 'target_only_M_candidate_screen',
                                    'claim_not_allowed': ['unique_m_recovery', 'full_key_recovery', 'scheme_break']},
              'inputs': {'target_trace_sha256': manifest['target_trace_sha256'],
                         'public_schedule_sha256': manifest['target_public_schedule_sha256'],
                         'upstream_sha256': {str(stage): PRODUCT.rank.digest(HERE / f't{stage:02d}-{name}.json')
                                             for stage, name in ((5, 'm-product-groups'), (6, 'm-add-anchors'),
                                                                 (7, 'm00-groups'), (8, 'm10-groups'))},
                         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    partial = OUT.with_suffix(OUT.suffix + '.partial')
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open('w') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    partial.replace(OUT)
    print(json.dumps({'matrix': matrix, 'full_matrix_candidates': len(full_candidates),
                      'group_pairs': len(pairs), 'rejected': len(rejected)}, indent=2))


if __name__ == '__main__':
    main()
