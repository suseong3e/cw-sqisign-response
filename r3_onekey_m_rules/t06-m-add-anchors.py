#!/usr/bin/env python3
"""Target-only first-byte search for addition stores after each product group."""
import json
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't06-m-add-anchors.json'
STRIDE, GAP = 4, 84
PRODUCT_GROUP_BEAM = 4


def product_branches(group):
    retained = group.get('retained_product_candidates')
    if retained:
        seen = set()
        for item in retained:
            if item['value'] not in seen:
                seen.add(item['value'])
                yield item['source'], item['value']
        return
    candidates = {name: [item['value'] for item in values]
                  for name, values in (group.get('split_full_candidates') or {}).items()}
    if not candidates:
        candidates = group.get('split_candidates', {}) or {'group_candidate': group['candidate']}
    seen = set()
    for source, values in candidates.items():
        if isinstance(values, str):
            values = [values]
        for candidate in values:
            candidate = hex(candidate) if isinstance(candidate, int) else candidate
            if candidate not in seen:
                seen.add(candidate)
                yield source, candidate


def select_groups(groups, cap=PRODUCT_GROUP_BEAM):
    ordered = sorted(groups, key=lambda group: (-group.get('min_validation_abs_r', float('-inf')),
                                                group['anchor'], group['candidate']))
    return (ordered, []) if cap is None else (ordered[:cap], ordered[cap:])


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads((HERE / 't05-m-product-groups.json').read_text())
    groups, excluded_groups = select_groups(prior['groups'], cap=None)
    rows = [json.loads(line) for line in (DATA / 'public_schedule.jsonl').read_text().splitlines()]
    traces = np.load(DATA / 'traces.npy', mmap_mode='r', allow_pickle=False)
    challenges = PRODUCT.rank.words(rows)
    ix = np.array([i for i, row in enumerate(rows) if row['block'] % 2 == 0])
    sub = [rows[i] for i in ix]
    guesses = np.arange(256, dtype=np.uint64)[:, None]
    findings = []
    for group in groups:
        if not group.get('source_order_valid', True):
            raise RuntimeError('product group violates its declared source order')
        for product_source, candidate in product_branches(group):
            secret = int(candidate, 16)
            _, state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
            labels = PRODUCT.limb.low.hw32((guesses + state[ix, 0][None, :]) & np.uint64(255)).astype(float).T
            start, stop = group['anchor'] + 6*PRODUCT.GAP, PRODUCT.STOP - 5*GAP
            if stop <= start:
                findings.append({'product_anchor': group['anchor'], 'product_candidate': candidate,
                                 'product_source': product_source, 'status': 'truncated_scan',
                                 'scan': [start, stop, STRIDE], 'source_order_valid': True, 'add_anchors': []})
                continue
            score = np.max(np.abs(PRODUCT.CONTROL.grid(PRODUCT.residual(labels, sub),
                                                       PRODUCT.residual(np.asarray(traces[ix, start:stop:STRIDE], dtype=float), sub))), axis=0)
            points = []
            for coarse in np.argsort(score)[::-1]:
                sample = start + int(coarse)*STRIDE
                if any(abs(sample - old['sample']) < 150 for old in points):
                    continue
                points.append({'sample': sample, 'train_max_abs_r': float(score[coarse])})
                if len(points) == 8:
                    break
            findings.append({'product_anchor': group['anchor'], 'product_candidate': candidate,
                             'product_source': product_source, 'product_split_candidates': group.get('split_candidates', {}),
                             'scan': [start, stop, STRIDE], 'source_order_valid': True, 'add_anchors': points,
                             'deferred_product_candidates': group.get('deferred_product_candidates', []),
                             'anchor_frontier': {'expanded': int(score.size), 'kept': len(points),
                                'dropped': int(score.size)-len(points),
                                'scores_by_coarse_sample': score.tolist()}})
            print(group['anchor'], product_source, points[:4], flush=True)
    status = ('analysis_incomplete' if prior.get('status') == 'analysis_incomplete' else
              'no_candidates' if not findings else 'diagnostic_only')
    report = {'schema': 'r3-target-only-m-add-anchors-v1', 'status': status, 'groups': findings,
              'excluded_product_groups': [{'anchor': group['anchor'], 'candidate': group['candidate'],
                                           'min_validation_abs_r': group.get('min_validation_abs_r'),
                                           'reason': 'product_group_rank_budget'} for group in excluded_groups],
              'computational_budget': {'product_group_beam': 'all',
                                       'retained_product_groups': len(groups),
                                       'excluded_product_groups': len(excluded_groups),
                                       'add_anchor_count_per_branch': 8,
                                       'elapsed_s': time.perf_counter()-started},
              'evidence_contract': {'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['m_recovery', 'key_recovery', 'scheme_break']}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
