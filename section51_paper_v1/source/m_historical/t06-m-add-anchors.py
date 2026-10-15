#!/usr/bin/env python3
"""Target-only first-byte search for addition stores after each product group."""
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't06-m-add-anchors.json'
STRIDE, GAP = 4, 84


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads((HERE / 't05-m-product-groups.json').read_text())
    groups = [item for item in prior['groups'] if item['split_agreement'] and item['min_validation_abs_r'] > 0.5]
    rows = [json.loads(line) for line in (DATA / 'public_schedule.jsonl').read_text().splitlines()]
    traces = np.load(DATA / 'traces.npy', mmap_mode='r', allow_pickle=False)
    challenges = PRODUCT.rank.words(rows)
    ix = np.array([i for i, row in enumerate(rows) if row['block'] % 2 == 0])
    sub = [rows[i] for i in ix]
    guesses = np.arange(256, dtype=np.uint64)[:, None]
    findings = []
    for group in groups:
        secret = int(group['candidate'], 16)
        _, state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
        labels = PRODUCT.limb.low.hw32((guesses + state[ix, 0][None, :]) & np.uint64(255)).astype(float).T
        start, stop = group['anchor'] + 6*PRODUCT.GAP, PRODUCT.STOP - 5*GAP
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
        findings.append({'product_anchor': group['anchor'], 'product_candidate': group['candidate'], 'scan': [start, stop, STRIDE],
                         'add_anchors': points})
        print(group['anchor'], points[:4], flush=True)
    report = {'schema': 'r3-target-only-m-add-anchors-v1', 'status': 'diagnostic_only', 'groups': findings,
              'evidence_contract': {'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['m_recovery', 'key_recovery', 'scheme_break']}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
