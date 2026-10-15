#!/usr/bin/env python3
"""Test source-ordered 84-sample addition groups for target-only M00."""
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = Path(os.environ['SECTION51_M_DATA'])
OUT = Path(os.environ['SECTION51_M_OUT']) / 't07-m00-groups.json'
GAP = 84


def recover(product, samples):
    carry = np.zeros(product.shape[0], dtype=np.uint64)
    words = []
    zero = np.zeros(product.shape[0], dtype=np.uint64)
    for row in range(6):
        base = product[:, row] + carry
        ranked = PRODUCT.rank.rank_word(base, zero, samples[:, row], 256, False, PRODUCT.limb)
        word = ranked['top1']
        words.append(word)
        carry = (base + np.uint64(word)) >> np.uint64(32)
    return sum(word << (32*row) for row, word in enumerate(words))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    products = json.loads((Path(os.environ['SECTION51_M_OUT']) / 't05-m-product-groups.json').read_text())['groups']
    add = json.loads((Path(os.environ['SECTION51_M_OUT']) / 't06-m-add-anchors.json').read_text())['groups']
    add_by_product = {entry['product_anchor']: entry for entry in add}
    rows = [json.loads(line) for line in (DATA / 'public_schedule.jsonl').read_text().splitlines()]
    traces = np.load(DATA / 'traces.npy', mmap_mode='r', allow_pickle=False)
    challenges = PRODUCT.rank.words(rows)
    splits = {name: np.array([i for i, row in enumerate(rows) if row['block'] % 2 == parity])
              for name, parity in [('discovery', 0), ('validation', 1)]}
    groups = []
    for product in products:
        product_anchor = product['anchor']
        if product_anchor > 15000 or not product['split_agreement'] or product['min_validation_abs_r'] < 0.5:
            continue
        secret = int(product['candidate'], 16)
        _, product_state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
        candidates = [entry['sample'] for entry in add_by_product[product_anchor]['add_anchors']
                      if product_anchor + 5*PRODUCT.GAP < entry['sample'] < 15000][:2]
        for anchor in candidates:
            points = [anchor + GAP*row for row in range(6)]
            guesses = {}
            for name, ix in splits.items():
                samples = np.asarray(traces[np.ix_(ix, points)], dtype=float)
                guesses[name] = hex(recover(product_state[ix], samples))
            value = int(guesses['discovery'], 16)
            correlations = {}
            for name, ix in splits.items():
                label, _ = PRODUCT.rank.add_labels(value, product_state[ix], PRODUCT.limb)
                subset = [rows[i] for i in ix]
                wave = np.asarray(traces[np.ix_(ix, points)], dtype=float)
                correlations[name] = np.diag(PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset),
                                                                   PRODUCT.residual(wave, subset))).tolist()
            result = {'product_anchor': product_anchor, 'product_candidate': product['candidate'],
                      'anchor': anchor, 'points': points, 'split_candidates': guesses,
                      'split_agreement': guesses['discovery'] == guesses['validation'],
                      'correlations': correlations,
                      'min_validation_abs_r': min(abs(x) for x in correlations['validation'])}
            groups.append(result)
            print(product_anchor, anchor, guesses, round(result['min_validation_abs_r'], 3), flush=True)
    groups.sort(key=lambda row: (-row['min_validation_abs_r'], row['product_anchor'], row['anchor']))
    report = {'schema': 'r3-target-only-m00-groups-v1', 'status': 'diagnostic_only',
              'method': {'add_gap': GAP, 'beam': 256, 'target_only': True}, 'groups': groups,
              'evidence_contract': {'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['unique_m_recovery', 'key_recovery', 'scheme_break']}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
