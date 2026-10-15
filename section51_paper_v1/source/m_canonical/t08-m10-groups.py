#!/usr/bin/env python3
"""Target-only source-order M10 addition after candidate M11 product groups."""
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = Path(os.environ['SECTION51_M_DATA'])
OUT = Path(os.environ['SECTION51_M_OUT']) / 't08-m10-groups.json'


def rank_sequence(product, challenges, traces, rows, index, anchor, points=None):
    subset = [rows[i] for i in index]
    carry = np.zeros(len(index), dtype=np.uint64)
    zero = np.zeros(len(index), dtype=np.uint64)
    selected, chosen_points, ties = [], [], []
    for row in range(6):
        base = product[index, row] + carry
        if points is None:
            if row == 0:
                sample = anchor
            else:
                left, right = chosen_points[-1] + 4, min(chosen_points[-1] + 41, 26008)
                guesses = np.arange(256, dtype=np.uint64)[:, None]
                label = PRODUCT.limb.low.hw32((guesses + base[None, :]) & np.uint64(255)).astype(float).T
                wave = np.asarray(traces[index, left:right], dtype=float)
                score = PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset), PRODUCT.residual(wave, subset))
                sample = left + int(np.unravel_index(np.argmax(np.abs(score)), score.shape)[1])
        else:
            sample = points[row]
        ranked = PRODUCT.rank.rank_word(base, zero, np.asarray(traces[index, sample], dtype=float),
                                        256, False, PRODUCT.limb)
        word = ranked['top1']
        selected.append(word)
        chosen_points.append(sample)
        if row == 0:
            top = ranked['stages'][-1]['top_k']
            ties = [item['value'] for item in top if abs(item['score'] - top[0]['score']) < 1e-12]
        carry = (base + np.uint64(word)) >> np.uint64(32)
    return sum(word << (32*row) for row, word in enumerate(selected)), chosen_points, ties


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
        if product_anchor < 15000 or not product['split_agreement'] or product['min_validation_abs_r'] < 0.5:
            continue
        secret = int(product['candidate'], 16)
        _, product_state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
        candidates = [entry['sample'] for entry in add_by_product[product_anchor]['add_anchors']
                      if product_anchor + 5*PRODUCT.GAP < entry['sample'] < 25800][:2]
        for anchor in candidates:
            value, points, ties = rank_sequence(product_state, challenges, traces, rows, splits['discovery'], anchor)
            validation, _, validation_ties = rank_sequence(product_state, challenges, traces, rows,
                                                           splits['validation'], anchor, points)
            correlations = {}
            for name, ix in splits.items():
                label, _ = PRODUCT.rank.add_labels(value, product_state[ix], PRODUCT.limb)
                subset = [rows[i] for i in ix]
                wave = np.asarray(traces[np.ix_(ix, points)], dtype=float)
                correlations[name] = np.diag(PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset),
                                                                   PRODUCT.residual(wave, subset))).tolist()
            result = {'product_anchor': product_anchor, 'product_candidate': product['candidate'],
                      'anchor': anchor, 'points': points,
                      'split_candidates': {'discovery': hex(value), 'validation': hex(validation)},
                      'first_word_ties': {'discovery': [hex(x) for x in ties],
                                          'validation': [hex(x) for x in validation_ties]},
                      'split_agreement': value == validation,
                      'correlations': correlations,
                      'min_validation_abs_r': min(abs(x) for x in correlations['validation'])}
            groups.append(result)
            print(product_anchor, anchor, points, result['split_candidates'],
                  round(result['min_validation_abs_r'], 3), flush=True)
    groups.sort(key=lambda row: (-row['min_validation_abs_r'], row['product_anchor'], row['anchor']))
    report = {'schema': 'r3-target-only-m10-groups-v1', 'status': 'diagnostic_only',
              'method': {'first_anchor_from_public_screen': True, 'subsequent_window': [4, 40],
                         'beam': 256, 'target_only': True}, 'groups': groups,
              'evidence_contract': {'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['unique_m_recovery', 'key_recovery', 'scheme_break']}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
