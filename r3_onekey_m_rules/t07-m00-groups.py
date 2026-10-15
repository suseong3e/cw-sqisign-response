#!/usr/bin/env python3
"""Test source-ordered 84-sample addition groups for target-only M00."""
import json
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't07-m00-groups.json'
GAP = 84
WORD_BEAM, BRANCH_BEAM = 8, 16


def recover_candidates(product, samples, word_beam=WORD_BEAM, branch_beam=BRANCH_BEAM,
                       prefix_cache=None):
    cache_key = ((product[:, :5].tobytes(), samples[:, :5].tobytes(), word_beam, branch_beam)
                 if prefix_cache is not None else None)
    cache_hit = cache_key in prefix_cache if prefix_cache is not None else False
    if cache_hit:
        branches, frontier = prefix_cache[cache_key]
        branches, frontier = list(branches), list(frontier)
    else:
        branches = [(0, np.zeros(product.shape[0], dtype=np.uint64), [], 0.0)]
        frontier = []
    zero = np.zeros(product.shape[0], dtype=np.uint64)
    calls = 0
    for row in range(5 if cache_hit else 0, 6):
        next_branches = []
        for value, carry, words, score in branches:
            base = product[:, row] + carry
            ranked = PRODUCT.rank.rank_word(base, zero, samples[:, row], 256, False,
                                            PRODUCT.limb, return_final_beam=True)
            calls += 1
            for item in ranked['scored_final_beam'][:word_beam]:
                word = item['value']
                next_branches.append((value | (word << (32 * row)),
                                      (base + np.uint64(word)) >> np.uint64(32), words + [word],
                                      score + item['score']))
        ordered = sorted(next_branches, key=lambda item: (-item[3], item[0]))
        branches = ordered[:branch_beam]
        frontier.append({'row': row, 'expanded': len(ordered), 'kept': len(branches),
                         'dropped': len(ordered)-len(branches),
                         'kept_candidates': [{'value': hex(item[0]), 'score': item[3]} for item in branches],
                         'dropped_candidates': [{'value': hex(item[0]), 'score': item[3]}
                                                for item in ordered[branch_beam:]]})
        if row == 4 and prefix_cache is not None:
            prefix_cache[cache_key] = (list(branches), list(frontier))
    return ([{'value': value, 'words': [hex(word) for word in words], 'rank_score': score}
             for value, _, words, score in branches], calls, frontier, cache_hit)


def recover(product, samples):
    return recover_candidates(product, samples)[0][0]['value']


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(OUT)
    add_report = json.loads((HERE / 't06-m-add-anchors.json').read_text())
    add = add_report['groups']
    later_product_anchors = {item['product_anchor'] for item in add
                             if item.get('status') != 'truncated_scan'}
    anchors_total = sum(any(later > anchor['sample'] for later in later_product_anchors)
                        for item in add for anchor in item.get('add_anchors', []))
    rows = [json.loads(line) for line in (DATA / 'public_schedule.jsonl').read_text().splitlines()]
    traces = np.load(DATA / 'traces.npy', mmap_mode='r', allow_pickle=False)
    challenges = PRODUCT.rank.words(rows)
    splits = {name: np.array([i for i, row in enumerate(rows) if row['block'] % 2 == parity])
              for name, parity in [('discovery', 0), ('validation', 1)]}
    groups, excluded_anchors, retained_anchor_count, rank_word_calls = [], [], 0, 0
    prefix_cache, cache_hits = {}, 0
    progress_path = OUT.with_suffix(OUT.suffix + '.progress.jsonl')
    for product in add:
        product_anchor = product['product_anchor']
        if product.get('status') == 'truncated_scan':
            continue
        secret = int(product['product_candidate'], 16)
        _, product_state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
        for anchor_info in product['add_anchors']:
            anchor = anchor_info['sample']
            if not any(later > anchor for later in later_product_anchors):
                excluded_anchors.append({'product_anchor': product_anchor, 'anchor': anchor,
                    'screen_score': anchor_info['train_max_abs_r'],
                    'reason': 'no_later_m11_product_in_source_order'})
                continue
            retained_anchor_count += 1
            points = [anchor + GAP*row for row in range(6)]
            if points[-1] >= traces.shape[1]:
                groups.append({'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                               'product_source': product['product_source'], 'anchor': anchor,
                               'status': 'truncated_points', 'points': points})
                continue
            attempts = []
            samples_by_split = {name: np.asarray(traces[np.ix_(ix, points)], dtype=float)
                                for name, ix in splits.items()}
            for word_beam, branch_beam in ((WORD_BEAM, BRANCH_BEAM), (16, 32)):
                candidates, frontiers = {}, {}
                for name, ix in splits.items():
                    candidates[name], calls, frontiers[name], hit = recover_candidates(
                        product_state[ix], samples_by_split[name], word_beam, branch_beam, prefix_cache)
                    rank_word_calls += calls
                    cache_hits += int(hit)
                shared = sorted(set(item['value'] for item in candidates['discovery']) &
                                set(item['value'] for item in candidates['validation']))
                attempts.append({'word_beam': word_beam, 'branch_beam': branch_beam,
                                 'candidates': candidates, 'frontier': frontiers,
                                 'shared_values': [hex(value) for value in shared],
                                 'prefix_cache_hits_cumulative': cache_hits})
                if shared:
                    break
            if not shared:
                groups.append({'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                               'product_source': product['product_source'], 'anchor': anchor, 'points': points,
                               'status': 'no_shared_full_candidate', 'split_full_candidates': candidates,
                               'search_attempts': attempts})
                with progress_path.open('a') as stream:
                    stream.write(json.dumps({'elapsed_s': time.perf_counter()-started,
                        'anchors_processed': retained_anchor_count, 'anchors_total': anchors_total,
                        'candidate_count': 0,
                        'rank_word_calls': rank_word_calls})+'\n')
                continue
            for value in shared:
                guesses = {name: hex(value) for name in splits}
                correlations = {}
                for name, ix in splits.items():
                    label, _ = PRODUCT.rank.add_labels(value, product_state[ix], PRODUCT.limb)
                    subset = [rows[i] for i in ix]
                    wave = np.asarray(traces[np.ix_(ix, points)], dtype=float)
                    correlations[name] = np.diag(PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset),
                                                                       PRODUCT.residual(wave, subset))).tolist()
                result = {'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                          'product_source': product['product_source'], 'anchor': anchor, 'points': points,
                          'split_candidates': guesses, 'split_agreement': True,
                          'split_full_candidates': candidates, 'correlations': correlations,
                          'search_attempts': attempts,
                          'min_validation_abs_r': min(abs(x) for x in correlations['validation'])}
                groups.append(result)
                print(product_anchor, anchor, guesses, round(result['min_validation_abs_r'], 3), flush=True)
            with progress_path.open('a') as stream:
                stream.write(json.dumps({'elapsed_s': time.perf_counter()-started,
                    'anchors_processed': retained_anchor_count, 'anchors_total': anchors_total,
                    'candidate_count': len(groups),
                    'rank_word_calls': rank_word_calls})+'\n')
    groups.sort(key=lambda row: (-row.get('min_validation_abs_r', float('-inf')), row['product_anchor'], row['anchor']))
    status = ('analysis_incomplete' if add_report.get('status') == 'analysis_incomplete' else
              'no_candidates' if not groups else 'diagnostic_only')
    report = {'schema': 'r3-target-only-m00-groups-v1', 'status': status,
              'method': {'add_gap': GAP, 'beam': 256, 'word_beam': WORD_BEAM, 'branch_beam': BRANCH_BEAM,
                         'add_anchor_beam': 'all', 'target_only': True},
              'groups': groups, 'excluded_anchors': excluded_anchors,
              'computational_budget': {'retained_anchors': retained_anchor_count,
                                       'excluded_anchor_count': len(excluded_anchors), 'rank_word_calls': rank_word_calls,
                                       'prefix_cache_hits': cache_hits,
                                       'elapsed_s': time.perf_counter()-started},
              'evidence_contract': {'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['unique_m_recovery', 'key_recovery', 'scheme_break']}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
