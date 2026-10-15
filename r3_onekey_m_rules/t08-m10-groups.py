#!/usr/bin/env python3
"""Target-only source-order M10 addition after candidate M11 product groups."""
import json
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PRODUCT = SourceFileLoader('product', str(HERE / 't05-m-product-groups.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't08-m10-groups.json'
WORD_BEAM, BRANCH_BEAM = 8, 16


def rank_sequence_candidates(product, challenges, traces, rows, index, anchor, points=None,
                             track_paths=False, word_beam=WORD_BEAM, branch_beam=BRANCH_BEAM,
                             prefix_cache=None):
    subset = [rows[i] for i in index]
    zero = np.zeros(len(index), dtype=np.uint64)
    cache_key = ((product[index, :5].tobytes(), index.tobytes(), anchor,
                  tuple(points[:5]) if points is not None else None,
                  track_paths, word_beam, branch_beam) if prefix_cache is not None else None)
    cache_hit = cache_key in prefix_cache if prefix_cache is not None else False
    if cache_hit:
        branches, chosen_points, ties, frontier = prefix_cache[cache_key]
        branches, chosen_points, ties, frontier = (list(branches), list(chosen_points),
                                                 list(ties), list(frontier))
    else:
        branches, chosen_points, ties, frontier = [
            (0, np.zeros(len(index), dtype=np.uint64), [], 0.0, [])], [], [], []
    calls = 0
    for row in range(5 if cache_hit else 0, 6):
        next_branches, location_proposals = [], []
        shared_sample = None
        if points is not None:
            shared_sample = points[row]
        elif not track_paths and row:
            prior = chosen_points[-1]
            left, right = prior + 4, min(prior + 41, traces.shape[1])
            guesses = np.arange(256, dtype=np.uint64)[:, None]
            base = product[index, row] + branches[0][1]
            label = PRODUCT.limb.low.hw32((guesses + base[None, :]) & np.uint64(255)).astype(float).T
            wave = np.asarray(traces[index, left:right], dtype=float)
            scan = PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset), PRODUCT.residual(wave, subset))
            shared_sample = left + int(np.unravel_index(np.argmax(np.abs(scan)), scan.shape)[1])
        for value, carry, words, score, branch_points in branches:
            if shared_sample is not None:
                sample, location_score = shared_sample, None
            elif row == 0:
                sample, location_score = anchor, None
            else:
                left, right = branch_points[-1] + 4, min(branch_points[-1] + 41, traces.shape[1])
                guesses = np.arange(256, dtype=np.uint64)[:, None]
                base = product[index, row] + carry
                label = PRODUCT.limb.low.hw32((guesses + base[None, :]) & np.uint64(255)).astype(float).T
                wave = np.asarray(traces[index, left:right], dtype=float)
                scan = PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset), PRODUCT.residual(wave, subset))
                best = np.unravel_index(np.argmax(np.abs(scan)), scan.shape)
                sample, location_score = left + int(best[1]), float(abs(scan[best]))
                location_proposals.append({'prefix': hex(value), 'window': [left, right],
                                           'sample': sample, 'score': location_score})
            base = product[index, row] + carry
            ranked = PRODUCT.rank.rank_word(base, zero, np.asarray(traces[index, sample], dtype=float),
                                            256, False, PRODUCT.limb, return_final_beam=True)
            calls += 1
            if row == 0 and not ties:
                top = ranked['stages'][-1]['top_k']
                ties = [item['value'] for item in top if abs(item['score'] - top[0]['score']) < 1e-12]
            for item in ranked['scored_final_beam'][:word_beam]:
                word = item['value']
                next_branches.append((value | (word << (32 * row)),
                                      (base + np.uint64(word)) >> np.uint64(32), words + [word],
                                      score + item['score'], branch_points + [sample]))
        ordered = sorted(next_branches, key=lambda item: (-item[3], item[0], item[4]))
        branches = ordered[:branch_beam]
        chosen_points.append(branches[0][4][-1])
        frontier.append({'row': row, 'expanded': len(ordered), 'kept': len(branches),
                         'dropped': len(ordered)-len(branches), 'location_proposals': location_proposals,
                         'kept_candidates': [{'value': hex(item[0]), 'score': item[3], 'points': item[4]}
                                             for item in branches],
                         'dropped_candidates': [{'value': hex(item[0]), 'score': item[3], 'points': item[4]}
                                                for item in ordered[branch_beam:]]})
        if row == 4 and prefix_cache is not None:
            prefix_cache[cache_key] = (list(branches), list(chosen_points), list(ties), list(frontier))
    return ([{'value': value, 'words': [hex(word) for word in words],
              'rank_score': score, 'points': path}
             for value, _, words, score, path in branches], chosen_points, ties, calls, frontier, cache_hit)


def rank_sequence(product, challenges, traces, rows, index, anchor, points=None):
    candidates, chosen_points, ties, _, _, _ = rank_sequence_candidates(product, challenges, traces, rows, index, anchor, points)
    return candidates[0]['value'], chosen_points, ties


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(OUT)
    add_report = json.loads((HERE / 't06-m-add-anchors.json').read_text())
    add = add_report['groups']
    m00_report = json.loads((HERE / 't07-m00-groups.json').read_text())
    prior_m00 = [item for item in m00_report['groups']
                 if item.get('status') is None and item.get('split_agreement')]
    anchors_total = sum(len(item.get('add_anchors', [])) for item in add
                        if item.get('status') != 'truncated_scan' and
                        any(prior['anchor'] < item['product_anchor'] for prior in prior_m00))
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
        if not any(item['anchor'] < product_anchor for item in prior_m00):
            excluded_anchors.extend({'product_anchor': product_anchor, 'anchor': item['sample'],
                'screen_score': item['train_max_abs_r'], 'reason': 'no_preceding_m00_add_in_source_order'}
                for item in product['add_anchors'])
            continue
        secret = int(product['product_candidate'], 16)
        _, product_state = PRODUCT.rank.product_labels(secret, challenges, PRODUCT.limb)
        for anchor_info in product['add_anchors']:
            retained_anchor_count += 1
            anchor = anchor_info['sample']
            discovery_candidates, points, ties, calls, discovery_frontier, hit = rank_sequence_candidates(
                product_state, challenges, traces, rows, splits['discovery'], anchor,
                prefix_cache=prefix_cache)
            rank_word_calls += calls
            cache_hits += int(hit)
            if points[-1] >= traces.shape[1]:
                groups.append({'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                               'product_source': product['product_source'], 'anchor': anchor,
                               'status': 'truncated_points', 'points': points})
                continue
            validation_candidates, _, validation_ties, calls, validation_frontier, hit = rank_sequence_candidates(
                product_state, challenges, traces, rows, splits['validation'], anchor, points,
                prefix_cache=prefix_cache)
            rank_word_calls += calls
            cache_hits += int(hit)
            shared = sorted(set(item['value'] for item in discovery_candidates) &
                            set(item['value'] for item in validation_candidates))
            attempts = [{'mode': 'shared_top_path', 'points': points,
                         'discovery_frontier': discovery_frontier,
                         'validation_frontier': validation_frontier,
                         'shared_values': [hex(value) for value in shared]}]
            shared_paths = [(value, points, discovery_candidates, validation_candidates,
                             ties, validation_ties) for value in shared]
            if not shared:
                branched, _, branched_ties, calls, branched_frontier, hit = rank_sequence_candidates(
                    product_state, challenges, traces, rows, splits['discovery'], anchor,
                    track_paths=True, word_beam=16, branch_beam=32, prefix_cache=prefix_cache)
                rank_word_calls += calls
                cache_hits += int(hit)
                by_path = {}
                for item in branched:
                    by_path.setdefault(tuple(item['points']), []).append(item)
                attempts.append({'mode': 'carry_path_rescan', 'discovery_frontier': branched_frontier,
                                 'path_count': len(by_path)})
                for path in by_path:
                    path_points = list(path)
                    checked, _, checked_ties, calls, checked_frontier, hit = rank_sequence_candidates(
                        product_state, challenges, traces, rows, splits['validation'], anchor, path_points,
                        word_beam=16, branch_beam=32, prefix_cache=prefix_cache)
                    rank_word_calls += calls
                    cache_hits += int(hit)
                    overlap = sorted({item['value'] for item in by_path[path]} &
                                     {item['value'] for item in checked})
                    attempts.append({'mode': 'carry_path_validation', 'points': path_points,
                                     'validation_frontier': checked_frontier,
                                     'shared_values': [hex(value) for value in overlap]})
                    shared_paths.extend((value, path_points, by_path[path], checked,
                                         branched_ties, checked_ties) for value in overlap)
            if not shared_paths:
                groups.append({'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                               'product_source': product['product_source'], 'anchor': anchor, 'points': points,
                               'status': 'no_shared_full_candidate',
                               'split_full_candidates': {'discovery': discovery_candidates,
                                                         'validation': validation_candidates},
                               'search_attempts': attempts})
            for value, candidate_points, selected_discovery, selected_validation, first_ties, valid_ties in shared_paths:
                correlations = {}
                for name, ix in splits.items():
                    label, _ = PRODUCT.rank.add_labels(value, product_state[ix], PRODUCT.limb)
                    subset = [rows[i] for i in ix]
                    wave = np.asarray(traces[np.ix_(ix, candidate_points)], dtype=float)
                    correlations[name] = np.diag(PRODUCT.CONTROL.grid(PRODUCT.residual(label, subset),
                                                                       PRODUCT.residual(wave, subset))).tolist()
                result = {'product_anchor': product_anchor, 'product_candidate': product['product_candidate'],
                          'product_source': product['product_source'], 'anchor': anchor, 'points': candidate_points,
                          'split_candidates': {'discovery': hex(value), 'validation': hex(value)},
                          'split_full_candidates': {'discovery': selected_discovery,
                                                    'validation': selected_validation},
                          'first_word_ties': {'discovery': [hex(x) for x in first_ties],
                                              'validation': [hex(x) for x in valid_ties]},
                          'search_attempts': attempts,
                          'split_agreement': True, 'correlations': correlations,
                          'min_validation_abs_r': min(abs(x) for x in correlations['validation'])}
                groups.append(result)
                print(product_anchor, anchor, candidate_points, result['split_candidates'],
                      round(result['min_validation_abs_r'], 3), flush=True)
            with progress_path.open('a') as stream:
                stream.write(json.dumps({'elapsed_s': time.perf_counter()-started,
                    'anchors_processed': retained_anchor_count, 'anchors_total': anchors_total,
                    'candidate_count': len(groups),
                    'rank_word_calls': rank_word_calls})+'\n')
    groups.sort(key=lambda row: (-row.get('min_validation_abs_r', float('-inf')), row['product_anchor'], row['anchor']))
    status = ('analysis_incomplete' if add_report.get('status') == 'analysis_incomplete' else
              'no_candidates' if not groups else 'diagnostic_only')
    report = {'schema': 'r3-target-only-m10-groups-v1', 'status': status,
              'method': {'first_anchor_from_public_screen': True, 'subsequent_window': [4, 40],
                         'beam': 256, 'word_beam': WORD_BEAM, 'branch_beam': BRANCH_BEAM,
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
