#!/usr/bin/env python3
"""Screen target traces for source-ordered M product hypotheses."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = next((parent for parent in HERE.parents if (parent / 'r3_inert_ideal_probe').is_dir()), None)
if ROOT is None:
    raise RuntimeError('repository root with r3_inert_ideal_probe is missing')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'r3_inert_ideal_probe'))
import challenge_matrix_cpa192 as limb
import r3_cpa_matrix_rank as rank


DATA = HERE / 'inputs/m'
OUT = HERE / 't05-m-product-groups.json'
STOP, STRIDE, GAP, ANCHORS = 26008, 4, 52, 14
PAIRED_HYPOTHESIS_CAP = 64
MOTIF_BEFORE, MOTIF_AFTER = 256, 768
WORD_BEAM, BRANCH_BEAM = 8, 16
FIRST_WORD_KEEP = 4


class Control:
    @staticmethod
    def grid(labels, wave):
        left = labels - labels.mean(axis=0)
        right = wave - wave.mean(axis=0)
        left_norm = np.sum(left * left, axis=0)
        right_norm = np.sum(right * right, axis=0)
        denominator = np.sqrt(left_norm[:, None] * right_norm[None, :])
        return np.divide(left.T @ right, denominator, out=np.zeros_like(denominator),
                         where=(left_norm[:, None] > 1e-8) & (right_norm[None, :] > 1e-8))


CONTROL = Control()


def residual(values, rows):
    blocks = np.array([row['block'] for row in rows])
    within = np.array([row['within_block'] for row in rows], dtype=float)
    byte = np.array([bytes.fromhex(row['challenge_hex'])[0] for row in rows], dtype=np.uint8)
    bits = ((byte[:, None] >> np.arange(8)) & 1).astype(float)
    design = np.column_stack((np.ones(len(rows)), (blocks[:, None] == np.unique(blocks)[1:]).astype(float), within, bits))
    basis, _ = np.linalg.qr(design, mode='reduced')
    return values - basis @ (basis.T @ values)


def anchors(traces, rows, index, challenges):
    subset = [rows[i] for i in index]
    first = challenges[index, 0]
    candidate = np.arange(256, dtype=np.uint64)[:, None]
    labels = limb.low.hw32((candidate * first[None, :]) & np.uint64(255)).astype(float).T
    score = np.max(np.abs(CONTROL.grid(residual(labels, subset),
                                        residual(np.asarray(traces[index, :STOP:STRIDE], dtype=float), subset))), axis=0)
    chosen = []
    for coarse in np.argsort(score)[::-1]:
        sample = int(coarse * STRIDE)
        if sample + 5 * GAP >= STOP or any(abs(sample - prior) < 200 for prior in chosen):
            continue
        chosen.append(sample)
        if len(chosen) == ANCHORS:
            break
    return chosen, {'expanded': int(score.size), 'kept': len(chosen),
                    'dropped': int(score.size)-len(chosen),
                    'coarse_stride': STRIDE, 'scores_by_coarse_sample': score.tolist(),
                    'kept_candidates': [{'sample': sample, 'score': float(score[sample//STRIDE])}
                                        for sample in chosen]}


def motif_scores(template, wave):
    width = len(template)
    size = 1 << (len(wave) + width - 2).bit_length()
    dot = np.fft.irfft(np.fft.rfft(wave, size) * np.fft.rfft(template[::-1], size), size)
    dot = dot[width-1:len(wave)]
    sums = np.r_[0.0, np.cumsum(wave)]
    squares = np.r_[0.0, np.cumsum(wave * wave)]
    segment_sum = sums[width:] - sums[:-width]
    segment_square = squares[width:] - squares[:-width]
    denominator = np.sqrt(np.maximum(segment_square - segment_sum * segment_sum / width, 1e-12)
                          * float(template @ template))
    return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)


def structural_pairs(traces, splits, groups):
    reference = sorted((group for group in groups if group['split_agreement']),
                       key=lambda group: -group['min_validation_abs_r'])[:2]
    if len(reference) != 2:
        return [], []
    reference.sort(key=lambda group: group['anchor'])
    means = {name: np.asarray(traces[index, :STOP], dtype=np.float64).mean(axis=0)
             for name, index in splits.items()}
    motif_matches = []
    for group in reference:
        center = group['anchor']
        template = means['discovery'][center-MOTIF_BEFORE:center+MOTIF_AFTER].copy()
        if len(template) != MOTIF_BEFORE + MOTIF_AFTER:
            return [], []
        template -= template.mean()
        discovery = motif_scores(template, means['discovery'])
        validation = motif_scores(template, means['validation'])
        matches = []
        for offset in np.argsort(np.minimum(np.abs(discovery), np.abs(validation)))[::-1]:
            sample = int(offset + MOTIF_BEFORE)
            if sample + 5 * GAP >= STOP or any(abs(sample - item['sample']) < len(template) for item in matches):
                continue
            matches.append({'sample': sample, 'train_abs_r': float(abs(discovery[offset])),
                            'validation_abs_r': float(abs(validation[offset]))})
            if len(matches) == 2:
                break
        motif_matches.append({'reference_anchor': center, 'matches': sorted(matches, key=lambda item: item['sample'])})
    if any(len(item['matches']) != 2 for item in motif_matches):
        return [], motif_matches
    pairs = [(motif_matches[0]['matches'][repeat]['sample'],
              motif_matches[1]['matches'][repeat]['sample']) for repeat in range(2)]
    return [pair for pair in pairs if pair[0] < pair[1]], motif_matches


def continue_product_candidates(first_word, challenges, samples):
    state = np.zeros((len(challenges), 12), dtype=np.uint64)
    limb.apply_row(state, first_word, challenges, 0)
    branches, calls, frontier = [(first_word, state, [first_word], 0.0)], 0, []
    for row in range(1, 6):
        next_branches = []
        for value, current, words, score in branches:
            ranked = rank.rank_word(current[:, row], challenges[:, 0], samples[:, row], 256,
                                    True, limb, return_final_beam=True)
            calls += 1
            for item in ranked['scored_final_beam'][:WORD_BEAM]:
                word = item['value']
                updated = current.copy()
                limb.apply_row(updated, word, challenges, row)
                next_branches.append((value | (word << (32 * row)), updated, words + [word], score + item['score']))
        ordered = sorted(next_branches, key=lambda item: (-item[3], item[0]))
        branches = ordered[:BRANCH_BEAM]
        frontier.append({'row': row, 'expanded': len(ordered), 'kept': len(branches),
                         'dropped': len(ordered)-len(branches),
                         'kept_candidates': [{'value': hex(item[0]), 'score': item[3]} for item in branches],
                         'dropped_candidates': [{'value': hex(item[0]), 'score': item[3]}
                                                for item in ordered[BRANCH_BEAM:]]})
    return ([{'value': value, 'words': [hex(word) for word in words], 'rank_score': score}
             for value, _, words, score in branches], calls, frontier)


def score_group(secret, challenges, traces, rows, index, points):
    modeled, _ = limb.leakage_samples(secret, challenges[index])
    subset = [rows[i] for i in index]
    return [float(value) for value in np.diag(CONTROL.grid(
        residual(modeled.T, subset), residual(np.asarray(traces[np.ix_(index, points)], dtype=float), subset)))]


def add_group(groups, anchor, points, guesses, correlations, source, early_anchor=None, first_words=None,
              split_full_candidates=None, retained_product_candidates=None,
              deferred_product_candidates=None, search_frontier=None):
    groups.append({'anchor': anchor, 'points': points, 'candidate': guesses['discovery'],
                   'split_candidates': guesses, 'split_agreement': guesses['discovery'] == guesses['validation'],
                   'correlations': correlations, 'min_validation_abs_r': min(abs(value) for value in correlations['validation']),
                   'product_source': source, 'early_anchor': early_anchor,
                   'source_order_valid': early_anchor is None or early_anchor < anchor, 'first_words': first_words,
                   'split_full_candidates': split_full_candidates,
                   'retained_product_candidates': retained_product_candidates,
                   'deferred_product_candidates': deferred_product_candidates,
                   'search_frontier': search_frontier})


def main():
    started = time.perf_counter()
    progress_path = OUT.with_suffix(OUT.suffix + '.progress.jsonl')
    def progress(**values):
        with progress_path.open('a') as stream:
            stream.write(json.dumps(values | {'elapsed_s': time.perf_counter()-started}, sort_keys=True)+'\n')
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['known_key_trace_count'] == 0 and manifest['target_truth_loaded'] is False
    trace_path, schedule_path = DATA / 'traces.npy', DATA / 'public_schedule.jsonl'
    assert rank.digest(trace_path) == manifest['target_trace_sha256']
    assert rank.digest(schedule_path) == manifest['target_public_schedule_sha256']
    traces = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    assert traces.shape == (768, 45056) and len(rows) == 768
    challenges = rank.words(rows)
    splits = {name: np.array([i for i, row in enumerate(rows) if row['block'] % 2 == parity])
              for name, parity in [('discovery', 0), ('validation', 1)]}
    selected, anchor_frontier = anchors(traces, rows, splits['discovery'], challenges)
    progress(stage='single_anchor_scan', anchors_selected=len(selected))
    groups, rank_word_calls = [], 0
    first_words = {}
    for anchor in selected:
        points = [anchor + GAP * row for row in range(6)]
        guesses, full_candidates, search_frontier = {}, {}, {}
        for name, index in splits.items():
            samples = np.asarray(traces[np.ix_(index, points)], dtype=float)
            zero = np.zeros(len(index), dtype=np.uint64)
            ranked = rank.rank_word(zero, challenges[index, 0], samples[:, 0], 64,
                                    True, limb, return_final_beam=True)
            first_options = ranked['scored_final_beam'][:FIRST_WORD_KEEP]
            alternatives, continuations = [], []
            rank_word_calls += 1
            for option in first_options:
                candidates, calls, frontier = continue_product_candidates(
                    option['value'], challenges[index], samples)
                rank_word_calls += calls
                alternatives.extend([item | {'rank_score': item['rank_score'] + option['score']}
                                     for item in candidates])
                continuations.append({'first_word': hex(option['value']), 'score': option['score'],
                                      'frontier': frontier})
            alternatives.sort(key=lambda item: (-item['rank_score'], item['value']))
            full_candidates[name] = alternatives
            guesses[name] = hex(alternatives[0]['value'])
            first_words.setdefault(anchor, {})[name] = first_options[0]['value']
            search_frontier[name] = {'first_word': {'expanded': len(ranked['scored_final_beam']),
                'kept': len(first_options), 'dropped': len(ranked['scored_final_beam'])-len(first_options),
                'kept_candidates': first_options,
                'dropped_candidates': ranked['scored_final_beam'][FIRST_WORD_KEEP:]},
                'continuations': continuations}
        common = sorted(set(item['value'] for item in full_candidates['discovery']) &
                        set(item['value'] for item in full_candidates['validation']))
        retained = ([{'source': 'split_consensus', 'value': hex(value)} for value in common]
                    if common else [{'source': name, 'value': guesses[name]} for name in splits])
        retained_values = {item['value'] for item in retained}
        deferred = [{'source': name, 'value': hex(item['value']), 'rank_score': item['rank_score'],
                     'reason': 'outside_split_consensus_or_fallback'}
                    for name, alternatives in full_candidates.items() for item in alternatives
                    if hex(item['value']) not in retained_values]
        candidate = int(guesses['discovery'], 16)
        correlations = {name: score_group(candidate, challenges, traces, rows, index, points)
                        for name, index in splits.items()}
        add_group(groups, anchor, points, guesses, correlations, 'single',
                  split_full_candidates=full_candidates,
                  retained_product_candidates=retained,
                  deferred_product_candidates=deferred,
                  search_frontier=search_frontier)
        progress(stage='single_product', anchors_processed=len(first_words),
                 anchors_total=len(selected), rank_word_calls=rank_word_calls)
    pairs, motif_matches = structural_pairs(traces, splits, groups)
    progress(stage='structural_pairs', pair_count=len(pairs), motif_matches=len(motif_matches))
    for early, _ in pairs:
        if early not in first_words:
            first_words[early] = {}
            for name, index in splits.items():
                zero = np.zeros(len(index), dtype=np.uint64)
                ranked = rank.rank_word(zero, challenges[index, 0],
                                        np.asarray(traces[index, early], dtype=float), 256, True, limb)
                first_words[early][name] = ranked['top1']
                rank_word_calls += 1
    processed_pairs = pairs[:PAIRED_HYPOTHESIS_CAP]
    for pair_rank, (early, late) in enumerate(processed_pairs, 1):
        points = [late + GAP * row for row in range(6)]
        guesses, paired_words, full_candidates = {}, {}, {}
        for name, index in splits.items():
            first = first_words[early][name]
            full_candidates[name], calls, frontier = continue_product_candidates(
                first, challenges[index], np.asarray(traces[np.ix_(index, points)], dtype=float))
            guesses[name] = hex(full_candidates[name][0]['value'])
            paired_words[name] = full_candidates[name][0]['words']
            rank_word_calls += calls
            paired_words.setdefault('frontier', {})[name] = frontier
        candidate = int(guesses['discovery'], 16)
        correlations = {name: score_group(candidate, challenges, traces, rows, index, points)
                        for name, index in splits.items()}
        common = sorted(set(item['value'] for item in full_candidates['discovery']) &
                        set(item['value'] for item in full_candidates['validation']))
        retained = ([{'source': 'split_consensus', 'value': hex(value)} for value in common]
                    if common else [{'source': name, 'value': guesses[name]} for name in splits])
        retained_values = {item['value'] for item in retained}
        deferred = [{'source': name, 'value': hex(item['value']), 'rank_score': item['rank_score'],
                     'reason': 'outside_split_consensus_or_fallback'}
                    for name, alternatives in full_candidates.items() for item in alternatives
                    if hex(item['value']) not in retained_values]
        add_group(groups, late, points, guesses, correlations, 'early_first_late_six_store', early, paired_words,
                  full_candidates, retained, deferred)
        progress(stage='paired_product', pairs_processed=pair_rank,
                 pairs_total=len(processed_pairs), rank_word_calls=rank_word_calls)
    groups.sort(key=lambda value: (-value['min_validation_abs_r'], value['anchor'], value['product_source'], value['early_anchor'] or -1))
    truncated = len(processed_pairs) != len(pairs)
    report = {'schema': 'r3-target-only-m-products-v2',
              'status': 'analysis_incomplete' if truncated else 'no_candidates' if not groups else 'diagnostic_only',
              'method': {'coarse_stride': STRIDE, 'sample_gap': GAP, 'max_anchors': ANCHORS,
                         'paired_hypothesis_cap': PAIRED_HYPOTHESIS_CAP,
                         'motif_window': [MOTIF_BEFORE, MOTIF_AFTER],
                         'word_beam': WORD_BEAM, 'branch_beam': BRANCH_BEAM,
                         'first_word_keep': FIRST_WORD_KEEP,
                         'hypotheses': ['single_six_store', 'early_first_late_six_store'],
                         'source_order_required': True},
              'groups': groups,
              'anchor_frontier': anchor_frontier,
              'target_only_motif_matches': motif_matches,
              'computational_budget': {'selected_anchors': len(selected), 'candidate_pairs': len(pairs),
                                       'processed_pairs': len(processed_pairs), 'rank_word_calls': rank_word_calls,
                                       'unprocessed_pairs': len(pairs) - len(processed_pairs)},
              'evidence_contract': {'evidence_origin': 'measured', 'target_truth_loaded': False,
                                    'known_key_traces_loaded': False,
                                    'claim_not_allowed': ['m_recovery', 'key_recovery', 'scheme_break']},
              'inputs': {'target_trace_sha256': manifest['target_trace_sha256'],
                         'public_schedule_sha256': manifest['target_public_schedule_sha256'],
                         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
