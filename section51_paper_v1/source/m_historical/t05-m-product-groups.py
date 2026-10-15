#!/usr/bin/env python3
"""Find ordered six-store M products from one key's measured traces only."""
import hashlib
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'r3_inert_ideal_probe'))
import challenge_matrix_cpa192 as limb
import r3_cpa_matrix_rank as rank

CONTROL = SourceFileLoader('control', str(HERE / 't02-m-public-control.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't05-m-product-groups.json'
STOP, STRIDE, GAP, ANCHORS = 26008, 4, 52, 14


def residual(values, rows):
    blocks = np.array([row['block'] for row in rows])
    within = np.array([row['within_block'] for row in rows], dtype=float)
    byte = np.array([bytes.fromhex(row['challenge_hex'])[0] for row in rows], dtype=np.uint8)
    bits = ((byte[:, None] >> np.arange(8)) & 1).astype(float)
    design = np.column_stack((np.ones(len(rows)), (blocks[:, None] == np.unique(blocks)[1:]).astype(float), within, bits))
    q, _ = np.linalg.qr(design, mode='reduced')
    return values - q @ (q.T @ values)


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
        if sample + 5*GAP >= STOP or any(abs(sample - prior) < 200 for prior in chosen):
            continue
        chosen.append(sample)
        if len(chosen) == ANCHORS:
            break
    return chosen


def score_group(secret, challenges, traces, rows, index, points):
    modeled, _ = limb.leakage_samples(secret, challenges[index])
    subset = [rows[i] for i in index]
    left = residual(modeled.T, subset)
    right = residual(np.asarray(traces[np.ix_(index, points)], dtype=float), subset)
    correlations = np.diag(CONTROL.grid(left, right))
    return [float(value) for value in correlations]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['known_key_trace_count'] == 0 and manifest['target_truth_loaded'] is False
    trace_path = DATA / 'traces.npy'
    schedule_path = DATA / 'public_schedule.jsonl'
    assert rank.digest(trace_path) == manifest['target_trace_sha256']
    assert rank.digest(schedule_path) == manifest['target_public_schedule_sha256']
    traces = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    assert traces.shape == (1024, 45056) and len(rows) == 1024
    challenges = rank.words(rows)
    splits = {name: np.array([i for i, row in enumerate(rows) if row['block'] % 2 == parity])
              for name, parity in [('discovery', 0), ('validation', 1)]}
    discovery = anchors(traces, rows, splits['discovery'], challenges)
    report_groups = []
    for anchor in discovery:
        points = [anchor + GAP*row for row in range(6)]
        guesses = {}
        for name, index in splits.items():
            samples = np.asarray(traces[np.ix_(index, points)], dtype=float)
            secret, _, _ = rank.rank_product(challenges[index], samples, limb)
            guesses[name] = hex(secret)
        candidate = int(guesses['discovery'], 16)
        correlations = {name: score_group(candidate, challenges, traces, rows, index, points)
                        for name, index in splits.items()}
        report_groups.append({'anchor': anchor, 'points': points, 'candidate': guesses['discovery'],
                              'split_candidates': guesses, 'split_agreement': guesses['discovery'] == guesses['validation'],
                              'correlations': correlations,
                              'min_validation_abs_r': min(abs(value) for value in correlations['validation'])})
        print(anchor, guesses, round(report_groups[-1]['min_validation_abs_r'], 3), flush=True)
    report_groups.sort(key=lambda value: (-value['min_validation_abs_r'], value['anchor']))
    report = {'schema': 'r3-target-only-m-products-v1', 'status': 'diagnostic_only',
              'method': {'coarse_stride': STRIDE, 'sample_gap': GAP, 'max_anchors': ANCHORS,
                         'nonoverlap_samples': 200, 'anchor_rank': 'max absolute first-byte partial Pearson',
                         'group_rank': 'minimum validation absolute Pearson across six ordered stores'},
              'groups': report_groups,
              'evidence_contract': {'evidence_origin': 'measured', 'target_truth_loaded': False,
                                    'known_key_traces_loaded': False, 'claim_not_allowed': ['m_recovery', 'key_recovery', 'scheme_break']},
              'inputs': {'target_trace_sha256': manifest['target_trace_sha256'],
                         'public_schedule_sha256': manifest['target_public_schedule_sha256'],
                         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('top groups', [(item['anchor'], item['candidate'], item['min_validation_abs_r']) for item in report_groups[:5]])


if __name__ == '__main__':
    main()
