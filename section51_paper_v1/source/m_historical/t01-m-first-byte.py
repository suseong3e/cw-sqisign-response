#!/usr/bin/env python3
"""Target-only M first-byte CPA and disjoint validation; no KAT or known-key traces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / 'inputs/m'
OUT = HERE / 't01-m-first-byte.json'
STOP = 26008
STRIDE = 8
SEED = 2026100611


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residualize(values: np.ndarray, rows: list[dict]) -> np.ndarray:
    blocks = np.array([row['block'] for row in rows])
    within = np.array([row['within_block'] for row in rows], dtype=float)
    unique = np.unique(blocks)
    design = np.column_stack((np.ones(len(rows)), (blocks[:, None] == unique[1:]).astype(float), within))
    basis, _ = np.linalg.qr(design, mode='reduced')
    return values - basis @ (basis.T @ values)


def predictions(rows: list[dict]) -> np.ndarray:
    challenge = np.array([int.from_bytes(bytes.fromhex(row['challenge_hex'])[:4], 'little') for row in rows], dtype=np.uint64)
    candidate = np.arange(256, dtype=np.uint64)[:, None]
    product = (candidate * challenge[None, :]) & np.uint64(255)
    return sum(((product >> np.uint64(bit)) & 1) for bit in range(8)).astype(float).T


def grid(label: np.ndarray, wave: np.ndarray) -> np.ndarray:
    x = label - label.mean(axis=0)
    y = wave - wave.mean(axis=0)
    denominator = np.sqrt(np.sum(x*x, axis=0)[:, None] * np.sum(y*y, axis=0)[None, :])
    return np.divide(x.T @ y, denominator, out=np.zeros((x.shape[1], y.shape[1])), where=denominator > 0)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if manifest['target_truth_loaded'] or manifest['known_key_trace_count'] or manifest['trace_count'] != 1024:
        raise ValueError('target-only input contract failed')
    trace_path, schedule_path = DATA / 'traces.npy', DATA / 'public_schedule.jsonl'
    if digest(trace_path) != manifest['target_trace_sha256'] or digest(schedule_path) != manifest['target_public_schedule_sha256']:
        raise ValueError('input hash mismatch')
    traces = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    if traces.shape != (1024, 45056) or len(rows) != 1024 or set(row['block'] for row in rows) != set(range(32)):
        raise ValueError('unexpected measured M target partition')
    index = {'train': np.array([i for i, row in enumerate(rows) if row['block'] < 16]),
             'validation': np.array([i for i, row in enumerate(rows) if 16 <= row['block'] < 24]),
             'heldout': np.array([i for i, row in enumerate(rows) if row['block'] >= 24])}
    labels = predictions(rows)
    train = index['train']
    label_train = residualize(labels[train], [rows[i] for i in train])
    wave_train = residualize(np.asarray(traces[train, :STOP:STRIDE], dtype=float), [rows[i] for i in train])
    coarse = grid(label_train, wave_train)
    candidate, coarse_position = np.unravel_index(np.argmax(np.abs(coarse)), coarse.shape)
    center = int(coarse_position * STRIDE)
    left, right = max(0, center-STRIDE), min(STOP, center+STRIDE+1)
    fine_wave = residualize(np.asarray(traces[train, left:right], dtype=float), [rows[i] for i in train])
    fine = grid(label_train[:, candidate:candidate+1], fine_wave)[0]
    sample = left + int(np.argmax(np.abs(fine)))
    chosen = int(candidate)
    evaluations = {}
    for name in ('validation', 'heldout'):
        ix = index[name]
        subset_rows = [rows[i] for i in ix]
        x = residualize(labels[ix], subset_rows)
        y = residualize(np.asarray(traces[ix, sample:sample+1], dtype=float), subset_rows)
        correlations = grid(x, y)[:, 0]
        rank = 1 + int(np.count_nonzero(np.abs(correlations) > abs(correlations[chosen]) + 1e-12))
        evaluations[name] = {'count': len(ix), 'selected_abs_r': float(abs(correlations[chosen])),
                             'selected_signed_r': float(correlations[chosen]), 'rank_of_train_candidate': rank,
                             'best_candidate': int(np.argmax(np.abs(correlations)))}
    ix = index['validation']
    validation_rows = [rows[i] for i in ix]
    label = residualize(labels[ix, chosen:chosen+1], validation_rows)[:, 0]
    wrong_wave = residualize(np.asarray(traces[ix, 36864:45056:16], dtype=float), validation_rows)
    wrong_max = float(np.max(np.abs(grid(label[:, None], wrong_wave))))
    rng = np.random.default_rng(SEED)
    observed = evaluations['validation']['selected_abs_r']
    wave = residualize(np.asarray(traces[ix, sample:sample+1], dtype=float), validation_rows)
    permuted = [abs(float(grid(label[rng.permutation(len(label)), None], wave)[0, 0])) for _ in range(999)]
    report = {'schema': 'r3-target-only-m-first-byte-v1', 'status': 'diagnostic_only',
              'evidence_contract': {'evidence_origin': 'measured', 'evidence_purpose': 'diagnostic',
                                    'blinding': 'target_truth_not_loaded', 'measurement_scope': 'isolated_primitive',
                                    'target_truth_loaded': False, 'known_key_traces_loaded': False,
                                    'claim_scope': 'round3_matrix_m:target_only_first_byte_screen',
                                    'claim_not_allowed': ['full_m_recovery', 'key_recovery', 'scheme_break']},
              'inputs': {'target_trace_sha256': manifest['target_trace_sha256'],
                         'public_schedule_sha256': manifest['target_public_schedule_sha256'],
                         'script_sha256': digest(Path(__file__))},
              'train_count': len(train), 'discovery_window': [0, STOP], 'stride': STRIDE,
              'first_byte_candidate': chosen, 'sample': sample,
              'train_abs_r': float(abs(fine[sample-left])),
              'validation': evaluations['validation'], 'heldout': evaluations['heldout'],
              'wrong_window_max_abs_r': wrong_max,
              'validation_permutation_p': (1+sum(value >= observed for value in permuted))/1000,
              'permutation_seed': SEED}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('first_byte_candidate', 'sample', 'train_abs_r',
                                                  'validation', 'heldout', 'wrong_window_max_abs_r',
                                                  'validation_permutation_p')}, indent=2))


if __name__ == '__main__':
    main()
