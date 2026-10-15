#!/usr/bin/env python3
"""Target-only byte screen after removing public-input main effects."""
import hashlib
import json
from pathlib import Path

import numpy as np

from importlib.machinery import SourceFileLoader


HERE = Path(__file__).resolve().parent
FIRST = SourceFileLoader('first', str(HERE / 't01-m-first-byte.py')).load_module()
DATA = HERE / 'inputs/m'
OUT = HERE / 't02b-m-public-control.json'
STOP = 26008
STRIDE = 4


def residual(values, rows):
    blocks = np.array([row['block'] for row in rows])
    within = np.array([row['within_block'] for row in rows], dtype=float)
    byte = np.array([bytes.fromhex(row['challenge_hex'])[0] for row in rows], dtype=np.uint8)
    bits = ((byte[:, None] >> np.arange(8)) & 1).astype(float)
    design = np.column_stack((np.ones(len(rows)), (blocks[:, None] == np.unique(blocks)[1:]).astype(float), within, bits))
    basis, upper = np.linalg.qr(design, mode='reduced')
    assert np.linalg.matrix_rank(upper) == design.shape[1]
    return values - basis @ (basis.T @ values)


def grid(labels, wave):
    x = labels - labels.mean(axis=0)
    y = wave - wave.mean(axis=0)
    xnorm = np.sum(x*x, axis=0)
    ynorm = np.sum(y*y, axis=0)
    denominator = np.sqrt(xnorm[:, None] * ynorm[None, :])
    return np.divide(x.T @ y, denominator, out=np.zeros_like(denominator),
                     where=(xnorm[:, None] > 1e-8) & (ynorm[None, :] > 1e-8))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['known_key_trace_count'] == 0 and manifest['target_truth_loaded'] is False
    trace_path = DATA / 'traces.npy'
    schedule_path = DATA / 'public_schedule.jsonl'
    assert FIRST.digest(trace_path) == manifest['target_trace_sha256']
    assert FIRST.digest(schedule_path) == manifest['target_public_schedule_sha256']
    traces = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    assert traces.shape == (1024, 45056) and len(rows) == 1024
    index = {name: np.array([i for i, row in enumerate(rows) if left <= row['block'] < right])
             for name, left, right in [('train', 0, 16), ('validation', 16, 24), ('heldout', 24, 32)]}
    labels = FIRST.predictions(rows)
    train = index['train']
    x = residual(labels[train], [rows[i] for i in train])
    y = residual(np.asarray(traces[train, :STOP:STRIDE], dtype=float), [rows[i] for i in train])
    scores = grid(x, y)
    flat = np.abs(scores).ravel()
    order = np.argsort(flat)[-20:][::-1]
    discoveries = []
    for position in order:
        candidate, coarse = np.unravel_index(position, scores.shape)
        center = int(coarse * STRIDE)
        left, right = max(0, center - STRIDE), min(STOP, center + STRIDE + 1)
        fine = grid(x[:, candidate:candidate+1], residual(np.asarray(traces[train, left:right], dtype=float), [rows[i] for i in train]))[0]
        sample = left + int(np.argmax(np.abs(fine)))
        evaluations = {}
        for name in ('validation', 'heldout'):
            ix = index[name]
            subset = [rows[i] for i in ix]
            correlations = grid(residual(labels[ix], subset), residual(np.asarray(traces[ix, sample:sample+1], dtype=float), subset))[:, 0]
            evaluations[name] = {'selected_abs_r': float(abs(correlations[candidate])),
                                 'rank': 1 + int(np.count_nonzero(np.abs(correlations) > abs(correlations[candidate]) + 1e-12)),
                                 'best_candidate': int(np.argmax(np.abs(correlations)))}
        discoveries.append({'candidate': int(candidate), 'sample': sample,
                            'train_abs_r': float(abs(fine[sample-left])), **evaluations})
    report = {'schema': 'r3-target-only-m-public-control-v2', 'status': 'diagnostic_only',
              'rule': 'project out capture block, within-block order, and all eight public input byte bits; scan first product byte HW',
              'evidence_contract': {'evidence_origin': 'measured', 'target_truth_loaded': False,
                                    'known_key_traces_loaded': False, 'claim_scope': 'target_only_first_byte_screen',
                                    'claim_not_allowed': ['m_recovery', 'key_recovery', 'scheme_break']},
              'inputs': {'target_trace_sha256': manifest['target_trace_sha256'],
                         'public_schedule_sha256': manifest['target_public_schedule_sha256'],
                         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'split_counts': {name: len(ix) for name, ix in index.items()},
              'discovery_window': [0, STOP], 'stride': STRIDE, 'discoveries': discoveries}
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(discoveries[:5], indent=2))


if __name__ == '__main__':
    main()
