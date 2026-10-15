import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path('/home/tntjd/cw-sqisign-response')
sys.path.insert(0, str(root))
import r3_cpa_onekey_single_split as split
import r3_cpa_crt_analyze as crt

run = root / 'runs/20261007T045953978815Z-r3-thirdkey-crt-manual'
data = run / 'inputs/crt'
original_path = run / 'single_split_frozen.json'
m_path = root / '.plans/2026-10-07-E-r3-single-split-cpa/t44-third-m-candidates-bound.json'
out = run / 'single_split_corrected_frozen.json'
assert not out.exists()
manifest = json.loads((data / 'manifest.json').read_text())
assert not manifest['target_truth_loaded'] and not manifest['known_key_trace_count']
assert crt.digest(data / 'traces.npy') == manifest['target_trace_sha256']
assert crt.digest(data / 'public_schedule.jsonl') == manifest['target_public_schedule_sha256']
original = json.loads(original_path.read_text())
assert original['status'] == 'frozen_before_heldout_and_KAT'
assert original['target_trace_sha256'] == manifest['target_trace_sha256']
assert original['target_public_schedule_sha256'] == manifest['target_public_schedule_sha256']
rows = [json.loads(line) for line in (data / 'public_schedule.jsonl').read_text().splitlines()]
traces = np.load(data / 'traces.npy', mmap_mode='r', allow_pickle=False)
m = json.loads(m_path.read_text())['matrix']
top = original['frozen'][0]
u = top['u'] & ((1 << 256) - 1)
assert u.bit_length() == 256
points = [row[:8] for row in top['multiply_points']]
assert len(points) == 6 and all(len(row) == 8 for row in points)
matrix = tuple(int(m[name], 16) for name in ('m00', 'm01')) + (
    int(top['m10'], 16), int(m['m11'], 16))
public = crt.oracle_public(rows, matrix, split.ORACLE)
indices = {coord: split.split_indices(rows, public, coord) for coord in ('x', 'y')}
unique = {}
for rank, candidate in enumerate(original['frozen'], 1):
    unique.setdefault((candidate['x'], candidate['y']), []).append(rank)
screened = []
for (x, y), ranks in unique.items():
    groups = {}
    for coord, secret in (('x', x), ('y', y)):
        groups[coord] = {}
        for part, ix in (('train', indices[coord][0]), ('validation', indices[coord][1])):
            _, per = split.product_score(secret, u, public[coord], traces, ix, points)
            groups[coord][part] = {'mean_abs_r': sum(per)/len(per),
                                   'min_abs_r': min(per), 'per_store_abs_r': per}
    screened.append({'x': x, 'y': y, 'original_ranks': ranks, 'groups': groups,
                     'train_min_abs_r': min(groups[c]['train']['min_abs_r'] for c in ('x', 'y')),
                     'validation_min_abs_r': min(groups[c]['validation']['min_abs_r'] for c in ('x', 'y'))})
screened.sort(key=lambda item: (-item['validation_min_abs_r'], -item['train_min_abs_r'],
                                item['x'], item['y']))
passing = [item for item in screened if item['train_min_abs_r'] > 0.5 and
           item['validation_min_abs_r'] > 0.5]
assert len(passing) == 1
chosen = passing[0]
candidate = {'m10': top['m10'], 'x': chosen['x'], 'y': chosen['y'], 'u': u,
             'u_word_count': 8, 'multiply_points': points,
             'selection_score': chosen['validation_min_abs_r'],
             'low_validation_mean': top['low_validation_mean'],
             'product_validation_mean': sum(chosen['groups'][c]['validation']['mean_abs_r']
                                            for c in ('x', 'y'))/2}
report = {'schema': 'r3-single-split-low-joint-frozen-corrected-v1',
          'status': 'frozen_before_heldout_and_KAT',
          'points': {'low': original['points']['low'],
                     'inverse_base': original['points']['inverse_base'],
                     'multiply': points},
          'target_trace_sha256': manifest['target_trace_sha256'],
          'target_public_schedule_sha256': manifest['target_public_schedule_sha256'],
          'm_candidates_sha256': crt.digest(m_path),
          'subtraction_poi_sha256': crt.digest(run / 'subtraction_poi.json'),
          'original_frozen_sha256': crt.digest(original_path),
          'diagnostic_sha256': {name: crt.digest(root / '.plans/2026-10-07-E-r3-single-split-cpa' / name)
                                for name in ('t45-third-high-gate-diagnostic.json',
                                             't46-third-row6-local-stores.json',
                                             't47-third-inverse-dynamic-poi.json')},
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'selection': {'rule': 'unique x/y pair with all 6x8 product stores above 0.5 in both train and validation for the low 256 bits of the original leading u',
                        'reason_for_width_revision': 'original nine-word model fails later stores in a row-dependent pattern; eight-word model matches all six rows',
                        'validation_reused_for_selection': True,
                        'original_ranks_of_selected_pair': chosen['original_ranks'],
                        'screened_unique_pairs': len(screened),
                        'passing_pairs': len(passing),
                        'runner_up_validation_min_abs_r': screened[1]['validation_min_abs_r']},
          'screened': screened,
          'frozen': [candidate],
          'evidence_contract': {'evidence_origin': 'measured',
                                'evidence_purpose': 'diagnostic',
                                'blinding': 'unblinded',
                                'measurement_scope': 'isolated_primitive',
                                'claim_scope': 'r3_single_split:one_key_coordinate_candidate_method_development',
                                'target_truth_loaded': False,
                                'known_key_traces_loaded': False,
                                'claim_not_allowed': ['blind_success_rate', 'scheme_break']}}
with out.open('x') as file:
    json.dump(report, file, indent=2)
    file.write('\n')
out.chmod(0o600)
print(json.dumps({'frozen': str(out), 'candidate_count': 1,
                  'train_min_abs_r': chosen['train_min_abs_r'],
                  'validation_min_abs_r': chosen['validation_min_abs_r'],
                  'runner_up_validation_min_abs_r': screened[1]['validation_min_abs_r']}))
