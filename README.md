# SQIsign Round-3 CPA offline analysis core

This repository is the minimal public implementation of CPA candidate ranking
and adaptive path selection for SQIsign Round-3 matrix operations. The Git
source tree does not contain raw traces, KAT files, secret keys, candidate
results, device-control code, or vendor snapshots with unresolved provenance.

## Scope

The measurement scope is `isolated_primitive`: the original arithmetic
functions are executed in isolation. The PC represents the honest signer, and
the analyzer represents the attacker. The analyzer receives only traces and
public challenges before candidate commitment; it does not receive KAT data or
secret operands.

A run that reduces the validated candidate set substantially without producing
a unique candidate is reported as a **partial success**. Reports must state the
candidate count, reduction, validation results, and whether the correct value
was found by the post-hoc KAT check.

The Section 5.1 release provides the paper, measured traces, public schedules,
the disposable experiment KAT, preserved source, and the complete clean run005
results. Private schedules are not included. The claim is limited to a one-key
`isolated_primitive` artifact replay.

## Contents

- `r3_cpa_matrix_rank.py`: truth-free matrix candidate ranking
- `r3_onekey_m_rules/`: adaptive M-candidate path rules
- `r3_inert_ideal_probe/`: leakage-model helpers used by the rules
- `test_r3_adaptive_paths.py`: regression checks that require no data or hardware
- `section51_reproduce.py`: Section 5.1 snapshot-integrity and numerical checks
- `section51_full_replay.py`: canonical Section 5.1 runner using the released experiment KAT
- `section51_paper_v1/source/`: preserved and path-independent canonical source
- `docs/section51_reproduction.md`: data layout, verification, and claim scope

## Quick start

Verified with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile \
  r3_cpa_matrix_rank.py test_r3_adaptive_paths.py \
  r3_onekey_m_rules/*.py r3_inert_ideal_probe/*.py
python test_r3_adaptive_paths.py
python test_section51_reproduce.py
python test_section51_full_replay.py
```

To verify an extracted Section 5.1 validation bundle:

```bash
python section51_reproduce.py --bundle section51-bundle --out section51-report.json
```

The paper, measured traces, validation bundle, clean run005 results, source
commit, runner, and asset SHA-256 values are bound in
`artifact/section51-full-replay-release.json`.

See `docs/section51_reproduction.md` for download, extraction, verification,
and scope details.

Measured data is distributed as GitHub Release assets because it exceeds Git
file-size limits. No project license has been selected; no reuse rights are
granted until an explicit license is added.
