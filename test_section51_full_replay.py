#!/usr/bin/env python3
"""Focused integrity checks for the Section 5.1 paper replay runner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import section51_full_replay as replay

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "section51_full_replay.py"
M_DATA = ROOT / "runs/20261007T025549776051Z-r3-onekey-m-only/evidence/inputs/m"
CRT_DATA = ROOT / "runs/20261007T045953978815Z-r3-thirdkey-crt-manual/inputs/crt"
KAT = ROOT / "runs/20261006T200749935815Z-r3-onekey-trial-0002/private/fresh_key.rsp"


class ReplayRunnerTest(unittest.TestCase):
    def command(self, out: Path, *extra: str) -> list[str]:
        return [sys.executable, str(RUNNER), "--m-data", str(M_DATA), "--crt-data", str(CRT_DATA),
                "--kat", str(KAT), "--out", str(out), *extra]

    def fixture(self, root: Path, paper_max: float = replay.EXPECTED_AUDIT["permutation_max_sum_abs_r"],
                replay_max: float = 82.53820607707073) -> Path:
        (root / "m").mkdir()
        (root / "crt").mkdir()
        values = {
            root / "candidate_commitment.json": {"stage": "before_posthoc_kat",
                "config": {"canonical_audit_sha256": "a", "historical_frozen_sha256": replay.HISTORICAL_FROZEN_SHA256,
                           "paper_v1_permutation_seed": replay.PAPER_V1_PERMUTATION_SEED}},
            root / "m/t44-third-m-candidates-bound.json": {"status": "one_candidate"},
            root / "crt/single_split_corrected_frozen.json": {"status": "frozen_before_heldout_and_KAT"},
            root / "crt/high_gate.json": {"status": "exploratory_candidates"},
            root / "crt/high_norm.json": {"status": "candidate_not_KAT_audited"},
            root / "crt/posthoc_kat.json": {"status": "diagnostic_exact_match",
                                               "field_matches": {key: True for key in ("M", "N", "x", "y", "u")},
                                               "kat_sha256": "0" * 64},
        }
        for path, value in values.items():
            path.write_text(json.dumps(value), encoding="utf-8")
        frozen = replay.digest(root / "crt/single_split_corrected_frozen.json")
        audit = {"status": "validated", **replay.EXPECTED_AUDIT,
                 "permutation_max_sum_abs_r": paper_max,
                 "replay_frozen_hash_permutation_max_sum_abs_r": replay_max,
                 "historical_frozen_sha256": replay.HISTORICAL_FROZEN_SHA256,
                 "replay_frozen_sha256": frozen, "frozen_sha256": frozen,
                 "permutation_seed": replay.PAPER_V1_PERMUTATION_SEED,
                 "replay_frozen_hash_permutation_seed": int(frozen[:16], 16),
                 "permutation_seed_rule": "paper-v1 fixed historical frozen SHA-256 prefix", "script_sha256": "a"}
        (root / "crt/heldout_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        (root / "figure_2.pdf").write_bytes(b"pdf")
        (root / "figure_2.png").write_bytes(b"png")
        return root

    def test_smoke_validates_dag_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fresh"
            result = subprocess.run(self.command(out, "--smoke"), cwd=ROOT, text=True,
                                    capture_output=True, check=True)
            steps = [json.loads(line)["step"] for line in result.stdout.splitlines()]
            self.assertEqual(steps[-2:], ["heldout-and-posthoc-kat", "figure-2"])
            self.assertFalse(out.exists())

    def test_final_fixture_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = replay.validate_final(self.fixture(Path(directory)), {"source_hashes_sha256": "a", "canonical_audit_sha256": "a"}, {"m": "b"}, {"crt": "c"})
            self.assertEqual(result["status"], "accepted")
            self.assertIn("numpy", result["runtime"])
            self.assertEqual(result["kat_sha256"], "0" * 64)

    def test_permutation_max_changes_are_rejected(self) -> None:
        for kwargs, field in (({"paper_max": 0.0}, "permutation_max_sum_abs_r"),
                              ({"replay_max": replay.EXPECTED_AUDIT["observed_sum_abs_r"] + 1},
                               "replay-frozen-hash permutation control")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, field):
                    replay.validate_final(self.fixture(Path(directory), **kwargs), {"canonical_audit_sha256": "a"}, {}, {})

    def test_candidate_commitment_has_no_kat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "m").mkdir()
            (out / "crt").mkdir()
            values = {out / "m/t44-third-m-candidates-bound.json": {"status": "one_candidate"},
                      out / "crt/single_split_corrected_frozen.json": {"status": "frozen_before_heldout_and_KAT"},
                      out / "crt/high_gate.json": {"status": "exploratory_candidates"},
                      out / "crt/high_norm.json": {"status": "candidate_not_KAT_audited", "candidate": {"x": "1"}}}
            for path, value in values.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            commitment = replay.candidate_commitment(out, {"canonical_audit_sha256": "a"}, {}, {})
            self.assertNotIn("kat_sha256", commitment)
            self.assertNotIn(str(KAT), json.dumps(commitment))

    def test_existing_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "occupied"
            out.mkdir()
            result = subprocess.run(self.command(out, "--smoke"), cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)


if __name__ == "__main__":
    unittest.main()
