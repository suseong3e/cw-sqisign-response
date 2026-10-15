#!/usr/bin/env python3
"""Small offline checks for the Section 5.1 snapshot integrity boundary."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from section51_reproduce import EXPECTED_METRICS, SCHEMA, reproduce


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(root: Path, *, secret: bool = False, observed: float | None = None,
                 bind_audit: bool = True, purpose: str = "diagnostic",
                 full_replay: bool = False) -> None:
    audit = {
        "status": "validated", **EXPECTED_METRICS,
        "evidence_contract": {"evidence_origin": "measured", "evidence_purpose": purpose, "blinding": "unblinded", "measurement_scope": "isolated_primitive", "claim_scope": "r3_single_split:one_key_coordinate_candidate", "target_truth_loaded": False},
    }
    if observed is not None:
        audit["observed_sum_abs_r"] = observed
    if secret:
        audit["sk"] = "not-permitted"
    artifact = root / "heldout_audit.json"
    artifact.write_text(json.dumps(audit), encoding="utf-8")
    files = {artifact.name: digest(artifact)} if bind_audit else {"other.txt": digest(root / "other.txt")}
    manifest = {"schema": SCHEMA, "files": files,
                "experiment": {"candidate_frozen_before_heldout": True}, "audit": {"report": artifact.name}}
    if full_replay:
        manifest["full_replay"] = {}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (root / "manifest.sha256").write_text(digest(manifest_path), encoding="ascii")


def expect_failure(root: Path) -> None:
    try:
        reproduce(root, root / "report.json")
    except ValueError:
        return
    raise AssertionError("expected integrity failure")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root)
        report = reproduce(root, root / "report.json")
        assert report["status"] == "verified_snapshot"
        assert report["metrics"] == EXPECTED_METRICS
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root, secret=True)
        expect_failure(root)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root, observed=EXPECTED_METRICS["observed_sum_abs_r"] + 1e-4)
        expect_failure(root)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root)
        (root / "heldout_audit.json").write_text("{}", encoding="utf-8")
        expect_failure(root)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "other.txt").write_text("", encoding="utf-8")
        write_bundle(root, bind_audit=False)
        expect_failure(root)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root, purpose="claim_bearing")
        expect_failure(root)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_bundle(root, full_replay=True)
        expect_failure(root)


if __name__ == "__main__":
    main()
