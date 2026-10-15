#!/usr/bin/env python3
"""Verify the integrity of a precomputed Section 5.1 held-out snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "round3-cpa-section51-snapshot-bundle-v1"
REPORT_SCHEMA = "round3-cpa-section51-snapshot-report-v1"
FORBIDDEN_FIELDS = frozenset({"a1", "a2", "m1", "m2", "expected", "split_v", "signature_hex", "sk"})
EXPECTED_METRICS = {
    "event_count": 136,
    "heldout_counts": {"x": 17, "y": 17},
    "permutation_count": 99,
    "observed_sum_abs_r": 116.80631243236115,
    "wrong_window_sum_abs_r": 25.061060481125295,
    "permutation_max_sum_abs_r": 80.89532561549119,
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def bundle_path(bundle: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        fail("bundle path must be a nonempty relative string")
    candidate = (bundle / name).resolve()
    try:
        candidate.relative_to(bundle)
    except ValueError as error:
        raise ValueError("bundle path escapes the bundle") from error
    return candidate


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error


def reject_secrets(value: Any, source: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_FIELDS:
                fail(f"secret schedule field in {source}: {key}")
            reject_secrets(nested, source)
    elif isinstance(value, list):
        for nested in value:
            reject_secrets(nested, source)


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path, digest_path = bundle / "manifest.json", bundle / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        fail("bundle requires manifest.json and manifest.sha256")
    expected_digest = digest_path.read_text(encoding="ascii").strip().lower()
    if len(expected_digest) != 64 or any(char not in "0123456789abcdef" for char in expected_digest):
        fail("manifest.sha256 must contain one SHA-256 digest")
    if sha256(manifest_path) != expected_digest:
        fail("manifest hash differs")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        fail("unsupported bundle manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("manifest files must be a nonempty object")
    for name, expected in files.items():
        path = bundle_path(bundle, name)
        if path == manifest_path or path == digest_path or not path.is_file():
            fail("manifest files must name existing non-manifest files")
        if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected.lower():
            fail(f"artifact hash differs: {name}")
    for path in bundle.rglob("*.json"):
        reject_secrets(load_json(path), path.relative_to(bundle).as_posix())
    for path in bundle.rglob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    reject_secrets(json.loads(line), path.relative_to(bundle).as_posix())
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL artifact: {path.relative_to(bundle)}") from error
    return manifest


def require_hashed_inputs(manifest: dict[str, Any], names: list[object]) -> None:
    files = manifest["files"]
    for name in names:
        if not isinstance(name, str) or name not in files:
            fail(f"replay input is not hash-bound by manifest: {name}")


def verify_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    source_contract = audit.get("evidence_contract")
    required_contract = {
        "evidence_origin": "measured",
        "evidence_purpose": "diagnostic",
        "blinding": "unblinded",
        "measurement_scope": "isolated_primitive",
        "claim_scope": "r3_single_split:one_key_coordinate_candidate",
        "target_truth_loaded": False,
    }
    if audit.get("status") != "validated" or not isinstance(source_contract, dict):
        fail("heldout audit is not a validated truth-free result")
    for key, expected in required_contract.items():
        if source_contract.get(key) != expected:
            fail(f"heldout evidence contract differs: {key}")
    for key, expected in EXPECTED_METRICS.items():
        actual = audit.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
                fail(f"heldout metric differs: {key}")
        elif actual != expected:
            fail(f"heldout metric differs: {key}")
    return {key: audit[key] for key in EXPECTED_METRICS}


def reproduce(bundle: Path, out: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        fail("bundle must be an existing directory")
    if out.exists():
        raise FileExistsError(out)
    manifest = verify_manifest(bundle)
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("candidate_frozen_before_heldout") is not True:
        fail("manifest must attest candidate_frozen_before_heldout")
    audit_spec = manifest.get("audit")
    if not isinstance(audit_spec, dict) or "report" not in audit_spec:
        fail("manifest audit must name a report")
    require_hashed_inputs(manifest, [audit_spec["report"]])
    if "full_replay" in manifest:
        fail("full_replay is not supported by the snapshot verifier")
    audit_path = bundle_path(bundle, audit_spec["report"])
    audit = load_json(audit_path)
    metrics = verify_metrics(audit)
    source_contract = audit["evidence_contract"]
    figure = {"status": "not_generated", "reason": "bundle does not declare Figure 2 artifacts"}
    figures = manifest.get("figures")
    if isinstance(figures, dict):
        figure_names = [figures.get(name) for name in ("figure_2_pdf", "figure_2_png", "renderer")]
        require_hashed_inputs(manifest, figure_names)
        figure = {"status": "verified_precomputed", "files": figure_names}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "verified_snapshot",
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "source_audit_sha256": sha256(audit_path),
        "metrics": metrics,
        "evidence_contract": source_contract,
        "verification_scope": "precomputed_snapshot_integrity",
        "figure_2": figure,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": reproduce(args.bundle, args.out)["status"]}))


if __name__ == "__main__":
    main()
