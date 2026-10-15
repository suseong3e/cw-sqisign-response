#!/usr/bin/env python3
"""Run the published Section 5.1 analysis DAG in a fresh directory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "section51_paper_v1" / "source"
HASHES = SOURCE / "source.sha256"
REQUIRED_SOURCE_HASHES = {
    "r3_cpa_onekey_poi.py": "bce628674dc59e300c68889928430277ba191f957484edcf71caa86dfa435d6c",
    "r3_cpa_onekey_single_split.py": "190c396eaf78bf0dd3112448a1bbcb916b19f82a6eebfa2049892ca422ef3eb8",
    "r3_cpa_onekey_high_gate.py": "0e84866187207a01d5ecb6df4a29bae15ba25e63052b711ccdd95ed80a5989da",
    "r3_cpa_onekey_high_norm.py": "5a28a428bab0dccab446a83d77d02beb81cd17b4d210a8b5ae7858ca3b7e6846",
    "r3_cpa_onekey_single_split_audit.py": "a6e5e78ec0de101861a5ac6970ee47ea3a546903c970fd2925b1b00143fa1e81",
}
T48_SOURCE = "m_canonical/t48-third-low-u-refreeze.py"
T48_SHA256 = "dcf30641e8839465ae86686c8a2b4fe2e5b425ed38fd6e4ae0a42142e9c109e6"
CANONICAL_AUDIT = "canonical/r3_cpa_onekey_single_split_audit.py"
HISTORICAL_FROZEN_SHA256 = "a31be289a898d7659a09ff2f81de7ab007818b63e7f3538417a02e6bb8cb093e"
PAPER_V1_PERMUTATION_SEED = int(HISTORICAL_FROZEN_SHA256[:16], 16)
EXPECTED_AUDIT = {
    "event_count": 136,
    "heldout_counts": {"x": 17, "y": 17},
    "permutation_count": 99,
    "observed_sum_abs_r": 116.80631243236115,
    "wrong_window_sum_abs_r": 25.061060481125295,
    "permutation_max_sum_abs_r": 80.89532561549119,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def runtime_binding() -> dict[str, str]:
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name)
            for name in ("numpy", "numba", "sympy", "matplotlib")}}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON object required: {path.name}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    os.replace(temp, path)


def source_binding() -> dict[str, str]:
    if not HASHES.is_file():
        fail("source snapshot hash list is missing")
    listed: dict[str, str] = {}
    for line in HASHES.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix("section51_paper_v1/source/")
        listed[name] = expected
        path = SOURCE / name
        if not path.is_file() or digest(path) != expected:
            fail(f"source hash differs: {name}")
    for name, expected in REQUIRED_SOURCE_HASHES.items():
        if listed.get(name) != expected:
            fail(f"recovered historical source hash differs: {name}")
    if listed.get(T48_SOURCE) != T48_SHA256:
        fail("t48 preserved source hash differs")
    return {"source_hashes_sha256": digest(HASHES), "source_snapshot_file_count": str(len(listed)),
            "canonical_audit_sha256": digest(SOURCE / CANONICAL_AUDIT)}


def data_binding(data: Path, expected_shape: tuple[int, int]) -> dict[str, str]:
    manifest = data / "manifest.json"
    traces, schedule = data / "traces.npy", data / "public_schedule.jsonl"
    if not manifest.is_file() or not traces.is_file() or not schedule.is_file():
        fail(f"missing target-only input under {data}")
    value = read_json(manifest)
    if value.get("target_truth_loaded") or value.get("known_key_trace_count"):
        fail("analysis input is not target-only")
    trace_hash, schedule_hash = digest(traces), digest(schedule)
    if trace_hash != value.get("target_trace_sha256"):
        fail("trace hash differs from manifest")
    if schedule_hash != value.get("target_public_schedule_sha256"):
        fail("public schedule hash differs from manifest")
    import numpy as np
    if np.load(traces, mmap_mode="r", allow_pickle=False).shape != expected_shape:
        fail(f"trace shape differs: expected {expected_shape}")
    return {"manifest_sha256": digest(manifest), "trace_sha256": trace_hash,
            "public_schedule_sha256": schedule_hash}


def commands(m_data: Path, crt_data: Path, kat: Path, out: Path) -> list[tuple[str, list[str]]]:
    m, crt = out / "m", out / "crt"
    return [
        ("m-product", [str(SOURCE / "m_canonical/t05-m-product-groups.py")]),
        ("m-add", [str(SOURCE / "m_canonical/t06-m-add-anchors.py")]),
        ("m00", [str(SOURCE / "m_canonical/t07-m00-groups.py")]),
        ("m10", [str(SOURCE / "m_canonical/t08-m10-groups.py")]),
        ("m-branch", [str(SOURCE / "m_canonical/t40-third-m-full-product-branch.py")]),
        ("m-screen", [str(SOURCE / "m_canonical/t41-third-m-branch-screen.py")]),
        ("m-freeze", [str(SOURCE / "m_canonical/t44-third-m-freeze-bound.py")]),
        ("poi", [str(SOURCE / "r3_cpa_onekey_poi.py"), "--data", str(crt_data), "--m-candidates", str(m / "t44-third-m-candidates-bound.json"), "--out", str(crt / "subtraction_poi.json")]),
        ("single-split", [str(SOURCE / "r3_cpa_onekey_single_split.py"), "--data", str(crt_data), "--m-candidates", str(m / "t44-third-m-candidates-bound.json"), "--subtraction-poi", str(crt / "subtraction_poi.json"), "--frozen", str(crt / "single_split_frozen.json")]),
        ("t48-refreeze", [str(SOURCE / "m_canonical/t48-third-low-u-refreeze.py")]),
        ("high-gate", [str(SOURCE / "r3_cpa_onekey_high_gate.py"), "--data", str(crt_data), "--m-candidates", str(m / "t44-third-m-candidates-bound.json"), "--frozen", str(crt / "single_split_corrected_frozen.json"), "--out", str(crt / "high_gate.json")]),
        ("high-norm", [str(SOURCE / "r3_cpa_onekey_high_norm.py"), "--data", str(crt_data), "--m-candidates", str(m / "t44-third-m-candidates-bound.json"), "--frozen", str(crt / "single_split_corrected_frozen.json"), "--gate", str(crt / "high_gate.json"), "--out", str(crt / "high_norm.json")]),
        ("heldout-and-posthoc-kat", [str(SOURCE / CANONICAL_AUDIT), "--data", str(crt_data), "--m-candidates", str(m / "t44-third-m-candidates-bound.json"), "--frozen", str(crt / "single_split_corrected_frozen.json"), "--gate", str(crt / "high_gate.json"), "--high", str(crt / "high_norm.json"), "--kat", str(kat), "--heldout-out", str(crt / "heldout_audit.json"), "--kat-out", str(crt / "posthoc_kat.json")]),
        ("figure-2", [str(SOURCE / "render_figure.py")]),
    ]


def candidate_commitment(out: Path, source: dict[str, str], m_input: dict[str, str],
                         crt_input: dict[str, str]) -> dict:
    m, crt = out / "m", out / "crt"
    paths = {"m_freeze": m / "t44-third-m-candidates-bound.json",
             "corrected_freeze": crt / "single_split_corrected_frozen.json",
             "high_gate": crt / "high_gate.json", "high_norm": crt / "high_norm.json"}
    values = {name: read_json(path) for name, path in paths.items()}
    required = {"m_freeze": "one_candidate", "corrected_freeze": "frozen_before_heldout_and_KAT",
                "high_gate": "exploratory_candidates", "high_norm": "candidate_not_KAT_audited"}
    for name, status in required.items():
        if values[name].get("status") != status:
            fail(f"candidate stage status differs: {name}")
    return {"schema": "section51-paper-v1-candidate-commitment-v1",
            "stage": "before_posthoc_kat", "source": source, "runtime": runtime_binding(),
            "inputs": {"m": m_input, "crt": crt_input},
            "config": {"m_freeze_rng_seed": 2026100701,
                       "m_freeze_config_sha256": digest(SOURCE / "m_canonical/t44-third-m-freeze-bound.py"),
                       "canonical_audit_sha256": source["canonical_audit_sha256"],
                       "historical_frozen_sha256": HISTORICAL_FROZEN_SHA256,
                       "paper_v1_permutation_seed": PAPER_V1_PERMUTATION_SEED,
                       "paper_v1_permutation_seed_rule": "historical frozen SHA-256 prefix"},
            "dependencies": {name: digest(path) for name, path in paths.items()},
            "candidate": values["high_norm"]["candidate"]}


def validate_final(out: Path, source: dict[str, str], m_input: dict[str, str],
                   crt_input: dict[str, str]) -> dict:
    m, crt = out / "m", out / "crt"
    commitment = out / "candidate_commitment.json"
    paths = {"candidate_commitment": commitment, "m_freeze": m / "t44-third-m-candidates-bound.json",
             "corrected_freeze": crt / "single_split_corrected_frozen.json", "high_gate": crt / "high_gate.json",
             "high_norm": crt / "high_norm.json", "heldout": crt / "heldout_audit.json",
             "posthoc_kat": crt / "posthoc_kat.json", "figure_pdf": out / "figure_2.pdf", "figure_png": out / "figure_2.png"}
    if not paths["figure_pdf"].is_file() or not paths["figure_png"].is_file():
        fail("Figure 2 PDF/PNG is missing")
    stages = {"candidate_commitment": "before_posthoc_kat", "m_freeze": "one_candidate",
              "corrected_freeze": "frozen_before_heldout_and_KAT", "high_gate": "exploratory_candidates",
              "high_norm": "candidate_not_KAT_audited"}
    stage_values = {name: read_json(path) for name, path in paths.items() if name in stages}
    for name, status in stages.items():
        if stage_values[name].get("stage" if name == "candidate_commitment" else "status") != status:
            fail(f"final stage status differs: {name}")
    config = stage_values["candidate_commitment"].get("config", {})
    if (config.get("canonical_audit_sha256") != source.get("canonical_audit_sha256") or
            config.get("historical_frozen_sha256") != HISTORICAL_FROZEN_SHA256 or
            config.get("paper_v1_permutation_seed") != PAPER_V1_PERMUTATION_SEED):
        fail("candidate commitment control binding differs")
    audit, kat = read_json(paths["heldout"]), read_json(paths["posthoc_kat"])
    if audit.get("status") != "validated" or audit.get("script_sha256") != source.get("canonical_audit_sha256"):
        fail("held-out audit is not validated or canonical")
    expected = dict(EXPECTED_AUDIT)
    for name, value in expected.items():
        actual = audit.get(name)
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-9):
                fail(f"held-out metric differs: {name}")
        elif actual != value:
            fail(f"held-out metric differs: {name}")
    replay_permutation_max = audit.get("replay_frozen_hash_permutation_max_sum_abs_r")
    if (not isinstance(replay_permutation_max, (int, float)) or
            not math.isfinite(replay_permutation_max) or replay_permutation_max < 0 or
            replay_permutation_max >= audit["observed_sum_abs_r"]):
        fail("replay-frozen-hash permutation control does not pass")
    expected["replay_frozen_hash_permutation_max_sum_abs_r"] = replay_permutation_max
    replay_frozen = digest(paths["corrected_freeze"])
    if (audit.get("historical_frozen_sha256") != HISTORICAL_FROZEN_SHA256 or
            audit.get("replay_frozen_sha256") != replay_frozen or audit.get("frozen_sha256") != replay_frozen or
            audit.get("permutation_seed") != PAPER_V1_PERMUTATION_SEED or
            audit.get("replay_frozen_hash_permutation_seed") != int(replay_frozen[:16], 16) or
            audit.get("permutation_seed_rule") != "paper-v1 fixed historical frozen SHA-256 prefix"):
        fail("permutation seed binding differs")
    if kat.get("status") != "diagnostic_exact_match" or kat.get("field_matches") != {key: True for key in ("M", "N", "x", "y", "u")}:
        fail("posthoc KAT does not exactly match M/N/x/y/u")
    result = {"schema": "section51-paper-v1-replay-result-v1", "status": "accepted",
              "source": source, "runtime": runtime_binding(), "inputs": {"m": m_input, "crt": crt_input},
              "dependencies": {name: digest(path) for name, path in paths.items() if name not in ("figure_pdf", "figure_png")},
              "outputs": {name: digest(path) for name, path in paths.items() if name in ("figure_pdf", "figure_png")},
              "validation": {"heldout_status": audit["status"], "metrics": expected,
                             "historical_frozen_sha256": HISTORICAL_FROZEN_SHA256,
                             "replay_frozen_sha256": replay_frozen,
                             "paper_v1_permutation_seed": PAPER_V1_PERMUTATION_SEED,
                             "posthoc_kat_status": kat["status"], "field_matches": kat["field_matches"]},
              "kat_sha256": kat["kat_sha256"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-data", type=Path, required=True)
    parser.add_argument("--crt-data", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="validate DAG and isolation without calculating results")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = source_binding()
    m_input, crt_input = data_binding(args.m_data, (1024, 45056)), data_binding(args.crt_data, (2112, 163840))
    if not args.kat.is_file():
        fail("posthoc KAT is missing")
    env = os.environ | {"SECTION51_SOURCE_ROOT": str(SOURCE), "SECTION51_M_DATA": str(args.m_data.resolve()), "SECTION51_CRT_DATA": str(args.crt_data.resolve()), "SECTION51_M_OUT": str((args.out / "m").resolve()), "SECTION51_CRT_OUT": str((args.out / "crt").resolve()), "SECTION51_FIGURE_OUT": str((args.out / "figure_2").resolve())}
    steps = commands(args.m_data, args.crt_data, args.kat, args.out)
    for name, argv in steps:
        print(json.dumps({"step": name, "command": [sys.executable, *argv]}))
    if args.dry_run or args.smoke:
        return
    args.out.mkdir()
    (args.out / "m").mkdir()
    (args.out / "crt").mkdir()
    log = args.out / "commands.jsonl"
    with log.open("x", encoding="utf-8") as stream:
        for name, argv in steps:
            result = subprocess.run([sys.executable, *argv], cwd=SOURCE, env=env, text=True, capture_output=True)
            stream.write(json.dumps({"step": name, "command": [sys.executable, *argv], "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}) + "\n")
            stream.flush()
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {log}")
            if name == "high-norm":
                atomic_json(args.out / "candidate_commitment.json", candidate_commitment(args.out, source, m_input, crt_input))
    atomic_json(args.out / "replay_result.json", validate_final(args.out, source, m_input, crt_input))


if __name__ == "__main__":
    main()
