#!/usr/bin/env python3
"""Audit frozen split-zero CRT coordinates before opening the private KAT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import r3_cpa_crt_analyze as crt
import r3_cpa_onekey_mul as mul
from r3_cpa_onekey_crt_prepare import fresh_key


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / ".plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def wrong_sample(point: int, sample_count: int) -> int:
    other = point + 30000 if point + 30000 < sample_count else point - 30000
    if other < 0:
        raise RuntimeError("wrong-window point is outside capture")
    return other


def prepare_events(frozen: dict, high: dict, matrix_candidates: dict,
                   rows: list[dict], traces: np.ndarray) -> tuple[list[dict], dict]:
    selected = frozen["frozen"][high["frozen_rank"]-1]
    m = matrix_candidates["matrix"]
    matrix = tuple(int(m[name], 16) for name in ("m00", "m01")) + (
        int(selected["m10"], 16), int(m["m11"], 16))
    public = crt.oracle_public(rows, matrix, ORACLE)
    n = int(high["candidate"]["N"], 16)
    x = int(high["candidate"]["x"], 16)
    y = int(high["candidate"]["y"], 16)
    u = int(high["candidate"]["u"], 16)
    if not (0 <= x < n and 0 <= y < n and pow(1 << 192, -1, n) == u):
        raise RuntimeError("frozen CRT candidate is not canonical")
    events = []
    summaries = {}
    for coord, secret in (("x", x), ("y", y)):
        ix = np.array([i for i, row in enumerate(rows) if row["heldout"] and
                       row["coord"] == coord and public["split"][i] == 0], dtype=int)
        if len(ix) < 8:
            raise RuntimeError(f"{coord} split-zero heldout is too small")
        known = public[coord][ix]
        low, _ = crt.low.subtraction_leakage(secret, known)
        diff, _ = crt.high.difference_words(secret, known)
        stores = mul.simulate_product(diff, u)
        inverse_points = (selected.get("multiply_points") or frozen["points"]["multiply"])
        word_count = selected.get("u_word_count", len(inverse_points[0]))
        if len(inverse_points) != 6 or any(len(row) != word_count for row in inverse_points):
            raise RuntimeError("inverse source-order point count differs")
        group_events = []
        for limb, point in enumerate(frozen["points"]["low"]):
            group_events.append(("low", limb, point, low[limb]))
        for row in range(6):
            for limb, point in enumerate(inverse_points[row]):
                group_events.append(("inverse", row*word_count+limb, point, stores[row, limb]))
        for item in high["groups"][coord]:
            if item.get("status") == "source_row_absent":
                continue
            for limb in range(7):
                point = item["start"] + 13*limb
                group_events.append(("high", item["row"]*7+limb, point,
                                     stores[item["row"], limb]))
        label = normalize(np.array([entry[3] for entry in group_events]))
        positive = normalize(np.array([traces[ix, entry[2]] for entry in group_events]))
        wrong = normalize(np.array([traces[ix, wrong_sample(entry[2], traces.shape[1])]
                                    for entry in group_events]))
        observed = np.abs(np.sum(label*positive, axis=1))
        negative = np.abs(np.sum(label*wrong, axis=1))
        for j, (stage, event, point, _) in enumerate(group_events):
            events.append({"coord": coord, "stage": stage, "event": event,
                           "sample": point, "abs_r": float(observed[j]),
                           "wrong_window_abs_r": float(negative[j])})
        summaries[coord] = {"trace_count": len(ix), "label": label, "positive": positive,
                            "capture_blocks": (ix//2)//4,
                            "wrong": wrong, "observed": float(observed.sum()),
                            "wrong_score": float(negative.sum())}
    return events, summaries


def heldout_audit(frozen_path: Path, gate_path: Path, high_path: Path,
                  m_path: Path, data: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    frozen = json.loads(frozen_path.read_text())
    gate = json.loads(gate_path.read_text())
    high = json.loads(high_path.read_text())
    if (frozen["status"] != "frozen_before_heldout_and_KAT" or
            gate["frozen_sha256"] != digest(frozen_path) or
            high["frozen_sha256"] != digest(frozen_path) or
            high["gate_sha256"] != digest(gate_path)):
        raise RuntimeError("candidate/POI freeze chain differs")
    manifest = json.loads((data / "manifest.json").read_text())
    if (manifest["target_truth_loaded"] or manifest["known_key_trace_count"] or
            manifest["target_trace_sha256"] != frozen["target_trace_sha256"] or
            manifest["target_public_schedule_sha256"] != frozen["target_public_schedule_sha256"] or
            digest(data / "traces.npy") != manifest["target_trace_sha256"] or
            digest(data / "public_schedule.jsonl") != manifest["target_public_schedule_sha256"]):
        raise RuntimeError("heldout input is not the frozen target-only capture")
    rows = [json.loads(line) for line in (data / "public_schedule.jsonl").read_text().splitlines()]
    traces = np.load(data / "traces.npy", mmap_mode="r", allow_pickle=False)
    matrix = json.loads(m_path.read_text())
    events, groups = prepare_events(frozen, high, matrix, rows, traces)
    observed = sum(group["observed"] for group in groups.values())
    wrong = sum(group["wrong_score"] for group in groups.values())
    rng = np.random.default_rng(int(digest(frozen_path)[:16], 16))
    permutation_scores = []
    for _ in range(99):
        score = 0.0
        for group in groups.values():
            shuffled = np.arange(group["label"].shape[1])
            for block in np.unique(group["capture_blocks"]):
                members = np.flatnonzero(group["capture_blocks"] == block)
                shuffled[members] = rng.permutation(members)
            score += float(np.abs(np.sum(group["label"][:, shuffled]*group["positive"], axis=1)).sum())
        permutation_scores.append(score)
    stage_median = {stage: float(np.median([event["abs_r"] for event in events
                                            if event["stage"] == stage]))
                    for stage in ("low", "inverse", "high")}
    passed = (observed > wrong and observed > max(permutation_scores) and
              all(value > 0.2 for value in stage_median.values()))
    report = {"schema": "r3-onekey-split-zero-heldout-audit-v1",
              "status": "validated" if passed else "blocked_negative_control",
              "frozen_sha256": digest(frozen_path), "gate_sha256": digest(gate_path),
              "high_sha256": digest(high_path), "m_candidates_sha256": digest(m_path),
              "input_manifest_sha256": digest(data / "manifest.json"),
              "target_trace_sha256": manifest["target_trace_sha256"],
              "target_public_schedule_sha256": manifest["target_public_schedule_sha256"],
              "script_sha256": digest(Path(__file__)),
              "event_count": len(events), "observed_sum_abs_r": observed,
              "wrong_window_sum_abs_r": wrong,
              "permutation_max_sum_abs_r": max(permutation_scores),
              "permutation_count": len(permutation_scores),
              "permutation_rule": "within_each_coordinate_and_four_challenge_capture_block",
              "stage_median_abs_r": stage_median,
              "heldout_counts": {coord: group["trace_count"] for coord, group in groups.items()},
              "events": events,
              "evidence_contract": {"evidence_origin": "measured",
                                    "evidence_purpose": "diagnostic",
                                    "blinding": "unblinded",
                                    "measurement_scope": "isolated_primitive",
                                    "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                                    "target_truth_loaded": False,
                                    "claim_not_allowed": ["blind_success_rate", "scheme_break"]}}
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    out.chmod(0o600)
    return report


def kat_audit(frozen_path: Path, high_path: Path, heldout_path: Path,
              m_path: Path, kat: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    heldout = json.loads(heldout_path.read_text())
    if heldout["frozen_sha256"] != digest(frozen_path) or heldout["high_sha256"] != digest(high_path):
        raise RuntimeError("KAT audit inputs differ from frozen candidate")
    high = json.loads(high_path.read_text())
    candidate = high["candidate"]
    matrix_truth, (n, x, y) = fresh_key(kat)
    matrix = json.loads(m_path.read_text())["matrix"]
    selected_m10 = json.loads(frozen_path.read_text())["frozen"][high["frozen_rank"]-1]["m10"]
    recovered_m = tuple(int(matrix[name], 16) for name in ("m00", "m01")) + (
        int(selected_m10, 16), int(matrix["m11"], 16))
    field_matches = {"M": recovered_m == matrix_truth,
                     "N": int(candidate["N"], 16) == n,
                     "x": int(candidate["x"], 16) == x,
                     "y": int(candidate["y"], 16) == y,
                     "u": int(candidate["u"], 16) == pow(1 << 192, -1, n)}
    status = ("diagnostic_exact_match" if all(field_matches.values()) and
              heldout["status"] == "validated" else "diagnostic_mismatch")
    report = {"schema": "r3-onekey-split-zero-kat-audit-v1", "status": status,
              "field_matches": field_matches,
              "frozen_sha256": digest(frozen_path), "high_sha256": digest(high_path),
              "heldout_sha256": digest(heldout_path), "kat_sha256": digest(kat),
              "script_sha256": digest(Path(__file__)),
              "evidence_contract": {"evidence_origin": "measured",
                                    "evidence_purpose": "diagnostic",
                                    "blinding": "unblinded",
                                    "measurement_scope": "isolated_primitive",
                                    "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                                    "target_truth_loaded_in_this_step": True,
                                    "claim_not_allowed": ["blind_success_rate", "scheme_break"]}}
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    out.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--m-candidates", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--heldout-out", type=Path, required=True)
    parser.add_argument("--kat-out", type=Path, required=True)
    args = parser.parse_args()
    heldout = heldout_audit(args.frozen, args.gate, args.high, args.m_candidates,
                            args.data, args.heldout_out)
    kat = kat_audit(args.frozen, args.high, args.heldout_out,
                    args.m_candidates, args.kat, args.kat_out)
    print(json.dumps({"heldout_status": heldout["status"], "kat_status": kat["status"]}))


if __name__ == "__main__":
    main()
