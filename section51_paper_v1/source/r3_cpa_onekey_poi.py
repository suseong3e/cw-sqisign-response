#!/usr/bin/env python3
"""Find one key's CRT subtraction stores from its target-only traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import r3_cpa_crt_analyze as crt
import r3_cpa_matrix_rank as rank


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / ".plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle"
BLOCK = 1024
MASK = np.uint64(0xffffffff)


def normalize(values: np.ndarray, axis: int) -> np.ndarray:
    values = values.astype(np.float32)
    values -= values.mean(axis=axis, keepdims=True)
    scale = np.sqrt(np.sum(values*values, axis=axis, keepdims=True))
    return np.divide(values, scale, out=np.zeros_like(values), where=scale > 1e-6)


def byte_labels(known: np.ndarray, borrow: np.ndarray) -> np.ndarray:
    candidate = np.arange(256, dtype=np.uint64)[:, None]
    stored = (candidate - known[None, :] - borrow[None, :]) & np.uint64(255)
    return normalize(crt.low.matrix_attack.limb.low.hw32(stored), axis=1)


def split_indices(rows: list[dict], public: dict, coord: str) -> tuple[np.ndarray, np.ndarray]:
    members = [i for i, row in enumerate(rows) if row["coord"] == coord and
               not row["heldout"] and public["split"][i] == 0]
    return tuple(np.array([i for i in members if
                           hashlib.sha256(rows[i]["challenge_hex"].encode()).digest()[0] & 1 == parity],
                          dtype=int) for parity in (0, 1))


def score_grid(model: np.ndarray, trace: np.ndarray) -> np.ndarray:
    return model @ normalize(trace, axis=0)


def discover_first(rows: list[dict], traces: np.ndarray, matrix_candidates: dict) -> tuple[dict, dict]:
    branches = []
    for m10 in matrix_candidates["m10_candidates"]:
        matrix = tuple(int(matrix_candidates[name], 16) for name in ("m00", "m01")) + (
            int(m10, 16), int(matrix_candidates["m11"], 16))
        public = crt.oracle_public(rows, matrix, ORACLE)
        groups = []
        for coord in ("x", "y"):
            train, valid = split_indices(rows, public, coord)
            if min(len(train), len(valid)) < 200:
                continue
            groups.append({"coord": coord, "train": train, "valid": valid,
                           "known": public[coord],
                           "model": byte_labels(public[coord][train, 0],
                                                np.zeros(len(train), dtype=np.uint64))})
        if len(groups) != 2:
            branches.append({"m10": m10, "status": "insufficient_split_zero_rows"})
            continue
        joint = np.zeros(traces.shape[1], dtype=np.float32)
        best = np.zeros((2, traces.shape[1]), dtype=np.uint8)
        for start in range(0, traces.shape[1], BLOCK):
            stop = min(start+BLOCK, traces.shape[1])
            for j, group in enumerate(groups):
                grid = score_grid(group["model"], np.asarray(traces[group["train"], start:stop]))
                chosen = np.argmax(np.abs(grid), axis=0)
                best[j, start:stop] = chosen.astype(np.uint8)
                joint[start:stop] += np.abs(grid[chosen, np.arange(stop-start)])
            if stop % 32768 == 0:
                print(f"subtraction first store {m10[-4:]} {stop}/{traces.shape[1]}", flush=True)
        peaks = []
        for offset in np.argsort(joint)[::-1]:
            sample = int(offset)
            if any(abs(sample - peak["sample"]) < 200 for peak in peaks):
                continue
            validation = []
            for j, group in enumerate(groups):
                candidate = int(best[j, sample])
                model = byte_labels(group["known"][group["valid"], 0],
                                    np.zeros(len(group["valid"]), dtype=np.uint64))
                scores = score_grid(model, np.asarray(traces[group["valid"], sample:sample+1]))[:, 0]
                validation.append({"coord": group["coord"], "candidate_byte": candidate,
                                   "abs_r": float(abs(scores[candidate])),
                                   "rank": 1 + int(np.count_nonzero(
                                       np.abs(scores) > abs(scores[candidate]) + 1e-6))})
            peaks.append({"sample": sample, "train_sum": float(joint[sample]),
                          "validation": validation})
            if len(peaks) == 16:
                break
        branches.append({"m10": m10, "status": "scored", "peaks": peaks,
                         "main_group_count": {g["coord"]: {"train": len(g["train"]),
                                                            "validation": len(g["valid"])} for g in groups}})
    applicable = [(branch, peak) for branch in branches if branch["status"] == "scored"
                  for peak in branch["peaks"] if all(item["rank"] == 1 and item["abs_r"] > 0.2
                                                    for item in peak["validation"])]
    if not applicable:
        raise RuntimeError("no split-zero subtraction first-store peak validates")
    branch, peak = max(applicable, key=lambda item: item[1]["train_sum"])
    report = {"branches": branches, "selected_m10": branch["m10"],
              "first_sample": peak["sample"], "first_validation": peak["validation"]}
    matrix = tuple(int(matrix_candidates[name], 16) for name in ("m00", "m01")) + (
        int(branch["m10"], 16), int(matrix_candidates["m11"], 16))
    return report, crt.oracle_public(rows, matrix, ORACLE)


def discover_second(rows: list[dict], traces: np.ndarray, public: dict,
                    first: int) -> tuple[int, dict]:
    left, right = first+8, min(first+512, traces.shape[1])
    if right-left < 100:
        raise RuntimeError("second subtraction store window is truncated")
    joint = np.zeros(right-left)
    groups = []
    helper = rank.helpers()
    for coord in ("x", "y"):
        train, valid = split_indices(rows, public, coord)
        base = (-public[coord][train, 0]) & MASK
        word0 = rank.rank_word(base, np.zeros(len(train), dtype=np.uint64),
                               np.asarray(traces[train, first], float), 64, False, helper)["top1"]
        borrow = (np.uint64(word0) < public[coord][train, 0]).astype(np.uint64)
        model = byte_labels(public[coord][train, 1], borrow)
        grid = score_grid(model, np.asarray(traces[train, left:right]))
        candidates = np.argmax(np.abs(grid), axis=0)
        joint += np.abs(grid[candidates, np.arange(right-left)])
        groups.append({"coord": coord, "word0": word0, "train": train, "valid": valid,
                       "candidate_by_sample": candidates})
    peaks = []
    for offset in np.argsort(joint)[::-1]:
        sample = left + int(offset)
        if any(abs(sample - peak["sample"]) < 8 for peak in peaks):
            continue
        validation = []
        for group in groups:
            coord, valid = group["coord"], group["valid"]
            borrow = (np.uint64(group["word0"]) < public[coord][valid, 0]).astype(np.uint64)
            model = byte_labels(public[coord][valid, 1], borrow)
            scores = score_grid(model, np.asarray(traces[valid, sample:sample+1]))[:, 0]
            candidate = int(group["candidate_by_sample"][offset])
            validation.append({"coord": coord, "candidate_byte": candidate,
                               "abs_r": float(abs(scores[candidate])),
                               "rank": 1 + int(np.count_nonzero(
                                   np.abs(scores) > abs(scores[candidate]) + 1e-6))})
        peaks.append({"sample": sample, "train_sum": float(joint[offset]),
                      "validation": validation})
        if len(peaks) == 12:
            break
    accepted = [peak for peak in peaks if all(item["rank"] == 1 and item["abs_r"] > 0.2
                                              for item in peak["validation"])]
    if not accepted:
        raise RuntimeError("no split-zero subtraction second-store peak validates")
    chosen = max(accepted, key=lambda peak: peak["train_sum"])
    return chosen["sample"], {"search_window": [left, right], "peaks": peaks,
                              "selected": chosen, "word0_by_coord": {group["coord"]: hex(group["word0"])
                                                                       for group in groups}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--m-candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["target_truth_loaded"] or manifest["known_key_trace_count"]:
        raise RuntimeError("POI discovery input is not target-only")
    trace_file = args.data / "traces.npy"
    schedule_file = args.data / "public_schedule.jsonl"
    if crt.digest(trace_file) != manifest["target_trace_sha256"] or crt.digest(schedule_file) != manifest["target_public_schedule_sha256"]:
        raise RuntimeError("POI discovery input hash mismatch")
    traces = np.load(trace_file, mmap_mode="r", allow_pickle=False)
    rows = [json.loads(line) for line in schedule_file.read_text().splitlines()]
    if traces.shape != (2112, 163840) or len(rows) != len(traces):
        raise RuntimeError("unexpected one-key CRT capture shape")
    matrix = json.loads(args.m_candidates.read_text())["matrix"]
    first_report, public = discover_first(rows, traces, matrix)
    second, second_report = discover_second(rows, traces, public, first_report["first_sample"])
    points = [first_report["first_sample"]] + [second+13*j for j in range(5)]
    report = {"schema": "r3-onekey-split-zero-subtraction-poi-v1", "status": "selected",
              "m10": first_report["selected_m10"], "low_points": points,
              "first": first_report, "second": second_report,
              "target_trace_sha256": manifest["target_trace_sha256"],
              "target_public_schedule_sha256": manifest["target_public_schedule_sha256"],
              "m_candidates_sha256": crt.digest(args.m_candidates),
              "script_sha256": crt.digest(Path(__file__)),
              "evidence_contract": {"evidence_origin": "measured",
                                    "evidence_purpose": "diagnostic",
                                    "blinding": "unblinded",
                                    "measurement_scope": "isolated_primitive",
                                    "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                                    "target_truth_loaded": False,
                                    "known_key_traces_loaded": False,
                                    "claim_not_allowed": ["blind_success_rate", "scheme_break"]}}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"m10": report["m10"], "low_points": points}))


if __name__ == "__main__":
    main()
