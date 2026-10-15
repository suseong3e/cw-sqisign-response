#!/usr/bin/env python3
"""Find row-six CRT stores from frozen single-key low/u candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / ".plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle"
import r3_cpa_crt_analyze as crt
import r3_cpa_matrix_rank as rank
import r3_cpa_onekey_mul as mul


def correlations(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    predictions = predictions.astype(np.float64)
    predictions -= predictions.mean(axis=1, keepdims=True)
    observed = observed.astype(np.float64) - float(np.mean(observed))
    denominator = np.linalg.norm(predictions, axis=1) * np.linalg.norm(observed)
    return np.divide(predictions @ observed, denominator,
                     out=np.zeros(len(predictions)), where=denominator > 0)


def first_byte(state: np.ndarray, borrow: np.ndarray, u0: np.uint64) -> np.ndarray:
    candidate = np.arange(256, dtype=np.uint64)[:, None]
    mask = np.uint64(255)
    stored = ((((candidate - borrow[None, :]) & mask) * (u0 & mask))
              + (state[None, :, 6] & mask)) & mask
    return crt.low.matrix_attack.limb.low.hw32(stored)


def indices(rows: list[dict], public: dict, coord: str) -> tuple[np.ndarray, np.ndarray]:
    members = [i for i, row in enumerate(rows) if row["coord"] == coord and
               not row["heldout"] and public["split"][i] == 0]
    return tuple(np.array([i for i in members if
                           hashlib.sha256(rows[i]["challenge_hex"].encode()).digest()[0] & 1 == parity],
                          dtype=int) for parity in (0, 1))


def normalized(values: np.ndarray, axis: int) -> np.ndarray:
    values = values.astype(np.float32)
    values -= values.mean(axis=axis, keepdims=True)
    scale = np.sqrt(np.sum(values*values, axis=axis, keepdims=True))
    return np.divide(values, scale, out=np.zeros_like(values), where=scale > 1e-6)


def search(candidate: dict, public: dict, rows: list[dict], traces: np.ndarray,
           left: int, right: int) -> list[dict]:
    u_words = mul.words(candidate["u"])
    groups = []
    for coord in ("x", "y"):
        train, _ = indices(rows, public, coord)
        state, borrow = mul.initial_state(candidate[coord], public[coord][train], u_words)
        groups.append((train, normalized(first_byte(state, borrow, u_words[0]), axis=1)))
    joint = np.zeros(right-left, dtype=np.float32)
    for start in range(left, right, 512):
        stop = min(start+512, right)
        for train, model in groups:
            grid = model @ normalized(np.asarray(traces[train, start:stop]), axis=0)
            joint[start-left:stop-left] += np.max(np.abs(grid), axis=0)
    peaks = []
    for offset in np.argsort(joint)[::-1]:
        sample = left + int(offset)
        if any(abs(sample-item["sample"]) < 100 for item in peaks):
            continue
        details = inspect(candidate, public, rows, traces, sample)
        peaks.append({"sample": sample, "train_sum": float(joint[offset]),
                      "validation_sum": sum(v["validation_abs_r"] for v in details.values()),
                      "validation_min": min(v["validation_abs_r"] for v in details.values()),
                      "wrong_window_sum": sum(v["wrong_window_abs_r"] for v in details.values()),
                      "groups": details})
        if len(peaks) == 8:
            break
    return peaks


def seven_store(candidate: dict, public: dict, rows: list[dict],
                traces: np.ndarray, sample: int, helper: object) -> dict:
    u_words = mul.words(candidate["u"])
    results = {}
    points = [sample+13*j for j in range(7)]
    for coord in ("x", "y"):
        train, valid = indices(rows, public, coord)
        states = {"train": mul.initial_state(candidate[coord], public[coord][train], u_words),
                  "validation": mul.initial_state(candidate[coord], public[coord][valid], u_words)}
        waves = {"train": np.asarray(traces[np.ix_(train, points)], float),
                 "validation": np.asarray(traces[np.ix_(valid, points)], float)}
        state, borrow = states["train"]
        base = (state[:, 6] - borrow*u_words[0]) & np.uint64(0xffffffff)
        multiplier = np.full(len(train), u_words[0], dtype=np.uint64)
        ranked = rank.rank_word(base, multiplier, waves["train"][:, 0], 64, True, helper)
        candidates = []
        for word in ranked["final_beam"]:
            scores = {}
            for part in ("train", "validation"):
                current, debt = states[part]
                _, predicted = mul.update_row(current, 6,
                    (np.uint64(word)-debt)&np.uint64(0xffffffff), u_words)
                scores[part] = [float(abs(correlations(predicted[j:j+1], waves[part][:, j])[0]))
                                for j in range(7)]
            candidates.append({"word": hex(word),
                               "train_sum": sum(scores["train"]),
                               "validation_sum": sum(scores["validation"]),
                               "validation_min": min(scores["validation"])})
        candidates.sort(key=lambda item: (-item["train_sum"], item["word"]))
        results[coord] = candidates[0]
    return results


def inspect(candidate: dict, public: dict, rows: list[dict], traces: np.ndarray,
            sample: int) -> dict:
    u_words = mul.words(int(candidate["u"], 16) if isinstance(candidate["u"], str) else candidate["u"])
    details = {}
    for coord in ("x", "y"):
        secret = candidate[coord]
        secret = int(secret, 16) if isinstance(secret, str) else int(secret)
        train, valid = indices(rows, public, coord)
        if min(len(train), len(valid)) < 200:
            raise RuntimeError(f"{coord}: too few split-zero rows for gate")
        train_state, train_borrow = mul.initial_state(secret, public[coord][train], u_words)
        train_scores = correlations(first_byte(train_state, train_borrow, u_words[0]),
                                    np.asarray(traces[train, sample], dtype=np.float64))
        byte = int(np.argmax(np.abs(train_scores)))
        valid_state, valid_borrow = mul.initial_state(secret, public[coord][valid], u_words)
        predicted = first_byte(valid_state, valid_borrow, u_words[0])
        validation = correlations(predicted, np.asarray(traces[valid, sample], dtype=np.float64))
        wrong_sample = sample + 30000 if sample + 30000 < traces.shape[1] else sample - 30000
        if wrong_sample < 0:
            raise RuntimeError("wrong-window sample is outside capture")
        wrong = correlations(predicted[byte:byte + 1],
                             np.asarray(traces[valid, wrong_sample], dtype=np.float64))[0]
        details[coord] = {"selected_byte": byte, "train_abs_r": float(abs(train_scores[byte])),
                          "validation_abs_r": float(abs(validation[byte])),
                          "validation_rank": 1 + int(np.count_nonzero(
                              np.abs(validation) > abs(validation[byte]) + 1e-12)),
                          "wrong_window_abs_r": float(abs(wrong)),
                          "train_count": len(train), "validation_count": len(valid)}
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--m-candidates", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads(args.frozen.read_text())
    frozen = source if isinstance(source, list) else source["frozen"]
    if not frozen:
        raise ValueError("frozen candidate set is empty")
    data = args.data
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["target_truth_loaded"] or manifest["known_key_trace_count"]:
        raise RuntimeError("diagnostic input is not target-only")
    rows = [json.loads(line) for line in (data / "public_schedule.jsonl").read_text().splitlines()]
    traces = np.load(data / "traces.npy", mmap_mode="r", allow_pickle=False)
    if (manifest["target_trace_sha256"] != source["target_trace_sha256"] or
            manifest["target_public_schedule_sha256"] != source["target_public_schedule_sha256"]):
        raise RuntimeError("high gate input hashes differ from candidate freeze")
    m = json.loads(args.m_candidates.read_text())["matrix"]
    public_by_m10 = {}
    for candidate in frozen:
        branch = candidate["m10"]
        if branch not in public_by_m10:
            matrix = tuple(int(m[name], 16) for name in ("m00", "m01")) + (
                int(branch, 16), int(m["m11"], 16))
            public_by_m10[branch] = crt.oracle_public(rows, matrix, ORACLE)
    alternatives = []
    for frozen_rank, candidate in enumerate(frozen, 1):
        points = candidate.get("multiply_points") or source["points"]["multiply"]
        lower = points[5][-1] + 1
        upper = min(lower + 2048, traces.shape[1]-7*13)
        if upper-lower < 1024:
            raise RuntimeError("row-six relative search window is truncated")
        peaks = search(candidate, public_by_m10[candidate["m10"]], rows, traces, lower, upper)
        for peak in peaks:
            if (peak["validation_min"] > 0.2 and peak["validation_sum"] > peak["wrong_window_sum"] and
                    all(item["validation_rank"] == 1 for item in peak["groups"].values())):
                alternatives.append({"frozen_rank": frozen_rank, "relative_window": [lower, upper], **peak})
        print(f"high row6 candidate {frozen_rank}/{len(frozen)}", flush=True)
    if not alternatives:
        raise RuntimeError("no row-six peak validates in both coordinates")
    by_sample = {}
    for item in alternatives:
        by_sample.setdefault(item["sample"], []).append(item)
    helper = rank.helpers()
    validated = []
    for sample, items in sorted(by_sample.items(), key=lambda pair: -max(z["train_sum"] for z in pair[1])):
        for item in sorted(items, key=lambda z: z["frozen_rank"]):
            candidate = frozen[item["frozen_rank"]-1]
            stores = seven_store(candidate, public_by_m10[candidate["m10"]],
                                 rows, traces, sample, helper)
            if all(value["validation_min"] > 0.2 for value in stores.values()):
                validated.append({**item, "seven_store": stores,
                                  "seven_store_train_sum": sum(value["train_sum"]
                                                               for value in stores.values()),
                                  "seven_store_validation_min": min(value["validation_min"]
                                                                      for value in stores.values())})
                break
    order = sorted(validated, key=lambda row: (-row["seven_store_train_sum"],
                                               row["frozen_rank"], row["sample"]))
    if not order:
        raise RuntimeError("no row-six seven-store peak validates in both coordinates")
    report = {"schema": "r3-single-split-high-seven-store-v1",
              "status": "exploratory_candidates", "sample": order[0]["sample"],
              "selected_frozen_rank": order[0]["frozen_rank"],
              "relative_window": order[0]["relative_window"],
              "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
              "target_trace_sha256": manifest["target_trace_sha256"],
              "target_public_schedule_sha256": manifest["target_public_schedule_sha256"],
              "m_candidates_sha256": crt.digest(args.m_candidates),
              "script_sha256": crt.digest(Path(__file__)),
              "mul_helper_sha256": crt.digest(ROOT / "r3_cpa_onekey_mul.py"),
              "oracle_sha256": crt.digest(ORACLE),
              "candidate_count": len(frozen), "ranked": order,
              "evidence_contract": {"evidence_origin": "measured",
                                    "evidence_purpose": "diagnostic",
                                    "blinding": "unblinded",
                                    "measurement_scope": "isolated_primitive",
                                    "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                                    "target_truth_loaded": False,
                                    "known_key_traces_loaded": False,
                                    "claim_not_allowed": ["blind_success_rate", "scheme_break"]}}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(frozen), "top5": order[:5]}, indent=2))


if __name__ == "__main__":
    main()
