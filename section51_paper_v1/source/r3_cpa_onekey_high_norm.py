#!/usr/bin/env python3
"""Recover high CRT words and norm from one frozen single-key candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / ".plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle"
import r3_cpa_crt_analyze as crt
import r3_cpa_matrix_rank as rank
import r3_cpa_onekey_mul as mul
from r3_cpa_onekey_norm import recover_norm


MASK = np.uint64(0xffffffff)


def partition(rows: list[dict], public: dict, coord: str) -> dict[int, np.ndarray]:
    members = [i for i, item in enumerate(rows) if item["coord"] == coord and
               not item["heldout"] and public["split"][i] == 0]
    return {parity: np.array([i for i in members if
                              hashlib.sha256(rows[i]["challenge_hex"].encode()).digest()[0] & 1 == parity],
                             dtype=int) for parity in (0, 1)}


def correlation(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    predicted = predicted - predicted.mean()
    observed = observed - observed.mean()
    denominator = np.linalg.norm(predicted) * np.linalg.norm(observed)
    return 0.0 if denominator == 0 else float(predicted @ observed / denominator)


def first_byte(state: np.ndarray, borrow: np.ndarray, row: int,
               u0: np.uint64) -> np.ndarray:
    candidates = np.arange(256, dtype=np.uint64)[:, None]
    byte = np.uint64(255)
    stored = ((((candidates - borrow[None, :]) & byte) * (u0 & byte))
              + (state[None, :, row] & byte)) & byte
    return crt.low.matrix_attack.limb.low.hw32(stored).astype(float)


def choose_start(state: np.ndarray, borrow: np.ndarray, row: int, u0: np.uint64,
                 traces: np.ndarray, indices: np.ndarray, previous: int) -> int:
    left, right = previous + 100, previous + 221
    model = first_byte(state, borrow, row, u0)
    model -= model.mean(axis=1, keepdims=True)
    wave = np.asarray(traces[indices, left:right], float)
    wave -= wave.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(model, axis=1)[:, None] * np.linalg.norm(wave, axis=0)[None, :]
    grid = np.divide(model @ wave, denominator, out=np.zeros((256, right-left)), where=denominator > 0)
    _, offset = np.unravel_index(np.argmax(np.abs(grid)), grid.shape)
    return left + int(offset)


def score_word(word: int, state: np.ndarray, borrow: np.ndarray, row: int,
               u_words: np.ndarray, observed: np.ndarray) -> tuple[float, list[float]]:
    difference = (np.uint64(word) - borrow) & MASK
    _, predicted = mul.update_row(state, row, difference, u_words)
    scores = [abs(correlation(predicted[j], observed[:, j])) for j in range(7)]
    return sum(scores), scores


def recover_coordinate(low_secret: int, public_words: np.ndarray,
                       traces: np.ndarray, by_parity: dict[int, np.ndarray],
                       u_words: np.ndarray, row6_start: int) -> list[tuple[int, list[dict]]]:
    helper = rank.helpers()
    states = {parity: mul.initial_state(low_secret, public_words[ix], u_words)
              for parity, ix in by_parity.items()}
    secret = low_secret
    previous = row6_start
    summary = []
    for row in (6, 7, 8):
        current, borrow = states[0]
        start = (row6_start if row == 6 else
                 choose_start(current, borrow, row, u_words[0], traces, by_parity[0], previous))
        points = [start + 13*j for j in range(7)]
        if points[-1] >= traces.shape[1]:
            raise RuntimeError("high-row stores exceed captured samples")
        observed = {parity: np.asarray(traces[np.ix_(ix, points)], float)
                    for parity, ix in by_parity.items()}
        if row == 8:
            candidates = list(range(1 << 11))
        else:
            base = (current[:, row] - borrow*u_words[0]) & MASK
            multiplier = np.full(len(by_parity[0]), u_words[0], dtype=np.uint64)
            ranked = rank.rank_word(base, multiplier, observed[0][:, 0], 64, True, helper)
            candidates = ranked["final_beam"]
        choices = []
        for word in candidates:
            train, _ = score_word(word, *states[0], row, u_words, observed[0])
            valid, per_store = score_word(word, *states[1], row, u_words, observed[1])
            choices.append((train, valid, word, per_store))
        choices.sort(key=lambda item: (-item[0], item[2]))
        train, valid, word, per_store = choices[0]
        if row == 8:
            absent = summary + [{"row": 8, "status": "source_row_absent",
                                 "word": hex(0), "searched_start": start}]
            options = [(secret, absent)]
            if word != 0 and min(per_store) >= 0.15:
                options.append((secret | word << (32*row), summary + [{
                    "row": row, "start": start, "word": hex(word),
                    "train_sum": train, "validation_sum": valid,
                    "min_validation_abs_r": min(per_store),
                    "candidate_count": len(choices),
                    "runner_up_train_sum": choices[1][0] if len(choices) > 1 else None}]))
            return options
        if min(per_store) < 0.15:
            raise RuntimeError(f"row {row} failed seven-store validation")
        secret |= word << (32*row)
        next_states = {}
        for parity, (current, borrow) in states.items():
            next_state, _ = mul.update_row(current, row, (np.uint64(word)-borrow)&MASK, u_words)
            next_states[parity] = (next_state, (np.uint64(word) < borrow).astype(np.uint64))
        states = next_states
        summary.append({"row": row, "start": start, "word": hex(word),
                        "train_sum": train, "validation_sum": valid,
                        "min_validation_abs_r": min(per_store),
                        "candidate_count": len(choices),
                        "runner_up_train_sum": choices[1][0] if len(choices) > 1 else None})
        previous = start
    raise RuntimeError("high coordinate rows did not terminate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--m-candidates", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frozen_bytes = args.frozen.read_bytes()
    frozen = json.loads(frozen_bytes)
    gate = json.loads(args.gate.read_text())
    if gate["frozen_sha256"] != hashlib.sha256(frozen_bytes).hexdigest():
        raise RuntimeError("high gate was run on a different candidate freeze")
    chosen = gate["ranked"][0]
    if chosen["validation_min"] <= 0.2 or chosen["validation_sum"] <= chosen["wrong_window_sum"]:
        raise RuntimeError("high first-byte gate lacks validation evidence")
    candidate = frozen["frozen"][chosen["frozen_rank"]-1]
    data = args.data
    manifest = json.loads((data / "manifest.json").read_text())
    if (manifest["target_truth_loaded"] or manifest["known_key_trace_count"] or
            manifest["target_trace_sha256"] != frozen["target_trace_sha256"] or
            manifest["target_public_schedule_sha256"] != frozen["target_public_schedule_sha256"]):
        raise RuntimeError("high/norm input is not the frozen target-only capture")
    rows = [json.loads(line) for line in (data / "public_schedule.jsonl").read_text().splitlines()]
    traces = np.load(data / "traces.npy", mmap_mode="r", allow_pickle=False)
    m = json.loads(args.m_candidates.read_text())["matrix"]
    matrix = tuple(int(m[name], 16) for name in ("m00", "m01")) + (
        int(candidate["m10"], 16), int(m["m11"], 16))
    public = crt.oracle_public(rows, matrix, ORACLE)
    u_words = mul.words(candidate["u"])
    options = {}
    for coord in ("x", "y"):
        by_parity = partition(rows, public, coord)
        if min(map(len, by_parity.values())) < 200:
            raise RuntimeError("too few train or validation rows")
        options[coord] = recover_coordinate(
            candidate[coord], public[coord], traces, by_parity, u_words, gate["sample"])
    header_path = ROOT / "vendor/r3-p500-27-m4f/quaternion_constants.h"
    header = header_path.read_text()
    field_prime = int(re.search(r'#define QUAT_P_HEX "([0-9a-f]+)"', header).group(1), 16)
    norm_candidates = []
    for x, x_rows in options["x"]:
        for y, y_rows in options["y"]:
            try:
                norm, cofactor = recover_norm(candidate["u"], x, y, field_prime)
            except (RuntimeError, ValueError):
                continue
            norm_candidates.append((norm, cofactor, x, y, {"x": x_rows, "y": y_rows}))
    if len(norm_candidates) != 1:
        raise RuntimeError(f"CRT norm relation selected {len(norm_candidates)} coordinate tuples")
    norm, cofactor, x, y, groups = norm_candidates[0]
    report = {"schema": "r3-single-split-high-norm-diagnostic-v1",
              "status": "candidate_not_KAT_audited", "frozen_rank": chosen["frozen_rank"],
              "frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
              "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
              "target_trace_sha256": manifest["target_trace_sha256"],
              "target_public_schedule_sha256": manifest["target_public_schedule_sha256"],
              "m_candidates_sha256": crt.digest(args.m_candidates),
              "script_sha256": crt.digest(Path(__file__)),
              "norm_helper_sha256": crt.digest(ROOT / "r3_cpa_onekey_norm.py"),
              "mul_helper_sha256": crt.digest(ROOT / "r3_cpa_onekey_mul.py"),
              "field_header_sha256": crt.digest(header_path),
              "oracle_sha256": crt.digest(ORACLE),
              "candidate": {"m10": candidate["m10"], "u": hex(candidate["u"]),
                            "x": hex(x), "y": hex(y),
                            "N": hex(norm), "gcd_cofactor": cofactor},
              "groups": groups,
              "evidence_contract": {"evidence_origin": "measured",
                                    "evidence_purpose": "diagnostic",
                                    "blinding": "unblinded",
                                    "measurement_scope": "isolated_primitive",
                                    "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                                    "target_truth_loaded": False,
                                    "claim_not_allowed": ["blind_success_rate", "scheme_break"]}}
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    args.out.chmod(0o600)
    print(json.dumps({"status": report["status"], "frozen_rank": chosen["frozen_rank"],
                      "cofactor": cofactor,
                      "min_validation_abs_r": min(row["min_validation_abs_r"]
                                                  for group in groups.values() for row in group
                                                  if row.get("status") != "source_row_absent")}, indent=2))


if __name__ == "__main__":
    main()
