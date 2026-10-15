#!/usr/bin/env python3
"""Sequentially recover the high words of the Round-3 CRT coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import challenge_matrix_cpa192 as product_attack
import crt_inverse_mul_cpa as inverse_attack
import crt_secret_low192 as low


SEED = 0x43525448
TRACE_COUNTS = (1024, 2048)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEY_INDICES = (0, 2)
HELD_OUT_PER_CLASS = 64
LOW_WORDS = 6
HIGH_ROWS = (6, 7, 8)
MUL_WORDS = 17
PRODUCT_WORDS = 34
PREFIX_BEAM = 64
PREFIX_TIE_CAP = 256
ROW_KEEP = 2
ROW_TIE_CAP = 8
COORDINATE_BEAM = 2
MASK32 = np.uint64(0xFFFFFFFF)
MOD192 = 1 << 192


def words(value: int, count: int = MUL_WORDS) -> np.ndarray:
    return np.array([(value >> (32 * i)) & 0xFFFFFFFF for i in range(count)], dtype=np.uint64)


def difference_words(secret: int, public: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    secret_words = words(secret)
    diff = np.empty((public.shape[0], MUL_WORDS), dtype=np.uint64)
    borrow = np.zeros(public.shape[0], dtype=np.uint64)
    borrow_before = np.empty_like(diff)
    for row in range(MUL_WORDS):
        borrow_before[:, row] = borrow
        known = public[:, row] if row < LOW_WORDS else np.uint64(0)
        subtrahend = known + borrow
        diff[:, row] = (secret_words[row] - subtrahend) & MASK32
        borrow = (secret_words[row] < subtrahend).astype(np.uint64)
    low.require(not np.any(borrow), "negative CRT difference is outside this simulation")
    return diff, borrow_before


def update_row(state: np.ndarray, row: int, diff_word: np.ndarray, u_words: np.ndarray,
               terminal_carry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    result = state.copy()
    stores = np.empty((MUL_WORDS, state.shape[0]), dtype=np.uint8)
    carry = np.zeros(state.shape[0], dtype=np.uint64)
    for inner in range(MUL_WORDS):
        total = diff_word * u_words[inner] + result[:, row + inner] + carry
        result[:, row + inner] = total & MASK32
        carry = total >> np.uint64(32)
        stores[inner] = low.matrix_attack.limb.low.hw32(result[:, row + inner])
    if terminal_carry:
        result[:, row + MUL_WORDS] = carry
    return result, stores


def simulate_product(diff: np.ndarray, u: int) -> tuple[np.ndarray, np.ndarray]:
    u_words = words(u)
    state = np.zeros((diff.shape[0], PRODUCT_WORDS), dtype=np.uint64)
    stores = np.empty((MUL_WORDS, MUL_WORDS, diff.shape[0]), dtype=np.uint8)
    for row in range(MUL_WORDS):
        state, stores[row] = update_row(state, row, diff[:, row], u_words)
    return stores, state


def scalar_reference(diff: int, u: int) -> tuple[np.ndarray, list[int]]:
    result = [0] * PRODUCT_WORDS
    stores = np.empty((MUL_WORDS, MUL_WORDS), dtype=np.uint8)
    d_words, u_words = words(diff), words(u)
    for row in range(MUL_WORDS):
        carry = 0
        for inner in range(MUL_WORDS):
            total = int(d_words[row]) * int(u_words[inner]) + result[row + inner] + carry
            result[row + inner] = total & 0xFFFFFFFF
            carry = total >> 32
            stores[row, inner] = int(result[row + inner]).bit_count()
        result[row + MUL_WORDS] = carry
    return stores, result


def parity_check(secret: int, public: np.ndarray, u: int) -> None:
    diff, _ = difference_words(secret, public[:4])
    stores, state = simulate_product(diff, u)
    for trace in range(diff.shape[0]):
        diff_value = sum(int(diff[trace, i]) << (32 * i) for i in range(MUL_WORDS))
        reference_stores, reference_state = scalar_reference(diff_value, u)
        low.require(np.array_equal(stores[:, :, trace], reference_stores),
                    "source-order store HW parity failed")
        low.require(state[trace].tolist() == reference_state,
                    "terminal-carry state parity failed")
        product = sum(int(word) << (32 * i) for i, word in enumerate(reference_state))
        low.require(product == diff_value * u, "schoolbook product does not match bigint")


def prefix_scores(candidates: np.ndarray, state_word: np.ndarray, borrow: np.ndarray,
                  u0: np.uint64, observed: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    centered = observed - observed.mean()
    observed_energy = float(centered @ centered)
    low.require(observed_energy > 0.0, "constant observed row leakage")
    for start in range(0, candidates.size, product_attack.CHUNK):
        stop = min(start + product_attack.CHUNK, candidates.size)
        diff = (candidates[start:stop, None] - borrow[None, :]) & mask
        stored = (diff * (u0 & mask) + (state_word[None, :] & mask)) & mask
        predicted = low.matrix_attack.limb.low.hw32(stored).astype(np.float64)
        if bits == 32:
            residual = predicted - observed[None, :]
            scores[start:stop] = -np.mean(residual * residual, axis=1)
        else:
            predicted -= predicted.mean(axis=1, keepdims=True)
            numerator = predicted @ centered
            denominator = np.sqrt(
                np.einsum("ij,ij->i", predicted, predicted) * observed_energy)
            scores[start:stop] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def full_row_score(candidate: int, state: np.ndarray, borrow: np.ndarray, row: int,
                   u_words: np.ndarray, observed: np.ndarray) -> float:
    diff = (np.uint64(candidate) - borrow) & MASK32
    predicted_state, predicted = update_row(state, row, diff, u_words)
    del predicted_state
    residual = predicted.astype(np.float64) - observed
    return -float(np.mean(residual * residual))


def attack_word(state: np.ndarray, borrow: np.ndarray, row: int, u_words: np.ndarray,
                observed: np.ndarray, truth: int, max_word: int) -> tuple[list[tuple[int, float]], dict[str, object]]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | byte[None, :]).reshape(-1)
        if bits == 32:
            prefixes = prefixes[prefixes <= np.uint64(max_word)]
        if not prefixes.size:
            stages.append({
                "bits": bits, "expanded": 0, "truth_rank": None,
                "truth_kept": False, "kept": 0,
            })
            return [], {
                "row": row, "truth_word": truth, "truth_evaluated": False,
                "truth_rank_full_row": None, "best_score_ties": 0, "stages": stages,
            }
        scores = prefix_scores(
            prefixes, state[:, row], borrow, u_words[0], observed[0], bits)
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        truth_position = np.flatnonzero(prefixes[order] == true_prefix)
        keep_count = min(PREFIX_BEAM, order.size)
        keep = order[:keep_count]
        tied_count = keep_count
        if keep_count:
            cutoff = scores[keep[-1]]
            tied_count = int(np.count_nonzero(scores[order] >= cutoff - 1e-12))
            keep = order[:min(tied_count, PREFIX_TIE_CAP)]
        stages.append({
            "bits": bits,
            "expanded": int(order.size),
            "truth_rank": int(truth_position[0] + 1) if truth_position.size else None,
            "truth_kept": bool(np.any(prefixes[keep] == true_prefix)),
            "kept": int(keep.size),
            "tie_cap_hit": tied_count > PREFIX_TIE_CAP,
        })
        prefixes = prefixes[keep]
    exact_scores = np.array([
        full_row_score(int(candidate), state, borrow, row, u_words, observed)
        for candidate in prefixes
    ])
    order = np.lexsort((prefixes, -exact_scores))
    keep_count = min(ROW_KEEP, order.size)
    keep = order[:keep_count]
    tied_count = keep_count
    if keep_count:
        cutoff = exact_scores[keep[-1]]
        tied_count = int(np.count_nonzero(exact_scores[order] >= cutoff - 1e-12))
        keep = order[:min(tied_count, ROW_TIE_CAP)]
    best = exact_scores[order[0]]
    return ([(int(prefixes[index]), float(exact_scores[index])) for index in keep], {
        "row": row,
        "truth_word": truth,
        "truth_evaluated": bool(np.any(prefixes == np.uint64(truth))),
        "truth_rank_full_row": (int(np.flatnonzero(prefixes[order] == np.uint64(truth))[0] + 1)
                                if np.any(prefixes == np.uint64(truth)) else None),
        "best_score_ties": int(np.count_nonzero(np.abs(exact_scores - best) <= 1e-12)),
        "row_tie_cap_hit": tied_count > ROW_TIE_CAP,
        "stages": stages,
    })


def initial_state(secret: int, public: np.ndarray, u_words: np.ndarray,
                  mode: str) -> tuple[np.ndarray, np.ndarray]:
    secret_words = words(secret, LOW_WORDS)
    diff = np.empty((public.shape[0], LOW_WORDS), dtype=np.uint64)
    borrow = np.zeros(public.shape[0], dtype=np.uint64)
    for row in range(LOW_WORDS):
        subtrahend = public[:, row] + borrow
        diff[:, row] = (secret_words[row] - subtrahend) & MASK32
        borrow = (secret_words[row] < subtrahend).astype(np.uint64)
    state = np.zeros((public.shape[0], PRODUCT_WORDS), dtype=np.uint64)
    if mode != "no_prior_state":
        for row in range(LOW_WORDS):
            state, _ = update_row(
                state, row, diff[:, row], u_words, terminal_carry=mode != "no_terminal_carry")
    return state, borrow


def attack_coordinate(secret: int, public: np.ndarray, observed: np.ndarray, u: int,
                      n: int, mode: str = "normal") -> dict[str, object]:
    u_words = words(u)
    state, borrow = initial_state(secret % MOD192, public, u_words, mode)
    hypotheses = [(secret % MOD192, state, borrow, 0.0, [])]
    truth_words = words(secret)
    row_summaries = []
    for offset, row in enumerate(HIGH_ROWS):
        expanded = []
        diagnostics = []
        for prefix, current_state, current_borrow, score, history in hypotheses:
            max_word = 0xFFFFFFFF
            if row == HIGH_ROWS[-1]:
                remaining = n - 1 - prefix
                if remaining < 0:
                    continue
                max_word = min(max_word, remaining >> (32 * row))
            candidates, detail = attack_word(
                current_state, current_borrow, row, u_words, observed[offset],
                int(truth_words[row]), max_word)
            diagnostics.append(detail)
            for candidate, candidate_score in candidates:
                diff = (np.uint64(candidate) - current_borrow) & MASK32
                next_state, _ = update_row(
                    current_state, row, diff, u_words,
                    terminal_carry=mode != "no_terminal_carry")
                next_borrow = (np.uint64(candidate) < current_borrow).astype(np.uint64)
                expanded.append((
                    prefix | candidate << (32 * row), next_state, next_borrow,
                    score + candidate_score, history + [detail]))
        expanded.sort(key=lambda item: (-item[3], item[0]))
        if not expanded:
            raise RuntimeError(f"no bounded candidates for coordinate row {row}")
        hypotheses = expanded[:COORDINATE_BEAM]
        truth_prefix = secret % (1 << (32 * (row + 1)))
        truth_positions = [i for i, item in enumerate(hypotheses) if item[0] == truth_prefix]
        row_summaries.append({
            "row": row,
            "expanded_hypotheses": len(expanded),
            "truth_survived": bool(truth_positions),
            "truth_rank": truth_positions[0] + 1 if truth_positions else None,
            "word_diagnostics": diagnostics,
        })
    recovered = hypotheses[0][0]
    return {
        "recovered": recovered,
        "success": recovered == secret,
        "candidate_count": len(hypotheses),
        "rows": row_summaries,
    }


def split_dataset(secret: tuple[int, int, int], matrix: tuple[int, int, int, int], count: int,
                  split: int, key_index: int, tag: int, oracle: Path) -> dict[str, object]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, key_index, count, tag]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(3 * count)]
    dataset = low.make_dataset(secret, matrix, challenges, oracle)
    selected = np.flatnonzero(dataset["split_v"] == split)
    low.require(selected.size >= count, "insufficient challenges in requested split class")
    selected = selected[:count]
    return {
        "public_x": dataset["public_x"][selected],
        "public_y": dataset["public_y"][selected],
        "challenges": [challenges[int(index)] for index in selected],
    }


def prepare(secret: tuple[int, int, int], matrix: tuple[int, int, int, int], count: int,
            key_index: int, oracle: Path) -> dict[str, object]:
    dataset = split_dataset(secret, matrix, count, 0, key_index, 0x54524143, oracle)
    n, x, y = secret
    u = pow(MOD192, -1, n)
    x_diff, _ = difference_words(x, dataset["public_x"])
    y_diff, _ = difference_words(y, dataset["public_y"])
    x_stores, _ = simulate_product(x_diff, u)
    y_stores, _ = simulate_product(y_diff, u)
    parity_check(x, dataset["public_x"], u)
    parity_check(y, dataset["public_y"], u)
    dataset.update({"u": u, "x_stores": x_stores, "y_stores": y_stores})
    return dataset


def attack_pair(secret: tuple[int, int, int], dataset: dict[str, object],
                observed_x: np.ndarray, observed_y: np.ndarray, u: int,
                mode: str = "normal") -> dict[str, object]:
    n, x, y = secret
    x_result = attack_coordinate(x, dataset["public_x"], observed_x, u, n, mode)
    y_result = attack_coordinate(y, dataset["public_y"], observed_y, u, n, mode)
    return {
        "x": x_result,
        "y": y_result,
        "pair_exact": x_result["success"] and y_result["success"],
    }


def observed_rows(dataset: dict[str, object], sigma: float, key_index: int,
                  traces: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, key_index, traces, int(sigma * 100)]))
    shape = (len(HIGH_ROWS), MUL_WORDS, traces)
    x = dataset["x_stores"][list(HIGH_ROWS)].astype(np.float64)
    y = dataset["y_stores"][list(HIGH_ROWS)].astype(np.float64)
    return x + rng.normal(0.0, sigma, shape), y + rng.normal(0.0, sigma, shape)


def held_out_validation(secret: tuple[int, int, int], matrix: tuple[int, int, int, int],
                        recovered_x: int, recovered_y: int, key_index: int,
                        oracle: Path) -> dict[str, object]:
    n, true_x, true_y = secret
    u = pow(MOD192, -1, n)
    exact = total = 0
    class_results = {}
    challenge_bytes = bytearray()
    for split in (0, -1):
        dataset = split_dataset(
            secret, matrix, HELD_OUT_PER_CLASS, split, key_index, 0x484F4C44 + split, oracle)
        challenge_bytes.extend(b"".join(
            value.to_bytes(24, "little") for value in dataset["challenges"]))
        truth = ((true_x, true_y) if split == 0
                 else low.transformed_coordinates(secret, split))
        recovered = ((recovered_x, recovered_y) if split == 0
                     else low.transformed_coordinates((n, recovered_x, recovered_y), split))
        class_exact = class_total = 0
        for public, expected_secret, guessed_secret in zip(
                (dataset["public_x"], dataset["public_y"]), truth, recovered):
            true_diff, _ = difference_words(expected_secret, public)
            guessed_diff, _ = difference_words(guessed_secret, public)
            true_stores, _ = simulate_product(true_diff, u)
            guessed_stores, _ = simulate_product(guessed_diff, u)
            target_true = true_stores[list(HIGH_ROWS)]
            target_guessed = guessed_stores[list(HIGH_ROWS)]
            class_exact += int(np.count_nonzero(target_true == target_guessed))
            class_total += int(target_true.size)
        class_results["split_1_0" if split == 0 else "split_1_minus1"] = {
            "store_hw_exact": class_exact,
            "store_hw_total": class_total,
        }
        exact += class_exact
        total += class_total
    transformed = low.transformed_coordinates((n, recovered_x, recovered_y), -1)
    inert_relation = (np.gcd(abs(2 * transformed[0] + 2 * recovered_y + 1),
                              abs(2 * transformed[1] - 2 * recovered_x + 1)) == n)
    return {
        "per_class": class_results,
        "store_hw_exact": exact,
        "store_hw_total": total,
        "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
        "coordinate_bounds": 0 <= recovered_x < n and 0 <= recovered_y < n,
        "inert_relation": bool(inert_relation),
    }


def controls(secret: tuple[int, int, int], dataset: dict[str, object], sigma: float,
             key_index: int) -> dict[str, dict[str, object]]:
    traces = dataset["public_x"].shape[0]
    observed_x, observed_y = observed_rows(dataset, sigma, key_index, traces)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x434F4E54]))
    permutation = rng.permutation(traces)
    shuffled = dict(dataset)
    shuffled["public_x"] = dataset["public_x"][permutation]
    shuffled["public_y"] = dataset["public_y"][permutation]
    noise_x = rng.normal(16.0, sigma, observed_x.shape)
    noise_y = rng.normal(16.0, sigma, observed_y.shape)
    def run(public: dict[str, object], x: np.ndarray, y: np.ndarray, u: int,
            mode: str = "normal") -> dict[str, object]:
        try:
            result = attack_pair(secret, public, x, y, u, mode)
            return {"status": "completed", "pair_exact": bool(result["pair_exact"])}
        except RuntimeError as error:
            low.require(str(error).startswith("no bounded candidates"),
                        f"unexpected negative-control failure: {error}")
            return {"status": "rejected_no_bounded_candidate", "pair_exact": False}

    results = {
        "challenge_shuffle": run(shuffled, observed_x, observed_y, dataset["u"]),
        "wrong_caller_window": run(dataset, observed_y, observed_x, dataset["u"]),
        "noise_only": run(dataset, noise_x, noise_y, dataset["u"]),
        "corrupted_u": run(dataset, observed_x, observed_y, dataset["u"] ^ 1),
        "no_prior_state": run(
            dataset, observed_x, observed_y, dataset["u"], "no_prior_state"),
        "no_terminal_carry": run(
            dataset, observed_x, observed_y, dataset["u"], "no_terminal_carry"),
    }
    u_words = words(dataset["u"])
    for name, mode in (("no_prior_state", "no_prior_state"),
                       ("no_terminal_carry", "no_terminal_carry")):
        normal, _ = initial_state(secret[1] % MOD192, dataset["public_x"], u_words, "normal")
        ablated, _ = initial_state(secret[1] % MOD192, dataset["public_x"], u_words, mode)
        results[name]["row6_entry_state_exact"] = bool(np.array_equal(normal, ablated))
    results["no_terminal_carry"]["interpretation"] = "inert_for_these_bounded_operands"
    return results


def verify_prerequisites(low_path: Path, inverse_path: Path) -> dict[str, object]:
    low_report = json.loads(low_path.read_text())
    inverse_report = json.loads(inverse_path.read_text())
    low.require(low_report["schema"] == "sqisign-r3-p500-m4-crt-secret-low192-v1",
                "unexpected low-coordinate prerequisite schema")
    low.require(inverse_report["schema"] == "sqisign-r3-p500-m4-crt-inverse-mul-cpa-v1",
                "unexpected full-N prerequisite schema")
    low.require(all(row["coordinate_low192_exact"]["split_1_0.x"] == row["secret_trials"]
                    and row["coordinate_low192_exact"]["split_1_0.y"] == row["secret_trials"]
                    for row in low_report["sweep"]), "low-coordinate prerequisite did not pass")
    low.require(all(row["n_exact"] == row["secret_trials"] for row in inverse_report["sweep"]),
                "full-N prerequisite did not pass")
    return {
        "low_coordinate_report_sha256": low.matrix_attack.limb.low.sha256(low_path),
        "full_n_report_sha256": low.matrix_attack.limb.low.sha256(inverse_path),
        "mode": "exact prior-stage outputs; truth values are used only because reports store hashes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--low-report", type=Path, required=True)
    parser.add_argument("--n-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    elf, kat, oracle, source, low_report, n_report = (
        path.resolve() for path in (args.elf, args.kat, args.oracle, args.source,
                                    args.low_report, args.n_report))
    low.require(oracle.is_file() and oracle.stat().st_mode & 0o111,
                "ideal oracle is not executable")
    target = inverse_attack.bind_target(elf)
    prerequisites = verify_prerequisites(low_report, n_report)
    secrets = low.kat_secrets(kat)
    matrices = low.matrix_attack.kat_matrices(kat)
    prepared = {
        (key_index, traces): prepare(
            secrets[key_index], matrices[key_index], traces, key_index, oracle)
        for traces in TRACE_COUNTS for key_index in TARGET_KEY_INDICES
    }
    noiseless = []
    for key_index in TARGET_KEY_INDICES:
        dataset = prepared[(key_index, TRACE_COUNTS[0])]
        exact_x = dataset["x_stores"][list(HIGH_ROWS)].astype(np.float64)
        exact_y = dataset["y_stores"][list(HIGH_ROWS)].astype(np.float64)
        result = attack_pair(secrets[key_index], dataset, exact_x, exact_y, dataset["u"])
        low.require(result["pair_exact"], f"noiseless high-coordinate recovery failed for key {key_index}")
        noiseless.append({"key_index": key_index, "result": result})
    sweep = []
    successful_results = {}
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            trials = []
            for key_index in TARGET_KEY_INDICES:
                dataset = prepared[(key_index, traces)]
                observed_x, observed_y = observed_rows(dataset, sigma, key_index, traces)
                result = attack_pair(
                    secrets[key_index], dataset, observed_x, observed_y, dataset["u"])
                trials.append({"key_index": key_index, "result": result})
                successful_results[(key_index, traces, sigma)] = result
            sweep.append({
                "traces_per_split_class": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(trials),
                "x_exact": sum(item["result"]["x"]["success"] for item in trials),
                "y_exact": sum(item["result"]["y"]["success"] for item in trials),
                "pair_exact": sum(item["result"]["pair_exact"] for item in trials),
                "trials": trials,
            })
    control_result = controls(
        secrets[TARGET_KEY_INDICES[0]], prepared[(TARGET_KEY_INDICES[0], TRACE_COUNTS[0])],
        NOISE_SIGMAS[0], TARGET_KEY_INDICES[0])
    primary_controls = ("challenge_shuffle", "wrong_caller_window", "noise_only", "corrupted_u")
    low.require(not any(control_result[name]["pair_exact"] for name in primary_controls),
                "a primary negative control recovered the exact pair")
    held_out = []
    for key_index in TARGET_KEY_INDICES:
        result = successful_results[(key_index, TRACE_COUNTS[-1], NOISE_SIGMAS[-1])]
        validation = held_out_validation(
            secrets[key_index], matrices[key_index], result["x"]["recovered"],
            result["y"]["recovered"], key_index, oracle)
        held_out.append({"key_index": key_index, **validation})
    report = {
        "schema": "sqisign-r3-p500-m4-crt-coordinate-high-cpa-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "unblinded",
            "measurement_scope": "instrumented_subpath",
            "claim_scope": "isk:r3_p500_m4_crt_xy_full_sequential_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": low.matrix_attack.limb.low.sha256(Path(__file__)),
            "low_helper_sha256": low.matrix_attack.limb.low.sha256(Path(low.__file__)),
            "inverse_helper_sha256": low.matrix_attack.limb.low.sha256(Path(inverse_attack.__file__)),
            "oracle_sha256": low.matrix_attack.limb.low.sha256(oracle),
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEED,
        },
        "source": {
            "commit_label": low.SOURCE_COMMIT,
            "git_metadata_present": (source / ".git").exists(),
            "ideal_c_sha256": low.matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/ideal.c"),
            "mp_c_sha256": low.matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/mp.c"),
        },
        "target_elf_sha256": low.matrix_attack.limb.low.ELF_SHA256,
        "kat_sha256": low.matrix_attack.limb.low.KAT_SHA256,
        "parameter_set": "p500_27",
        "target_key_indices": list(TARGET_KEY_INDICES),
        "target": target,
        "dynamic_scope": {
            "split_class": "(1,0)",
            "x_crt_caller": target["intersection_x_crt_call"],
            "y_crt_caller": target["intersection_y_crt_call"],
            "ibz_crt_inverse_product_call": target["crt_inverse_product_call"],
            "ibz_mul_store": target["ibz_mul_inner_store"],
            "outer_rows": list(HIGH_ROWS),
            "inner_rows": list(range(MUL_WORDS)),
            "dynamic_store_occurrences": {
                str(row): [row * MUL_WORDS, row * MUL_WORDS + MUL_WORDS - 1]
                for row in HIGH_ROWS
            },
        },
        "prerequisites": prerequisites,
        "model": ("17 source-order ibz_mul inner-store HW samples per outer row, with exact "
                  "prior-row product state and subtraction borrow plus independent Gaussian noise"),
        "attack": ("8/16/24-bit signed-Pearson prefix search on j=0, then 32-bit low-store "
                   "negative-MSE and exact 17-store row likelihood with a two-state coordinate "
                   "beam; non-informative control ties are capped at 256 prefixes and 8 row "
                   "candidates"),
        "source_equivalent_dimensions": {"diff_words": 17, "u_words": 17, "product_words": 34},
        "noiseless": noiseless,
        "sweep": sweep,
        "negative_controls": {
            name: control_result[name] for name in primary_controls
        },
        "ablations": {
            name: control_result[name] for name in ("no_prior_state", "no_terminal_carry")
        },
        "held_out": held_out,
        "allowed_statement": ("For fixed KAT keys 0 and 2, exact prior low-coordinate and full-N/u "
                              "outputs, and the declared simulated aligned HW model, the reported "
                              "sequential full x/y recovery rates were observed."),
        "blocked_statements": [
            "prior matrix, low-coordinate, N/u, and high-coordinate stages were jointly error-propagated",
            "physical STM32F4 leakage, trigger alignment, instruction mixing, or HD leakage were modeled",
            "a full signing key or forgery was recovered",
            "the result generalizes beyond the fixed KAT keys, ELF, positive-difference branch, and model",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(args.out.suffix + ".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.chmod(0o600)
    staged.replace(args.out)
    for row in sweep:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces_per_split_class']:4d} "
              f"x={row['x_exact']}/{row['secret_trials']} "
              f"y={row['y_exact']}/{row['secret_trials']} "
              f"pair={row['pair_exact']}/{row['secret_trials']}", flush=True)
    print("controls=" + ",".join(
        f"{name}:{int(item['pair_exact'])}" for name, item in control_result.items()))


if __name__ == "__main__":
    main()
