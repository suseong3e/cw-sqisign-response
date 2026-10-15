#!/usr/bin/env python3
"""Recover Round-3 inert-ideal low limbs from the CRT subtraction HW model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

import challenge_matrix_full_cpa as matrix_attack


SEED = 0x4352544E
TRACE_COUNTS = (1024, 2048)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEYS = 3
CONTROL_KEYS = 1
HELD_OUT_CHALLENGES = 64
LIMBS = 6
PREFIX_BEAM = 8
COORDINATE_BEAM = 8
MASK32 = np.uint64(0xFFFFFFFF)
MOD192 = 1 << 192
PUBLICKEY_BYTES = 129
FP_ENCODED_BYTES = 64
SOURCE_COMMIT = "6d017708db403bf83977fa70770fc4f7f9e9ff21"
CRT_SUB_STORE_PC = 0x08025A52


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def words(value: int) -> np.ndarray:
    return np.array([(value >> (32 * i)) & 0xFFFFFFFF for i in range(LIMBS)], dtype=np.uint64)


def integers_to_words(values: list[int]) -> np.ndarray:
    return np.array([words(value) for value in values], dtype=np.uint64)


def kat_secrets(path: Path) -> list[tuple[int, int, int]]:
    require(matrix_attack.limb.low.sha256(path) == matrix_attack.limb.low.KAT_SHA256,
            "Round-3 p500_27 KAT hash mismatch")
    keys = [bytes.fromhex(value) for value in re.findall(r"^sk = ([0-9A-F]+)$", path.read_text(), re.M)]
    require(len(keys) == 100 and all(len(key) == 417 for key in keys),
            "unexpected KAT key count or size")
    offset = PUBLICKEY_BYTES
    result = []
    for key in keys:
        result.append(tuple(int.from_bytes(
            key[offset + field * FP_ENCODED_BYTES:offset + (field + 1) * FP_ENCODED_BYTES], "little")
            for field in range(3)))
    require(all(n > 0 and n & 1 and 0 <= x < n and 0 <= y < n for n, x, y in result),
            "invalid encoded inert ideal")
    return result


def bind_target(elf: Path) -> dict[str, str]:
    target = matrix_attack.limb.low.bind_target(elf)
    crt = matrix_attack.limb.low.checked(
        ["arm-none-eabi-objdump", "-d", "--disassemble=ibz_crt", str(elf)], elf.parent)
    intersection = matrix_attack.limb.low.checked(
        ["arm-none-eabi-objdump", "-d", "--disassemble=quat_ideal_intersect_O0", str(elf)], elf.parent)
    require(re.search(r"(?m)^\s*8025a52:.*\bstr\.w\s+r2, \[r9, #4\]!$", crt) is not None,
            "CRT subtraction store binding mismatch")
    for pc in ("80155ba", "80155c8"):
        require(re.search(rf"(?m)^\s*{pc}:.*\bbl\s+\S+ <ibz_crt>$", intersection) is not None,
                f"intersection CRT call binding mismatch at {pc}")
    target.update({
        "ibz_crt_subtraction_store": f"0x{CRT_SUB_STORE_PC:08x}",
        "intersection_x_crt_call": "0x080155ba",
        "intersection_y_crt_call": "0x080155c8",
    })
    return target


def transformed_coordinates(secret: tuple[int, int, int], split_v: int) -> tuple[int, int]:
    n, x, y = secret
    if split_v == 0:
        return x, y
    require(split_v == -1, f"unsupported challenge split coordinate {split_v}")
    inv2 = pow(2, -1, n)
    tmp_x = (-2 * (2 * y + 1)) % (2 * n)
    tmp_y = (4 * x) % (2 * n)
    return ((tmp_x // 2) * inv2) % n, (((tmp_y - 2) // 2) * inv2) % n


def subtraction_leakage(secret: int, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    secret_words = words(secret)
    borrow = np.zeros(known.shape[0], dtype=np.uint64)
    output = np.empty_like(known)
    samples = np.empty((LIMBS, known.shape[0]), dtype=np.float64)
    for row in range(LIMBS):
        subtrahend = known[:, row] + borrow
        output[:, row] = (secret_words[row] - subtrahend) & MASK32
        samples[row] = matrix_attack.limb.low.hw32(output[:, row])
        borrow = (secret_words[row] < subtrahend).astype(np.uint64)
    return samples, output


def score_candidates(candidates: np.ndarray, known: np.ndarray, borrow: np.ndarray,
                     leakage: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    centered = leakage - leakage.mean()
    leakage_energy = float(centered @ centered)
    require(leakage_energy > 0.0, "constant subtraction leakage vector")
    for start in range(0, candidates.size, matrix_attack.limb.CHUNK):
        stop = min(start + matrix_attack.limb.CHUNK, candidates.size)
        predicted = (candidates[start:stop, None] - known[None, :] - borrow[None, :]) & mask
        predicted = matrix_attack.limb.low.hw32(predicted).astype(np.float64)
        if bits == 32:
            residual = predicted - leakage[None, :]
            scores[start:stop] = -np.mean(residual * residual, axis=1)
        else:
            predicted -= predicted.mean(axis=1, keepdims=True)
            numerator = predicted @ centered
            denominator = np.sqrt(np.einsum("ij,ij->i", predicted, predicted) * leakage_energy)
            scores[start:stop] = np.divide(np.abs(numerator), denominator,
                                           out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def attack_word(known: np.ndarray, borrow: np.ndarray, leakage: np.ndarray, truth: int) -> dict[str, object]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    final_scores = np.empty(0)
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            next_byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | next_byte[None, :]).reshape(-1)
        scores = score_candidates(prefixes, known & np.uint64((1 << bits) - 1), borrow,
                                  leakage, bits)
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        positions = np.flatnonzero(prefixes[order] == true_prefix)
        stages.append({
            "bits": bits,
            "truth_survived": bool(positions.size),
            "truth_rank_within_expanded_beam": int(positions[0] + 1) if positions.size else None,
        })
        keep = order[:min(PREFIX_BEAM, order.size)]
        prefixes = prefixes[keep]
        final_scores = scores[keep]
    return {
        "recovered": int(prefixes[0]),
        "success": bool(int(prefixes[0]) == truth),
        "best_score": float(final_scores[0]),
        "candidates": [int(value) for value in prefixes],
        "candidate_scores": [float(value) for value in final_scores],
        "stages": stages,
    }


def attack_coordinate(secret: int, known: np.ndarray, leakage: np.ndarray) -> tuple[
        int, list[dict[str, object]], list[dict[str, float | int]]]:
    truth = words(secret)
    states: list[tuple[int, np.ndarray, float]] = [
        (0, np.zeros(known.shape[0], dtype=np.uint64), 0.0)
    ]
    details = []
    for row in range(LIMBS):
        expanded = []
        for prefix, borrow, cumulative_score in states:
            result = attack_word(known[:, row], borrow, leakage[row], int(truth[row]))
            for word, word_score in zip(result["candidates"], result["candidate_scores"]):
                next_borrow = (np.uint64(word) < known[:, row] + borrow).astype(np.uint64)
                expanded.append((prefix | word << (32 * row), next_borrow,
                                 cumulative_score + word_score))
        states = sorted(expanded, key=lambda state: (-state[2], state[0]))[:COORDINATE_BEAM]
        truth_prefix = secret % (1 << (32 * (row + 1)))
        positions = [index for index, state in enumerate(states) if state[0] == truth_prefix]
        details.append({
            "limb": row,
            "state_count": len(states),
            "truth_survived": bool(positions),
            "truth_rank": positions[0] + 1 if positions else None,
        })
    candidates = [{"value": state[0], "score": state[2]} for state in states]
    return states[0][0], details, candidates


def derive_n_low(x0: int, y0: int, x1: int, y1: int) -> tuple[int | None, list[dict[str, int]]]:
    a = (2 * x1 + 2 * y0 + 1) % MOD192
    b = (2 * y1 - 2 * x0 + 1) % MOD192
    matches = []
    for sign, candidate in ((1, b), (-1, (-b) % MOD192)):
        for coefficient in (1, 3):
            if candidate & 1 and a == coefficient * candidate % MOD192:
                matches.append({"n_low": candidate, "b_sign": sign, "a_coefficient": coefficient})
    unique = sorted({item["n_low"] for item in matches})
    return (unique[0] if len(unique) == 1 else None), matches


def make_dataset(secret: tuple[int, int, int], matrix: tuple[int, int, int, int],
                 challenges: list[int], oracle: Path) -> dict[str, object]:
    rows = matrix_attack.oracle_batch(oracle, [matrix_attack.uv(matrix, challenge) for challenge in challenges])
    public_x = integers_to_words([int(row[0], 16) for row in rows])
    public_y = integers_to_words([int(row[1], 16) for row in rows])
    split_v = np.array([int(row[5], 16) for row in rows], dtype=np.int8)
    require(set(split_v.tolist()) <= {0, -1}, "unexpected official challenge split class")
    representations = {value: transformed_coordinates(secret, value) for value in (0, -1)}
    exact_x = np.zeros((LIMBS, len(rows)), dtype=np.float64)
    exact_y = np.zeros((LIMBS, len(rows)), dtype=np.float64)
    for value in (0, -1):
        selected = split_v == value
        exact_x[:, selected], _ = subtraction_leakage(representations[value][0], public_x[selected])
        exact_y[:, selected], _ = subtraction_leakage(representations[value][1], public_y[selected])
    return {
        "public_x": public_x,
        "public_y": public_y,
        "split_v": split_v,
        "representations": representations,
        "exact_x": exact_x,
        "exact_y": exact_y,
    }


def attack_dataset(dataset: dict[str, object], observed_x: np.ndarray, observed_y: np.ndarray,
                   shuffle_rng: np.random.Generator | None = None) -> dict[str, object]:
    recovered: dict[int, tuple[int | None, int | None]] = {}
    coordinate_success = {}
    candidate_sets: dict[str, list[dict[str, float | int]]] = {}
    representation_truth = dataset["representations"]
    for value, name in ((0, "split_1_0"), (-1, "split_1_minus1")):
        selected = np.flatnonzero(dataset["split_v"] == value)
        if not selected.size:
            recovered[value] = None, None
            coordinate_success[name] = {
                "traces": 0,
                "eligible": False,
                "x_low192": False,
                "y_low192": False,
            }
            continue
        public_x = dataset["public_x"][selected].copy()
        public_y = dataset["public_y"][selected].copy()
        if shuffle_rng is not None:
            public_x = public_x[shuffle_rng.permutation(selected.size)]
            public_y = public_y[shuffle_rng.permutation(selected.size)]
        truth_x, truth_y = representation_truth[value]
        recovered_x, _, candidates_x = attack_coordinate(truth_x, public_x, observed_x[:, selected])
        recovered_y, _, candidates_y = attack_coordinate(truth_y, public_y, observed_y[:, selected])
        recovered[value] = recovered_x, recovered_y
        candidate_sets[f"{name}.x"] = candidates_x
        candidate_sets[f"{name}.y"] = candidates_y
        coordinate_success[name] = {
            "traces": int(selected.size),
            "eligible": True,
            "x_low192": False,
            "y_low192": False,
        }
    x0, y0 = recovered[0]
    x1, y1 = recovered[-1]
    relation_eligible = all(value is not None for value in (x0, y0, x1, y1))
    relation_candidates = []
    if relation_eligible:
        names = ("split_1_0.x", "split_1_0.y", "split_1_minus1.x", "split_1_minus1.y")
        for candidate_tuple in itertools.product(*(candidate_sets[name] for name in names)):
            values = tuple(int(candidate["value"]) for candidate in candidate_tuple)
            candidate_n, matches = derive_n_low(*values)
            if candidate_n is not None:
                relation_candidates.append({
                    "values": values,
                    "n_low": candidate_n,
                    "score": sum(float(candidate["score"]) for candidate in candidate_tuple),
                    "matches": matches,
                })
    relation_candidates.sort(key=lambda candidate: (
        -candidate["score"], candidate["n_low"], candidate["values"]))
    if relation_candidates:
        chosen = relation_candidates[0]
        x0, y0, x1, y1 = chosen["values"]
        recovered = {0: (x0, y0), -1: (x1, y1)}
        n_low = chosen["n_low"]
        relation_matches = chosen["matches"]
    else:
        n_low, relation_matches = (derive_n_low(x0, y0, x1, y1)
                                   if relation_eligible else (None, []))
    for value, name in ((0, "split_1_0"), (-1, "split_1_minus1")):
        if coordinate_success[name]["eligible"]:
            truth_x, truth_y = representation_truth[value]
            coordinate_success[name]["x_low192"] = recovered[value][0] == truth_x % MOD192
            coordinate_success[name]["y_low192"] = recovered[value][1] == truth_y % MOD192
    return {
        "recovered": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "n": n_low},
        "coordinate_success": coordinate_success,
        "relation_eligible": relation_eligible,
        "relation_candidate_count": len(relation_candidates),
        "relation_matches": relation_matches,
    }


def held_out_validation(secret: tuple[int, int, int], matrix: tuple[int, int, int, int],
                        recovered: dict[str, int | None], key_index: int, oracle: Path) -> dict[str, object]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x484F4C44, key_index]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(HELD_OUT_CHALLENGES)]
    rows = matrix_attack.oracle_batch(oracle, [matrix_attack.uv(matrix, challenge) for challenge in challenges])
    matched = 0
    for row in rows:
        value = int(row[5], 16)
        true_x, true_y = transformed_coordinates(secret, value)
        guessed_x = recovered["x0" if value == 0 else "x1"]
        guessed_y = recovered["y0" if value == 0 else "y1"]
        public_x, public_y = int(row[0], 16), int(row[1], 16)
        matched += (guessed_x is not None and guessed_y is not None and
                    (guessed_x - public_x) % MOD192 == (true_x - public_x) % MOD192 and
                    (guessed_y - public_y) % MOD192 == (true_y - public_y) % MOD192)
    return {
        "challenges": HELD_OUT_CHALLENGES,
        "challenge_sha256": hashlib.sha256(
            b"".join(value.to_bytes(24, "little") for value in challenges)).hexdigest(),
        "crt_diff_pair_low192_exact": matched,
    }


def self_check(secret: tuple[int, int, int], matrix: tuple[int, int, int, int], oracle: Path) -> None:
    x0, y0 = transformed_coordinates(secret, 0)
    x1, y1 = transformed_coordinates(secret, -1)
    require(math.gcd(abs(2 * x1 + 2 * y0 + 1), abs(2 * y1 - 2 * x0 + 1)) == secret[0],
            "split-coordinate relations do not recover the KAT norm")
    rng = np.random.default_rng(SEED)
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(512)]
    dataset = make_dataset(secret, matrix, challenges, oracle)
    result = attack_dataset(dataset, dataset["exact_x"], dataset["exact_y"])
    require(all(field[coordinate] for field in result["coordinate_success"].values()
                for coordinate in ("x_low192", "y_low192")), "noiseless coordinate recovery failed")
    require(result["recovered"]["n"] == secret[0] % MOD192, "noiseless N-low recovery failed")


def trial(secret: tuple[int, int, int], matrix: tuple[int, int, int, int], traces: int,
          sigma: float, key_index: int, oracle: Path, control: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    challenge_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index]))
    challenges = [int.from_bytes(challenge_rng.bytes(24), "little") for _ in range(traces)]
    dataset = make_dataset(secret, matrix, challenges, oracle)
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index, int(sigma * 100)]))
    observed_x = dataset["exact_x"] + noise_rng.normal(0.0, sigma, (LIMBS, traces))
    observed_y = dataset["exact_y"] + noise_rng.normal(0.0, sigma, (LIMBS, traces))
    result = attack_dataset(dataset, observed_x, observed_y)
    result["n_low192_success"] = result["recovered"]["n"] == secret[0] % MOD192
    result["held_out"] = held_out_validation(secret, matrix, result["recovered"], key_index, oracle)
    result["recovered_sha256"] = hashlib.sha256(b"".join(
        int(result["recovered"][name] or 0).to_bytes(24, "little")
        for name in ("x0", "y0", "x1", "y1", "n"))).hexdigest()
    negative = None
    if control:
        negative = attack_dataset(dataset, observed_x, observed_y, noise_rng)
        negative["n_low192_success"] = negative["recovered"]["n"] == secret[0] % MOD192
    return result, negative


def sweep(secrets: list[tuple[int, int, int]], matrices: list[tuple[int, int, int, int]],
          oracle: Path) -> list[dict[str, object]]:
    rows = []
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            positive = []
            negative = []
            for key_index in range(TARGET_KEYS):
                result, control = trial(secrets[key_index], matrices[key_index], traces, sigma,
                                        key_index, oracle, key_index < CONTROL_KEYS)
                positive.append(result)
                if control is not None:
                    negative.append(control)
            coordinate_names = ("split_1_0", "split_1_minus1")
            rows.append({
                "traces": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(positive),
                "coordinate_low192_exact": {
                    f"{name}.{coordinate}": sum(
                        item["coordinate_success"][name][f"{coordinate}_low192"] for item in positive)
                    for name in coordinate_names for coordinate in ("x", "y")
                },
                "coordinate_eligible_trials": {
                    f"{name}.{coordinate}": sum(
                        item["coordinate_success"][name]["eligible"] for item in positive)
                    for name in coordinate_names for coordinate in ("x", "y")
                },
                "all_four_coordinate_eligible_trials": sum(item["relation_eligible"] for item in positive),
                "all_four_coordinates_low192_exact": sum(item["relation_eligible"] and all(
                    item["coordinate_success"][name][f"{coordinate}_low192"]
                    for name in coordinate_names for coordinate in ("x", "y")) for item in positive),
                "n_relation_eligible_trials": sum(item["relation_eligible"] for item in positive),
                "n_low192_exact": sum(item["n_low192_success"] for item in positive),
                "held_out_crt_diff_pair_low192_exact": sum(
                    item["held_out"]["crt_diff_pair_low192_exact"] for item in positive),
                "held_out_trials": len(positive) * HELD_OUT_CHALLENGES,
                "shuffled_input_trials": len(negative),
                "shuffled_all_four_coordinates_low192_exact": sum(item["relation_eligible"] and all(
                    item["coordinate_success"][name][f"{coordinate}_low192"]
                    for name in coordinate_names for coordinate in ("x", "y")) for item in negative),
                "shuffled_n_low192_exact": sum(item["n_low192_success"] for item in negative),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    elf, kat, oracle, source = (path.resolve() for path in (args.elf, args.kat, args.oracle, args.source))
    require(oracle.is_file() and oracle.stat().st_mode & 0o111, "ideal oracle is not executable")
    target = bind_target(elf)
    secrets = kat_secrets(kat)
    matrices = matrix_attack.kat_matrices(kat)
    self_check(secrets[0], matrices[0], oracle)
    results = sweep(secrets, matrices, oracle)
    report = {
        "schema": "sqisign-r3-p500-m4-crt-secret-low192-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "unblinded",
            "measurement_scope": "instrumented_subpath",
            "claim_scope": "isk:r3_p500_m4_crt_xy_nlow192_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": matrix_attack.limb.low.sha256(Path(__file__)),
            "matrix_helper_sha256": matrix_attack.limb.low.sha256(Path(matrix_attack.__file__)),
            "oracle_sha256": matrix_attack.limb.low.sha256(oracle),
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEED,
        },
        "source": {
            "commit": SOURCE_COMMIT,
            "ideal_c_sha256": matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/ideal.c"),
            "mp_c_sha256": matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/mp.c"),
            "id2iso_c_sha256": matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/id2iso.c"),
        },
        "target_elf_sha256": matrix_attack.limb.low.ELF_SHA256,
        "kat_sha256": matrix_attack.limb.low.KAT_SHA256,
        "parameter_set": "p500_27",
        "target": target,
        "known_input": "public challenge coefficient mapped through the recovered basis matrix to official ideal_chall_two.x/y and chall_split",
        "secret_target": "low 192 bits of both CRT secret-coordinate representations, then N mod 2^192 from their exact inert-coordinate relations",
        "window": "six perfectly aligned 32-bit result stores at the inlined diff_a=a2-a1 loop in each x/y ibz_crt call",
        "model": "HW of each subtraction result word plus independent Gaussian noise",
        "attack": ("split-class-conditioned byte-prefix recovery with cross-limb borrow beam search; "
                   f"prefix beam={PREFIX_BEAM}, coordinate beam={COORDINATE_BEAM}; inert-coordinate "
                   "relations select the joint four-coordinate candidate; calibrated zero-offset "
                   "unit-scale HW likelihood at each 32-bit stage"),
        "held_out": "independent challenges unused by recovery; exact low-192 CRT difference pair checked",
        "target_values_sha256": hashlib.sha256(b"".join(
            value.to_bytes(FP_ENCODED_BYTES, "little")
            for secret in secrets[:TARGET_KEYS] for value in secret)).hexdigest(),
        "sweep": results,
        "allowed_statement": "Under the declared synthetic model, the reported low-192 coordinate and norm residues were recovered at the reported rates and validated on independent challenges.",
        "blocked_statements": [
            "the challenge-independent high coordinate limbs are identifiable from this CPA surface",
            "complete secret ideals, signing keys, or forgeries were recovered",
            "physical STM32F4 leakage, alignment, or full signing traces were modeled",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(args.out.suffix + ".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.chmod(0o600)
    staged.replace(args.out)
    for row in results:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces']:4d} "
              f"coords={row['all_four_coordinates_low192_exact']}/{row['secret_trials']} "
              f"Nlow={row['n_low192_exact']}/{row['secret_trials']} "
              f"heldout={row['held_out_crt_diff_pair_low192_exact']}/{row['held_out_trials']} "
              f"shuffleN={row['shuffled_n_low192_exact']}/{row['shuffled_input_trials']}")


if __name__ == "__main__":
    main()
