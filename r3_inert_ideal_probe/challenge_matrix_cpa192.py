#!/usr/bin/env python3
"""Recover all six m01 words from aligned M4 j=0 store HW samples.

The simulated challenge is a uniform 24-byte random-oracle output. Each trace
contains six perfectly aligned observations, one for the first inner iteration
of each informative secret-limb row in the p500 Cortex-M4 ``ibz_mul`` loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

import challenge_matrix_cpa as low


SEED = 0x19232427
TRACE_COUNTS = (512, 1024)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEYS = 6
CONTROL_KEYS = 2
LIMBS = 6
BEAM = 64
CHUNK = 512
MASK32 = np.uint64(0xFFFFFFFF)


def kat_m01_values(path: Path) -> list[int]:
    low.require(low.sha256(path) == low.KAT_SHA256, "Round-3 p500_27 KAT hash mismatch")
    keys = [bytes.fromhex(value) for value in re.findall(r"^sk = ([0-9A-F]+)$", path.read_text(), re.M)]
    low.require(len(keys) == 100 and all(len(key) == 417 for key in keys), "unexpected KAT key count or size")
    values = [int.from_bytes(key[low.M01_OFFSET:low.M01_OFFSET + 24], "little") for key in keys]
    low.require(all(0 <= value < 1 << 192 for value in values), "m01 value exceeds its encoded bound")
    low.require(len(set(values)) == len(values), "duplicate m01 values in KAT")
    return values


def words(value: int) -> list[int]:
    return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(LIMBS)]


def integer(challenge_words: np.ndarray) -> int:
    return sum(int(word) << (32 * i) for i, word in enumerate(challenge_words))


def apply_row(state: np.ndarray, secret_word: int, challenges: np.ndarray, row: int) -> None:
    carry = np.zeros(challenges.shape[0], dtype=np.uint64)
    word = np.uint64(secret_word)
    for column in range(LIMBS):
        total = word * challenges[:, column] + state[:, row + column] + carry
        state[:, row + column] = total & MASK32
        carry = total >> np.uint64(32)
    state[:, row + LIMBS] = carry


def leakage_samples(secret: int, challenges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal store HW and final schoolbook state for the six informative rows."""
    state = np.zeros((challenges.shape[0], 2 * LIMBS), dtype=np.uint64)
    samples = np.empty((LIMBS, challenges.shape[0]), dtype=np.float64)
    for row, secret_word in enumerate(words(secret)):
        stored = (np.uint64(secret_word) * challenges[:, 0] + state[:, row]) & MASK32
        samples[row] = low.hw32(stored)
        apply_row(state, secret_word, challenges, row)
    return samples, state


def correlations(candidates: np.ndarray, challenge0: np.ndarray, accumulator: np.ndarray,
                 leakage: np.ndarray, bits: int) -> np.ndarray:
    centered_leakage = leakage - leakage.mean()
    leakage_energy = float(centered_leakage @ centered_leakage)
    low.require(leakage_energy > 0.0, "constant leakage vector")
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    for start in range(0, candidates.size, CHUNK):
        stop = min(start + CHUNK, candidates.size)
        stored = (candidates[start:stop, None] * challenge0[None, :] + accumulator[None, :]) & mask
        predictions = low.hw32(stored).astype(np.float64)
        predictions -= predictions.mean(axis=1, keepdims=True)
        numerator = predictions @ centered_leakage
        denominator = np.sqrt(np.einsum("ij,ij->i", predictions, predictions) * leakage_energy)
        scores[start:stop] = np.divide(np.abs(numerator), denominator,
                                       out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def attack_word(challenge0: np.ndarray, accumulator: np.ndarray, leakage: np.ndarray,
                truth: int) -> dict[str, object]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    final_scores = np.empty(0)
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            next_byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | next_byte[None, :]).reshape(-1)
        scores = correlations(prefixes, challenge0, accumulator, leakage, bits)
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        positions = np.flatnonzero(prefixes[order] == true_prefix)
        stages.append({
            "bits": bits,
            "truth_survived": bool(positions.size),
            "truth_rank_within_expanded_beam": int(positions[0] + 1) if positions.size else None,
        })
        keep = order[:min(BEAM, order.size)]
        prefixes = prefixes[keep]
        final_scores = scores[keep]
    return {
        "recovered": int(prefixes[0]),
        "success": bool(int(prefixes[0]) == truth),
        "best_score": float(final_scores[0]),
        "stages": stages,
    }


def attack(secret: int, challenges: np.ndarray, leakage: np.ndarray) -> dict[str, object]:
    truth = words(secret)
    recovered = []
    state = np.zeros((challenges.shape[0], 2 * LIMBS), dtype=np.uint64)
    limb_results = []
    for row in range(LIMBS):
        result = attack_word(challenges[:, 0], state[:, row], leakage[row], truth[row])
        recovered.append(result["recovered"])
        limb_results.append(result)
        apply_row(state, result["recovered"], challenges, row)
    recovered_value = sum(word << (32 * i) for i, word in enumerate(recovered))
    return {
        "success": recovered_value == secret,
        "correct_limbs": sum(result["success"] for result in limb_results),
        "first_wrong_limb": next((i for i, result in enumerate(limb_results) if not result["success"]), None),
        "limbs": limb_results,
    }


def trial(secret: int, traces: int, sigma: float, key_index: int,
          control: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    challenge_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index, int(sigma * 100)]))
    challenges = challenge_rng.integers(0, 1 << 32, size=(traces, LIMBS), dtype=np.uint32).astype(np.uint64)
    ideal, _ = leakage_samples(secret, challenges)
    observed = ideal + noise_rng.normal(0.0, sigma, ideal.shape)
    positive = attack(secret, challenges, observed)
    negative = None
    if control:
        permutation = noise_rng.permutation(traces)
        negative = attack(secret, challenges[permutation], observed)
    return positive, negative


def self_check(secret: int) -> None:
    rng = np.random.default_rng(SEED)
    challenges = rng.integers(0, 1 << 32, size=(512, LIMBS), dtype=np.uint32).astype(np.uint64)
    ideal, state = leakage_samples(secret, challenges)
    for trace in range(4):
        product = secret * integer(challenges[trace])
        modeled = sum(int(state[trace, i]) << (32 * i) for i in range(2 * LIMBS))
        low.require(modeled == product, "schoolbook state differs from exact 192x192 product")
    low.require(attack(secret, challenges, ideal)["success"], "noiseless full-192 CPA self-check failed")


def sweep(secrets: list[int]) -> list[dict[str, object]]:
    rows = []
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            positive = []
            negative = []
            for key_index, secret in enumerate(secrets[:TARGET_KEYS]):
                result, control = trial(secret, traces, sigma, key_index, key_index < CONTROL_KEYS)
                positive.append(result)
                if control is not None:
                    negative.append(control)
            rows.append({
                "traces": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(positive),
                "exact_192_recoveries": sum(result["success"] for result in positive),
                "exact_192_success_rate": round(sum(result["success"] for result in positive) / len(positive), 4),
                "per_limb_success_rate": [
                    round(sum(result["limbs"][i]["success"] for result in positive) / len(positive), 4)
                    for i in range(LIMBS)
                ],
                "median_correct_limbs": float(np.median([result["correct_limbs"] for result in positive])),
                "shuffled_challenge_control_trials": len(negative),
                "shuffled_challenge_exact_recoveries": sum(result["success"] for result in negative),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    elf = args.elf.resolve()
    kat = args.kat.resolve()
    target = low.bind_target(elf)
    secrets = kat_m01_values(kat)
    self_check(secrets[0])
    results = sweep(secrets)
    report = {
        "schema": "sqisign-r3-p500-m4-m01-cpa192-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "not_applicable",
            "measurement_scope": "isolated_primitive",
            "claim_scope": "isk:r3_p500_m4_m01_192_cpa_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": low.sha256(Path(__file__)),
            "low32_helper_sha256": low.sha256(Path(low.__file__)),
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "objdump": low.checked(["arm-none-eabi-objdump", "--version"], elf.parent).splitlines()[0],
            "seed": SEED,
        },
        "target_elf_sha256": low.ELF_SHA256,
        "kat_sha256": low.KAT_SHA256,
        "parameter_set": "p500_27",
        "target": target,
        "known_input": "uniform 192-bit public challenge coefficient modeled as SHAKE256 random-oracle output",
        "secret_target": "all six 32-bit magnitude limbs of mat_BAcan_to_BA0_two.m[0][1]",
        "target_values_sha256": hashlib.sha256(
            b"".join(value.to_bytes(24, "little") for value in secrets[:TARGET_KEYS])).hexdigest(),
        "window": "six dynamic j=0 occurrences of the bound UMLAL/ADDS/ADC/STR loop, one per informative secret limb",
        "model": "six perfectly aligned samples per trace: HW(low32(m01_i * challenge_0 + prior-row accumulator_i)) plus independent Gaussian noise",
        "attack": f"low-to-high limb recovery; each limb uses 8/16/24/32-bit absolute-Pearson CPA with beam={BEAM}",
        "sweep": results,
        "allowed_statement": "Under the declared simulated leakage and challenge models, the reported fraction of six fixed KAT m01 values was recovered exactly at 192 bits.",
        "blocked_statements": [
            "full signing transcripts, HD leakage, jitter, adjacent-instruction mixing, or physical STM32F4 leakage were modeled",
            "the remaining matrix entries, full matrix, secret ideal (N,x,y), or signing key were recovered",
            "SQIsign key recovery or forgery was demonstrated",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(args.out)
    for row in results:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces']:4d} "
              f"exact192={row['exact_192_recoveries']}/{row['secret_trials']} "
              f"limbs={row['per_limb_success_rate']} "
              f"control={row['shuffled_challenge_exact_recoveries']}/{row['shuffled_challenge_control_trials']}",
              flush=True)


if __name__ == "__main__":
    main()
