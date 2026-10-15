#!/usr/bin/env python3
"""Run an evidence-bound synthetic CPA on the Round-3 M4 m01*challenge low word."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


ELF_SHA256 = "8746008ea3e42306e6e9d7798baa551ae2a80757e20781f87598f00dbdb1bc41"
KAT_SHA256 = "986a28c7752428ba0eff2784af4f07b996d259646b192b16407a647a584d7537"
SEED = 0x32427
TRACE_COUNTS = (128, 512, 1024)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEYS = 6
CONTROL_KEYS = 2
BEAM = 64
CHUNK = 512
M01_OFFSET = 129 + 3 * 64 + 24
MATRIX_CALL_PC = 0x08008EDE
MUL_PC = 0x0801F2E2
STORE_PC = 0x0801F2EE
MASK32 = np.uint64(0xFFFFFFFF)
HW8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=45)
    require(result.returncode == 0, f"{command[0]} failed ({result.returncode}): {result.stderr[-1200:]}")
    return result.stdout


def kat_m01_words(path: Path) -> list[int]:
    require(sha256(path) == KAT_SHA256, "Round-3 p500_27 KAT hash mismatch")
    keys = [bytes.fromhex(value) for value in re.findall(r"^sk = ([0-9A-F]+)$", path.read_text(), re.M)]
    require(len(keys) == 100 and all(len(key) == 417 for key in keys), "unexpected KAT key count or size")
    words = [int.from_bytes(key[M01_OFFSET:M01_OFFSET + 4], "little") for key in keys]
    require(len(set(words)) == len(words), "duplicate m01 low words in KAT")
    return words


def bind_target(elf: Path) -> dict[str, str]:
    require(sha256(elf) == ELF_SHA256, "canonical Round-3 p500_27 M4 ELF hash mismatch")
    matrix = checked(["arm-none-eabi-objdump", "-d", "--disassemble=ibz_mat_2x2_eval", str(elf)], elf.parent)
    multiply = checked(["arm-none-eabi-objdump", "-d", "--disassemble=ibz_mul", str(elf)], elf.parent)
    require(re.search(r"(?m)^\s*8008ede:.*\bbl\s+801f1d8 <ibz_mul>$", matrix) is not None,
            "m01*challenge call-site binding mismatch")
    require(re.search(r"(?m)^\s*801f2e2:.*\bumlal\s+r3, ip, r5, r4$", multiply) is not None,
            "M4 multiply instruction binding mismatch")
    require(re.search(r"(?m)^\s*801f2ee:.*\bstr\s+r3, \[r0, #0\]$", multiply) is not None,
            "M4 store instruction binding mismatch")
    return {
        "ibz_mat_2x2_eval_m01_challenge_call": f"0x{MATRIX_CALL_PC:08x}",
        "ibz_mul_lowword_umlal": f"0x{MUL_PC:08x}",
        "ibz_mul_lowword_store": f"0x{STORE_PC:08x}",
    }


def hw32(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.uint32, copy=False)
    octets = words.view(np.uint8).reshape(words.shape + (4,))
    return np.add.reduce(HW8[octets], axis=-1, dtype=np.uint8)


def correlations(candidates: np.ndarray, challenges: np.ndarray, leakage: np.ndarray, bits: int) -> np.ndarray:
    centered_leakage = leakage - leakage.mean()
    leakage_energy = float(centered_leakage @ centered_leakage)
    require(leakage_energy > 0.0, "constant leakage vector")
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    for start in range(0, candidates.size, CHUNK):
        stop = min(start + CHUNK, candidates.size)
        products = (candidates[start:stop, None] * challenges[None, :]) & mask
        predictions = hw32(products).astype(np.float64)
        predictions -= predictions.mean(axis=1, keepdims=True)
        numerator = predictions @ centered_leakage
        denominator = np.sqrt(np.einsum("ij,ij->i", predictions, predictions) * leakage_energy)
        scores[start:stop] = np.divide(np.abs(numerator), denominator,
                                               out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def attack(challenges: np.ndarray, leakage: np.ndarray, truth: int) -> dict[str, object]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    final_scores = np.empty(0)
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            next_byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | next_byte[None, :]).reshape(-1)
        scores = correlations(prefixes, challenges, leakage, bits)
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        true_positions = np.flatnonzero(prefixes[order] == true_prefix)
        stages.append({
            "bits": bits,
            "truth_survived": bool(true_positions.size),
            "truth_rank_within_expanded_beam": int(true_positions[0] + 1) if true_positions.size else None,
        })
        keep = order[:min(BEAM, order.size)]
        prefixes = prefixes[keep]
        final_scores = scores[keep]
    return {
        "success": bool(int(prefixes[0]) == truth),
        "best_score": float(final_scores[0]),
        "stages": stages,
    }


def one_trial(word: int, traces: int, sigma: float, key_index: int,
              control: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, int(sigma * 10), key_index]))
    challenges = rng.integers(0, 1 << 32, size=traces, dtype=np.uint32).astype(np.uint64)
    ideal = hw32((np.uint64(word) * challenges) & MASK32).astype(np.float64)
    leakage = ideal + rng.normal(0.0, sigma, traces)
    positive = attack(challenges, leakage, word)
    negative = attack(challenges[rng.permutation(traces)], leakage, word) if control else None
    return positive, negative


def self_check(word: int) -> None:
    require(hw32(np.array([0, 1, 0xFFFFFFFF], dtype=np.uint64)).tolist() == [0, 1, 32],
            "HW implementation failed")
    rng = np.random.default_rng(SEED)
    challenges = rng.integers(0, 1 << 32, size=512, dtype=np.uint32).astype(np.uint64)
    leakage = hw32((np.uint64(word) * challenges) & MASK32).astype(np.float64)
    require(attack(challenges, leakage, word)["success"], "noiseless staged CPA self-check failed")


def sweep(words: list[int]) -> list[dict[str, object]]:
    results = []
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            positive = []
            negative = []
            for key_index, word in enumerate(words[:TARGET_KEYS]):
                attack_result, control_result = one_trial(
                    word, traces, sigma, key_index, key_index < CONTROL_KEYS)
                positive.append(attack_result)
                if control_result is not None:
                    negative.append(control_result)
            results.append({
                "traces": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(positive),
                "exact_low32_recoveries": sum(item["success"] for item in positive),
                "exact_low32_success_rate": round(sum(item["success"] for item in positive) / len(positive), 4),
                "stage_truth_survival_rates": {
                    str(bits): round(sum(item["stages"][bits // 8 - 1]["truth_survived"] for item in positive) / len(positive), 4)
                    for bits in (8, 16, 24, 32)
                },
                "median_winning_absolute_correlation": round(float(np.median([item["best_score"] for item in positive])), 4),
                "shuffled_challenge_control_trials": len(negative),
                "shuffled_challenge_exact_recoveries": sum(item["success"] for item in negative),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--kat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    elf = args.elf.resolve()
    kat = args.kat.resolve()
    target = bind_target(elf)
    words = kat_m01_words(kat)
    self_check(words[0])
    report = {
        "schema": "sqisign-r3-p500-m4-challenge-matrix-cpa-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "not_applicable",
            "measurement_scope": "isolated_instruction_model",
            "claim_scope": "isk:r3_p500_m4_m01_low32_cpa_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": sha256(Path(__file__)),
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "objdump": checked(["arm-none-eabi-objdump", "--version"], elf.parent).splitlines()[0],
            "seed": SEED,
        },
        "target_elf_sha256": ELF_SHA256,
        "kat_sha256": KAT_SHA256,
        "parameter_set": "p500_27",
        "target": target,
        "known_input": "public challenge coefficient low 32-bit limb",
        "secret_target": "mat_BAcan_to_BA0_two.m[0][1] low 32-bit magnitude limb",
        "target_words_sha256": hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in words[:TARGET_KEYS])).hexdigest(),
        "model": "one perfectly aligned sample: HW((m01_low32 * challenge_low32) mod 2^32) plus independent Gaussian noise",
        "attack": f"four-stage 8/16/24/32-bit absolute-Pearson CPA with beam={BEAM}",
        "sweep": sweep(words),
        "allowed_statement": "Under the declared synthetic model, the reported fraction of six fixed KAT words was recovered exactly.",
        "blocked_statements": [
            "STM32F4 physical leakage or trace alignment was demonstrated",
            "the full 192-bit matrix entry, full matrix, signing key, or (N,x,y) was recovered",
            "SQIsign was broken",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(args.out)
    for row in report["sweep"]:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces']:4d} "
              f"success={row['exact_low32_recoveries']}/{row['secret_trials']} "
              f"control={row['shuffled_challenge_exact_recoveries']}/{row['shuffled_challenge_control_trials']}")


if __name__ == "__main__":
    main()
