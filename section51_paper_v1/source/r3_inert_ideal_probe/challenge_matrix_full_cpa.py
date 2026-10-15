#!/usr/bin/env python3
"""Recover the Round-3 p500 2x2 secret basis matrix in a synthetic M4 HW model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import challenge_matrix_cpa192 as limb


SEED = 0xF0110192
TRACE_COUNTS = (512, 1024)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEYS = 3
CONTROL_KEYS = 1
HELD_OUT_CHALLENGES = 4
ADD_BEAM = 256
MATRIX_OFFSET = limb.low.M01_OFFSET - 24
MASK192 = (1 << 192) - 1
SOURCE_COMMIT = "6d017708db403bf83977fa70770fc4f7f9e9ff21"
ADD_STORE_PC = 0x0801F042


def kat_matrices(path: Path) -> list[tuple[int, int, int, int]]:
    limb.low.require(limb.low.sha256(path) == limb.low.KAT_SHA256, "Round-3 p500_27 KAT hash mismatch")
    keys = [bytes.fromhex(value) for value in re.findall(r"^sk = ([0-9A-F]+)$", path.read_text(), re.M)]
    limb.low.require(len(keys) == 100 and all(len(key) == 417 for key in keys),
                     "unexpected KAT key count or size")
    matrices = [tuple(int.from_bytes(key[MATRIX_OFFSET + 24 * i:MATRIX_OFFSET + 24 * (i + 1)], "little")
                      for i in range(4)) for key in keys]
    limb.low.require(all(all(0 <= value < 1 << 192 for value in matrix) for matrix in matrices),
                     "matrix entry exceeds 192 bits")
    limb.low.require(len(set(matrices)) == len(matrices), "duplicate KAT matrices")
    return matrices


def bind_target(elf: Path) -> dict[str, str]:
    target = limb.low.bind_target(elf)
    matrix = limb.low.checked(["arm-none-eabi-objdump", "-d", "--disassemble=ibz_mat_2x2_eval", str(elf)],
                              elf.parent)
    addition = limb.low.checked(["arm-none-eabi-objdump", "-d", "--disassemble=ibz_add", str(elf)], elf.parent)
    calls = {
        "m00_times_one": ("8008ec6", "ibz_mul"),
        "m01_times_challenge": ("8008ede", "ibz_mul"),
        "row0_add": ("8008ee8", "ibz_add"),
        "m10_times_one": ("8008ef4", "ibz_mul"),
        "m11_times_challenge": ("8008f08", "ibz_mul"),
        "row1_add": ("8008f12", "ibz_add"),
    }
    for name, (pc, symbol) in calls.items():
        limb.low.require(re.search(rf"(?m)^\s*{pc}:.*\bbl\s+\S+ <{symbol}>$", matrix) is not None,
                         f"{name} call-site binding mismatch")
        target[name] = f"0x0{pc}"
    limb.low.require(re.search(r"(?m)^\s*801f042:.*\bstr\.w\s+r1, \[r9, #4\]!$", addition) is not None,
                     "M4 addition store binding mismatch")
    target["ibz_add_overlap_store"] = f"0x{ADD_STORE_PC:08x}"
    return target


def add_leakage(addend: int, product: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    carry = np.zeros(product.shape[0], dtype=np.uint64)
    samples = np.empty((limb.LIMBS, product.shape[0]), dtype=np.float64)
    output = np.empty((product.shape[0], limb.LIMBS), dtype=np.uint64)
    for row, word in enumerate(limb.words(addend)):
        total = product[:, row] + np.uint64(word) + carry
        output[:, row] = total & limb.MASK32
        samples[row] = limb.low.hw32(output[:, row])
        carry = total >> np.uint64(32)
    return samples, output


def add_correlations(candidates: np.ndarray, base: np.ndarray, leakage: np.ndarray, bits: int) -> np.ndarray:
    centered = leakage - leakage.mean()
    leakage_energy = float(centered @ centered)
    limb.low.require(leakage_energy > 0.0, "constant addition leakage vector")
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    for start in range(0, candidates.size, limb.CHUNK):
        stop = min(start + limb.CHUNK, candidates.size)
        stored = (candidates[start:stop, None] + base[None, :]) & mask
        predictions = limb.low.hw32(stored).astype(np.float64)
        predictions -= predictions.mean(axis=1, keepdims=True)
        numerator = predictions @ centered
        denominator = np.sqrt(np.einsum("ij,ij->i", predictions, predictions) * leakage_energy)
        scores[start:stop] = np.divide(np.abs(numerator), denominator,
                                       out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def add_likelihood(candidates: np.ndarray, base: np.ndarray, leakage: np.ndarray) -> np.ndarray:
    scores = np.empty(candidates.size, dtype=np.float64)
    for start in range(0, candidates.size, limb.CHUNK):
        stop = min(start + limb.CHUNK, candidates.size)
        stored = (candidates[start:stop, None] + base[None, :]) & limb.MASK32
        residual = limb.low.hw32(stored).astype(np.float64) - leakage[None, :]
        scores[start:stop] = -np.mean(residual * residual, axis=1)
    return scores


def attack_add_word(base: np.ndarray, leakage: np.ndarray, truth: int) -> dict[str, object]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    final_scores = np.empty(0)
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            next_byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | next_byte[None, :]).reshape(-1)
        scores = (add_likelihood(prefixes, base, leakage) if bits == 32
                  else add_correlations(prefixes, base, leakage, bits))
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        positions = np.flatnonzero(prefixes[order] == true_prefix)
        stages.append({
            "bits": bits,
            "truth_survived": bool(positions.size),
            "truth_rank_within_expanded_beam": int(positions[0] + 1) if positions.size else None,
        })
        keep = order[:min(ADD_BEAM, order.size)]
        prefixes = prefixes[keep]
        final_scores = scores[keep]
    return {
        "recovered": int(prefixes[0]),
        "success": bool(int(prefixes[0]) == truth),
        "best_score": float(final_scores[0]),
        "stages": stages,
    }


def recover_product(secret: int, challenges: np.ndarray, leakage: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    truth = limb.words(secret)
    state = np.zeros((challenges.shape[0], 2 * limb.LIMBS), dtype=np.uint64)
    recovered = []
    results = []
    for row in range(limb.LIMBS):
        result = limb.attack_word(challenges[:, 0], state[:, row], leakage[row], truth[row])
        recovered.append(result["recovered"])
        results.append(result)
        limb.apply_row(state, result["recovered"], challenges, row)
    return sum(word << (32 * i) for i, word in enumerate(recovered)), results


def recover_addend(secret: int, product: np.ndarray, leakage: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    truth = limb.words(secret)
    carry = np.zeros(product.shape[0], dtype=np.uint64)
    recovered = []
    results = []
    for row in range(limb.LIMBS):
        result = attack_add_word(product[:, row] + carry, leakage[row], truth[row])
        recovered.append(result["recovered"])
        results.append(result)
        carry = (product[:, row] + np.uint64(result["recovered"]) + carry) >> np.uint64(32)
    return sum(word << (32 * i) for i, word in enumerate(recovered)), results


def simulate(matrix: tuple[int, int, int, int], challenges: np.ndarray) -> np.ndarray:
    m00, m01, m10, m11 = matrix
    mul01, product01 = limb.leakage_samples(m01, challenges)
    add00, _ = add_leakage(m00, product01)
    mul11, product11 = limb.leakage_samples(m11, challenges)
    add10, _ = add_leakage(m10, product11)
    return np.stack((mul01, add00, mul11, add10))


def attack_matrix(matrix: tuple[int, int, int, int], challenges: np.ndarray,
                  leakage: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, bool]]:
    m00, m01, m10, m11 = matrix
    recovered01, _ = recover_product(m01, challenges, leakage[0])
    recovered11, _ = recover_product(m11, challenges, leakage[2])
    _, product01 = limb.leakage_samples(recovered01, challenges)
    _, product11 = limb.leakage_samples(recovered11, challenges)
    recovered00, _ = recover_addend(m00, product01, leakage[1])
    recovered10, _ = recover_addend(m10, product11, leakage[3])
    recovered = (recovered00, recovered01, recovered10, recovered11)
    return recovered, {name: got == want for name, got, want in zip(
        ("m00", "m01", "m10", "m11"), recovered, matrix)}


def oracle_batch(oracle: Path, pairs: list[tuple[int, int]]) -> list[tuple[str, ...]]:
    payload = "".join(f"{u:x} {v:x}\n" for u, v in pairs)
    result = subprocess.run([str(oracle)], input=payload, text=True, capture_output=True, timeout=60)
    limb.low.require(result.returncode == 0, f"ideal oracle failed ({result.returncode}): {result.stderr[-1200:]}")
    rows = [tuple(line.split("\t")) for line in result.stdout.splitlines()]
    limb.low.require(len(rows) == len(pairs) and all(len(row) == 8 for row in rows),
                     "unexpected ideal oracle output")
    return rows


def uv(matrix: tuple[int, int, int, int], challenge: int) -> tuple[int, int]:
    m00, m01, m10, m11 = matrix
    return ((m00 + m01 * challenge) & MASK192, (m10 + m11 * challenge) & MASK192)


def held_out_validation(oracle: Path, truth: tuple[int, int, int, int],
                        recovered: tuple[int, int, int, int], key_index: int) -> dict[str, object]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x484F4C44, key_index]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(HELD_OUT_CHALLENGES)]
    expected_pairs = [uv(truth, challenge) for challenge in challenges]
    recovered_pairs = [uv(recovered, challenge) for challenge in challenges]
    outputs = oracle_batch(oracle, expected_pairs + recovered_pairs)
    expected_outputs = outputs[:HELD_OUT_CHALLENGES]
    recovered_outputs = outputs[HELD_OUT_CHALLENGES:]
    return {
        "challenges": HELD_OUT_CHALLENGES,
        "challenge_sha256": hashlib.sha256(b"".join(value.to_bytes(24, "little") for value in challenges)).hexdigest(),
        "uv_exact": sum(a == b for a, b in zip(expected_pairs, recovered_pairs)),
        "ideal_and_split_exact": sum(a == b for a, b in zip(expected_outputs, recovered_outputs)),
    }


def self_check(matrix: tuple[int, int, int, int], oracle: Path) -> None:
    rng = np.random.default_rng(SEED)
    challenges = rng.integers(0, 1 << 32, size=(512, limb.LIMBS), dtype=np.uint32).astype(np.uint64)
    for addend, multiplier in ((matrix[0], matrix[1]), (matrix[2], matrix[3])):
        _, product = limb.leakage_samples(multiplier, challenges)
        _, output = add_leakage(addend, product)
        for trace in range(4):
            modeled = sum(int(output[trace, i]) << (32 * i) for i in range(limb.LIMBS))
            exact = (addend + multiplier * limb.integer(challenges[trace])) & MASK192
            limb.low.require(modeled == exact, "addition state differs from exact matrix row evaluation")
    recovered, _ = attack_matrix(matrix, challenges, simulate(matrix, challenges))
    limb.low.require(recovered == matrix, "noiseless full-matrix recovery failed")
    check = held_out_validation(oracle, matrix, recovered, 0)
    limb.low.require(check["uv_exact"] == HELD_OUT_CHALLENGES and
                     check["ideal_and_split_exact"] == HELD_OUT_CHALLENGES,
                     "held-out oracle self-check failed")


def trial(matrix: tuple[int, int, int, int], traces: int, sigma: float, key_index: int,
          oracle: Path, control: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    challenge_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index, int(sigma * 100)]))
    challenges = challenge_rng.integers(0, 1 << 32, size=(traces, limb.LIMBS), dtype=np.uint32).astype(np.uint64)
    observed = simulate(matrix, challenges) + noise_rng.normal(0.0, sigma, (4, limb.LIMBS, traces))
    recovered, entry_success = attack_matrix(matrix, challenges, observed)
    held_out = held_out_validation(oracle, matrix, recovered, key_index)
    positive = {
        "success": recovered == matrix,
        "entry_success": entry_success,
        "recovered_sha256": hashlib.sha256(b"".join(value.to_bytes(24, "little") for value in recovered)).hexdigest(),
        "held_out": held_out,
    }
    negative = None
    if control:
        permutation = noise_rng.permutation(traces)
        control_recovered, control_entries = attack_matrix(matrix, challenges[permutation], observed)
        negative = {"success": control_recovered == matrix, "entry_success": control_entries}
    return positive, negative


def sweep(matrices: list[tuple[int, int, int, int]], oracle: Path) -> list[dict[str, object]]:
    rows = []
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            positive = []
            negative = []
            for key_index, matrix in enumerate(matrices[:TARGET_KEYS]):
                result, control = trial(matrix, traces, sigma, key_index, oracle, key_index < CONTROL_KEYS)
                positive.append(result)
                if control is not None:
                    negative.append(control)
            rows.append({
                "traces": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(positive),
                "full_matrix_exact_recoveries": sum(item["success"] for item in positive),
                "entry_exact_recoveries": {
                    name: sum(item["entry_success"][name] for item in positive)
                    for name in ("m00", "m01", "m10", "m11")
                },
                "held_out_uv_exact": sum(item["held_out"]["uv_exact"] for item in positive),
                "held_out_ideal_and_split_exact": sum(
                    item["held_out"]["ideal_and_split_exact"] for item in positive),
                "held_out_trials": len(positive) * HELD_OUT_CHALLENGES,
                "shuffled_challenge_control_trials": len(negative),
                "shuffled_challenge_full_matrix_recoveries": sum(item["success"] for item in negative),
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
    limb.low.require(oracle.is_file() and oracle.stat().st_mode & 0o111, "ideal oracle is not executable")
    target = bind_target(elf)
    matrices = kat_matrices(kat)
    self_check(matrices[0], oracle)
    results = sweep(matrices, oracle)
    report = {
        "schema": "sqisign-r3-p500-m4-full-matrix-heldout-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "unblinded",
            "measurement_scope": "isolated_primitive",
            "claim_scope": "isk:r3_p500_m4_full_matrix_heldout_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": limb.low.sha256(Path(__file__)),
            "m01_helper_sha256": limb.low.sha256(Path(limb.__file__)),
            "low32_helper_sha256": limb.low.sha256(Path(limb.low.__file__)),
            "oracle_sha256": limb.low.sha256(oracle),
            "oracle_source_sha256": limb.low.sha256(Path(__file__).with_name("ideal_chall_oracle.c")),
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEED,
        },
        "source": {
            "commit": SOURCE_COMMIT,
            "sign_c_sha256": limb.low.sha256(source / "src/pqm4/sqisign_p500_27/m4f/sign.c"),
            "dim2_c_sha256": limb.low.sha256(source / "src/pqm4/sqisign_p500_27/m4f/dim2.c"),
            "id2iso_c_sha256": limb.low.sha256(source / "src/pqm4/sqisign_p500_27/m4f/id2iso.c"),
        },
        "target_elf_sha256": limb.low.ELF_SHA256,
        "kat_sha256": limb.low.KAT_SHA256,
        "parameter_set": "p500_27",
        "target": target,
        "known_input": "uniform public 192-bit challenge coefficient modeled as SHAKE256 random-oracle output",
        "secret_target": "all four 192-bit entries of mat_BAcan_to_BA0_two",
        "target_values_sha256": hashlib.sha256(b"".join(
            value.to_bytes(24, "little") for matrix in matrices[:TARGET_KEYS] for value in matrix)).hexdigest(),
        "window": "24 perfectly aligned dynamic stores per trace: six limbs for each variable multiply and row addition",
        "model": "HW of six m01/m11 multiply stores and six m00/m10 row-add stores plus independent Gaussian noise",
        "attack": (f"sequential 32-bit limb recovery: absolute-Pearson byte prefixes for all entries; "
                   f"calibrated HW-Gaussian likelihood at each addend word's 32-bit stage; "
                   f"multiply beam={limb.BEAM}, add beam={ADD_BEAM}"),
        "held_out": "independent challenges unused by recovery; official host-ref id2iso conversion from the pinned source",
        "sweep": results,
        "allowed_statement": "Under the declared synthetic model, the reported KAT matrices were recovered and tested on independent challenges at the reported rates.",
        "blocked_statements": [
            "physical STM32F4 leakage, trace alignment, or full signing transcripts were modeled",
            "the secret ideal (N,x,y), complete signing key, or a forgery was recovered",
            "host-ref ideal conversion equality is physical M4 attack evidence",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged_path = args.out.with_suffix(args.out.suffix + ".tmp")
    staged_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged_path.chmod(0o600)
    staged_path.replace(args.out)
    for row in results:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces']:4d} "
              f"matrix={row['full_matrix_exact_recoveries']}/{row['secret_trials']} "
              f"uv={row['held_out_uv_exact']}/{row['held_out_trials']} "
              f"ideal={row['held_out_ideal_and_split_exact']}/{row['held_out_trials']} "
              f"control={row['shuffled_challenge_full_matrix_recoveries']}/"
              f"{row['shuffled_challenge_control_trials']}")


if __name__ == "__main__":
    main()
