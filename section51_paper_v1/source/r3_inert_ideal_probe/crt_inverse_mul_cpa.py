#!/usr/bin/env python3
"""Recover N through the first CRT inverse-multiplication row HW model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

import challenge_matrix_cpa192 as product_attack
import crt_secret_low192 as low


SEED = 0x43525455
TRACE_COUNTS = (1024, 2048)
NOISE_SIGMAS = (2.0, 4.0)
TARGET_KEY_INDICES = (0, 2)
HELD_OUT_CHALLENGES = 64
LIMBS = 6
MASK32 = np.uint64(0xFFFFFFFF)
MOD192 = 1 << 192
MUL_STORE_PC = 0x0801F2EE
CRT_MUL_CALL_PC = 0x08025AA2


def bind_target(elf: Path) -> dict[str, str]:
    target = low.bind_target(elf)
    crt = low.matrix_attack.limb.low.checked(
        ["arm-none-eabi-objdump", "-d", "--disassemble=ibz_crt", str(elf)], elf.parent)
    mul = low.matrix_attack.limb.low.checked(
        ["arm-none-eabi-objdump", "-d", "--disassemble=ibz_mul", str(elf)], elf.parent)
    low.require(re.search(r"(?m)^\s*8025aa2:.*\bbl\s+\S+ <ibz_mul>$", crt) is not None,
                "CRT inverse-product call binding mismatch")
    low.require(re.search(r"(?m)^\s*801f2ee:\s+6003\s+str\s+r3, \[r0, #0\]$", mul) is not None,
                "ibz_mul inner store binding mismatch")
    target.update({
        "crt_inverse_product_call": f"0x{CRT_MUL_CALL_PC:08x}",
        "ibz_mul_inner_store": f"0x{MUL_STORE_PC:08x}",
    })
    return target


def inverse_parameters(n: int) -> tuple[int, int, int]:
    n_low = n % MOD192
    k = (-pow(n_low, -1, MOD192)) % MOD192
    c, remainder = divmod(k * n_low + 1, MOD192)
    low.require(remainder == 0 and k & 1, "invalid CRT inverse decomposition")
    u = pow(MOD192, -1, n)
    low.require(MOD192 * u - 1 == k * n, "CRT inverse multiplier identity mismatch")
    return u, k, c


def reconstruct_n(n_low: int, u_low: int) -> int:
    k = (-pow(n_low, -1, MOD192)) % MOD192
    c, remainder = divmod(k * n_low + 1, MOD192)
    low.require(remainder == 0 and k & 1, "invalid known-low reconstruction constants")
    high = ((u_low - c) * pow(k, -1, MOD192)) % MOD192
    return n_low + MOD192 * high


def known_diff0(dataset: dict[str, object]) -> np.ndarray:
    public_x = dataset["public_x"][:, 0]
    split_v = dataset["split_v"]
    representations = dataset["representations"]
    secret_x = np.array(
        [representations[int(value)][0] & 0xFFFFFFFF for value in split_v], dtype=np.uint64)
    for index, value in enumerate(split_v):
        public_value = sum(
            int(word) << (32 * limb) for limb, word in enumerate(dataset["public_x"][index]))
        low.require(representations[int(value)][0] > public_value,
                    "CRT difference is not positive")
    return (secret_x - public_x) & MASK32


def leakage_samples(u: int, diff0: np.ndarray) -> np.ndarray:
    leakage = np.empty((LIMBS, diff0.size), dtype=np.float64)
    carry = np.zeros(diff0.size, dtype=np.uint64)
    for limb, word in enumerate(low.words(u)):
        total = diff0 * word + carry
        leakage[limb] = low.matrix_attack.limb.low.hw32(total & MASK32)
        carry = total >> np.uint64(32)
    return leakage


def score_candidates(candidates: np.ndarray, diff0: np.ndarray, carry: np.ndarray,
                     leakage: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    scores = np.empty(candidates.size, dtype=np.float64)
    centered = leakage - leakage.mean()
    leakage_energy = float(centered @ centered)
    low.require(leakage_energy > 0.0, "constant inverse-product leakage vector")
    for start in range(0, candidates.size, product_attack.CHUNK):
        stop = min(start + product_attack.CHUNK, candidates.size)
        stored = (candidates[start:stop, None] * diff0[None, :] + carry[None, :]) & mask
        predicted = low.matrix_attack.limb.low.hw32(stored).astype(np.float64)
        if bits == 32:
            residual = predicted - leakage[None, :]
            scores[start:stop] = -np.mean(residual * residual, axis=1)
        else:
            predicted -= predicted.mean(axis=1, keepdims=True)
            numerator = predicted @ centered
            denominator = np.sqrt(
                np.einsum("ij,ij->i", predicted, predicted) * leakage_energy)
            scores[start:stop] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return scores


def attack_word(diff0: np.ndarray, carry: np.ndarray, leakage: np.ndarray,
                truth: int) -> dict[str, object]:
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    final_scores = np.empty(0)
    for stage in range(4):
        bits = 8 * (stage + 1)
        if stage:
            next_byte = np.arange(256, dtype=np.uint64) << (8 * stage)
            prefixes = (prefixes[:, None] | next_byte[None, :]).reshape(-1)
        scores = score_candidates(prefixes, diff0, carry, leakage, bits)
        order = np.lexsort((prefixes, -scores))
        true_prefix = np.uint64(truth & ((1 << bits) - 1))
        positions = np.flatnonzero(prefixes[order] == true_prefix)
        stages.append({
            "bits": bits,
            "truth_survived": bool(positions.size),
            "truth_rank_within_expanded_beam": int(positions[0] + 1) if positions.size else None,
        })
        keep = order[:min(product_attack.BEAM, order.size)]
        prefixes = prefixes[keep]
        final_scores = scores[keep]
    return {
        "recovered": int(prefixes[0]),
        "success": int(prefixes[0]) == truth,
        "best_score": float(final_scores[0]),
        "stages": stages,
    }


def attack(u: int, diff0: np.ndarray, leakage: np.ndarray) -> dict[str, object]:
    truth_words = low.words(u)
    carry = np.zeros(diff0.size, dtype=np.uint64)
    recovered_words = []
    limbs = []
    for limb in range(LIMBS):
        result = attack_word(diff0, carry, leakage[limb], int(truth_words[limb]))
        recovered_words.append(result["recovered"])
        limbs.append(result)
        carry = (diff0 * np.uint64(result["recovered"]) + carry) >> np.uint64(32)
    recovered = sum(word << (32 * index) for index, word in enumerate(recovered_words))
    return {
        "u_low192": recovered,
        "u_low192_success": recovered == u % MOD192,
        "correct_limbs": sum(result["success"] for result in limbs),
        "per_limb": limbs,
    }


def algebra_self_check(secrets: list[tuple[int, int, int]]) -> None:
    for n, _, _ in secrets:
        low.require(n < MOD192 * MOD192, "N exceeds unique reconstruction range")
        u, _, _ = inverse_parameters(n)
        low.require(reconstruct_n(n % MOD192, u % MOD192) == n,
                    "u-low does not reconstruct the KAT norm")


def target_self_check(secret: tuple[int, int, int], matrix: tuple[int, int, int, int],
                      oracle: Path) -> None:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x53454C46]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(512)]
    dataset = low.make_dataset(secret, matrix, challenges, oracle)
    diff0 = known_diff0(dataset)
    u, _, _ = inverse_parameters(secret[0])
    result = attack(u, diff0, leakage_samples(u, diff0))
    low.require(result["u_low192_success"], "noiseless inverse-product recovery failed")
    low.require(reconstruct_n(secret[0] % MOD192, result["u_low192"]) == secret[0],
                "noiseless norm reconstruction failed")


def held_out_validation(secret: tuple[int, int, int], matrix: tuple[int, int, int, int],
                        recovered_u: int, key_index: int, traces: int, sigma: float,
                        oracle: Path) -> dict[str, int]:
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, 0x484F4C44, key_index, traces, int(sigma * 100)]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(HELD_OUT_CHALLENGES)]
    dataset = low.make_dataset(secret, matrix, challenges, oracle)
    diff0 = known_diff0(dataset)
    true_u, _, _ = inverse_parameters(secret[0])
    expected = leakage_samples(true_u, diff0)
    replayed = leakage_samples(recovered_u, diff0)
    return {
        "store_hw_exact": int(np.count_nonzero(expected == replayed)),
        "store_hw_total": int(expected.size),
    }


def trial(secret: tuple[int, int, int], matrix: tuple[int, int, int, int], traces: int,
          sigma: float, key_index: int, oracle: Path,
          controls: bool) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, traces, key_index]))
    challenges = [int.from_bytes(rng.bytes(24), "little") for _ in range(traces)]
    dataset = low.make_dataset(secret, matrix, challenges, oracle)
    diff0 = known_diff0(dataset)
    n = secret[0]
    u, k, _ = inverse_parameters(n)
    ideal = leakage_samples(u, diff0)
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([SEED, traces, key_index, int(sigma * 100)]))
    observed = ideal + noise_rng.normal(0.0, sigma, ideal.shape)
    positive = attack(u, diff0, observed)
    positive["n_recovered"] = reconstruct_n(n % MOD192, positive["u_low192"])
    positive["n_success"] = positive["n_recovered"] == n
    positive["n_bits"] = n.bit_length()
    positive["u_bits"] = u.bit_length()
    positive["k_bits"] = k.bit_length()
    positive["held_out"] = held_out_validation(
        secret, matrix, positive["u_low192"], key_index, traces, sigma, oracle)
    negative = {}
    if controls:
        shuffled = attack(u, diff0[noise_rng.permutation(traces)], observed)
        wrong_carry = np.zeros(traces, dtype=np.uint64)
        for word in low.words(u):
            wrong_carry = (diff0 * word + wrong_carry) >> np.uint64(32)
        wrong_word = np.uint64((u >> (32 * LIMBS)) & 0xFFFFFFFF)
        wrong_store = low.matrix_attack.limb.low.hw32(
            (diff0 * wrong_word + wrong_carry) & MASK32).astype(np.float64)
        wrong_window = attack(u, diff0, np.tile(wrong_store, (LIMBS, 1)))
        noise_only = attack(u, diff0, noise_rng.normal(16.0, sigma, ideal.shape))
        for name, item in (("shuffled", shuffled), ("wrong_window", wrong_window),
                           ("noise_only", noise_only)):
            item["n_recovered"] = reconstruct_n(n % MOD192, item["u_low192"])
            item["n_success"] = item["n_recovered"] == n
            negative[name] = item
    return positive, negative


def sweep(secrets: list[tuple[int, int, int]], matrices: list[tuple[int, int, int, int]],
          oracle: Path) -> list[dict[str, object]]:
    rows = []
    for sigma in NOISE_SIGMAS:
        for traces in TRACE_COUNTS:
            positive = []
            controls = []
            for key_index in TARGET_KEY_INDICES:
                result, control = trial(secrets[key_index], matrices[key_index], traces, sigma,
                                        key_index, oracle, key_index == TARGET_KEY_INDICES[0])
                positive.append(result)
                if control:
                    controls.append(control)
            rows.append({
                "traces": traces,
                "sigma_hw_units": sigma,
                "secret_trials": len(positive),
                "u_low192_exact": sum(item["u_low192_success"] for item in positive),
                "n_exact": sum(item["n_success"] for item in positive),
                "per_limb_exact": [
                    sum(item["per_limb"][limb]["success"] for item in positive)
                    for limb in range(LIMBS)
                ],
                "stage_truth_survival": [[
                    sum(item["per_limb"][limb]["stages"][stage]["truth_survived"]
                        for item in positive)
                    for stage in range(4)] for limb in range(LIMBS)],
                "held_out_store_hw_exact": sum(
                    item["held_out"]["store_hw_exact"] for item in positive),
                "held_out_store_hw_total": sum(
                    item["held_out"]["store_hw_total"] for item in positive),
                "negative_control_trials": len(controls),
                "shuffled_n_exact": sum(item["shuffled"]["n_success"] for item in controls),
                "wrong_window_n_exact": sum(
                    item["wrong_window"]["n_success"] for item in controls),
                "noise_only_n_exact": sum(item["noise_only"]["n_success"] for item in controls),
                "target_bit_lengths": [
                    {name: item[name] for name in ("n_bits", "u_bits", "k_bits")}
                    for item in positive
                ],
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
    elf, kat, oracle, source = (
        path.resolve() for path in (args.elf, args.kat, args.oracle, args.source))
    low.require(oracle.is_file() and oracle.stat().st_mode & 0o111,
                "ideal oracle is not executable")
    target = bind_target(elf)
    secrets = low.kat_secrets(kat)
    matrices = low.matrix_attack.kat_matrices(kat)
    algebra_self_check(secrets)
    target_self_check(secrets[TARGET_KEY_INDICES[0]], matrices[TARGET_KEY_INDICES[0]], oracle)
    results = sweep(secrets, matrices, oracle)
    report = {
        "schema": "sqisign-r3-p500-m4-crt-inverse-mul-cpa-v1",
        "evidence_contract": {
            "evidence_origin": "simulated",
            "evidence_purpose": "diagnostic",
            "blinding": "unblinded",
            "measurement_scope": "instrumented_subpath",
            "claim_scope": "isk:r3_p500_m4_crt_inverse_mul_n_recovery_diagnostic_v1",
        },
        "execution_binding": {
            "script_sha256": low.matrix_attack.limb.low.sha256(Path(__file__)),
            "crt_low_helper_sha256": low.matrix_attack.limb.low.sha256(Path(low.__file__)),
            "product_cpa_helper_sha256": low.matrix_attack.limb.low.sha256(
                Path(product_attack.__file__)),
            "full_matrix_helper_sha256": low.matrix_attack.limb.low.sha256(
                Path(low.matrix_attack.__file__)),
            "base_cpa_helper_sha256": low.matrix_attack.limb.low.sha256(
                Path(low.matrix_attack.limb.low.__file__)),
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
            "protocol_c_sha256": low.matrix_attack.limb.low.sha256(
                source / "src/pqm4/sqisign_p500_27/m4f/protocol.c"),
        },
        "target_elf_sha256": low.matrix_attack.limb.low.ELF_SHA256,
        "kat_sha256": low.matrix_attack.limb.low.KAT_SHA256,
        "parameter_set": "p500_27",
        "target_key_indices": list(TARGET_KEY_INDICES),
        "target": target,
        "dynamic_scope": {
            "intersection_x_crt_caller": target["intersection_x_crt_call"],
            "ibz_crt_inverse_product_call": f"0x{CRT_MUL_CALL_PC:08x}",
            "ibz_mul_outer_index": 0,
            "ibz_mul_inner_indices": list(range(LIMBS)),
            "ibz_mul_store_pc": f"0x{MUL_STORE_PC:08x}",
        },
        "prerequisite": ("exact N mod 2^192 and split-conditioned secret x mod 2^192 from the "
                         "previous CRT-subtraction stage"),
        "known_input": ("low word of positive diff_a=a2-a1, using public ideal_chall_two.x and "
                        "the prerequisite split-conditioned a2 low residue"),
        "secret_target": "six low words of u=(2^192)^(-1) mod N, followed by exact N reconstruction",
        "window": ("first ibz_mul outer row i=0 and its first six dynamic inner stores at the "
                   "single bound store PC"),
        "model": ("six perfectly aligned samples per trace: "
                  "HW(low32(diff0*u_j+carry_j)) plus independent Gaussian noise"),
        "attack": ("low-to-high 8/16/24-bit signed-Pearson CPA and final 32-bit HW likelihood "
                   "with beam=64; each recovered word determines the next carry"),
        "reconstruction": ("for the checked N<2^384 bounded targets: k=-N_low^(-1) mod "
                           "2^192; c=(k*N_low+1)/2^192; "
                           "N_high=(u_low-c)*k^(-1) mod 2^192"),
        "target_values_sha256": hashlib.sha256(b"".join(
            value.to_bytes(low.FP_ENCODED_BYTES, "little")
            for index in TARGET_KEY_INDICES for value in secrets[index])).hexdigest(),
        "sweep": results,
        "allowed_statement": ("Under the declared synthetic model, exact prior low-residue "
                              "prerequisite, and checked N<2^384 bound, the reported KAT norms "
                              "were reconstructed exactly."),
        "blocked_statements": [
            "the prerequisite low-residue recovery and this second stage were jointly error-propagated",
            "physical STM32F4 leakage, trigger alignment, adjacent-instruction mixing, or HD leakage were modeled",
            "the full signing key or a forgery was recovered",
            "N uniqueness holds without the checked N<2^384 target bound",
            "the result generalizes beyond the fixed KAT keys, ELF, source files, and leakage model",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(args.out.suffix + ".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    staged.chmod(0o600)
    staged.replace(args.out)
    for row in results:
        print(f"sigma={row['sigma_hw_units']:.1f} traces={row['traces']:4d} "
              f"u192={row['u_low192_exact']}/{row['secret_trials']} "
              f"N={row['n_exact']}/{row['secret_trials']} "
              f"limbs={row['per_limb_exact']} "
              f"controls={row['shuffled_n_exact']},{row['wrong_window_n_exact']},"
              f"{row['noise_only_n_exact']}/{row['negative_control_trials']}", flush=True)


if __name__ == "__main__":
    main()
