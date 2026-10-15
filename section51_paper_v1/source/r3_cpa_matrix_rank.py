#!/usr/bin/env python3
"""Truth-free ranking for the frozen first-CPA matrix-M procedure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange


KAT_SHA256 = "986a28c7752428ba0eff2784af4f07b996d259646b192b16407a647a584d7537"
EVENTS, LIMBS, MASK32, BEAM_MUL, BEAM_ADD, LIKELIHOOD_CHUNK = 24, 6, np.uint64(0xFFFFFFFF), 64, 256, 512
WRONG_START, WRONG_STOP, SHUFFLE_SEED = 36864, 45056, 2026100210
TIE_TOLERANCE = np.finfo(np.float64).eps * 32


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def helpers() -> object:
    root = Path.cwd() / "r3_inert_ideal_probe"
    require(root.is_dir(), "run from the repository root containing r3_inert_ideal_probe")
    sys.path.insert(0, str(root))
    import challenge_matrix_cpa192 as limb
    return limb


def first_two_matrices(kat: Path, limb: object) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Decode only calibration-key matrices; no key-2 material is parsed."""
    require(digest(kat) == KAT_SHA256, "Round-3 p500_27 KAT hash mismatch")
    matrices = []
    pattern = re.compile(r"^sk = ([0-9A-F]+)$")
    offset = limb.low.M01_OFFSET - 24
    with kat.open(encoding="ascii") as source:
        for line in source:
            match = pattern.match(line.rstrip("\n"))
            if not match:
                continue
            key = bytes.fromhex(match.group(1))
            require(len(key) == 417, "unexpected KAT secret-key length")
            matrices.append(tuple(int.from_bytes(key[offset + 24 * i:offset + 24 * (i + 1)], "little")
                                  for i in range(4)))
            if len(matrices) == 2:
                break
    require(len(matrices) == 2, "KAT lacks calibration keys")
    return matrices[0], matrices[1]


def inputs(run: Path) -> tuple[Path, Path, np.ndarray, list[dict[str, object]]]:
    traces_path = run / "traces.npy"
    schedule_path = run / "provenance" / "schedule.jsonl"
    if not schedule_path.is_file():
        schedule_path = run / "schedule.jsonl"
    require(traces_path.is_file() and schedule_path.is_file(), "run lacks traces.npy or matching schedule.jsonl")
    traces = np.load(traces_path, mmap_mode="r")
    require(traces.ndim == 2 and traces.shape[1] >= WRONG_STOP and np.isfinite(traces).all(),
            "traces must be finite (n, >=45056)")
    rows = [json.loads(line) for line in schedule_path.read_text().splitlines() if line]
    require(len(rows) == traces.shape[0], "trace/schedule row count mismatch")
    require(all({"key_index", "block", "challenge_hex"} <= row.keys() for row in rows),
            "schedule lacks key, block, or challenge")
    return traces_path, schedule_path, traces, rows


def words(rows: list[dict[str, object]]) -> np.ndarray:
    values = [np.frombuffer(bytes.fromhex(str(row["challenge_hex"])), dtype="<u4") for row in rows]
    require(all(value.shape == (LIMBS,) for value in values), "challenge is not 24 bytes")
    return np.asarray(values, dtype=np.uint64)


def product_labels(secret: int, challenges: np.ndarray, limb: object) -> tuple[np.ndarray, np.ndarray]:
    return limb.leakage_samples(secret, challenges)


def add_labels(addend: int, product: np.ndarray, limb: object) -> tuple[np.ndarray, np.ndarray]:
    carry = np.zeros(product.shape[0], dtype=np.uint64)
    output = np.empty((product.shape[0], LIMBS), dtype=np.uint64)
    labels = np.empty((product.shape[0], LIMBS), dtype=np.float64)
    for row, word in enumerate(limb.words(addend)):
        total = product[:, row] + np.uint64(word) + carry
        output[:, row] = total & MASK32
        labels[:, row] = limb.low.hw32(output[:, row])
        carry = total >> np.uint64(32)
    return labels, output


@njit(inline="always")
def _hw32(value: np.uint64) -> int:
    value &= np.uint64(0xFFFFFFFF)
    value -= (value >> np.uint64(1)) & np.uint64(0x55555555)
    value = (value & np.uint64(0x33333333)) + ((value >> np.uint64(2)) & np.uint64(0x33333333))
    return int(((((value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F)) * np.uint64(0x01010101)) >> np.uint64(24)) & np.uint64(0x3F))


@njit(parallel=True)
def _pearson_scores(candidates: np.ndarray, base: np.ndarray, challenge0: np.ndarray,
                    centered: np.ndarray, mask: np.uint64, multiply: bool, leakage_energy: float) -> np.ndarray:
    scores = np.empty(candidates.size, dtype=np.float64)
    rows = base.size
    for index in prange(candidates.size):
        total = squared = numerator = 0.0
        candidate = candidates[index]
        for row in range(rows):
            stored = (candidate * challenge0[row] + base[row] if multiply else candidate + base[row]) & mask
            prediction = _hw32(stored)
            total += prediction
            squared += prediction * prediction
            numerator += prediction * centered[row]
        denominator = np.sqrt((squared - total * total / rows) * leakage_energy)
        scores[index] = abs(numerator) / denominator if denominator > 0.0 else 0.0
    return scores


def pearson(candidates: np.ndarray, base: np.ndarray, challenge0: np.ndarray, leakage: np.ndarray,
            bits: int, multiply: bool, limb: object) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    centered = leakage - leakage.mean()
    return _pearson_scores(candidates, base, challenge0, centered, mask, multiply, float(centered @ centered))


def precise_pearson(candidates: np.ndarray, base: np.ndarray, challenge0: np.ndarray, leakage: np.ndarray,
                    bits: int, multiply: bool, limb: object) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    centered = leakage - leakage.mean()
    prediction = ((candidates[:, None] * challenge0[None, :] + base[None, :] if multiply
                   else candidates[:, None] + base[None, :]) & mask)
    prediction = limb.low.hw32(prediction).astype(np.float64)
    prediction -= prediction.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.einsum("ij,ij->i", prediction, prediction) * float(centered @ centered))
    return np.divide(np.abs(prediction @ centered), denominator,
                     out=np.zeros(candidates.size), where=denominator > 0)


def likelihood(candidates: np.ndarray, base: np.ndarray, challenge0: np.ndarray, leakage: np.ndarray,
               alpha: float, beta: float, sigma: float, multiply: bool, limb: object) -> np.ndarray:
    require(sigma > 0.0, "Gaussian fit has non-positive sigma")
    scores = np.empty(candidates.size)
    for start in range(0, candidates.size, LIKELIHOOD_CHUNK):
        stop = min(start + LIKELIHOOD_CHUNK, candidates.size)
        stored = ((candidates[start:stop, None] * challenge0[None, :] + base[None, :] if multiply
                   else candidates[start:stop, None] + base[None, :]) & MASK32)
        predicted = alpha * limb.low.hw32(stored).astype(np.float64) + beta
        scores[start:stop] = -0.5 * np.mean((leakage[None, :] - predicted) ** 2, axis=1) / (sigma * sigma) - np.log(sigma)
    return scores


def expand(prefixes: np.ndarray, stage: int) -> np.ndarray:
    return (prefixes[:, None] | (np.arange(256, dtype=np.uint64)[None, :] << np.uint64(8 * stage))).reshape(-1)


def rank_word(base: np.ndarray, challenge0: np.ndarray, leakage: np.ndarray, beam: int, multiply: bool,
              limb: object, alpha: float | None = None, beta: float | None = None,
              sigma: float | None = None, return_final_beam: bool = False) -> dict[str, object]:
    charge_rank()
    prefixes = np.arange(256, dtype=np.uint64)
    stages = []
    for stage in range(4):
        if stage:
            prefixes = expand(prefixes, stage)
        bits = 8 * (stage + 1)
        require(alpha is None or (beta is not None and sigma is not None), "incomplete Gaussian fit")
        scores = (likelihood(prefixes, base, challenge0, leakage, alpha, beta, sigma, multiply, limb)
                  if bits == 32 and alpha is not None else
                  pearson(prefixes, base, challenge0, leakage, bits, multiply, limb))
        order = np.lexsort((prefixes, -scores))
        if alpha is None or bits != 32:
            cutoff = scores[order[min(beam, order.size) - 1]]
            uncertain = np.flatnonzero(np.abs(scores - cutoff) <= TIE_TOLERANCE)
            exact = np.unique(np.concatenate((order[:min(beam, order.size)], uncertain)))
            scores[exact] = precise_pearson(prefixes[exact], base, challenge0, leakage, bits, multiply, limb)
            order = np.lexsort((prefixes, -scores))
        kept = order[:min(beam, order.size)]
        top = [{"value": int(prefixes[index]), "score": float(scores[index])} for index in kept[:8]]
        stages.append({"bits": bits, "metric": "gaussian_likelihood" if bits == 32 and alpha is not None else "absolute_pearson",
                       "top_k": top, "margin": float(scores[kept[0]] - scores[kept[1]]) if kept.size > 1 else None})
        prefixes = prefixes[kept]
    result = {"top1": int(prefixes[0]), "final_beam": [int(value) for value in prefixes[:8]], "stages": stages}
    if return_final_beam:
        result["scored_final_beam"] = [{"value": int(value), "score": float(score)}
                                       for value, score in zip(prefixes, scores[kept])]
    return result


def charge_rank() -> None:
    budget = os.environ.get("R3_RANK_BUDGET_PATH")
    if not budget:
        return
    path = Path(budget)
    state = json.loads(path.read_text())
    calls = int(state["rank_word_calls"])
    limit = int(state["rank_word_limit"])
    if calls >= limit:
        state["exhausted"] = True
        path.write_text(json.dumps(state) + "\n")
        raise RuntimeError("budget_exceeded: global rank_word call limit")
    temporary = path.with_suffix(".tmp")
    state["rank_word_calls"] = calls + 1
    temporary.write_text(json.dumps(state) + "\n")
    temporary.replace(path)


def rank_product(challenges: np.ndarray, samples: np.ndarray, limb: object) -> tuple[int, dict[str, object], np.ndarray]:
    state = np.zeros((challenges.shape[0], 2 * LIMBS), dtype=np.uint64)
    selected, metadata = [], []
    for row in range(LIMBS):
        result = rank_word(state[:, row], challenges[:, 0], samples[:, row], BEAM_MUL, True, limb)
        selected.append(result["top1"])
        metadata.append(result)
        limb.apply_row(state, result["top1"], challenges, row)
    return sum(value << (32 * row) for row, value in enumerate(selected)), {"limbs": metadata}, state


def rank_addend(product: np.ndarray, samples: np.ndarray, fit: list[dict[str, float]], limb: object) -> tuple[int, dict[str, object]]:
    carry = np.zeros(product.shape[0], dtype=np.uint64)
    selected, metadata = [], []
    challenge0 = np.zeros(product.shape[0], dtype=np.uint64)
    for row in range(LIMBS):
        model = fit[row]
        result = rank_word(product[:, row] + carry, challenge0, samples[:, row], BEAM_ADD, False, limb,
                           model["alpha"], model["beta"], model["sigma"])
        selected.append(result["top1"])
        metadata.append(result)
        carry = (product[:, row] + np.uint64(result["top1"]) + carry) >> np.uint64(32)
    return sum(value << (32 * row) for row, value in enumerate(selected)), {"limbs": metadata}


def fit_line(labels: np.ndarray, samples: np.ndarray) -> list[dict[str, float]]:
    fits = []
    require(labels.shape == samples.shape and labels.shape[1] == 2 * LIMBS,
            "expected both six-limb addition rows for analog calibration")
    for event in range(labels.shape[1]):
        design = np.column_stack((labels[:, event], np.ones(labels.shape[0])))
        alpha, beta = np.linalg.lstsq(design, samples[:, event], rcond=None)[0]
        residual = samples[:, event] - (alpha * labels[:, event] + beta)
        fits.append({"alpha": float(alpha), "beta": float(beta), "sigma": float(np.sqrt(np.mean(residual * residual))),
                     "residual_rms": float(np.sqrt(np.mean(residual * residual)))})
    return fits


def transfer(key0: list[dict[str, float]], key1: list[dict[str, float]]) -> dict[str, object]:
    diagnostics = []
    for first, second in zip(key0, key1):
        ratio = abs(second["alpha"] / first["alpha"]) if first["alpha"] else float("inf")
        diagnostics.append({"alpha_sign_agrees": bool(np.sign(first["alpha"]) == np.sign(second["alpha"])),
                            "alpha_magnitude_ratio": ratio,
                            "rms_ratio": second["residual_rms"] / first["residual_rms"] if first["residual_rms"] else float("inf")})
    passed = all(item["alpha_sign_agrees"] and .5 <= item["alpha_magnitude_ratio"] <= 2 and item["rms_ratio"] <= 2
                 for item in diagnostics)
    return {"passed": passed, "events": diagnostics}


def block_permutation(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = np.empty(blocks.size, dtype=np.int64)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        permutation[indices] = rng.permutation(indices)
    return permutation


def rank_matrix(challenges: np.ndarray, samples: np.ndarray, add_fit0: list[dict[str, float]], limb: object) -> dict[str, object]:
    m01, meta01, product01 = rank_product(challenges, samples[:, :6], limb)
    m11, meta11, product11 = rank_product(challenges, samples[:, 12:18], limb)
    m00, meta00 = rank_addend(product01, samples[:, 6:12], add_fit0[:6], limb)
    m10, meta10 = rank_addend(product11, samples[:, 18:24], add_fit0[6:], limb)
    return {"top1_matrix_candidate": {"m00": hex(m00), "m01": hex(m01), "m10": hex(m10), "m11": hex(m11)},
            "entries": {"m01": meta01, "m11": meta11, "m00": meta00, "m10": meta10}}


def calibration(calibration: Path) -> dict[str, object]:
    value = json.loads(calibration.read_text())
    require(value.get("schema") == "sqisign-r3-first-cpa-matrix-calibration-v1", "wrong calibration schema")
    gate = value.get("gate", {})
    require(gate.get("calibration_accepted") is True and gate.get("positive_24_of_24") is True and
            gate.get("controls_rejected") is True, "calibration gates were not accepted")
    poi = value.get("key0_discovery", {}).get("pois")
    require(isinstance(poi, list) and len(poi) == EVENTS and all(isinstance(x, int) for x in poi), "invalid frozen POIs")
    require(all(a + 4 <= b for a, b in zip(poi, poi[1:])), "POI gap is not frozen at four samples")
    return value


def write_output(out: Path, report: dict[str, object]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(out)


def self_test() -> None:
    class Low:
        @staticmethod
        def hw32(values: np.ndarray) -> np.ndarray:
            return sum((values >> np.uint64(bit)) & np.uint64(1) for bit in range(32))
    class Limb:
        low = Low()
    rng = np.random.default_rng(12)
    challenge = rng.integers(0, 1 << 32, 96, dtype=np.uint64)
    accumulator = rng.integers(0, 1 << 32, 96, dtype=np.uint64)
    secret = np.uint64(173)
    prefix_leakage = Low.hw32((secret * challenge + accumulator) & np.uint64(255)).astype(float)
    ranked = rank_word(accumulator, challenge, prefix_leakage, 64, True, Limb())
    assert ranked["stages"][0]["top_k"][0]["value"] == int(secret)
    candidate = np.arange(256, dtype=np.uint64)
    old_model = pearson(candidate, np.zeros(96, dtype=np.uint64), challenge, prefix_leakage, 8, True, Limb())
    assert int(candidate[np.argmax(old_model)]) != int(secret)
    leakage = Low.hw32((secret * challenge + accumulator) & MASK32).astype(float)
    likelihood_scores = likelihood(candidate, accumulator, challenge, leakage, 1.0, 0.0, 1.0, True, Limb())
    assert int(candidate[np.argmax(likelihood_scores)]) == int(secret)

    def reference(bits: int, multiply: bool) -> np.ndarray:
        mask = np.uint64((1 << bits) - 1)
        centered = leakage - leakage.mean()
        prediction = ((wide[:, None] * challenge[None, :] + accumulator[None, :] if multiply
                       else wide[:, None] + accumulator[None, :]) & mask)
        prediction = Low.hw32(prediction).astype(np.float64)
        prediction -= prediction.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.einsum("ij,ij->i", prediction, prediction) * float(centered @ centered))
        return np.divide(np.abs(prediction @ centered), denominator,
                         out=np.zeros(wide.size), where=denominator > 0)

    wide = np.arange(4096, dtype=np.uint64)
    for bits, multiply in ((8, True), (16, True), (24, False), (32, False)):
        expected = reference(bits, multiply)
        optimized = pearson(wide, accumulator, challenge, leakage, bits, multiply, Limb())
        assert np.allclose(expected, optimized, rtol=1e-12, atol=1e-12)
    labels = rng.normal(size=(96, 2 * LIMBS))
    samples = labels * 2.0 + rng.normal(scale=0.01, size=labels.shape)
    fitted = fit_line(labels, samples)
    assert len(fitted) == 2 * LIMBS and all(item["alpha"] > 1.9 for item in fitted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--kat", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
        return
    require(all(value is not None for value in (args.run, args.kat, args.calibration, args.out)),
            "--run, --kat, --calibration, and --out are required")
    run, kat, calibration_path, out = args.run.resolve(), args.kat.resolve(), args.calibration.resolve(), args.out.resolve()
    require(run.is_dir() and kat.is_file() and calibration_path.is_file(), "missing input")
    require(not out.exists() and not out.with_suffix(out.suffix + ".tmp").exists(), "refusing to overwrite output")
    cal = calibration(calibration_path)
    require(cal.get("inputs", {}).get("kat_sha256") == digest(kat), "calibration KAT binding mismatch")
    limb = helpers()
    traces_path, schedule_path, traces, rows = inputs(run)
    require(cal.get("inputs", {}).get("traces_sha256") == digest(traces_path) and
            cal.get("inputs", {}).get("schedule_sha256") == digest(schedule_path),
            "calibration run binding mismatch")
    indices = {key: np.array([i for i, row in enumerate(rows) if int(row["key_index"]) == key]) for key in (0, 1, 2)}
    require(len(indices[0]) == len(indices[1]) == 512 and len(indices[2]) == 1024, "unexpected key trace counts")
    matrix0, matrix1 = first_two_matrices(kat, limb)
    pois = cal["key0_discovery"]["pois"]
    selected = {key: (words([rows[i] for i in indices[key]]), np.asarray(traces[indices[key]][:, pois], dtype=float))
                for key in (0, 1)}
    add_fits = []
    for matrix, key in ((matrix0, 0), (matrix1, 1)):
        challenge, samples = selected[key]
        _, product01 = product_labels(matrix[1], challenge, limb)
        labels00, _ = add_labels(matrix[0], product01, limb)
        _, product11 = product_labels(matrix[3], challenge, limb)
        labels10, _ = add_labels(matrix[2], product11, limb)
        add_fits.append(fit_line(np.column_stack((labels00, labels10)), np.column_stack((samples[:, 6:12], samples[:, 18:24]))))
    transfer_gate = transfer(add_fits[0], add_fits[1])
    if not transfer_gate["passed"]:
        write_output(out, {
            "schema": "sqisign-r3-first-cpa-matrix-ranking-v1",
            "status": "blocked_transfer_gate",
            "evidence_contract": {"evidence_origin": "measured", "evidence_purpose": "claim_bearing",
                                  "blinding": "unblinded", "target_truth_loaded": False,
                                  "measurement_scope": "isolated_primitive",
                                  "claim_scope": "truth_free_matrix_candidate_ranking_only"},
            "inputs": {"run": str(run), "traces_sha256": digest(traces_path), "schedule_sha256": digest(schedule_path),
                       "kat_sha256": digest(kat), "calibration_sha256": digest(calibration_path)},
            "frozen_pois": pois, "transfer_gate": transfer_gate,
            "fit_key0": add_fits[0], "fit_key1": add_fits[1],
            "execution_binding": {"analyzer_sha256": digest(Path(__file__)), "numpy": np.__version__},
        })
        return
    wrong_pois = np.rint(np.linspace(WRONG_START, WRONG_STOP - 1, EVENTS)).astype(int)
    challenge2 = words([rows[i] for i in indices[2]])
    samples2 = np.asarray(traces[indices[2]][:, pois], dtype=float)
    wrong2 = np.asarray(traces[indices[2]][:, wrong_pois], dtype=float)
    blocks2 = np.array([int(rows[i]["block"]) for i in indices[2]])
    target = rank_matrix(challenge2, samples2, add_fits[0], limb)
    shuffled = rank_matrix(challenge2[block_permutation(blocks2, np.random.default_rng(SHUFFLE_SEED))], samples2, add_fits[0], limb)
    wrong = rank_matrix(challenge2, wrong2, add_fits[0], limb)
    report = {
        "schema": "sqisign-r3-first-cpa-matrix-ranking-v1",
        "evidence_contract": {"evidence_origin": "measured", "evidence_purpose": "claim_bearing", "blinding": "unblinded",
                              "target_truth_loaded": False, "measurement_scope": "isolated_primitive",
                              "claim_scope": "truth_free_matrix_candidate_ranking_only"},
        "inputs": {"run": str(run), "traces_sha256": digest(traces_path), "schedule_sha256": digest(schedule_path),
                   "kat_sha256": digest(kat), "calibration_sha256": digest(calibration_path)},
        "frozen_pois": pois, "wrong_window_pois": wrong_pois.tolist(),
        "transfer_gate": transfer_gate, "fit_key0": add_fits[0], "fit_key1": add_fits[1],
        "target_ranking": target, "controls": {"shuffled_challenges": shuffled, "wrong_window": wrong},
        "execution_binding": {"analyzer_sha256": digest(Path(__file__)), "numpy": np.__version__, "shuffle_seed": SHUFFLE_SEED},
    }
    write_output(out, report)


if __name__ == "__main__":
    main()
