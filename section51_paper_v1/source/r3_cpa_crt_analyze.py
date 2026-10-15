#!/usr/bin/env python3
"""Host-only CPA analysis for the scheduled Round-3 ``ibz_crt`` capture."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "r3_inert_ideal_probe"))
import challenge_matrix_full_cpa as matrix
import crt_coordinate_high_cpa as high
import crt_inverse_mul_cpa as inverse
import crt_secret_low192 as low


MASK192 = (1 << 192) - 1
BEAM = 8
SHUFFLE_SEED = 2026100209
EXACT_ELF_SHA256 = "bf7bc4767948e69726609b83d3132c9b9506c69cac95f8cd59fe8fdadaa20f68"
STORE_BINDING = {"crt_sub_call": "0x08003410", "crt_mul_call": "0x0800341a",
                 "sub_store_r3": "0x080024d8", "mul_store_r3": "0x080025fc"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write(path: Path, value: object) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rows(path: Path) -> list[dict[str, object]]:
    value = [json.loads(line) for line in path.read_text().splitlines() if line]
    fields = {"key_index", "challenge_hex", "coord", "heldout"}
    require(value and all(set(row) == fields for row in value), "public schedule schema mismatch")
    require(all(type(row["key_index"]) is int and row["coord"] in ("x", "y") and
                isinstance(row["challenge_hex"], str) and row["challenge_hex"].startswith("0x") and
                type(row["heldout"]) is bool for row in value), "invalid public schedule row")
    return value


def branch_file(path: Path, ranking: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    value = json.loads(path.read_text())
    require(value.get("schema") == "r3-cpa-m-branches-v1" and isinstance(value.get("branches"), list),
            "M branch schema mismatch")
    require(ranking.is_file() and value.get("source_ranking_sha256") == digest(ranking),
            "M branches do not bind the supplied t08 ranking")
    result = []
    for item in value["branches"]:
        values = item.get("matrix")
        require(isinstance(item.get("label"), str) and isinstance(values, list) and len(values) == 4,
                "invalid M branch")
        result.append((item["label"], tuple(int(x, 0) if isinstance(x, str) else int(x) for x in values)))
    require(1 <= len(result) <= 8 and len({x[1] for x in result}) == len(result), "expected one to eight distinct M branches")
    return result


def calibration_keys(kat: Path) -> tuple[dict[int, tuple[int, int, int]], dict[int, tuple[int, int, int, int]]]:
    secrets, matrices = {}, {}
    count = None
    with kat.open(encoding="utf-8") as source:
        for line in source:
            match = re.fullmatch(r"count = (\d+)\n?", line)
            if match:
                count = int(match[1])
                if count > 1:
                    break  # Target key 2 is never parsed by this analyzer.
            elif count in (0, 1) and line.startswith("sk = "):
                encoded = bytes.fromhex(line[5:].strip())
                require(len(encoded) == 417, "calibration KAT secret size mismatch")
                fields = tuple(int.from_bytes(encoded[129 + 64*i:193 + 64*i], "little") for i in range(3))
                require(fields[0] > 1 and fields[0] & 1 and all(0 <= x < fields[0] for x in fields[1:]), "invalid calibration N/x/y")
                secrets[count] = fields
                matrices[count] = tuple(int.from_bytes(encoded[offset:offset + 24], "little") for offset in (321, 345, 369, 393))
    require(set(secrets) == {0, 1} and set(matrices) == {0, 1}, "missing calibration KAT keys")
    return secrets, matrices


def run_inputs(run: Path, schedule: Path) -> tuple[np.ndarray, dict[str, object], Path]:
    manifest_path, trace_path = run / "manifest.json", run / "traces.npy"
    require(manifest_path.is_file() and trace_path.is_file(), "capture run lacks manifest or traces")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "ok" and manifest.get("stage") == "original_ibz_crt", "not a successful ibz_crt run")
    protocol = json.loads((run / "protocol.json").read_text())
    frozen = run / "schedule.jsonl"
    if frozen.is_file():
        require(digest(frozen) == manifest.get("schedule_sha256"), "frozen private schedule binding mismatch")
    else:
        require(protocol.get("private_schedule_sha256") == manifest.get("schedule_sha256"),
                "sanitized run lacks its private schedule hash binding")
    traces = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    require(traces.ndim == 2, "traces must be rank-two data")
    for start in range(0, traces.shape[0], 16):
        require(bool(np.isfinite(traces[start:start + 16]).all()), "traces contain nonfinite data")
    require(len(rows(schedule)) == traces.shape[0], "public schedule/trace count mismatch")
    return traces, manifest, trace_path


def source_binding(run: Path, manifest: dict[str, object]) -> dict[str, object]:
    files = manifest.get("provenance", {}).get("frozen", {}).get("files", {})
    require(isinstance(files, dict), "capture manifest lacks frozen provenance")
    required = ("firmware_elf", "source_ideal", "source_mp")
    require(all(name in files for name in required), "capture manifest lacks ELF/source binding")
    bound = {}
    for name in required:
        item = files[name]; path = run / item["path"]
        require(path.is_file() and digest(path) == item["sha256"], f"frozen {name} hash mismatch")
        bound[name] = {"path": str(path.relative_to(run)), "sha256": item["sha256"]}
    current = {"source_ideal": ROOT / "vendor/r3-p500-27-m4f/ideal.c", "source_mp": ROOT / "vendor/r3-p500-27-m4f/mp.c"}
    for name, path in current.items():
        require(path.is_file() and digest(path) == bound[name]["sha256"], f"current {name} differs from captured source")
    protocol = json.loads((run / "protocol.json").read_text())
    require(bound["firmware_elf"]["sha256"] == EXACT_ELF_SHA256 and
            protocol.get("firmware_elf_sha256") == EXACT_ELF_SHA256 and
            protocol.get("store_binding") == STORE_BINDING and
            protocol.get("analyzer_sha256") == digest(Path(__file__)),
            "current ELF or CRT store binding differs from the frozen analysis model")
    return bound


def oracle_public(schedule: list[dict[str, object]], candidate: tuple[int, int, int, int], oracle: Path) -> dict[str, np.ndarray]:
    challenges = [int(str(row["challenge_hex"]), 0) for row in schedule]
    output = matrix.oracle_batch(oracle, [matrix.uv(candidate, challenge) for challenge in challenges])
    split = np.array([int(row[5], 16) for row in output], dtype=np.int8)
    require(set(split.tolist()) <= {0, -1}, "oracle returned unsupported split")
    return {"x": low.integers_to_words([int(row[0], 16) for row in output]),
            "y": low.integers_to_words([int(row[1], 16) for row in output]), "split": split}


def labels(secret: tuple[int, int, int], public: dict[str, np.ndarray], coord: str) -> dict[str, np.ndarray]:
    known = public[coord]
    sub = np.empty((6, known.shape[0]), dtype=float)
    for split in (0, -1):
        selected = public["split"] == split
        sub[:, selected], _ = low.subtraction_leakage(
            low.transformed_coordinates(secret, split)[0 if coord == "x" else 1], known[selected])
    stores = np.empty((3, high.MUL_WORDS, known.shape[0]), dtype=float)
    inv = np.empty((6, known.shape[0]), dtype=float)
    u = pow(1 << 192, -1, secret[0])
    for split in (0, -1):
        selected = public["split"] == split
        representation = low.transformed_coordinates(secret, split)[0 if coord == "x" else 1]
        d, _ = high.difference_words(representation, known[selected])
        full, _ = high.simulate_product(d, u)
        stores[:, :, selected] = full[list(high.HIGH_ROWS)]
        inv[:, selected] = inverse.leakage_samples(u, d[:, 0])
    return {"low": sub, "inverse": inv, "high": stores}


def corr(label: np.ndarray, samples: np.ndarray) -> np.ndarray:
    x, y = label.astype(float) - float(label.mean()), samples.astype(float) - samples.mean(axis=0)
    den = np.sqrt(float(x @ x) * np.einsum("ij,ij->j", y, y))
    return np.divide(x @ y, den, out=np.zeros(samples.shape[1]), where=den > 0)


def discover(traces: np.ndarray, index: np.ndarray, label: np.ndarray, stop: int, stride: int) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    require(stride > 0, "discovery stride must be positive")
    # Coarse scan avoids expanding event_count × key0 × the full capture window.
    wave = np.asarray(traces[index, :stop:stride], dtype=float)
    pois, fits = [], []
    for event in label:
        if float(np.var(event)) == 0.0:
            pois.append(-1); fits.append({"poi": None, "constant_calibration_label": True})
            continue
        coarse = int(np.argmax(np.abs(corr(event, wave)))) * stride
        left, right = max(0, coarse - stride), min(stop, coarse + stride + 1)
        local = np.asarray(traces[index, left:right], dtype=float)
        point = left + int(np.argmax(np.abs(corr(event, local))))
        design = np.column_stack((event, np.ones(event.size)))
        sample = local[:, point - left]
        alpha, beta = np.linalg.lstsq(design, sample, rcond=None)[0]
        residual = sample - (alpha * event + beta)
        require(abs(alpha) > 1e-12, "zero calibration slope")
        pois.append(point); fits.append({"poi": point, "correlation": float(corr(event, sample[:, None])[0]),
                                         "alpha": float(alpha), "beta": float(beta),
                                         "residual_rms": float(np.sqrt(np.mean(residual * residual)))})
    return np.array(pois, dtype=int), fits


def transfer(traces: np.ndarray, index: np.ndarray, label: np.ndarray, fits: list[dict[str, float | int]]) -> dict[str, object]:
    rows = []
    for event, fit in zip(label, fits):
        if fit["poi"] is None:
            rows.append({"excluded_constant_calibration_label": True})
            continue
        sample = np.asarray(traces[index, int(fit["poi"])], dtype=float)
        slope, intercept = np.linalg.lstsq(np.column_stack((event, np.ones(event.size))), sample, rcond=None)[0]
        residual = sample - (slope * event + intercept)
        ratio = abs(slope / float(fit["alpha"]))
        rms_ratio = float(np.sqrt(np.mean(residual * residual))) / float(fit["residual_rms"]) if fit["residual_rms"] else float("inf")
        rows.append({"alpha_sign_agrees": bool(np.sign(slope) == np.sign(fit["alpha"])), "alpha_ratio": float(ratio), "rms_ratio": rms_ratio})
    used = [x for x in rows if not x.get("excluded_constant_calibration_label")]
    return {"passed": bool(used) and all(x["alpha_sign_agrees"] and .5 <= x["alpha_ratio"] <= 2 and x["rms_ratio"] <= 2 for x in used),
            "calibrated_events": len(used), "excluded_events": len(rows) - len(used), "events": rows}


def rank(score, beam: int = BEAM) -> list[tuple[int, float]]:
    prefixes = np.arange(256, dtype=np.uint64)
    for stage in range(4):
        if stage: prefixes = (prefixes[:, None] | (np.arange(256, dtype=np.uint64) << np.uint64(8 * stage))).reshape(-1)
        values = score(prefixes, 8 * (stage + 1))
        prefixes = prefixes[np.lexsort((prefixes, -values))[:min(beam, prefixes.size)]]
    values = score(prefixes, 32)
    order = np.lexsort((prefixes, -values))
    return [(int(prefixes[i]), float(values[i])) for i in order[:beam]]


def low_attack(known: np.ndarray, observed: np.ndarray) -> list[tuple[int, float]]:
    states = [(0, np.zeros(known.shape[0], dtype=np.uint64), 0.0)]
    for limb in range(6):
        expanded = []
        for prefix, borrow, total in states:
            for word, score in rank(lambda c, b: low.score_candidates(c, known[:, limb], borrow, observed[limb], b)):
                expanded.append((prefix | word << (32 * limb), (np.uint64(word) < known[:, limb] + borrow).astype(np.uint64), total + score))
        states = sorted(expanded, key=lambda x: (-x[2], x[0]))[:BEAM]
    return [(x[0], x[2]) for x in states]


def inverse_attack(diff0: np.ndarray, observed: np.ndarray) -> list[tuple[int, float]]:
    carry = np.zeros(diff0.size, dtype=np.uint64); result = 0; score = 0.0
    for limb in range(6):
        word, value = rank(lambda c, b: inverse.score_candidates(c, diff0, carry, observed[limb], b))[0]
        result |= word << (32 * limb); score += value; carry = (diff0 * np.uint64(word) + carry) >> np.uint64(32)
    return [(result, score)]


def high_attack(low_value: int, public: np.ndarray, observed: np.ndarray, n: int, u: int) -> list[tuple[int, float]]:
    words = high.words(u); state, borrow = high.initial_state(low_value, public, words, "normal")
    hypotheses = [(low_value, state, borrow, 0.0)]
    for offset, row in enumerate(high.HIGH_ROWS):
        expanded = []
        for prefix, current, current_borrow, total in hypotheses:
            max_word = min(0xffffffff, (n - 1 - prefix) >> (32 * row)) if row == high.HIGH_ROWS[-1] else 0xffffffff
            prefixes = [(word, score) for word, score in rank(
                lambda c, b: high.prefix_scores(c, current[:, row], current_borrow, words[0], observed[offset, 0], b), 64)
                        if word <= max_word]
            exact = []
            valid = np.isfinite(observed[offset])
            require(bool(valid.any()), "no calibrated high-store features")
            for word, _ in prefixes:
                d = (np.uint64(word) - current_borrow) & high.MASK32
                _, predicted = high.update_row(current, row, d, words)
                residual = predicted[valid].astype(float) - observed[offset][valid]
                exact.append((word, -float(np.mean(residual * residual))))
            candidates = sorted(exact, key=lambda x: (-x[1], x[0]))[:2]
            for word, score in candidates:
                d = (np.uint64(word) - current_borrow) & high.MASK32
                next_state, _ = high.update_row(current, row, d, words)
                expanded.append((prefix | word << (32 * row), next_state, (np.uint64(word) < current_borrow).astype(np.uint64), total + score))
        require(expanded, "no bounded high-coordinate candidate")
        hypotheses = sorted(expanded, key=lambda x: (-x[3], x[0]))[:2]
    return [(x[0], x[3]) for x in hypotheses]


def self_test() -> None:
    known = np.random.default_rng(7).integers(0, 1 << 32, (512, 6), dtype=np.uint64)
    secret = sum((173 + i * 0x10001) << (32 * i) for i in range(6)); leakage, _ = low.subtraction_leakage(secret, known)
    assert low_attack(known, leakage)[0][0] == secret
    assert low.derive_n_low(1, 2, 3, 4)[0] is None
    sample_label = np.arange(32, dtype=float) % 17
    trial = np.zeros((32, 64), dtype=np.float32)
    trial[:, 23] = sample_label
    trial[:, 24] = sample_label / 2
    poi, fit = discover(trial, np.arange(32), sample_label[None, :], 64, 8)
    assert poi.tolist() == [23] and abs(fit[0]["alpha"] - 1) < 1e-8


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", type=Path); parser.add_argument("--public-schedule", type=Path); parser.add_argument("--m-branches", type=Path); parser.add_argument("--ranking", type=Path)
    parser.add_argument("--kat", type=Path); parser.add_argument("--oracle", type=Path); parser.add_argument("--out", type=Path)
    parser.add_argument("--discovery-stride", type=int, default=16)
    args = parser.parse_args(); self_test()
    if args.self_test: print("self-test passed"); return
    require(all((args.run, args.public_schedule, args.m_branches, args.ranking, args.kat, args.oracle, args.out)), "all analysis paths are required")
    run, schedule_path, branch_path, ranking, kat, oracle, out = (x.resolve() for x in (args.run, args.public_schedule, args.m_branches, args.ranking, args.kat, args.oracle, args.out))
    require(oracle.is_file() and oracle.stat().st_mode & 0o111, "oracle must be executable")
    schedule = rows(schedule_path); traces, manifest, trace_path = run_inputs(run, schedule_path); binding = source_binding(run, manifest)
    protocol = json.loads((run / "protocol.json").read_text())
    require(protocol.get("public_schedule_sha256") == digest(schedule_path) and
            protocol.get("m_branches_sha256") == digest(branch_path) and
            protocol.get("calibration_kat_sha256") == digest(kat),
            "public inputs differ from frozen capture protocol")
    secrets, matrices = calibration_keys(kat)
    key = {i: np.array([j for j, row in enumerate(schedule) if row["key_index"] == i and not row["heldout"]]) for i in (0, 1, 2)}
    require(all(key[i].size for i in key), "all calibration/transfer/target keys need traces")
    quality = json.loads((run / "capture_summary.json").read_text())["rig_quality"]
    stop = int(quality["trigger_high_recorded_min"])
    require(quality["decimate"] == 4 and stop > 1 and stop <= traces.shape[1] and quality["overflow_count"] == 0,
            "invalid trigger-high or overflow evidence")
    calibration, gates = {}, {}
    public0, public1 = (oracle_public(schedule, matrices[i], oracle) for i in (0, 1))
    for coord in ("x", "y"):
        for split in (0, -1):
            for stage in ("low", "inverse", "high"):
                shape = labels(secrets[0], public0, coord)[stage]
                e0 = shape.reshape((-1, shape.shape[-1])); e1 = labels(secrets[1], public1, coord)[stage].reshape((-1, shape.shape[-1]))
                ix0 = key[0][np.array([schedule[i]["coord"] == coord and public0["split"][i] == split for i in key[0]])]
                ix1 = key[1][np.array([schedule[i]["coord"] == coord and public1["split"][i] == split for i in key[1]])]
                require(ix0.size >= 16, "insufficient key0 calibration rows")
                e0, e1 = e0[:, ix0], e1[:, ix1]
                if split == 0:
                    require(ix1.size >= 16, "insufficient key1 transfer rows")
                    discovery, validation, validation_label, mode = ix0, ix1, e1, "key1_transfer"
                else:
                    cut = ix0.size // 2; require(cut >= 16 and ix0.size - cut >= 16, "insufficient split-minus-one key0 rows")
                    discovery, validation, validation_label, e0, mode = ix0[:cut], ix0[cut:], e0[:, cut:], e0[:, :cut], "key0_disjoint_validation"
                poi, fits = discover(traces, discovery, e0, stop, args.discovery_stride); gate = transfer(traces, validation, validation_label, fits)
                if stage != "high":
                    require(gate["excluded_events"] == 0, f"{stage} has an uncalibrated store")
                else:
                    require(all(sum(fit["poi"] is not None for fit in fits[row*17:(row+1)*17]) >= 4 for row in range(3)),
                            "high-stage calibration has fewer than four varying stores in a row")
                calibration[f"{stage}:{coord}:{split}"] = {"pois": poi.tolist(), "fits": fits, "validation": mode}
                gates[f"{stage}:{coord}:{split}"] = gate
    report = {"schema": "sqisign-r3-crt-cpa-analysis-v1", "status": "blocked_transfer_gate", "evidence_contract": {"evidence_origin": "measured", "evidence_purpose": "diagnostic", "blinding": "unblinded", "measurement_scope": "isolated_primitive", "target_truth_loaded": False, "claim_scope": "round3_ibz_crt:exploratory_candidates", "claim_not_allowed": ["key_recovery", "full_key_recovery", "scheme_break"]}, "inputs": {"run": str(run), "traces_sha256": digest(trace_path), "public_schedule_sha256": digest(schedule_path), "m_branches_sha256": digest(branch_path), "ranking_sha256": digest(ranking), "kat_sha256": digest(kat), "oracle_sha256": digest(oracle)}, "binding": binding, "trigger_high_min": stop, "discovery_stride": args.discovery_stride, "calibration_key0": calibration, "transfer_key1": gates, "execution_binding": {"analyzer_sha256": digest(Path(__file__)), "numpy": np.__version__}}
    if not all(item["passed"] for item in gates.values()): write(out, report); return
    def observed(stage: str, coord: str, split: int, index: np.ndarray) -> np.ndarray:
        item = calibration[f"{stage}:{coord}:{split}"]; fits = item["fits"]
        values = np.array([((np.asarray(traces[index, int(f["poi"])], dtype=float) - float(f["beta"])) / float(f["alpha"]))
                           if f["poi"] is not None else np.full(index.size, np.nan) for f in fits])
        return values.reshape((3, 17, index.size)) if stage == "high" else values
    target = {}
    for label, candidate in branch_file(branch_path, ranking):
        public = oracle_public(schedule, candidate, oracle)
        groups = {(coord, split): key[2][np.array([schedule[i]["coord"] == coord and public["split"][i] == split for i in key[2]])]
                  for coord in ("x", "y") for split in (0, -1)}
        if not all(index.size >= 16 for index in groups.values()) or not np.any(public["x"]) or not np.any(public["y"]):
            target[label] = {"status": "rejected_public_oracle", "split_counts": {str(split): int(np.count_nonzero(public["split"] == split)) for split in (0, -1)},
                             "nonzero_public_x": bool(np.any(public["x"])), "nonzero_public_y": bool(np.any(public["y"]))}
            continue
        candidates = {f"{coord}{split}": low_attack(public[coord][index], observed("low", coord, split, index))
                      for (coord, split), index in groups.items()}
        relations = []
        for choice in itertools.product(*(candidates[name] for name in ("x0", "y0", "x-1", "y-1"))):
            n_low, matches = low.derive_n_low(*(item[0] for item in choice))
            if n_low is not None: relations.append((sum(item[1] for item in choice), n_low, tuple(item[0] for item in choice), matches))
        relations.sort(key=lambda x: (-x[0], x[1], x[2])); require(relations, f"{label}: no low-coordinate relation candidate")
        branch_results = []
        for low_score, n_low, values, matches in relations[:BEAM]:
            diff = np.empty(key[2].size, dtype=np.uint64)
            ordered = key[2]
            for pos, row_index in enumerate(ordered):
                coord, split = schedule[row_index]["coord"], int(public["split"][row_index])
                secret_low = values[(0 if coord == "x" else 1) + (0 if split == 0 else 2)]
                diff[pos] = (np.uint64(secret_low) - public[coord][row_index, 0]) & low.MASK32
            inv_observed = np.empty((6, ordered.size))
            for pos, row_index in enumerate(ordered):
                coord, split = schedule[row_index]["coord"], int(public["split"][row_index])
                inv_observed[:, pos] = observed("inverse", coord, split, np.array([row_index]))[:, 0]
            u_low, inv_score = inverse_attack(diff, inv_observed)[0]
            n = inverse.reconstruct_n(n_low, u_low); u = pow(1 << 192, -1, n)
            recovered = {}
            for coord, split in groups:
                index = groups[(coord, split)]; low_value = values[(0 if coord == "x" else 1) + (0 if split == 0 else 2)]
                recovered[f"{coord}{split}"] = high_attack(low_value, public[coord][index], observed("high", coord, split, index), n, u)
            branch_results.append({"low_score": low_score, "n_low192": hex(n_low), "u_low192": hex(u_low), "n": hex(n), "u": hex(u), "relation_matches": matches,
                                   "relation_low_coordinates": {name: hex(value) for name, value in zip(("x0", "y0", "x-1", "y-1"), values)},
                                   "low_coordinates": {name: hex(value[0]) for name, value in candidates.items()},
                                   "high_coordinates": {name: [{"value": hex(value), "score": score} for value, score in value_set] for name, value_set in recovered.items()}})
        target[label] = branch_results
    report["status"] = "diagnostic_only"; report["target_key2"] = target
    report["controls"] = {"status": "not_run", "reason": "shuffled-label, post-low, noise, and held-out gates are required before any recovery claim"}
    write(out, report)


if __name__ == "__main__": main()
