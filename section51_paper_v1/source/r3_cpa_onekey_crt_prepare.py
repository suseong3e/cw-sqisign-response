#!/usr/bin/env python3
"""Prepare a one-fresh-key CRT capture schedule before any CPA analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable

import r3_cpa_capture as base
import r3_cpa_matrix_capture as matrix_capture

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "r3_inert_ideal_probe"))
import challenge_matrix_full_cpa as full  # noqa: E402
import crt_secret_low192 as low  # noqa: E402


ORACLE = ROOT / ".plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle"
B = 1 << 192
FIELD_OFFSETS = (129, 193, 257)
APPLICABILITY_RULE = {"schema": "r3-onekey-crt-split-zero-applicability-v2",
                      "pilot_per_class": 4, "main_per_partition_min": 200,
                      "heldout_split_zero_min": 8,
                      "partition": "sha256(hex(challenge).encode()).digest()[0] & 1"}
OracleBatch = Callable[[Path, list[tuple[int, int]]], list[tuple[str, ...]]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fresh_key(kat: Path) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Read exactly the count-zero p500_27 key needed for this target-only run."""
    text = kat.read_text(encoding="ascii")
    current: int | None = None
    entries: list[tuple[int, bytes]] = []
    counts: list[int] = []
    for line in text.splitlines():
        count = re.fullmatch(r"count = (\d+)", line)
        if count:
            current = int(count.group(1))
            counts.append(current)
            continue
        secret = re.fullmatch(r"sk = ([0-9A-Fa-f]+)", line)
        if secret and current is not None:
            entries.append((current, bytes.fromhex(secret.group(1))))
    _require(counts == [0] and len(entries) == 1 and entries[0][0] == 0,
             "KAT must contain exactly one count = 0 key")
    encoded = entries[0][1]
    _require(len(encoded) == 417, "fresh p500_27 secret key must be 417 bytes")
    matrices = matrix_capture.parse_kat_text(text)
    _require(set(matrices) == {0}, "KAT must contain exactly one parseable matrix at count 0")
    secret = tuple(int.from_bytes(encoded[offset:offset + 64], "little") for offset in FIELD_OFFSETS)
    n, x, y = secret
    _require(n > 1 and n & 1 and 0 <= x < n and 0 <= y < n, "KAT has an invalid inert ideal")
    return matrices[0], secret


def public_key(kat: Path) -> bytes:
    keys = re.findall(r"^pk = ([0-9A-Fa-f]+)$", kat.read_text(encoding="ascii"), re.MULTILINE)
    _require(len(keys) == 1, "KAT must contain exactly one public key")
    encoded = bytes.fromhex(keys[0])
    _require(len(encoded) == 129, "fresh p500_27 public key must be 129 bytes")
    return encoded


def source_rows(main_schedule: Path, heldout_schedule: Path) -> list[tuple[bool, dict[str, object]]]:
    main = matrix_capture.load_schedule(main_schedule)
    heldout = matrix_capture.load_schedule(heldout_schedule)
    _require(len(main) == 1024, "main schedule must contain exactly 1024 signer-verified challenges")
    _require(len(heldout) == 32, "held-out schedule must contain exactly 32 signer-verified challenges")
    _require(all(row["key_index"] == 0 for row in main + heldout), "one-key schedules must use key_index 0")
    main_challenges = {int(row["challenge"]) for row in main}
    heldout_challenges = {int(row["challenge"]) for row in heldout}
    _require(len(main_challenges) == 1024 and len(heldout_challenges) == 32,
             "source schedules must have unique challenges")
    _require(main_challenges.isdisjoint(heldout_challenges), "main and held-out challenges overlap")
    return [(False, row) for row in main] + [(True, row) for row in heldout]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.chmod(path, 0o600)


def prepare(kat: Path, main_schedule: Path, heldout_schedule: Path, out: Path,
            oracle: Path = ORACLE, oracle_batch: OracleBatch = full.oracle_batch) -> dict[str, object]:
    """Classify CRT split support before emitting capture inputs for a fresh key.

    The output is refused when it already exists so a run cannot silently replace its
    key-bound schedule.  ``oracle_batch`` is injectable only for deterministic local checks.
    """
    if out.exists():
        raise FileExistsError(out)
    if not oracle.is_file():
        raise FileNotFoundError(oracle)
    matrix, secret = fresh_key(kat)
    encoded_public_key = public_key(kat)
    sources = source_rows(main_schedule, heldout_schedule)
    pairs = [full.uv(matrix, int(row["challenge"])) for _, row in sources]
    oracle_rows: list[tuple[str, ...]] = []
    for start in range(0, len(pairs), 128):
        oracle_rows.extend(oracle_batch(oracle, pairs[start:start + 128]))
    _require(len(oracle_rows) == len(sources) and all(len(row) == 8 for row in oracle_rows),
             "oracle returned an unexpected row count or shape")

    private: list[dict[str, object]] = []
    public: list[dict[str, object]] = []
    pilot_indices: list[int] = []
    pilot_counts: dict[tuple[int, str], int] = {}
    split_counts = {"main": {"0": 0, "-1": 0}, "heldout": {"0": 0, "-1": 0}}
    partition_counts = {name: {"train": 0, "validation": 0} for name in ("0", "-1")}
    n, _, _ = secret
    for (heldout, source), oracle_row in zip(sources, oracle_rows, strict=True):
        split = int(oracle_row[5], 16)
        _require(split in (0, -1), "oracle returned an unsupported split")
        split_counts["heldout" if heldout else "main"][str(split)] += 1
        if not heldout:
            partition = "validation" if hashlib.sha256(hex(int(source["challenge"])).encode()).digest()[0] & 1 else "train"
            partition_counts[str(split)][partition] += 1
        coordinates = low.transformed_coordinates(secret, split)
        for coord, column in (("x", 0), ("y", 1)):
            a1, a2 = int(oracle_row[column], 16), coordinates[column]
            _require(0 <= a1 < B and a1 < a2 < n, "oracle operands are not canonical CRT operands")
            expected = a1 + B * (((a2 - a1) * pow(B, -1, n)) % n)
            _require(expected % B == a1 and expected % n == a2, "host CRT parity check failed")
            private.append({"key_index": 0, "challenge_hex": hex(int(source["challenge"])), "coord": coord,
                            "split_v": split, "a1": hex(a1), "a2": hex(a2), "m1": hex(B),
                            "m2": hex(n), "expected": hex(expected)})
            public.append({"key_index": 0, "challenge_hex": hex(int(source["challenge"])),
                           "coord": coord, "heldout": heldout})
            class_key = (split, coord)
            if not heldout and split == 0 and pilot_counts.get(class_key, 0) < APPLICABILITY_RULE["pilot_per_class"]:
                pilot_counts[class_key] = pilot_counts.get(class_key, 0) + 1
                pilot_indices.append(len(private) - 1)
    expected_classes = {(0, coord) for coord in ("x", "y")}
    if (pilot_counts != {key: APPLICABILITY_RULE["pilot_per_class"] for key in expected_classes} or
            min(partition_counts["0"].values()) < APPLICABILITY_RULE["main_per_partition_min"] or
            split_counts["heldout"]["0"] < APPLICABILITY_RULE["heldout_split_zero_min"]):
        out.mkdir(parents=True)
        os.chmod(out, 0o700)
        manifest = {"schema": "r3-cpa-onekey-crt-preparation-v1", "status": "method_not_applicable",
                    "reason": "insufficient_crt_class_samples", "split_challenge_counts": split_counts,
                    "main_partition_counts": partition_counts,
                    "source_challenges": len(sources), "physical_attempts": 0,
                    "kat_sha256": base.sha256_file(kat), "oracle_sha256": base.sha256_file(oracle),
                    "source_schedule_sha256": {"main": base.sha256_file(main_schedule),
                                               "heldout": base.sha256_file(heldout_schedule)},
                    "prepare_script_sha256": base.sha256_file(Path(__file__).resolve()),
                    "applicability_rule": APPLICABILITY_RULE,
                    "claim_scope": "r3_target_only:single_new_key_diagnostic"}
        base.write_json(out / "prepare_manifest.json", manifest)
        os.chmod(out / "prepare_manifest.json", 0o600)
        return manifest
    _require(len(private) == 2112 and len(public) == 2112 and len(pilot_indices) == 8,
             "unexpected prepared row or pilot count")

    out.mkdir(parents=True)
    os.chmod(out, 0o700)
    (out / "public_key.bin").write_bytes(encoded_public_key)
    os.chmod(out / "public_key.bin", 0o600)
    _write_jsonl(out / "private_schedule.jsonl", private)
    _write_jsonl(out / "public_schedule.jsonl", public)
    protocol = {"schema": "r3-cpa-crt-capture-v1", "mode": "pilot",
                "capture": {"sample_count": 300000, "post_low": 8192, "min_swing": 0.2, "decimate": 4},
                "programmer": {"baud": 38400}, "pilot_indices": pilot_indices,
                "claim_scope": "r3_target_only:single_new_key_diagnostic",
                "capture_script_sha256": base.sha256_file(ROOT / "r3_cpa_crt_capture.py"),
                "firmware_elf_sha256": base.sha256_file(ROOT / "simpleserial-r3-cpa-CW308_STM32F4.elf"),
                "oracle_sha256": base.sha256_file(oracle), "kat_sha256": base.sha256_file(kat),
                "private_schedule_sha256": base.sha256_file(out / "private_schedule.jsonl"),
                "public_schedule_sha256": base.sha256_file(out / "public_schedule.jsonl"),
                "source_schedule_sha256": {"main": base.sha256_file(main_schedule),
                                           "heldout": base.sha256_file(heldout_schedule)}}
    base.write_json(out / "pilot_protocol.json", protocol)
    os.chmod(out / "pilot_protocol.json", 0o600)
    manifest = {"schema": "r3-cpa-onekey-crt-preparation-v1", "status": "ready", "rows": len(private),
                "source_challenges": len(sources), "pilot_indices": pilot_indices,
                "split_challenge_counts": split_counts, "main_partition_counts": partition_counts,
                "prepare_script_sha256": base.sha256_file(Path(__file__).resolve()),
                "applicability_rule": APPLICABILITY_RULE,
                "physical_attempts_main_expected": len(private) + 1,
                "physical_attempts_pilot_expected": len(pilot_indices) + 1,
                "artifacts_sha256": {name: base.sha256_file(out / name) for name in
                                     ("private_schedule.jsonl", "public_schedule.jsonl", "pilot_protocol.json",
                                      "public_key.bin")},
                "claim_scope": protocol["claim_scope"]}
    base.write_json(out / "prepare_manifest.json", manifest)
    os.chmod(out / "prepare_manifest.json", 0o600)
    return manifest


def self_test() -> None:
    """Exercise the one-key schema and pilot selection without a board or real oracle."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def fake_oracle(_: Path, pairs: list[tuple[int, int]]) -> list[tuple[str, ...]]:
        return [("0", "0", "", "", "", "-1" if pair[0] & 1 else "0", "", "") for pair in pairs]

    def schedule_row(challenge: int) -> dict[str, object]:
        signature = bytearray(306)
        signature[280:304] = challenge.to_bytes(24, "little")
        return {"key_index": 0, "signature_hex": signature.hex(), "challenge_hex": challenge.to_bytes(24, "little").hex(),
                "verified": True}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        encoded = bytearray(417)
        encoded[129:193] = (65).to_bytes(64, "little")
        encoded[193:257] = (5).to_bytes(64, "little")
        encoded[257:321] = (7).to_bytes(64, "little")
        for offset in (321, 345, 369, 393):
            encoded[offset:offset + 24] = (1).to_bytes(24, "little")
        kat = root / "fresh.rsp"
        kat.write_text("count = 0\npk = " + (b"\x01" * 129).hex() +
                       "\nsk = " + encoded.hex() + "\n", encoding="ascii")
        main, heldout = root / "main.jsonl", root / "heldout.jsonl"
        main.write_text("".join(json.dumps(schedule_row(value)) + "\n" for value in range(1, 1025)), encoding="utf-8")
        heldout.write_text("".join(json.dumps(schedule_row(value)) + "\n" for value in range(1025, 1057)), encoding="utf-8")
        oracle = root / "oracle"; oracle.write_bytes(b"test")
        out = root / "prepared"
        manifest = prepare(kat, main, heldout, out, oracle, fake_oracle)
        require(manifest["status"] == "ready" and manifest["rows"] == 2112 and
                len(manifest["pilot_indices"]) == 8, "prepared counts differ")
        protocol = json.loads((out / "pilot_protocol.json").read_text(encoding="utf-8"))
        require(protocol["programmer"] == {"baud": 38400} and "m_branches_sha256" not in protocol,
                "pilot protocol has an unexpected M dependency")
        require((out / "private_schedule.jsonl").stat().st_mode & 0o777 == 0o600,
                "private schedule permissions differ")
        require((out / "public_key.bin").read_bytes() == b"\x01" * 129 and
                manifest["artifacts_sha256"]["public_key.bin"] == base.sha256_file(out / "public_key.bin"),
                "prepared public key is not bound to the manifest")
        only_zero = lambda _oracle, batch: [("0", "0", "", "", "", "0", "", "") for _ in batch]
        missing = prepare(kat, main, heldout, root / "missing", oracle, only_zero)
        require(missing["status"] == "ready" and
                missing["split_challenge_counts"] == {"main": {"0": 1024, "-1": 0},
                                                       "heldout": {"0": 32, "-1": 0}} and
                len(missing["pilot_indices"]) == 8,
                "split-zero-only key did not prepare capture inputs")
        seen = 0
        def scarce_zero(_: Path, batch: list[tuple[int, int]]) -> list[tuple[str, ...]]:
            nonlocal seen
            rows = []
            for _ in batch:
                rows.append(("0", "0", "", "", "", "0" if seen < 3 else "-1", "", ""))
                seen += 1
            return rows
        scarce = prepare(kat, main, heldout, root / "scarce", oracle, scarce_zero)
        require(scarce["status"] == "method_not_applicable" and
                scarce["split_challenge_counts"]["main"] == {"0": 3, "-1": 1021},
                "undersampled split-zero class was treated as ready")
        seen = 0
        def thin_zero(_: Path, batch: list[tuple[int, int]]) -> list[tuple[str, ...]]:
            nonlocal seen
            rows = []
            for _ in batch:
                split = "0" if seen < 390 or (seen >= 1024 and seen % 2 == 0) else "-1"
                rows.append(("0", "0", "", "", "", split, "", ""))
                seen += 1
            return rows
        thin = prepare(kat, main, heldout, root / "thin", oracle, thin_zero)
        require(thin["status"] == "method_not_applicable" and
                thin["split_challenge_counts"]["main"]["0"] == 390 and
                thin["split_challenge_counts"]["heldout"]["0"] == 16,
                "underpowered split-zero partition was treated as ready")
        seen = 0
        def missing_heldout(_: Path, batch: list[tuple[int, int]]) -> list[tuple[str, ...]]:
            nonlocal seen
            rows = []
            for _ in batch:
                split = "0" if seen < 1024 else "-1"
                rows.append(("0", "0", "", "", "", split, "", ""))
                seen += 1
            return rows
        unheld = prepare(kat, main, heldout, root / "unheld", oracle, missing_heldout)
        require(unheld["status"] == "method_not_applicable" and
                unheld["split_challenge_counts"]["heldout"]["0"] == 0,
                "missing split-zero heldout class was treated as ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kat", type=Path)
    parser.add_argument("--main-schedule", type=Path)
    parser.add_argument("--heldout-schedule", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.kat, args.main_schedule, args.heldout_schedule, args.out)):
            parser.error("--self-test does not take preparation paths")
        self_test()
        return
    if any(value is None for value in (args.kat, args.main_schedule, args.heldout_schedule, args.out)):
        parser.error("--kat, --main-schedule, --heldout-schedule, and --out are required")
    print(json.dumps(prepare(args.kat, args.main_schedule, args.heldout_schedule, args.out, args.oracle), indent=2))


if __name__ == "__main__":
    main()
