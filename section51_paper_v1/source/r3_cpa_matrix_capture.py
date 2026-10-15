#!/usr/bin/env python3
"""Capture schedule-bound 192-bit 2x2 matrix evaluations on the STM32F4."""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

import r3_cpa_capture as base

KAT_SHA256 = "986a28c7752428ba0eff2784af4f07b996d259646b192b16407a647a584d7537"
FRAME_BYTES = 320
SCALAR_BYTES = 324
RESULT_BYTES = 2 * SCALAR_BYTES
MATRIX_OFFSETS = (321, 345, 369, 393)
PILOT_CLAIM_SCOPE = "round3_ibz_mat_2x2_eval:diagnostic_measured_isolated_primitive"
ATTACK_CLAIM_SCOPE = "round3_matrix_M_recovery:measured_isolated_primitive"
PILOT_MANIFEST_SHA256 = "a3165b3d8f40fbc46cbe3f945de1121b028e8533aa69974a01256309fe29ad91"
ATTACK_SCHEDULE_SHA256 = "b69fddc2c876d5d44d6934e965ae77743dcb37651863e5f26c25480506108cca"
HELDOUT_SCHEDULE_SHA256 = "89931f75402530ff015f661a086af4b8a1e7360066fa00fda3b240fb8ef874a4"
ATTACK_EXPECTED_TRIGGER = 26008
ATTACK_MAIN_COUNTS = (512, 512, 1024)
ATTACK_HELDOUT_COUNTS = (32, 32, 32)
PROGRAMMER_BAUD = 38400


def sha256(path: pathlib.Path) -> str:
    return base.sha256_file(path)


def write_json(path: pathlib.Path, value: object) -> None:
    base.write_json(path, value)


def append_json(path: pathlib.Path, value: object) -> None:
    base.append_json(path, value)


def frame(value: int) -> bytes:
    if not 0 <= value < (1 << 192):
        raise ValueError("operand must fit 192 bits")
    return value.to_bytes(24, "little") + bytes(FRAME_BYTES - 24)


def parse_kat_text(text: str) -> dict[int, tuple[int, int, int, int]]:
    matrices: dict[int, tuple[int, int, int, int]] = {}
    count: int | None = None
    for line in text.splitlines():
        match = re.fullmatch(r"count = (\d+)", line)
        if match:
            count = int(match.group(1))
            continue
        match = re.fullmatch(r"sk = ([0-9A-Fa-f]+)", line)
        if match and count is not None:
            secret = bytes.fromhex(match.group(1))
            if len(secret) != 417:
                raise ValueError(f"KAT secret key {count} is {len(secret)} bytes, expected 417")
            values = tuple(int.from_bytes(secret[offset:offset + 24], "little") for offset in MATRIX_OFFSETS)
            if any(not 0 < value < (1 << 192) for value in values):
                raise ValueError(f"KAT matrix {count} has non-positive or oversized entry")
            matrices[count] = values
    if not matrices:
        raise ValueError("KAT contains no secret keys")
    return matrices


def load_matrices(path: pathlib.Path) -> dict[int, tuple[int, int, int, int]]:
    if sha256(path) != KAT_SHA256:
        raise RuntimeError("Round-3 p500_27 KAT hash mismatch")
    return parse_kat_text(path.read_text(encoding="utf-8"))


def load_schedule(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: dict[int, set[int]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"schedule line {number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"schedule line {number} is not JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"schedule line {number} is not an object")
        try:
            key = row["key_index"]
            signature = bytes.fromhex(row["signature_hex"])
            challenge = bytes.fromhex(row["challenge_hex"])
            verified = row["verified"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"schedule line {number} lacks valid signer fields") from exc
        if type(key) is not int or not 0 <= key <= 2:
            raise ValueError(f"schedule line {number} has invalid key_index")
        if len(signature) != 306 or len(challenge) != 24 or challenge != signature[280:304]:
            raise ValueError(f"schedule line {number} challenge/signature binding failed")
        value = int.from_bytes(challenge, "little")
        if not 0 <= value < (1 << 192) or verified is not True:
            raise ValueError(f"schedule line {number} is unverified or out of range")
        if value in seen.setdefault(key, set()):
            raise ValueError(f"schedule line {number} repeats a challenge for key {key}")
        seen[key].add(value)
        row = dict(row)
        row["schedule_index"] = len(rows)
        row["challenge"] = value
        rows.append(row)
    if not rows:
        raise ValueError("schedule is empty")
    return rows


def valid_attack_partition(rows: list[dict[str, object]], heldout: list[dict[str, object]]) -> bool:
    return (len(rows) == sum(ATTACK_MAIN_COUNTS) and
            all(sum(row["key_index"] == key for row in rows) == count
                for key, count in enumerate(ATTACK_MAIN_COUNTS)) and
            len(heldout) == sum(ATTACK_HELDOUT_COUNTS) and
            all(sum(row["key_index"] == key for row in heldout) == count
                for key, count in enumerate(ATTACK_HELDOUT_COUNTS)) and
            not ({(row["key_index"], row["challenge"]) for row in rows} &
                 {(row["key_index"], row["challenge"]) for row in heldout}))


def ack(target, command: str, payload: bytes = b"") -> None:
    base._ack(target, command, payload)


def load_operand(target, command: str, value: int) -> None:
    encoded = frame(value)
    for offset in range(0, FRAME_BYTES, base.CHUNK_BYTES):
        ack(target, command, encoded[offset:offset + base.CHUNK_BYTES])


def load_matrix(target, matrix: tuple[int, int, int, int]) -> None:
    base._reset(target, "E")
    for command, value in zip("4567", matrix, strict=True):
        load_operand(target, command, value)


def parse_board(raw: bytes, matrix: tuple[int, int, int, int], value: int, challenge: int) -> list[dict[str, object]]:
    if len(raw) != RESULT_BYTES:
        raise RuntimeError(f"board result length is {len(raw)}, expected {RESULT_BYTES}")
    expected = (matrix[0] * value + matrix[1] * challenge, matrix[2] * value + matrix[3] * challenge)
    decoded = []
    for index, wanted in enumerate(expected):
        encoded = raw[index * SCALAR_BYTES:(index + 1) * SCALAR_BYTES]
        bitlen = int.from_bytes(encoded[:4], "little", signed=True)
        observed = int.from_bytes(encoded[4:], "little")
        decoded.append({"bitlen": bitlen, "value": observed, "exact": bitlen == 386 and observed == wanted})
    return decoded


def read_board(target, matrix: tuple[int, int, int, int], value: int, challenge: int) -> list[dict[str, object]]:
    target.simpleserial_write("l", bytearray())
    size = target.simpleserial_read("r", 3, timeout=5_000)
    if size is None or len(size) != 3 or size[0] != 1 or ((size[1] << 8) | size[2]) != RESULT_BYTES:
        raise RuntimeError(f"result size reply invalid: {list(size or [])}")
    raw = bytearray()
    for index in range((RESULT_BYTES + base.CHUNK_BYTES - 1) // base.CHUNK_BYTES):
        target.simpleserial_write("g", bytearray([index]))
        chunk = target.simpleserial_read("r", base.CHUNK_BYTES, timeout=5_000)
        if chunk is None or len(chunk) != base.CHUNK_BYTES:
            raise RuntimeError(f"result chunk {index} missing")
        raw.extend(chunk)
    return parse_board(bytes(raw[:RESULT_BYTES]), matrix, value, challenge)


def execute_attempt(scope, target, matrix: tuple[int, int, int, int], row: dict[str, object], path: pathlib.Path,
                    samples: int, post_low: int, min_swing: float | None,
                    expected_trigger: int | None) -> tuple[np.ndarray, dict[str, object]]:
    value = 1
    challenge = int(row["challenge"])
    base._reset(target, "V")
    load_operand(target, "8", value)
    load_operand(target, "9", challenge)
    target.flush()
    scope.arm()
    target.simpleserial_write("e", bytearray())
    timeout = bool(scope.capture())
    wave = np.asarray(scope.get_last_trace(), dtype=np.float32)
    np.save(path, wave)
    trigger_count = int(scope.adc.trig_count)
    overflow = base.scope_overflow(scope)
    response = target.simpleserial_read("r", 1, timeout=5_000)
    board = read_board(target, matrix, value, challenge)
    swing = float(np.ptp(wave)) if wave.size else 0.0
    checks = {"capture_timeout": not timeout, "ack": response == bytearray([1]), "sample_count": wave.size == samples,
              "finite": bool(np.isfinite(wave).all()), "trigger_full_pulse": 0 < trigger_count < samples,
              "post_low": samples - trigger_count >= post_low, "fifo_overflow": not overflow,
              "board_result": all(item["exact"] for item in board)}
    if expected_trigger is not None:
        checks["trigger_alignment"] = trigger_count == expected_trigger
    if min_swing is not None:
        checks["swing"] = swing >= min_swing
    return wave, {"accepted": all(checks.values()), "checks": checks, "ack": list(response or []), "board": board,
                  "trigger_high_count": trigger_count, "post_low_observed": samples - trigger_count, "fifo_overflow": overflow,
                  "trace_swing": swing, "wave_path": str(path.relative_to(path.parent.parent)), "wave_sha256": sha256(path)}


def freeze_file(run: pathlib.Path, source: pathlib.Path, name: str) -> dict[str, str]:
    destination = run / "provenance" / name
    digest = sha256(source)
    shutil.copyfile(source, destination)
    if sha256(source) != digest or sha256(destination) != digest:
        raise RuntimeError(f"source changed while freezing provenance: {source}")
    return {"path": str(destination.relative_to(run)), "sha256": digest}


def artifact(run: pathlib.Path, path: pathlib.Path, claim_scope: str) -> dict[str, object]:
    entry: dict[str, object] = {"filename": str(path.relative_to(run)), "sha256": sha256(path), "size_bytes": path.stat().st_size,
                                "role": "capture_evidence", "claim_scope": claim_scope}
    if path.suffix == ".npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            entry.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        except (OSError, ValueError) as exc:
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def write_blocked(run: pathlib.Path, reason: str, attempts: list[dict[str, object]], config: dict[str, object],
                  started_at: int, provenance: dict[str, object]) -> None:
    artifacts = [artifact(run, path, config["claim_scope"]) for path in run.rglob("*") if path.is_file() and path.name not in {"blocked.json", "manifest.json"}]
    write_json(run / "blocked.json", {"status": "blocked", "reason": reason, "run_id": run.name, "started_at_unix": started_at,
               "finished_at_unix": int(time.time()), "attempt_count": len(attempts), "attempts": attempts,
               "config_sha256": sha256(run / "config.json"), "provenance": provenance, "evidence_origin": "measured",
               "evidence_purpose": config["evidence_purpose"], "blinding": "unblinded", "measurement_scope": "isolated_primitive",
               "claim_scope": config["claim_scope"], "claim_not_allowed": config["claim_not_allowed"], "artifact_roles": artifacts,
               "artifact_sha256": {item["filename"]: item["sha256"] for item in artifacts},
               "host": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__}})


def diagnostic(path: pathlib.Path, traces: np.ndarray, run_id: str) -> None:
    import matplotlib.pyplot as plt

    mean, deviation = traces.mean(axis=0), traces.std(axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140, sharex=True)
    axes[0].plot(mean); axes[0].fill_between(np.arange(mean.size), mean - deviation, mean + deviation, alpha=0.3)
    axes[0].set_title("Trace mean and variability")
    axes[1].plot(traces[0]); axes[1].set_title("Scheduled trace 0")
    figure.suptitle(f"Measured matrix-evaluation capture quality ({run_id}) — {traces.shape[0]} traces")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def capture(args: argparse.Namespace) -> pathlib.Path:
    if args.samples < 2 or not 0 < args.post_low < args.samples:
        raise ValueError("require 0 < --post-low < --samples")
    if args.min_swing is not None and args.min_swing <= 0:
        raise ValueError("--min-swing must be positive when provided")
    pilot = None
    if args.purpose == "attack":
        if args.pilot_manifest is None or args.expected_trigger is None or args.protocol is None or args.heldout is None:
            raise ValueError("attack capture requires --pilot-manifest, --expected-trigger, --protocol, and --heldout")
        if sha256(args.pilot_manifest) != PILOT_MANIFEST_SHA256:
            raise RuntimeError("pilot manifest hash mismatch")
        pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
        if pilot["status"] != "ok" or pilot["attempt_count"] != 17:
            raise RuntimeError("pilot manifest status or attempt count mismatch")
        if args.expected_trigger != ATTACK_EXPECTED_TRIGGER or args.samples != 45056 or args.post_low != 8192 or args.min_swing != 0.2:
            raise ValueError("attack capture settings differ from the frozen pilot-derived protocol")
    run = args.out.resolve()
    if run.exists():
        raise FileExistsError(run)
    rows, matrices = load_schedule(args.schedule.resolve()), load_matrices(args.kat.resolve())
    if pilot:
        heldout = load_schedule(args.heldout.resolve())
        if sha256(args.schedule) != ATTACK_SCHEDULE_SHA256 or sha256(args.heldout) != HELDOUT_SCHEDULE_SHA256:
            raise RuntimeError("attack or held-out schedule hash mismatch")
        if not valid_attack_partition(rows, heldout):
            raise ValueError("capture or held-out challenge partition differs from protocol")
    if any(int(row["key_index"]) not in matrices for row in rows):
        raise ValueError("schedule key_index is absent from KAT")
    run.mkdir(parents=True); (run / "attempts").mkdir(); (run / "logs").mkdir()
    started_at = int(time.time())
    claim_scope = ATTACK_CLAIM_SCOPE if args.purpose == "attack" else PILOT_CLAIM_SCOPE
    config: dict[str, object] = {"run_id": run.name, "purpose": args.purpose, "trace_count": len(rows), "samples": args.samples,
                                 "programmer_baud": PROGRAMMER_BAUD,
                                 "post_low": args.post_low, "min_swing": args.min_swing, "operand_bytes": 24, "frame_bytes": FRAME_BYTES,
                                 "input_chunks": 10, "expected_trigger": args.expected_trigger, "pilot_manifest_sha256": PILOT_MANIFEST_SHA256 if pilot else None,
                                 "protocol_sha256": sha256(args.protocol) if pilot else None,
                                 "heldout_sha256": sha256(args.heldout) if pilot else None,
                                 "evidence_origin": "measured", "evidence_purpose": "claim_bearing" if pilot else "diagnostic", "blinding": "unblinded",
                                 "measurement_scope": "isolated_primitive", "claim_scope": claim_scope,
                                 "claim_allowed": ["isolated M recovery only after independent analysis gates"] if pilot else
                                                  ["diagnostic measured acquisition and exact isolated matrix-evaluation readback"],
                                 "claim_not_allowed": ["full_signer_path", "complete_signing_key", "secret_ideal_N_x_y", "forgery"] if pilot else
                                                      ["leakage_detection", "key_recovery", "full_algorithm_path", "full_kem_path", "scheme_break"]}
    write_json(run / "config.json", config)
    attempts: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    scope = target = None
    try:
        provenance = base.freeze_provenance(run, args.build_log.resolve(), args.physical_setup_record.resolve(), None)
        if pilot and provenance["files"]["firmware_hex"]["sha256"] != pilot["provenance"]["frozen"]["files"]["firmware_hex"]["sha256"]:
            raise RuntimeError("firmware HEX differs from pilot")
        provenance["files"].update({"matrix_capture_script": freeze_file(run, pathlib.Path(__file__).resolve(), "r3_cpa_matrix_capture.py"),
                                    "schedule": freeze_file(run, args.schedule.resolve(), "schedule.jsonl"), "kat": freeze_file(run, args.kat.resolve(), "kat.rsp"),
                                    "config": freeze_file(run, run / "config.json", "config.json")})
        if pilot:
            provenance["files"]["pilot_manifest"] = freeze_file(run, args.pilot_manifest, "pilot_manifest.json")
            provenance["files"]["attack_protocol"] = freeze_file(run, args.protocol, "attack_protocol.md")
            provenance["files"]["heldout_schedule"] = freeze_file(run, args.heldout, "heldout.jsonl")
            if (provenance["files"]["schedule"]["sha256"] != ATTACK_SCHEDULE_SHA256 or
                    provenance["files"]["heldout_schedule"]["sha256"] != HELDOUT_SCHEDULE_SHA256 or
                    provenance["files"]["pilot_manifest"]["sha256"] != PILOT_MANIFEST_SHA256):
                raise RuntimeError("frozen pilot or challenge schedule differs from protocol")
            root = pathlib.Path(__file__).resolve().parent
            provenance["files"]["signer_source"] = freeze_file(run, root / "r3_cpa_signer_challenges.c", "r3_cpa_signer_challenges.c")
            provenance["files"]["signer_binary"] = freeze_file(run, root / "r3_cpa_signer_challenges", "r3_cpa_signer_challenges")
        import chipwhisperer as cw

        scope = cw.scope(); target = cw.target(scope)
        before = {"samples": int(scope.adc.samples), "gain_db": float(scope.gain.db), "clkgen_freq": float(scope.clock.clkgen_freq), "adc_freq": float(scope.clock.adc_freq)}
        scope.default_setup(); scope.adc.samples = args.samples
        if not hasattr(scope.adc, "fifoOverflow"):
            raise RuntimeError("this capture requires a Husky scope with scope.adc.fifoOverflow")
        if int(scope.adc.decimate) != 1 or int(scope.adc.offset) != 0:
            raise RuntimeError("capture requires decimate=1 and offset=0 for trigger coverage")
        after = {"samples": int(scope.adc.samples), "decimate": int(scope.adc.decimate), "offset": int(scope.adc.offset), "timeout": float(scope.adc.timeout),
                 "gain_db": float(scope.gain.db), "clkgen_freq": float(scope.clock.clkgen_freq), "adc_freq": float(scope.clock.adc_freq),
                 "scope_serial": str(getattr(scope, "sn", "")), "trigger": str(scope.trigger.triggers), "overflow_api": "scope.adc.fifoOverflow()"}
        if pilot:
            before_scope = pilot["provenance"]["scope"]
            invariant = ("scope_serial", "gain_db", "clkgen_freq", "adc_freq", "decimate", "offset", "trigger")
            if any(after[key] != before_scope[key] for key in invariant):
                raise RuntimeError("scope settings differ from pilot")
        write_json(run / "scope_settings.json", {"before": before, "after": after})
        log = run / "logs" / "capture.jsonl"; append_json(log, {"event": "scope_settings", "before": before, "after": after})
        with (run / "logs" / "flash_verify_console.log").open("w", encoding="utf-8") as console, \
             contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            cw.program_target(scope, cw.programmers.STM32FProgrammer,
                              str(run / provenance["files"]["firmware_hex"]["path"]), baud=PROGRAMMER_BAUD)
        if "Verified flash OK" not in (run / "logs" / "flash_verify_console.log").read_text(encoding="utf-8"):
            raise RuntimeError("flash readback verification text missing")
        time.sleep(0.2); target.flush()
        active_key: int | None = None
        traces = np.empty((len(rows), args.samples), dtype=np.float32)
        for eligible, row in [(False, rows[0]), *[(True, row) for row in rows]]:
            key = int(row["key_index"])
            if key != active_key:
                load_matrix(target, matrices[key]); active_key = key
            attempt_id = len(attempts); path = run / "attempts" / f"{attempt_id:06d}.npy"
            try:
                wave, record = execute_attempt(scope, target, matrices[key], row, path, args.samples, args.post_low, args.min_swing,
                                               args.expected_trigger)
                record.update({"physical_attempt": attempt_id, "eligible": eligible, "reason": "first_arm_warmup" if not eligible else "scheduled_capture", "query": row})
            except (Exception, KeyboardInterrupt) as exc:
                record = {"physical_attempt": attempt_id, "eligible": eligible, "reason": "first_arm_warmup" if not eligible else "scheduled_capture", "query": row,
                          "accepted": False, "error": f"{type(exc).__name__}: {exc}", "wave_path": str(path.relative_to(run)) if path.exists() else None,
                          "wave_sha256": sha256(path) if path.exists() else None}
                attempts.append(record); append_json(log, record)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError(record["error"]) from exc
            attempts.append(record); append_json(log, record)
            if not record["accepted"]:
                raise RuntimeError("capture integrity failed: " + ", ".join(key for key, ok in record["checks"].items() if not ok))
            if eligible:
                traces[int(row["schedule_index"])] = wave
        np.save(run / "traces.npy", traces)
        diagnostic(run / "capture_diagnostic.png", traces, run.name)
        accepted = [record for record in attempts if record["eligible"]]
        write_json(run / "capture_summary.json", {"backend": "hardware", "query_count": len(rows), "purpose": args.purpose,
                   "rig_quality": {"sample_count": args.samples, "minimum_trace_swing": args.min_swing, "trigger_high_count_min": min(record["trigger_high_count"] for record in accepted),
                                   "post_low_min": min(record["post_low_observed"] for record in accepted), "power_trace_swing_min": min(record["trace_swing"] for record in accepted),
                                   "power_trace_swing_mean": float(np.mean([record["trace_swing"] for record in accepted])),
                                   "power_trace_std": float(traces.std()), "overflow_count": sum(bool(record["fifo_overflow"]) for record in attempts),
                                   "sampling_settings": {key: after[key] for key in ("decimate", "offset", "adc_freq", "clkgen_freq", "gain_db", "trigger")}}})
        artifacts = [artifact(run, path, claim_scope) for path in run.rglob("*") if path.is_file() and path.name != "manifest.json"]
        manifest = {"run_id": run.name, "status": "ok", "backend": "hardware", "started_at_unix": started_at, "finished_at_unix": int(time.time()),
                    "attempt_count": len(attempts), "config_sha256": sha256(run / "config.json"), "evidence_origin": "measured", "evidence_purpose": config["evidence_purpose"],
                    "blinding": "unblinded", "measurement_scope": "isolated_primitive", "claim_scope": claim_scope, "claim_allowed": config["claim_allowed"],
                    "claim_not_allowed": config["claim_not_allowed"], "provenance": {"frozen": provenance, "target_board": {"model": "CW308_STM32F4", "target_type": type(target).__name__, "serial": str(getattr(target, "sn", ""))}, "scope": after},
                    "artifacts": [item["filename"] for item in artifacts], "artifact_roles": artifacts, "artifact_sha256": {item["filename"]: item["sha256"] for item in artifacts},
                    "host": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "chipwhisperer": cw.__version__}}
        write_json(run / "manifest.json", manifest)
    except (Exception, KeyboardInterrupt) as exc:
        (run / "logs" / "capture_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_blocked(run, f"{type(exc).__name__}: {exc}", attempts, config, started_at, provenance)
        raise
    finally:
        if target is not None: target.dis()
        if scope is not None: scope.dis()
    return run


def self_test() -> None:
    matrix = tuple((1 << 191) - offset for offset in range(1, 5))
    board_matrix = ((1 << 385), 1, (1 << 385) - 1, 1)
    encoded = b"".join((386).to_bytes(4, "little", signed=True) + value.to_bytes(320, "little")
                       for value in (board_matrix[0] + 2, board_matrix[2] + 2))
    assert frame((1 << 192) - 1)[:24] == bytes([255]) * 24 and len(frame(1)) == FRAME_BYTES
    assert all(item["exact"] for item in parse_board(encoded, board_matrix, 1, 2))
    assert not parse_board((385).to_bytes(4, "little", signed=True) + encoded[4:], board_matrix, 1, 2)[0]["exact"]
    secret = bytearray(417)
    for offset, value in zip(MATRIX_OFFSETS, matrix, strict=True): secret[offset:offset + 24] = value.to_bytes(24, "little")
    assert parse_kat_text("count = 2\nsk = " + secret.hex().upper() + "\n")[2] == matrix
    signature = bytes(range(256)) + bytes(range(50)); challenge = signature[280:304]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory); schedule = root / "schedule.jsonl"
        schedule.write_text(json.dumps({"key_index": 2, "signature_hex": signature.hex(), "challenge_hex": challenge.hex(), "verified": True}) + "\n", encoding="utf-8")
        assert load_schedule(schedule)[0]["challenge"] == int.from_bytes(challenge, "little")
        try:
            load_matrices(schedule)
            raise AssertionError("KAT hash guard accepted a schedule")
        except RuntimeError as exc:
            assert "hash mismatch" in str(exc)
        run = root / "run"; (run / "logs").mkdir(parents=True); write_json(run / "config.json", {"claim_not_allowed": []}); (run / "logs" / "capture.jsonl").write_text("failure\n", encoding="utf-8")
        (run / "partial.npy").write_bytes(b"not an npy file")
        write_blocked(run, "self-test failure", [], {"claim_not_allowed": [], "claim_scope": "test", "evidence_purpose": "diagnostic"}, 1, {})
        blocked = json.loads((run / "blocked.json").read_text())
        assert blocked["status"] == "blocked" and any("read_error" in item for item in blocked["artifact_roles"])


def main() -> None:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--out", type=pathlib.Path, required=True); capture_parser.add_argument("--schedule", type=pathlib.Path, required=True)
    capture_parser.add_argument("--kat", type=pathlib.Path, required=True); capture_parser.add_argument("--samples", type=int, required=True)
    capture_parser.add_argument("--post-low", type=int, required=True); capture_parser.add_argument("--min-swing", type=float)
    capture_parser.add_argument("--build-log", type=pathlib.Path, required=True); capture_parser.add_argument("--physical-setup-record", type=pathlib.Path, required=True)
    capture_parser.add_argument("--purpose", choices=("pilot", "attack"), required=True)
    capture_parser.add_argument("--pilot-manifest", type=pathlib.Path)
    capture_parser.add_argument("--expected-trigger", type=int)
    capture_parser.add_argument("--protocol", type=pathlib.Path)
    capture_parser.add_argument("--heldout", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "self-test": self_test()
    else: print(capture(args))


if __name__ == "__main__":
    main()
