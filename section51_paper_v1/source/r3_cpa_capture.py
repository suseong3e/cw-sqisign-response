#!/usr/bin/env python3
"""Capture diagnostic STM32F4 traces for the SHA-bound Round-3 ibz_mul harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
FRAME_BYTES = 320
CHUNK_BYTES = 32
RESULT_BYTES = 324
RESULT_CHUNKS = (RESULT_BYTES + CHUNK_BYTES - 1) // CHUNK_BYTES
SECRET = 0xD3C1A5E79B2468F1
FIXED_PUBLIC = 0xC75E3A196B842FD1
CONTRAST_PUBLIC_HW0 = 0x8000000000000000
CONTRAST_PUBLIC_HW32 = 0x80000000CFDA77EF


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_json(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def operand_frame(value: int) -> bytes:
    if not (0 < value < (1 << 64)):
        raise ValueError("operand must be a positive 64-bit integer")
    return value.to_bytes(8, "little") + bytes(FRAME_BYTES - 8)


def decode_result(raw: bytes, expected: int) -> dict[str, object]:
    if len(raw) != RESULT_BYTES:
        raise RuntimeError(f"board result length is {len(raw)}, expected {RESULT_BYTES}")
    bitlen = int.from_bytes(raw[:4], "little", signed=True)
    limbs = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "little") for i in range(80)]
    observed = sum(limb << (32 * i) for i, limb in enumerate(limbs))
    return {"bitlen": bitlen, "limbs": limbs, "product": observed, "exact": bitlen == 129 and observed == expected}


def schedule(trace_count: int, seed: int, contrast_hw_store_j0: bool = False) -> list[dict[str, int]]:
    if trace_count < 20 or trace_count % 20:
        raise ValueError("--traces must be a positive multiple of 20")
    generator = random.Random(seed)
    rows: list[dict[str, int]] = []
    seen_publics: set[int] = {FIXED_PUBLIC}
    for block in range(trace_count // 20):
        fixed = CONTRAST_PUBLIC_HW0 if contrast_hw_store_j0 else FIXED_PUBLIC
        block_rows = [{"kind": 0, "public": fixed, "block": block} for _ in range(10)]
        if contrast_hw_store_j0:
            block_rows.extend({"kind": 1, "public": CONTRAST_PUBLIC_HW32, "block": block} for _ in range(10))
        else:
            while len(block_rows) < 20:
                candidate = generator.randrange(1 << 63, 1 << 64)
                if candidate not in seen_publics:
                    seen_publics.add(candidate)
                    block_rows.append({"kind": 1, "public": candidate, "block": block})
        generator.shuffle(block_rows)
        rows.extend(block_rows)
    for index, row in enumerate(rows):
        row["schedule_index"] = index
        row["expected_product"] = SECRET * row["public"]
        row["public_bitlen"] = row["public"].bit_length()
    return rows


def crossed_schedule(capture: dict[str, object]) -> list[dict[str, int]]:
    """Build the protocol-bound Latin schedule for the crossed diagnostic."""
    required = {"mode", "secret_operands", "public_operands", "shuffle_seed", "blocks", "traces", "seed", "samples", "post_low"}
    missing = required.difference(capture)
    if missing:
        raise ValueError("crossed protocol capture missing: " + ", ".join(sorted(missing)))
    if capture["mode"] != "crossed_secret_public":
        raise ValueError("crossed protocol capture mode must be crossed_secret_public")
    secrets = capture["secret_operands"]
    publics = capture["public_operands"]
    if not isinstance(secrets, list) or len(secrets) != 12 or not isinstance(publics, list) or len(publics) != 24:
        raise ValueError("crossed protocol requires exactly 12 secret_operands and 24 public_operands")
    if not all(type(value) is int and (1 << 63) <= value < (1 << 64) and value & 1 for value in secrets + publics):
        raise ValueError("crossed protocol operands must be high-bit odd 64-bit integers")
    if len(set(secrets)) != len(secrets) or len(set(publics)) != len(publics):
        raise ValueError("crossed protocol secret_operands and public_operands must be distinct")
    if capture["shuffle_seed"] != 2026100207 or capture["seed"] != 2026100207:
        raise ValueError("crossed protocol seed must be 2026100207")
    if capture["blocks"] != 24 or capture["traces"] != 288:
        raise ValueError("crossed protocol requires 24 blocks and 288 traces")
    generator = random.Random(capture["shuffle_seed"])
    rows: list[dict[str, int]] = []
    for block in range(24):
        block_rows = [{"secret": secret, "secret_index": secret_index,
                       "public_index": (block + secret_index) % 24, "block": block}
                      for secret_index, secret in enumerate(secrets)]
        for row in block_rows:
            row["public"] = publics[row["public_index"]]
        generator.shuffle(block_rows)
        rows.extend(block_rows)
    for index, row in enumerate(rows):
        row["expected_product"] = row["secret"] * row["public"]
        row["schedule_index"] = index
        row["public_bitlen"] = row["public"].bit_length()
    return rows


def crossed_protocol_preflight(capture: dict[str, object], args: argparse.Namespace) -> None:
    if capture["samples"] != args.samples or capture["post_low"] != args.post_low:
        raise ValueError("--samples and --post-low must match the crossed protocol")
    hashes = {
        "firmware_hex_sha256": ROOT / "simpleserial-r3-cpa-CW308_STM32F4.hex",
        "capture_script_sha256": pathlib.Path(__file__).resolve(),
        "physical_setup_sha256": args.physical_setup_record.resolve(),
    }
    for key, path in hashes.items():
        if key in capture and capture[key] != sha256_file(path):
            raise ValueError(f"crossed protocol {key} does not match {path}")


def bound_files() -> dict[str, dict[str, str]]:
    paths = {
        "firmware_elf": ROOT / "simpleserial-r3-cpa-CW308_STM32F4.elf",
        "firmware_hex": ROOT / "simpleserial-r3-cpa-CW308_STM32F4.hex",
        "firmware_listing": ROOT / "simpleserial-r3-cpa-CW308_STM32F4.lss",
        "firmware_map": ROOT / "simpleserial-r3-cpa-CW308_STM32F4.map",
        "harness": ROOT / "simpleserial-r3-cpa.c",
        "makefile": ROOT / "Makefile.r3-cpa",
        "host_vector_source": ROOT / "r3_cpa_host_vector.c",
        "host_vector_binary": ROOT / "r3_cpa_host_vector",
        "source_mp": ROOT / "vendor/r3-p500-27-m4f/mp.c",
        "source_ideal": ROOT / "vendor/r3-p500-27-m4f/ideal.c",
        "source_manifest": ROOT / "vendor/r3-p500-27-m4f/SNAPSHOT_MANIFEST.md",
        "capture_script": pathlib.Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact: " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def freeze_provenance(run: pathlib.Path, build_log: pathlib.Path, setup_record: pathlib.Path,
                      protocol_record: pathlib.Path | None) -> dict[str, object]:
    """Freeze the exact flash image and build inputs before opening hardware."""
    for path in (build_log, setup_record):
        if not path.is_file():
            raise FileNotFoundError(f"provenance input missing: {path}")
    directory = run / "provenance"
    directory.mkdir()
    frozen: dict[str, dict[str, str]] = {}
    for name, source in bound_files().items():
        path = pathlib.Path(source["path"])
        destination = directory / f"{name}_{path.name}"
        shutil.copyfile(path, destination)
        if sha256_file(destination) != source["sha256"] or sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"source changed while freezing provenance: {name}")
        frozen[name] = {"path": str(destination.relative_to(run)), "sha256": source["sha256"]}
    destination = directory / "build.log"
    shutil.copyfile(build_log, destination)
    frozen["build_log"] = {"path": str(destination.relative_to(run)), "sha256": sha256_file(destination)}
    destination = directory / "physical_setup.json"
    setup_hash = sha256_file(setup_record)
    shutil.copyfile(setup_record, destination)
    if sha256_file(destination) != setup_hash or sha256_file(setup_record) != setup_hash:
        raise RuntimeError("physical setup record changed while freezing")
    frozen["physical_setup"] = {"path": str(destination.relative_to(run)), "sha256": setup_hash}
    if protocol_record is not None:
        if not protocol_record.is_file():
            raise FileNotFoundError(f"experiment protocol missing: {protocol_record}")
        destination = directory / "experiment_protocol.json"
        protocol_hash = sha256_file(protocol_record)
        shutil.copyfile(protocol_record, destination)
        if sha256_file(destination) != protocol_hash or sha256_file(protocol_record) != protocol_hash:
            raise RuntimeError("experiment protocol changed while freezing")
        frozen["experiment_protocol"] = {"path": str(destination.relative_to(run)), "sha256": protocol_hash}
    for name, source in (("round3_source", ROOT / "vendor/r3-p500-27-m4f"),
                         ("chipwhisperer_firmware", ROOT / "vendor/chipwhisperer-firmware")):
        archive = directory / f"{name}.tar"
        with tarfile.open(archive, "w") as handle:
            handle.add(source, arcname=name)
        frozen[name] = {"path": str(archive.relative_to(run)), "sha256": sha256_file(archive)}
    commands = subprocess.run(["make", "-f", "Makefile.r3-cpa", "PLATFORM=CW308_STM32F4", "-Bn"],
                              cwd=ROOT, text=True, capture_output=True, check=True)
    command_log = directory / "build_commands.log"
    command_log.write_text(commands.stdout + commands.stderr, encoding="utf-8")
    frozen["build_commands"] = {"path": str(command_log.relative_to(run)), "sha256": sha256_file(command_log)}
    compiler = subprocess.check_output(["arm-none-eabi-gcc", "--version"], text=True).splitlines()[0]
    return {"files": frozen, "compiler": compiler, "source_commit_declared": "6d017708db403bf83977fa70770fc4f7f9e9ff21",
            "source_commit_derivation": "unverified; SHA-256-bound local snapshot"}


def _ack(target, command: str, payload: bytes = b"") -> list[int]:
    target.simpleserial_write(command, bytearray(payload))
    reply = target.simpleserial_read("r", 1, timeout=5_000)
    if reply != bytearray([1]):
        raise RuntimeError(f"{command} ACK failed: {list(reply or [])}")
    return list(reply)


def _reset(target, command: str) -> None:
    target.simpleserial_write(command, bytearray())
    status = target.simpleserial_wait_ack(timeout=5_000)
    if status != 0:
        raise RuntimeError(f"{command} reset ACK failed: {status}")


def load_secret(target, value: int = SECRET) -> None:
    _reset(target, "K")
    frame = operand_frame(value)
    for offset in range(0, FRAME_BYTES, CHUNK_BYTES):
        _ack(target, "k", frame[offset:offset + CHUNK_BYTES])


def load_public(target, value: int) -> None:
    _reset(target, "x")
    frame = operand_frame(value)
    for offset in range(0, FRAME_BYTES, CHUNK_BYTES):
        _ack(target, "a", frame[offset:offset + CHUNK_BYTES])


def board_result(target) -> bytes:
    target.simpleserial_write("l", bytearray())
    size = target.simpleserial_read("r", 3, timeout=5_000)
    if size is None or len(size) != 3 or size[0] != 1:
        raise RuntimeError(f"result size reply invalid: {list(size or [])}")
    length = (size[1] << 8) | size[2]
    if length != RESULT_BYTES:
        raise RuntimeError(f"board result reports {length} bytes, expected {RESULT_BYTES}")
    output = bytearray()
    for index in range(RESULT_CHUNKS):
        target.simpleserial_write("g", bytearray([index]))
        block = target.simpleserial_read("r", CHUNK_BYTES, timeout=5_000)
        if block is None or len(block) != CHUNK_BYTES:
            raise RuntimeError(f"result chunk {index} missing")
        output.extend(block)
    return bytes(output[:length])


def scope_overflow(scope) -> bool:
    if not hasattr(scope.adc, "fifoOverflow"):
        raise RuntimeError("Husky fifo overflow API scope.adc.fifoOverflow is unavailable")
    return bool(scope.adc.fifoOverflow())


def execute_attempt(scope, target, row: dict[str, int], path: pathlib.Path, samples: int, min_swing: float | None,
                    post_low: int, reload_secret: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    if reload_secret:
        load_secret(target, row["secret"])
    load_public(target, row["public"])
    target.flush()
    scope.arm()
    target.simpleserial_write("m", bytearray())
    timeout = bool(scope.capture())
    wave = np.asarray(scope.get_last_trace(), dtype=np.float32)
    np.save(path, wave)
    trig_count = int(scope.adc.trig_count)
    overflow = scope_overflow(scope)
    ack = target.simpleserial_read("r", 1, timeout=5_000)
    decoded = decode_result(board_result(target), row["expected_product"])
    swing = float(np.ptp(wave)) if wave.size else 0.0
    post_low_observed = samples - trig_count
    checks = {
        "capture_timeout": not timeout,
        "ack": ack == bytearray([1]),
        "sample_count": wave.size == samples,
        "finite": bool(np.isfinite(wave).all()),
        "trigger_seen": trig_count > 0,
        "post_low": post_low_observed >= post_low,
        "fifo_overflow": not overflow,
        "board_product": bool(decoded["exact"]),
    }
    if min_swing is not None:
        checks["swing"] = swing >= min_swing
    return wave, {
        "ack": list(ack or []), "accepted": all(checks.values()), "checks": checks,
        "wave_path": str(path.relative_to(path.parent.parent)), "wave_sha256": sha256_file(path), "trace_swing": swing,
        "fifo_overflow": overflow, "trigger_high_count": trig_count,
        "post_low_observed": post_low_observed, "board": decoded,
    }


def diagnostic(path: pathlib.Path, traces: np.ndarray, groups: np.ndarray, run_id: str, contrast_hw_store_j0: bool) -> None:
    import matplotlib.pyplot as plt

    fixed, random_rows = traces[groups == 0], traces[groups == 1]
    mean_fixed, mean_random = fixed.mean(axis=0), random_rows.mean(axis=0)
    noise = (fixed.var(axis=0) + random_rows.var(axis=0)) / 2
    snr = np.divide((mean_fixed - mean_random) ** 2, noise, out=np.zeros_like(noise), where=noise > 0)
    poi = int(np.argmax(snr))
    begin, end = max(0, poi - 2), min(traces.shape[1], poi + 3)
    feature = traces[:, begin:end].mean(axis=1)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    labels = ["HW0", "HW32"] if contrast_hw_store_j0 else ["fixed", "random"]
    axes[0, 0].plot(snr); axes[0, 0].axvline(poi, color="tab:red"); axes[0, 0].set_title("Group count-SNR and POI")
    axes[0, 1].plot(mean_fixed, label=labels[0]); axes[0, 1].plot(mean_random, label=labels[1]); axes[0, 1].legend(); axes[0, 1].set_title("Mean trace by label")
    axes[1, 0].scatter(groups, feature, s=12); axes[1, 0].set_xticks([0, 1], labels); axes[1, 0].set_title("POI feature by label")
    axes[1, 1].scatter(np.arange(feature.size), feature, c=groups, s=12); axes[1, 1].set_title("POI feature across schedule")
    figure.suptitle(f"Measured ibz_mul diagnostic ({run_id}) — {traces.shape[0]} traces")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def crossed_diagnostic(path: pathlib.Path, traces: np.ndarray, run_id: str) -> None:
    import matplotlib.pyplot as plt

    mean, deviation = traces.mean(axis=0), traces.std(axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140, sharex=True)
    axes[0].plot(mean, label="mean")
    axes[0].fill_between(np.arange(mean.size), mean - deviation, mean + deviation, alpha=0.3, label="±1 standard deviation")
    axes[0].legend(); axes[0].set_title("Trace mean and variability")
    axes[1].plot(traces[0]); axes[1].set_title("Representative trace (schedule index 0)")
    figure.suptitle(f"Measured crossed-secret/public capture quality ({run_id}) — {traces.shape[0]} traces")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def artifact_entry(run: pathlib.Path, path: pathlib.Path) -> dict[str, object]:
    entry: dict[str, object] = {"filename": str(path.relative_to(run)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                                "role": "capture_evidence", "claim_scope": "round3_ibz_mul:diagnostic_measured_isolated_primitive"}
    if path.suffix == ".npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            entry.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        except (OSError, ValueError) as exc:
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def write_blocked(run: pathlib.Path, reason: str, attempts: list[dict[str, object]], config: dict, started_at: int, provenance: dict) -> None:
    """Keep failure evidence and hashes without presenting a partial run as success."""
    artifacts = [artifact_entry(run, path) for path in run.rglob("*") if path.is_file() and path.name not in ("blocked.json", "manifest.json")]
    scope_path = run / "scope_settings.json"
    write_json(run / "blocked.json", {
        "status": "blocked", "reason": reason, "run_id": run.name, "started_at_unix": started_at, "finished_at_unix": int(time.time()),
        "config_path": "config.json", "config_sha256": sha256_file(run / "config.json"),
        "evidence_origin": config["evidence_origin"], "evidence_purpose": config["evidence_purpose"],
        "blinding": config["blinding"], "measurement_scope": config["measurement_scope"], "claim_scope": config["claim_scope"],
        "claim_allowed": ["diagnostic record of a failed physical acquisition attempt"], "claim_not_allowed": config["claim_not_allowed"],
        "attempt_count": len(attempts), "attempts": attempts,
        "provenance": provenance, "scope_settings": json.loads(scope_path.read_text()) if scope_path.is_file() else None,
        "host": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__},
        "artifact_roles": artifacts, "artifact_sha256": {entry["filename"]: entry["sha256"] for entry in artifacts},
    })


def capture(args: argparse.Namespace) -> pathlib.Path:
    """Acquire one diagnostic run; every physical attempt is recorded before quality decisions."""
    if args.samples < 2 or args.post_low < 1 or args.post_low >= args.samples:
        raise ValueError("require 0 < --post-low < --samples")
    if args.min_swing is not None and args.min_swing <= 0:
        raise ValueError("--min-swing must be positive when provided")
    if not args.crossed_design and (args.traces < 20 or args.traces % 20):
        raise ValueError("--traces must be a positive multiple of 20")
    if args.contrast_hw_store_j0 and args.protocol_record is None:
        raise ValueError("contrast capture requires --protocol-record")
    if args.crossed_design and args.protocol_record is None:
        raise ValueError("crossed capture requires --protocol-record")
    if args.contrast_hw_store_j0 and args.crossed_design:
        raise ValueError("choose either contrast or crossed capture")
    crossed_capture: dict[str, object] | None = None
    if args.crossed_design:
        protocol = json.loads(args.protocol_record.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict) or not isinstance(protocol.get("capture"), dict):
            raise ValueError("crossed protocol requires a capture object")
        crossed_capture = protocol["capture"]
        rows = crossed_schedule(crossed_capture)
        if args.seed != crossed_capture["seed"] or args.traces != crossed_capture["traces"]:
            raise ValueError("--seed and --traces must match the crossed protocol")
        crossed_protocol_preflight(crossed_capture, args)
    else:
        rows = schedule(args.traces, args.seed, args.contrast_hw_store_j0)
    run = args.out.resolve()
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    started_at = int(time.time())
    (run / "attempts").mkdir(); (run / "queries").mkdir(); (run / "logs").mkdir()
    fixed_public = CONTRAST_PUBLIC_HW0 if args.contrast_hw_store_j0 else FIXED_PUBLIC
    write_json(run / "queries" / "queries.json", rows)
    config = {"run_id": run.name, "seed": args.seed, "samples": args.samples, "min_swing": args.min_swing, "post_low": args.post_low,
              "trace_count": args.traces, "chunk_bytes": CHUNK_BYTES,
              "mode": "crossed_secret_public" if args.crossed_design else "contrast_hw_store_j0" if args.contrast_hw_store_j0 else "fixed_random",
              "protocol_sha256": sha256_file(args.protocol_record.resolve()) if args.protocol_record else None,
              "contrast_public": CONTRAST_PUBLIC_HW32 if args.contrast_hw_store_j0 else None,
              "build_log_input": str(args.build_log.resolve()), "declared_source_commit": "6d017708db403bf83977fa70770fc4f7f9e9ff21",
              "source_commit_derivation": "unverified; SHA-256-bound local snapshot",
              "evidence_origin": "measured", "evidence_purpose": "diagnostic", "blinding": "unblinded",
              "measurement_scope": "isolated_primitive", "claim_scope": "round3_ibz_mul:diagnostic_measured_isolated_primitive",
              "claim_allowed": ["diagnostic measured acquisition and exact isolated ibz_mul readback on this firmware"],
              "claim_not_allowed": ["leakage_detection", "key_recovery", "full_algorithm_path", "full_kem_path", "scheme_break"]}
    if crossed_capture is not None:
        config["crossed_design"] = {key: crossed_capture[key] for key in ("secret_operands", "public_operands", "shuffle_seed", "blocks")}
    else:
        config.update({"secret": SECRET, "fixed_public": fixed_public})
    write_json(run / "config.json", config)
    attempts: list[dict[str, object]] = []
    scope = target = None
    traces = np.empty((args.traces, args.samples), dtype=np.float32)
    groups = np.empty(args.traces, dtype=np.uint8) if not args.crossed_design else None
    operands = np.empty(args.traces, dtype=np.uint64)
    secret_operands = np.empty(args.traces, dtype=np.uint64) if args.crossed_design else None
    secret_indices = np.empty(args.traces, dtype=np.uint8) if args.crossed_design else None
    public_indices = np.empty(args.traces, dtype=np.uint8) if args.crossed_design else None
    log_path = run / "logs" / "capture.log"
    provenance: dict[str, object] = {}
    try:
        provenance = freeze_provenance(run, args.build_log.resolve(), args.physical_setup_record.resolve(),
                                       args.protocol_record.resolve() if args.protocol_record else None)
        if args.protocol_record and provenance["files"]["experiment_protocol"]["sha256"] != config["protocol_sha256"]:
            raise RuntimeError("experiment protocol changed before capture")
        import chipwhisperer as cw

        scope = cw.scope(); target = cw.target(scope)
        before = {"samples": int(scope.adc.samples), "gain_db": float(scope.gain.db), "clkgen_freq": float(scope.clock.clkgen_freq), "adc_freq": float(scope.clock.adc_freq)}
        scope.default_setup(); scope.adc.samples = args.samples
        if not hasattr(scope.adc, "fifoOverflow"):
            raise RuntimeError("this capture requires a Husky scope with scope.adc.fifoOverflow")
        if int(scope.adc.decimate) != 1 or int(scope.adc.offset) != 0:
            raise RuntimeError("capture requires decimate=1 and offset=0 for trigger coverage")
        after = {"samples": int(scope.adc.samples), "decimate": int(scope.adc.decimate), "offset": int(scope.adc.offset),
                 "timeout": float(scope.adc.timeout), "gain_db": float(scope.gain.db), "clkgen_freq": float(scope.clock.clkgen_freq), "adc_freq": float(scope.clock.adc_freq),
                 "scope_serial": str(getattr(scope, "sn", "")), "trigger": str(scope.trigger.triggers), "overflow_api": "scope.adc.fifoOverflow()"}
        write_json(run / "scope_settings.json", {"before": before, "after": after})
        log_path.write_text(json.dumps({"event": "scope_settings", "before": before, "after": after}, sort_keys=True) + "\n", encoding="utf-8")
        cw.program_target(scope, cw.programmers.STM32FProgrammer, str(run / provenance["files"]["firmware_hex"]["path"]))
        time.sleep(0.2); target.flush()
        if args.crossed_design:
            warmup_row = dict(rows[0], schedule_index=-1)
        else:
            load_secret(target)
            warmup_row = {"kind": 0, "public": fixed_public, "expected_product": SECRET * fixed_public, "schedule_index": -1, "public_bitlen": 64}
        from tqdm import tqdm
        for eligible, row in tqdm([(False, warmup_row)] + [(True, row) for row in rows], file=sys.stderr, desc="physical attempts"):
            attempt_id = len(attempts)
            wave_path = run / "attempts" / f"{attempt_id:06d}.npy"
            try:
                wave, record = execute_attempt(scope, target, row, wave_path, args.samples, args.min_swing, args.post_low,
                                               reload_secret=args.crossed_design)
                record.update({"physical_attempt": attempt_id, "eligible": eligible, "reason": "first_arm_warmup" if not eligible else "scheduled_capture", "query": row})
            except Exception as exc:
                record = {"physical_attempt": attempt_id, "eligible": eligible, "reason": "first_arm_warmup" if not eligible else "scheduled_capture", "query": row,
                          "accepted": False, "error": f"{type(exc).__name__}: {exc}", "wave_path": str(wave_path.relative_to(run)) if wave_path.exists() else None,
                          "wave_sha256": sha256_file(wave_path) if wave_path.exists() else None}
                attempts.append(record); append_json(log_path, record)
                raise RuntimeError(record["error"]) from exc
            attempts.append(record); append_json(log_path, record)
            if not eligible and not record["accepted"]:
                raise RuntimeError("warm-up integrity failed: " + ", ".join(key for key, value in record["checks"].items() if not value))
            if eligible:
                if not record["accepted"]:
                    raise RuntimeError("capture integrity failed: " + ", ".join(key for key, value in record["checks"].items() if not value))
                schedule_index = row["schedule_index"]
                traces[schedule_index] = wave; operands[schedule_index] = row["public"]
                if args.crossed_design:
                    secret_operands[schedule_index] = row["secret"]
                    secret_indices[schedule_index] = row["secret_index"]
                    public_indices[schedule_index] = row["public_index"]
                else:
                    groups[schedule_index] = row["kind"]
        np.save(run / "traces.npy", traces); np.save(run / "public_operands.npy", operands)
        if args.crossed_design:
            np.save(run / "secret_operands.npy", secret_operands)
            np.save(run / "secret_indices.npy", secret_indices)
            np.save(run / "public_indices.npy", public_indices)
            crossed_diagnostic(run / "capture_diagnostic.png", traces, run.name)
        else:
            np.save(run / "fixed_random.npy", groups)
            diagnostic(run / "capture_diagnostic.png", traces, groups, run.name, args.contrast_hw_store_j0)
        accepted = [record for record in attempts if record["eligible"] and record["accepted"]]
        summary = {"backend": "hardware", "query_count": len(rows), "stage": "round3_original_ibz_mul", "rig_quality": {
            "sample_count": args.samples, "minimum_trace_swing": args.min_swing, "swing_gate": "measured-only diagnostic" if args.min_swing is None else "configured minimum",
            "required_post_low": args.post_low,
            "power_trace_swing_min": min(record["trace_swing"] for record in accepted), "power_trace_swing_mean": float(np.mean([record["trace_swing"] for record in accepted])),
            "trigger_swing_min_mv": None, "trigger_swing_evidence": "digital trigger count; analog trigger voltage not sampled",
            "trigger_high_count_min": min(record["trigger_high_count"] for record in accepted), "post_low_min": min(record["post_low_observed"] for record in accepted),
            "overflow_count": sum(bool(record["fifo_overflow"]) for record in attempts)}}
        write_json(run / "capture_summary.json", summary)
        artifacts = [artifact_entry(run, path) for path in run.rglob("*") if path.is_file() and path.name != "manifest.json"]
        if any("read_error" in entry for entry in artifacts):
            raise RuntimeError("a saved NPY artifact cannot be read")
        board = {"model": "CW308_STM32F4", "target_type": type(target).__name__, "serial": str(getattr(target, "sn", ""))}
        frozen_makefile = run / provenance["files"]["makefile"]["path"]
        compile_flags = next(line.partition("+=")[2].strip() for line in frozen_makefile.read_text().splitlines() if line.startswith("CFLAGS_LAST"))
        manifest = {"run_id": run.name, "scheme": "SQIsign Round-3 p500_27/m4f SHA-bound local snapshot", "stage": "original_ibz_mul", "backend": "hardware", "seed": args.seed,
                    "config_path": "config.json", "config_sha256": sha256_file(run / "config.json"),
                    "started_at_unix": started_at, "finished_at_unix": int(time.time()), "attempt_count": len(attempts), "status": "ok", "host": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "chipwhisperer": cw.__version__},
                    "evidence_origin": config["evidence_origin"], "evidence_purpose": config["evidence_purpose"], "blinding": config["blinding"], "measurement_scope": config["measurement_scope"],
                    "claim": config["claim_allowed"][0], "claim_scope": config["claim_scope"], "claim_allowed": config["claim_allowed"], "claim_not_allowed": config["claim_not_allowed"],
                    "provenance": {"frozen": provenance, "compile_flags_last": compile_flags,
                                   "prepared_queries": {"path": "queries/queries.json", "sha256": sha256_file(run / "queries" / "queries.json"), "query_count": len(rows)},
                                   "target_board": board,
                                   "command_route": "K/k each physical attempt; x/a each query; m; l/g full readback" if args.crossed_design else "K/k once; x/a each query; m; l/g full readback",
                                   "scope": after},
                    "artifacts": [entry["filename"] for entry in artifacts], "artifact_roles": artifacts, "artifact_sha256": {entry["filename"]: entry["sha256"] for entry in artifacts}}
        write_json(run / "manifest.json", manifest)
    except Exception as exc:
        (run / "logs" / "capture_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_blocked(run, f"{type(exc).__name__}: {exc}", attempts, config, started_at, provenance)
        raise
    finally:
        if target is not None: target.dis()
        if scope is not None: scope.dis()
    return run


def self_test() -> None:
    from types import SimpleNamespace

    class FakeSerial11:
        def __init__(self):
            self.pending = []
            self.commands = []
            self.payloads = []

        def simpleserial_write(self, command, payload):
            self.commands.append(command)
            self.payloads.append(bytes(payload))
            if command in ("K", "x"):
                assert not payload
                self.pending.append(("z", 0))
            elif command in ("k", "a"):
                assert len(payload) == CHUNK_BYTES
                self.pending.extend((("r", 1), ("z", 0)))
            else:
                raise AssertionError(command)

        def simpleserial_wait_ack(self, timeout=500):
            command, value = self.pending.pop(0)
            return value if command == "z" else None

        def simpleserial_read(self, command, length, timeout=250, ack=True):
            received, value = self.pending.pop(0)
            if received != command:
                return None
            assert length == 1 and ack and self.simpleserial_wait_ack(timeout) == 0
            return bytearray([value])

    fake = FakeSerial11()
    test_secret = (1 << 63) + 9
    load_secret(fake, test_secret)
    load_public(fake, FIXED_PUBLIC)
    assert fake.commands == ["K"] + ["k"] * 10 + ["x"] + ["a"] * 10 and not fake.pending
    assert b"".join(fake.payloads[1:11]) == operand_frame(test_secret)
    assert not scope_overflow(SimpleNamespace(adc=SimpleNamespace(fifoOverflow=lambda: False)))
    assert scope_overflow(SimpleNamespace(adc=SimpleNamespace(fifoOverflow=lambda: True)))
    assert operand_frame(SECRET)[:8] == SECRET.to_bytes(8, "little")
    assert len(operand_frame(SECRET)) == FRAME_BYTES
    rows = schedule(40, 7)
    assert all(sum(row["kind"] == 0 for row in rows[20 * block:20 * (block + 1)]) == 10 for block in range(2))
    assert all(row["public_bitlen"] == 64 for row in rows)
    contrast = schedule(200, 7, True)
    assert all(sum(row["kind"] == 0 for row in contrast[20 * block:20 * (block + 1)]) == 10 for block in range(10))
    assert all(row["public"] == (CONTRAST_PUBLIC_HW0 if row["kind"] == 0 else CONTRAST_PUBLIC_HW32) for row in contrast)
    assert ((SECRET * CONTRAST_PUBLIC_HW0) & 0xffffffff) == 0
    assert ((SECRET * CONTRAST_PUBLIC_HW32) & 0xffffffff) == 0xffffffff
    crossed_capture = {"mode": "crossed_secret_public", "secret_operands": [(1 << 63) + 2 * value + 1 for value in range(12)],
                       "public_operands": [(1 << 63) + 2 * value + 101 for value in range(24)], "shuffle_seed": 2026100207,
                       "blocks": 24, "traces": 288, "seed": 2026100207, "samples": 24400, "post_low": 8192}
    crossed = crossed_schedule(crossed_capture)
    assert len(crossed) == 288
    for block in range(24):
        block_rows = crossed[12 * block:12 * (block + 1)]
        assert {row["secret_index"] for row in block_rows} == set(range(12))
        assert all(row["public_index"] == (block + row["secret_index"]) % 24 for row in block_rows)
        assert all(row["expected_product"] == row["secret"] * row["public"] for row in block_rows)
    invalid_crossed = dict(crossed_capture, secret_operands=[crossed_capture["secret_operands"][0]] * 12)
    try:
        crossed_schedule(invalid_crossed)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate crossed secrets accepted")
    expected = SECRET * FIXED_PUBLIC
    raw = (129).to_bytes(4, "little", signed=True) + expected.to_bytes(320, "little")
    assert decode_result(raw, expected)["exact"]
    assert not decode_result((128).to_bytes(4, "little", signed=True) + expected.to_bytes(320, "little"), expected)["exact"]
    with tempfile.TemporaryDirectory() as directory:
        run = pathlib.Path(directory)
        (run / "logs").mkdir()
        (run / "logs" / "capture.log").write_text("failure\n")
        config = {"evidence_origin": "measured", "evidence_purpose": "diagnostic", "blinding": "unblinded",
                  "measurement_scope": "isolated_primitive", "claim_scope": "round3_ibz_mul:diagnostic_measured_isolated_primitive",
                  "claim_not_allowed": ["leakage_detection"]}
        write_json(run / "config.json", config)
        write_blocked(run, "self-test failure", [], config, 1, {})
        blocked = json.loads((run / "blocked.json").read_text())
        assert blocked["attempt_count"] == 0 and blocked["config_sha256"] == sha256_file(run / "config.json")
        assert blocked["artifact_sha256"]["logs/capture.log"] == sha256_file(run / "logs" / "capture.log")


def main() -> None:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--out", type=pathlib.Path, required=True)
    capture_parser.add_argument("--traces", type=int, default=20)
    capture_parser.add_argument("--seed", type=int, default=1)
    capture_parser.add_argument("--samples", type=int, required=True)
    capture_parser.add_argument("--min-swing", type=float)
    capture_parser.add_argument("--post-low", type=int, required=True)
    capture_parser.add_argument("--build-log", type=pathlib.Path, required=True)
    capture_parser.add_argument("--physical-setup-record", type=pathlib.Path, required=True)
    capture_parser.add_argument("--protocol-record", type=pathlib.Path)
    capture_parser.add_argument("--contrast-hw-store-j0", action="store_true")
    capture_parser.add_argument("--crossed-design", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        print(capture(args))


if __name__ == "__main__":
    main()
