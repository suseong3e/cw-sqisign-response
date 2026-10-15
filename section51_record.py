#!/usr/bin/env python3
"""Build or verify the Section 5.1 historical experiment record."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
KAT_SHA256 = "96c6d3d48331101ef3bb19068d73458a410c9294d0fca27780a48e8b87d85e6b"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    """Archive names mapped to repo-relative source paths; this is the publication allowlist."""
    m = "runs/20261007T025549776051Z-r3-onekey-m-only/evidence"
    crt = "runs/20261007T045953978815Z-r3-thirdkey-crt-manual"
    capture = "runs/20261007T031239169793Z-r3-onekey-trial2-crt-main-target-first"
    plan = ".plans/2026-10-07-E-r3-single-split-cpa"
    result = {
        "analysis-input/m/public_schedule.jsonl": f"{m}/inputs/m/public_schedule.jsonl",
        "analysis-input/m/manifest.json": f"{m}/inputs/m/manifest.json",
        "analysis-input/crt/public_schedule.jsonl": f"{crt}/inputs/crt/public_schedule.jsonl",
        "analysis-input/crt/manifest.json": f"{crt}/inputs/crt/manifest.json",
        "frozen-results/crt/subtraction_poi.json": f"{crt}/subtraction_poi.json",
        "frozen-results/crt/single_split_frozen.json": f"{crt}/single_split_frozen.json",
        "frozen-results/crt/single_split_corrected_frozen.json": f"{crt}/single_split_corrected_frozen.json",
        "frozen-results/crt/high_gate_corrected.json": f"{crt}/high_gate_corrected.json",
        "frozen-results/crt/high_norm.json": f"{crt}/high_norm.json",
        "posthoc-truth/heldout_audit.json": f"{crt}/heldout_audit.json",
        "posthoc-truth/posthoc_kat.json": f"{crt}/posthoc_kat.json",
        "posthoc-truth/trial-0002/fresh_key.rsp": "runs/20261006T200749935815Z-r3-onekey-trial-0002/private/fresh_key.rsp",
        "posthoc-truth/trial-0002/config.json": "runs/20261006T200749935815Z-r3-onekey-trial-0002/config.json",
        "posthoc-truth/trial-0002/crt_private_schedule.jsonl": f"{capture}/schedule.jsonl",
        "provenance/crt-analysis-config.json": f"{crt}/config.json",
        "provenance/capture/config.json": f"{capture}/config.json",
        "provenance/capture/protocol.json": f"{capture}/protocol.json",
        "provenance/capture/capture_summary.json": f"{capture}/capture_summary.json",
        "provenance/capture/build.log": f"{capture}/provenance/build.log",
        "provenance/capture/build_commands.log": f"{capture}/provenance/build_commands.log",
        "provenance/capture/capture_script_r3_cpa_capture.py": f"{capture}/provenance/capture_script_r3_cpa_capture.py",
        "provenance/capture/r3_cpa_crt_capture.py": f"{capture}/provenance/r3_cpa_crt_capture.py",
        "provenance/capture/experiment_protocol.json": f"{capture}/provenance/experiment_protocol.json",
        "provenance/capture/firmware.elf": f"{capture}/provenance/firmware_elf_simpleserial-r3-cpa-CW308_STM32F4.elf",
        "provenance/capture/firmware.hex": f"{capture}/provenance/firmware_hex_simpleserial-r3-cpa-CW308_STM32F4.hex",
        "provenance/capture/harness_simpleserial-r3-cpa.c": f"{capture}/provenance/harness_simpleserial-r3-cpa.c",
    }
    for n in ("t05-m-product-groups", "t06-m-add-anchors", "t07-m00-groups", "t08-m10-groups"):
        for ext in ("py", "json"):
            result[f"frozen-results/m/{n}.{ext}"] = f"{m}/{n}.{ext}"
    for n in ("t40-third-m-full-product-branch", "t41-third-m-branch-screen", "t44-third-m-freeze-bound"):
        for ext in ("py", "json"):
            source = f"{plan}/{n}.{ext}"
            if (ROOT / source).is_file():
                result[f"frozen-results/m/{n}.{ext}"] = source
    result["frozen-results/m/t44-third-m-candidates-bound.json"] = f"{plan}/t44-third-m-candidates-bound.json"
    for n in ("t45-third-high-gate-diagnostic", "t46-third-row6-local-stores", "t47-third-inverse-dynamic-poi", "t48-third-low-u-refreeze"):
        for ext in ("py", "json"):
            source = f"{plan}/{n}.{ext}"
            if (ROOT / source).is_file():
                result[f"frozen-results/crt/{n}.{ext}"] = source
    for n in ("run-poi.sh", "run-single-split.sh", "run-high-gate.sh", "run-high-gate-corrected.sh", "run-high-norm.sh", "run-heldout.sh"):
        result[f"provenance/wrappers/{n}"] = f"{crt}/{n}"
    for n in ("r3_cpa_onekey_poi.py", "r3_cpa_onekey_single_split.py", "r3_cpa_onekey_high_gate.py", "r3_cpa_onekey_high_norm.py", "r3_cpa_onekey_single_split_audit.py"):
        result[f"provenance/current-analysis-source/{n}"] = n
    return result


README = """# Section 5.1 historical experiment record\n\nThis archive is a publication record for the measured Section 5.1\nanalysis chronology.  It is **not** an executable or bit-exact full replay.\nThe two raw-trace Release assets are referenced in `trace-assets.json` rather\nthan duplicated here.\n\nPhase order: `analysis-input` (public trace schedules/manifests) →\n`frozen-results` (M then CRT candidate paths) → `posthoc-truth` (held-out\naudit and disposable trial-0002 truth).  The private schedule and KAT are\nintentionally only in `posthoc-truth`; they were not analyst input before\ncandidate freezing.\n\n`provenance/current-analysis-source` preserves the currently available source\nfor inspection.  Its hashes differ from the hashes bound into most historical\nCRT output reports, and the exact historical held-out/posthoc source is not\npresent in this repository.  Thus use the archive to inspect, verify hashes,\nand reproduce the recorded evidence flow—not to claim an executable full CPA\nreplay.  Paths in stored wrappers are historical local paths.\n\nThe trial-0002 KAT is a disposable experiment key, published under the user's\nexplicit authorization.  No vendor/source archives, device serials, or raw\ntrace arrays are included.\n"""


def metadata(files):
    return {
        "schema": "round3-cpa-section51-historical-record-v1",
        "record_kind": "historical_experiment_record",
        "executable_full_replay": False,
        "evidence_contract": {
            "evidence_origin": "measured",
            "evidence_purpose": "diagnostic",
            "blinding": "unblinded",
            "blinding_detail": "precomputed snapshot of candidate-frozen heldout validation",
            "measurement_scope": "isolated_primitive",
            "claim_scope": "r3_single_split:one_key_coordinate_candidate",
        },
        "measurement_scope": "isolated_primitive",
        "analysis_phase_order": ["analysis-input", "frozen-results", "posthoc-truth"],
        "trace_assets": {
            "section51-matrix-v1.tar.zst": "7d77a132bae0a38641cc2637baf80a7835e55cc71e0aeea78fc3d771ae11fe74",
            "section51-crt-v1.tar.zst": "2e820779e186faa7fe5b235b6f5d17818ce7d38e698571a8ec83cea7b0de76e1",
        },
        "known_limitations": [
            "Historical CRT output script hashes differ from current preserved sources.",
            "The exact historical held-out/posthoc analyzer source is unavailable.",
        ],
        "files": files,
    }


def safe_name(name):
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and name == str(path)


def build(output):
    entries = sources()
    blobs = {"README.md": README.encode(), "trace-assets.json": json.dumps(metadata({})["trace_assets"], indent=2, sort_keys=True).encode() + b"\n"}
    file_meta = {}
    for arcname, source in entries.items():
        path = ROOT / source
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing or unsafe source: {source}")
        data = path.read_bytes()
        blobs[arcname] = data
        file_meta[arcname] = {"sha256": sha256_bytes(data), "bytes": len(data), "source": source}
    kat = blobs["posthoc-truth/trial-0002/fresh_key.rsp"]
    if sha256_bytes(kat) != KAT_SHA256:
        raise SystemExit("trial-0002 KAT hash mismatch")
    for name, data in blobs.items():
        if not safe_name(name):
            raise SystemExit(f"unsafe archive member: {name}")
        if name.endswith(".rsp") and not name.startswith("posthoc-truth/"):
            raise SystemExit("KAT outside posthoc-truth")
        if name.endswith(".jsonl") and "private" in name and not name.startswith("posthoc-truth/"):
            raise SystemExit("private schedule outside posthoc-truth")
        file_meta.setdefault(name, {"sha256": sha256_bytes(data), "bytes": len(data), "source": "generated"})
    manifest = json.dumps(metadata(file_meta), indent=2, sort_keys=True).encode() + b"\n"
    blobs["manifest.json"] = manifest
    blobs["manifest.sha256"] = (sha256_bytes(manifest) + "  manifest.json\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w|") as archive:
        for name in sorted(blobs):
            info = tarfile.TarInfo(name)
            info.size = len(blobs[name])
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(blobs[name]))
    verify(output)


def verify(archive_path):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not member.isfile() or not safe_name(member.name) for member in members):
            raise SystemExit("archive contains a non-file or unsafe member")
        if len({member.name for member in members}) != len(members):
            raise SystemExit("archive contains duplicate member names")
        names = {member.name for member in members}
        if "manifest.json" not in names or "manifest.sha256" not in names:
            raise SystemExit("archive has no manifest")
        read = lambda name: archive.extractfile(name).read()
        manifest = read("manifest.json")
        expected = read("manifest.sha256").decode().split()[0]
        if sha256_bytes(manifest) != expected:
            raise SystemExit("manifest hash mismatch")
        listed = json.loads(manifest)["files"]
        if set(listed) != names - {"manifest.json", "manifest.sha256"}:
            raise SystemExit("manifest member set mismatch")
        for name, detail in listed.items():
            data = read(name)
            if len(data) != detail["bytes"] or sha256_bytes(data) != detail["sha256"]:
                raise SystemExit(f"member hash mismatch: {name}")
        if sha256_bytes(read("posthoc-truth/trial-0002/fresh_key.rsp")) != KAT_SHA256:
            raise SystemExit("KAT hash mismatch")
        if any(name.endswith(".rsp") and not name.startswith("posthoc-truth/") for name in names):
            raise SystemExit("KAT placement violation")
        if any(name.endswith(".jsonl") and "private" in name and not name.startswith("posthoc-truth/") for name in names):
            raise SystemExit("private schedule placement violation")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.out:
        build(args.out)
        print(f"{sha256_file(args.out)}  {args.out}")
    else:
        verify(args.verify)
        print(f"OK  {args.verify}")


if __name__ == "__main__":
    main()
