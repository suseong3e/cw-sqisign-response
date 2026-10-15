#!/usr/bin/env python3
"""Plot one measured CRT trace and a held-out CPA time scan."""

from pathlib import Path
import hashlib
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import r3_cpa_crt_analyze as crt
from r3_cpa_onekey_single_split_audit import ORACLE
WRONG_WINDOW_OFFSET = 30000


RUN = Path(os.environ['SECTION51_CRT_OUT'])
DATA = Path(os.environ['SECTION51_CRT_DATA'])
MATRIX = Path(os.environ['SECTION51_M_OUT']) / 't44-third-m-candidates-bound.json'
OUT = Path(os.environ['SECTION51_FIGURE_OUT'])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    audit = json.loads((RUN / "heldout_audit.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    high = json.loads((RUN / "high_norm.json").read_text())
    matrix = json.loads(MATRIX.read_text())["matrix"]
    trace_file, schedule_file = DATA / "traces.npy", DATA / "public_schedule.jsonl"
    contract = {"evidence_origin": "measured", "evidence_purpose": "diagnostic",
                "blinding": "unblinded", "measurement_scope": "isolated_primitive",
                "claim_scope": "r3_single_split:one_key_coordinate_candidate",
                "target_truth_loaded": False,
                "claim_not_allowed": ["blind_success_rate", "scheme_break"]}
    if (audit["status"] != "validated" or audit["heldout_counts"]["x"] != 17 or
            audit["evidence_contract"] != contract or
            digest(RUN / "high_norm.json") != audit["high_sha256"] or
            digest(RUN / "single_split_corrected_frozen.json") != audit["frozen_sha256"] or
            digest(MATRIX) != audit["m_candidates_sha256"] or
            digest(trace_file) != audit["target_trace_sha256"] or
            digest(schedule_file) != audit["target_public_schedule_sha256"] or
            manifest["target_trace_sha256"] != audit["target_trace_sha256"]):
        raise ValueError("measured held-out audit or source binding differs")

    rows = [json.loads(line) for line in schedule_file.read_text().splitlines()]
    traces = np.load(trace_file, mmap_mode="r", allow_pickle=False)
    if traces.shape != (len(rows), 163840):
        raise ValueError("capture shape differs")
    m = tuple(int(matrix[name], 16) for name in ("m00", "m01")) + (
        int(matrix["m10_candidates"][0], 16), int(matrix["m11"], 16))
    heldout = np.array([i for i, row in enumerate(rows) if row["heldout"]])
    public = crt.oracle_public([rows[i] for i in heldout], m, ORACLE)
    selected = np.array([j for j, i in enumerate(heldout)
                         if rows[i]["coord"] == "x" and public["split"][j] == 0])
    indices = heldout[selected]
    if len(indices) != audit["heldout_counts"]["x"]:
        raise ValueError("held-out x-coordinate branch count differs")
    x = int(high["candidate"]["x"], 16)
    labels, _ = crt.low.subtraction_leakage(x, public["x"][selected])
    event = next(e for e in audit["events"]
                 if e["coord"] == "x" and e["stage"] == "low" and e["event"] == 1)
    point, radius = event["sample"], 220
    offsets = np.arange(-radius, radius + 1)
    samples = point + offsets
    predicted = labels[1]
    observed = np.abs(crt.corr(predicted, np.asarray(traces[np.ix_(indices, samples)], float)))
    displaced = np.abs(crt.corr(predicted, np.asarray(
        traces[np.ix_(indices, samples + WRONG_WINDOW_OFFSET)], float)))
    if (not math.isclose(observed[radius], event["abs_r"], abs_tol=1e-9) or
            not math.isclose(displaced[radius], event["wrong_window_abs_r"], abs_tol=1e-9)):
        raise ValueError("plotted CPA values differ from the frozen held-out audit")

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], hspace=0.55, wspace=0.30)
    full, local, cpa = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    single = np.asarray(traces[indices[0]], float)
    bins = single.reshape(-1, 256)
    centers = np.arange(len(bins)) * 256 + 127.5
    full.fill_between(centers, bins.min(axis=1), bins.max(axis=1), color="#0072B2", alpha=0.55, linewidth=0)
    full.plot(centers, bins.mean(axis=1), color="#005A8D", linewidth=0.55)
    full.axvline(point, color="#D55E00", linewidth=1)
    full.set(xlim=(0, len(single)), ylabel="ADC amplitude (a.u.)",
             xlabel="Sample index", title="(a) One measured CRT trace (256-sample envelope)")
    local.plot(offsets, single[samples], color="#0072B2", linewidth=0.65)
    local.axvline(0, color="#D55E00", linestyle="--", linewidth=1)
    local.set(xlim=(-radius, radius), ylabel="ADC amplitude (a.u.)",
              xlabel="Sample offset from frozen store", title="(b) Raw samples near the store")
    cpa.plot(offsets, observed, color="#0072B2", linewidth=1, label="Frozen candidate")
    cpa.plot(offsets, displaced, color="0.55", linewidth=0.9, label="Shifted-window control")
    cpa.axvline(0, color="#D55E00", linestyle="--", linewidth=1)
    cpa.scatter([0], [observed[radius]], color="#D55E00", s=20, zorder=3)
    cpa.set(xlim=(-radius, radius), ylim=(0, 1.04), ylabel=r"Absolute Pearson $|r|$",
            xlabel="Relative sample offset within each window", title="(c) Held-out CPA, 17 traces")
    cpa.legend(frameon=False, fontsize=7, loc="upper left")
    for ax in (full, local, cpa):
        ax.grid(alpha=0.17)
        ax.set_axisbelow(True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"{OUT.with_suffix('.pdf')}\n{OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
