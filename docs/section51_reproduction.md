# Section 5.1 reproduction

This repository provides two verification paths. `section51_reproduce.py`
checks the hashes and numerical values of the published validation snapshot.
`section51_full_replay.py` is the canonical analysis runner; its post-hoc KAT
stage requires a separately supplied KAT that is not part of the public release.

`section51_reproduce.py` verifies the **unblinded snapshot integrity** of the
held-out audit and Figure 2 after candidate commitment. It does not establish
generalization to unseen keys, reconstruct the complete signer, or recover the
complete secret key. Its measurement scope is `isolated_primitive`.

## Bundle format

The validation bundle contains `manifest.json`, `manifest.sha256` holding the
SHA-256 of the raw manifest bytes, and the artifacts enumerated by hash in the
manifest. The manifest uses the following fields:

```json
{
  "schema": "round3-cpa-section51-snapshot-bundle-v1",
  "files": {"heldout_audit.json": "<sha256>"},
  "experiment": {"candidate_frozen_before_heldout": true},
  "audit": {"report": "heldout_audit.json"}
}
```

Every path must be relative to the bundle. The verifier rejects the fields
`a1`, `a2`, `m1`, `m2`, `expected`, `split_v`, `signature_hex`, and `sk` in any
JSON or JSONL file. Private schedules, KAT data, and secret keys therefore do
not belong in this bundle.

`audit.report` must be bound by SHA-256 in `files`. The verifier rejects an
unbound report. It validates only the stored audit hashes, contract, and
numerical values; it does not recompute candidates or permutation statistics
from raw traces, and it explicitly rejects a `full_replay` declaration.

## Snapshot verification

Verify the downloaded final-release assets against the checksum file committed
to the source tree:

```bash
(cd section51-download && \
  sha256sum -c ../artifact/section51-full-replay-release.sha256)
```

Extract the validation bundle only after all five assets report `OK`. The
bundle's internal `manifest.sha256` detects corruption; the committed external
checksum file is the trust anchor.

```bash
python3 section51_reproduce.py --bundle path/to/bundle --out path/to/report.json
```

A passing report requires 136 events, 17 held-out x events, 17 held-out y
events, 99 permutations, and the following values within an absolute tolerance
of `1e-9`: observed sum 116.80631243236115, shifted-window sum
25.061060481125295, and permutation maximum 80.89532561549119. The report
contract matches the source audit:
`measured` / `diagnostic` / `unblinded` / `isolated_primitive` /
`r3_single_split:one_key_coordinate_candidate`. The verification scope is
`precomputed_snapshot_integrity`.

If the bundle declares the Figure 2 PDF, PNG, and renderer, the verifier checks
all three manifest hashes and reports `verified_precomputed`. If the bundle
does not declare the figure, the verifier reports `not_generated` instead of
guessing or generating a replacement.

## Public reproduction assets

Python 3.12 on Linux x86-64 is required. The included challenge oracle is a
dynamically linked x86-64 ELF binary; portability to other operating systems or
architectures is outside the current claim.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

gh release download section51-full-replay-v1 \
  --pattern Power_Analysis_on_SQIsign.pdf \
  --pattern section51-full-replay-v1.tar.zst \
  --pattern section51-matrix-v1.tar.zst \
  --pattern section51-crt-v1.tar.zst \
  --pattern section51-validation-v1.tar.zst \
  --dir section51-download

(cd section51-download && \
  sha256sum -c ../artifact/section51-full-replay-release.sha256)

mkdir -p section51-data section51-validation section51-run005
tar --zstd -xf section51-download/section51-matrix-v1.tar.zst -C section51-data
tar --zstd -xf section51-download/section51-crt-v1.tar.zst -C section51-data
tar --zstd -xf section51-download/section51-validation-v1.tar.zst -C section51-validation
tar --zstd -xf section51-download/section51-full-replay-v1.tar.zst -C section51-run005
```

`section51-full-replay-v1.tar.zst` contains the clean run005 command log,
candidate commitment, held-out audit, result JSON files, and Figure 2 in PDF and
PNG formats. KAT data and private schedules are not included in the public
release.

Clean run005 completed from start to finish with exit code 0 and produced:

- 136 events, 17 held-out x events, 17 held-out y events, and 99 permutations
- observed sum: `116.80631243236115`
- wrong-window sum: `25.061060481125295`
- paper fixed-seed permutation maximum: `80.89532561549119`
- run005 frozen-file-hash-seeded permutation maximum: `87.05374536975015`
- exact post-hoc KAT matches for M, N, x, y, and u

Both permutation-control maxima are below the observed sum. The second control
is seeded from frozen-file hashes that include provenance metadata such as run
time, so its exact value may vary between executions. The verifier binds the
seed for each run and checks that the resulting maximum remains below the
observed sum instead of enforcing a universal constant.

The supported claim is that the one-key Section 5.1 `isolated_primitive`
artifact computation was reproduced from the published measured traces and
canonical source. Byte-identical reconstruction of historical paths,
unseen-key success rates, full-signer execution, complete key recovery, and a
break of SQIsign are outside the claim.
