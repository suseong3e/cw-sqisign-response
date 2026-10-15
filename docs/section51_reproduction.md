# Section 5.1 스냅샷 검증

`section51_reproduce.py`는 후보 동결 뒤 저장된 held-out 감사와 Figure 2의
**비맹검 스냅샷 무결성**을 검증한다. 이는 새 키 일반화, 전체 서명기 재구성 또는 전체 키 복구를 주장하지
않으며, 측정 범위는 `isolated_primitive`다.

## 번들 형식

번들 디렉터리는 `manifest.json`, 그 원시 바이트 SHA-256만 담은
`manifest.sha256`, 그리고 manifest가 해시로 열거한 산출물을 가진다. manifest는
다음 필드를 사용한다.

```json
{
  "schema": "round3-cpa-section51-snapshot-bundle-v1",
  "files": {"heldout_audit.json": "<sha256>"},
  "experiment": {"candidate_frozen_before_heldout": true},
  "audit": {"report": "heldout_audit.json"}
}
```

모든 경로는 번들 내부 상대 경로여야 한다. 모든 JSON/JSONL의 `a1`, `a2`, `m1`,
`m2`, `expected`, `split_v`, `signature_hex`, `sk` 필드는 거부된다. 따라서 비밀
스케줄, KAT, 개인키를 이 번들에 넣지 않는다.

`audit.report`는 `files`에 SHA-256으로 결속되어야 한다. 검증기는 결속되지 않은
보고서를 거부한다. 원시 파형에서 후보나 순열 통계를 다시 계산하지 않고 저장된
감사의 해시·계약·수치만 검증하며, `full_replay` 선언도 명시적으로 거부한다.

## 실행

Release 자산을 받은 디렉터리에서 소스 트리의 체크섬 파일로 먼저 검증한다.

```bash
sha256sum -c /path/to/repo/artifact/section51-release.sha256
```

세 항목이 모두 `OK`일 때만 검증 번들을 푼다. 내부 `manifest.sha256`은 번들 손상
검사용이며, 외부 신뢰 기준은 커밋된 이 체크섬 파일이다.


```bash
python3 section51_reproduce.py --bundle path/to/bundle --out path/to/report.json
```

통과하려면 이벤트 136개, x/y held-out 각 17개, 순열 99개 및 다음 합계가 절대 오차
`1e-9` 이내여야 한다: 관측 116.80631243236115, shifted-window
25.061060481125295, 순열 최대 80.89532561549119. 보고서의 계약은 원본 감사와 같은 `measured` / `diagnostic` / `unblinded` /
`isolated_primitive` / `r3_single_split:one_key_coordinate_candidate`이며, 검증 범위는
`precomputed_snapshot_integrity`다.

검증 번들이 Figure 2 PDF/PNG와 렌더러를 선언하면 세 파일의 manifest 해시를
검증하고 `verified_precomputed`로 기록한다. 번들이 그림을 선언하지 않으면 새 그림을
추측해 생성하지 않고 `not_generated`로 기록한다.
