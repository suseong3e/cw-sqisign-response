# Section 5.1 재현

이 저장소는 두 가지 검증을 제공한다. `section51_reproduce.py`는 기존 결과의
해시·수치를 빠르게 검사하고, `section51_full_replay.py`는 공개 원시 파형에서
후보·held-out 통계·사후 KAT·Figure 2를 새 출력 디렉터리에 다시 계산한다.

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

네 항목이 모두 `OK`일 때만 검증 번들을 푼다. 내부 `manifest.sha256`은 번들 손상
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

## 역사적 실험 기록

`section51-experiment-record-v1.tar.gz`는 분석 당시의 자료를 시점별로 보존한
추가 Release 자산이다. `analysis-input/`에는 공개 스케줄과 원시 파형 자산의
정확한 이름·SHA-256 참조가, `frozen-results/`에는 M·CRT 경로의 스크립트와
동결 결과가, `posthoc-truth/`에는 held-out 감사·KAT·CRT 비공개 스케줄이 있다.
KAT와 비공개 스케줄은 후보 동결 전에 분석기에 제공된 입력이 아니다.

아카이브 자체는 다음처럼 검증한다.

```bash
sha256sum -c artifact/section51-release.sha256
python3 section51_record.py --verify section51-experiment-record-v1.tar.gz
```

아카이브의 manifest는 각 멤버 해시와 원본의 저장소 상대 경로를 결속한다. raw
trace는 기존 `section51-matrix-v1.tar.zst`와 `section51-crt-v1.tar.zst`를 받는다.
장비 식별자, 벤더·소스 아카이브, 원시 trace 배열은 이 기록에 넣지 않았다.

이 아카이브는 **역사적 실험 기록**이며 단독 실행기는 아니다. 이후 세션 기록에서
논문 당시 분석 소스를 SHA-256 일치 상태로 복구했고, 경로 메타데이터를 분리한
canonical 실행 경로를 `section51_paper_v1/source/`에 추가했다.

## 원시 파형부터 전체 재실행

Python 3.12와 Linux x86-64 환경이 필요하다. 포함된 challenge oracle이 동적 링크된
x86-64 ELF이므로 다른 운영체제·아키텍처 재현은 현재 주장하지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

gh release download section51-v1 \
  --pattern section51-matrix-v1.tar.zst \
  --pattern section51-crt-v1.tar.zst \
  --pattern section51-experiment-record-v1.tar.gz \
  --dir section51-download

mkdir -p section51-data section51-record
tar --zstd -xf section51-download/section51-matrix-v1.tar.zst -C section51-data
tar --zstd -xf section51-download/section51-crt-v1.tar.zst -C section51-data
tar -xzf section51-download/section51-experiment-record-v1.tar.gz -C section51-record

python section51_full_replay.py \
  --m-data section51-data/matrix \
  --crt-data section51-data/crt \
  --kat section51-record/posthoc-truth/trial-0002/fresh_key.rsp \
  --out section51-replay
```

실행기는 기존 출력 디렉터리를 거부한다. 후보 생성에는 파형과 공개 schedule만 쓰며,
`candidate_commitment.json`을 기록한 뒤에만 held-out 감사와 실험용 KAT을 실행한다.
성공하면 `replay_result.json`의 상태가 `accepted`이고 Figure PDF/PNG가 생성된다.

run004는 다음 결과를 만들었다.

- 이벤트 136개, x/y held-out 각 17개, 순열 99개
- 관측 합 `116.80631243236115`
- wrong-window 합 `25.061060481125295`
- 논문 고정 시드 순열 최대 `80.89532561549119`
- run004 frozen-file 해시 시드 순열 최대 `81.64536918804018`
- 사후 KAT의 M/N/x/y/u 전부 exact match

두 순열 대조 최대가 모두 관측 합보다 낮다. 두 번째 값은 실행시간 등 provenance
메타데이터가 포함된 frozen-file 해시에 따라 실행마다 달라질 수 있으므로 상수로
강제하지 않고, 해당 실행의 시드 결속과 관측 합 미만 여부를 검증한다.

### run004 편차 기록

run004의 14개 계산 단계는 모두 종료 코드 0으로 끝났고 후보 동결, held-out/KAT,
Figure까지 생성했다. 최상위 실행기는 run003에서 관측한 두 번째 대조값
`82.53820607707073`을 잘못 상수로 강제해 마지막 검증에서 종료 코드 1을 반환했다.
분석 산출물은 수정하지 않은 채 최종 검증기를 실행별 유한 대조값이 관측 합보다
낮은지 확인하도록 고쳤고, 동일 run004 산출물에 다시 적용해
`replay_result.json` 상태 `accepted`를 생성했다. 따라서 run004는 수정된 실행기로
처음부터 끝까지 중단 없이 exit 0인 실행이라고 주장하지 않는다.

허용 범위는 공개 원시 파형과 canonical source로 논문 Section 5.1의 한 키
`isolated_primitive` artifact 계산을 재생성했다는 것이다. 당시 파일 경로까지 같은
byte-exact 환경, 새 키 성공률, 전체 서명 경로·전체 키 복구, SQIsign scheme break는
주장하지 않는다.
