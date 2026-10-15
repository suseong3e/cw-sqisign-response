# SQIsign Round-3 CPA offline analysis core

SQIsign Round-3의 행렬 연산을 대상으로 한 CPA 후보 순위화와 적응형 경로
선택 코드의 최소 공개본이다. 이 Git 소스 트리는 원시 파형, KAT, 비밀키, 후보 결과,
장비 제어 코드 및 출처가 아직 확정되지 않은 벤더 스냅샷을 포함하지 않는다.

## 범위

전체 프로젝트에서 주장하는 측정 범위는 원본 산술 함수를 분리 실행한
`isolated_primitive`다. PC는 정직한 서명기이고 공격자는 분석기다. 분석기에는
파형과 공개 challenge만 제공하며, KAT와 비밀 피연산자는 후보 동결 전에
제공하지 않는다.

정답을 보지 않고 검증을 통과한 후보 집합을 의미 있게 줄였지만 유일하게 만들지
못한 경우는 **부분성공**이다. 후보 수, 축소 범위, 검증 결과와 사후 KAT 포함
여부를 각각 기록해야 한다.

Section 5.1 Release는 원시 파형과 공개 schedule, 처분 가능한 실험용 KAT,
보존 소스 및 canonical 실행기를 제공한다. `section51_full_replay.py`는 새 출력
디렉터리에서 후보 생성부터 held-out 감사, 사후 KAT, Figure 2까지 다시 계산한다.
이는 한 키의 `isolated_primitive` artifact replay이며 새 키 일반화를 뜻하지 않는다.

## 구성

- `r3_cpa_matrix_rank.py`: truth-free 행렬 후보 순위화
- `r3_onekey_m_rules/`: 적응형 M 후보 경로 규칙
- `r3_inert_ideal_probe/`: 규칙이 사용하는 누설 모델 헬퍼 두 개
- `test_r3_adaptive_paths.py`: 데이터·하드웨어 비필수 회귀 검사
- `section51_reproduce.py`: 논문 Section 5.1 스냅샷 무결성·수치 검사
- `section51_full_replay.py`: 원시 파형부터 Section 5.1 전체 분석 재실행
- `section51_paper_v1/source/`: 역사 소스와 경로 독립 canonical 소스 스냅샷
- `docs/section51_reproduction.md`: 데이터와 재현 범위

## 실행

Python 3.12에서 확인했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile \
  r3_cpa_matrix_rank.py test_r3_adaptive_paths.py \
  r3_onekey_m_rules/*.py r3_inert_ideal_probe/*.py
python test_r3_adaptive_paths.py
python test_section51_reproduce.py
python test_section51_full_replay.py
```

Section 5.1 검증 번들을 받은 뒤에는 다음처럼 실행한다.

```bash
python section51_reproduce.py --bundle section51-bundle --out section51-report.json
```

최종 Release의 논문, 원시 파형, 실험 기록, 검증 번들, run005 전체 결과와
소스 커밋·실행기·자산 SHA-256은
`artifact/section51-full-replay-release.json`에 결속한다.

`section51-experiment-record-v1.tar.gz`는 Section 5.1의 공개 입력, M/CRT 후보
동결 경로, 사후 held-out 감사, 그리고 **처분 가능한 trial-0002 KAT·CRT 비공개
스케줄**을 시점별로 묶은 역사적 실험 기록이다. 원시 파형은 중복하지 않고 기존
Release 자산을 해시로 참조한다. 역사 기록 자체는 실행기가 아니며, 전체 재실행은
복구·보존한 소스와 `section51_full_replay.py`를 사용한다. 다운로드, 압축 해제,
실행 명령과 검증 범위는 `docs/section51_reproduction.md`에 명시한다.

측정 데이터는 Git 파일 크기 제한 때문에 같은 저장소의 독립 Release 자산으로
제공한다. 프로젝트 자체 라이선스는 아직 선택하지 않았으므로 명시적 라이선스가
추가되기 전에는 재사용 권한을 부여하지 않는다.
