# experiments_v4 — AuditRAG v4 실험 하네스

서비스의 `src.rag.retrieval.retrieve()`를 그대로 호출하는 실험 스크립트다(지시서 §6).
실데이터·실행 결과(`data/ runs/ results/ cache/`)는 커밋하지 않는다.

## 준비

- 케이스 색인: 서비스 인덱싱으로 `hdr-qwen3` 프로필 생성 → 실험 프로필 추가
  `python -m scripts.build_profiles <CASE_ID> --profiles nohdr-qwen3 hdr-bgem3` (프로젝트 루트에서)
- `config.yaml`의 `case_id`, `b_ratio` 확인
- `data/queries.jsonl` 작성 (감사인, 시스템 결과를 보기 전에 확정)
  `{"qid": "q001", "text": "...", "type": "exact|semantic|relation", "split": "test|dev", "query_lang": "ko",
  "target_evidence_lang": "zh", "source_question": "2.6", "note": ""}`
  - `target_evidence_lang`(선택): 질의를 쓸 때 예상한 결정적 증거 언어. 현지어 증거 질의를 5개 이상
    확보했는지 판정 전에 확인하는 용도. 최종 언어 분해는 정답(등급 2 단위 언어)으로 한다.
  - `source_question`(선택): 부록 A 감사인 질문 번호(예 "1.3"). 없으면 빈칸.
  - 입력 검증(경고만, 실행은 계속): test 60~80개, dev 20개, test 유형별 20개 이상,
    현지어 증거 예상 test 질의 5개 이상. `python -m ev4 queries` 로 T0 와 경고를 본다.

### 질의 3유형 (논문 부록 [표 A-1])

| `type` | 논문 이름 | 정의 | 주요 검색 신호 |
|---|---|---|---|
| `exact` | 정확 일치 | 인명, 업체명, 계약번호, 금액, 특정 문구처럼 본문이나 파일명에 같은 문자열로 나타나는 대상을 찾는 질의 | 문자열 일치(BM25) |
| `semantic` | 정황 | 본문에 직접 드러나지 않는 행위, 회피 표현, 사내 은어를 서술하여 찾는 질의 | 의미 유사도(밀집 검색) |
| `relation` | 관계 | 특정 인물이나 업체 사이 소통, 또는 날짜 선후를 조건으로 하는 질의 | 메타데이터 헤더와 의미 유사도 |

## 실행 순서 (`experiments_v4/` 에서)

```bash
python -m ev4 corpus                                  # T1
python -m ev4 queries                                 # T0 (판정 전) + 입력 검증 경고
python -m ev4 run --split dev --runs R2 R4 R6 R8 R9   # 동작 점검 (모델 선정 아님)
ollama stop gemma4:12b                                # 검색 실험은 LLM 내린 상태 (V11)
python -m ev4 run --split test --runs all
python -m ev4 pool --split test                       # data/judgments/pool_A.csv, pool_B.csv
#   → 판정기준.md 를 보고 감사인 판정 (grade 0/1/2, annotator) · 풀 통계 results/J1_pool.md
python -m ev4 judgments import data/judgments/pool_A.csv data/judgments/pool_B.csv
python -m ev4 judgments agreement                     # 불일치 → data/judgments/adjudication.csv
#   → adjudicated_grade 입력
python -m ev4 judgments finalize                      # data/qrels.txt
python -m ev4 evaluate                                # T2~T6 results/tables.md
python -m ev4 latency                                 # T7 (세 모델 적재)
python -m ev4 failures                                # F1 (category 는 사용자가 입력, 재실행해도 보존)
python -m ev4 env                                     # E1
```

## selftest (GPU·실데이터 없이)

```bash
AST_FAKE_MODELS=1 python -m ev4 selftest          # 임시 폴더, 끝나면 삭제
python -m ev4 selftest --keep ../data/ev4_selftest  # 산출물 남기기
pytest experiments_v4/tests -q                    # 프로젝트 루트에서
```

## 산출물

| 파일 | 내용 |
|---|---|
| `results/T0_queries.md` | 질의 구성 T0 (유형×평가/개발, 현지어 증거 예상·정답), query_lang·source_question 분포, 입력 검증 경고 |
| `results/T1_corpus.md` | 단위 통계 (색인 기준) |
| `results/tables.md` | T2 구성별 성능 · T3 헤더 효과(유형별 차이) · 사전 지정 비교 11개 · T4 유형 · T5 증거 언어(Recall@10) · T6 R9 vs R8 |
| `results/per_query.csv`, `comparisons.csv` | 질의별 지표 / 비교별 nDCG@10 차이·짝지은 bootstrap 95% CI·fisher p·Holm p·교차 확인 p |
| `results/J1_pool.md` | 판정 풀 크기(질의당 평균·최소·최대), 번역·절단 단위 수 |
| `results/agreement.json`, `J3_agreement.md` | 일치도 (완전 일치·관련성≥1·QWK n_levels=3, 불일치 수, B 판정 질의 비율·유형별) |
| `results/J4_qrels.md` | 최종 정답 통계 (등급별 단위 수, 등급 2 없는 질의) |
| `data/judgments/판정기준.md` | 등급 기준([표 9])과 질의 유형 정의 (pool 이 생성) |
| `results/T7_latency.md` | 단계별 p50/p95, 재순위 절단 비율, 생성 시간, VRAM |
| `results/F1_failures.csv`, `F1_summary.md` | 놓친 등급 2 단위 전체 시트(category_hint 자동·category 수동, 질의 문장 포함 → P0) / 분류별 단위 수·비율 |
| `results/E1_env.md`, `env.json` | 실험 환경 |
| `runs/pool_provenance.jsonl` | 판정 대상 단위의 출처 실행 (블라인드용 분리) |

평가 규칙: 관련 단위(등급 ≥1)가 없는 test 질의는 평가에서 빼고 표에 수를 적는다.
`recall@10-l2` 와 증거 언어 분해(T5·T6 언어 행)는 등급 2 단위가 있는 질의만 쓴다.
질의 5개 미만 그룹은 `†사례 수준` 표시를 붙인다. ranx fisher 는 |평균 차이|를 비교하는 양측 검정이다.
