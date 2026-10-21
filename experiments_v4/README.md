# experiments_v4 — AuditRAG v4 실험 하네스

서비스의 `src.rag.retrieval.retrieve()`를 그대로 호출하는 실험 스크립트다(지시서 §6).
실데이터·실행 결과(`data/ runs/ results/ cache/`)는 커밋하지 않는다.

## 준비

- 케이스 색인(v4.4 §1): 프로필은 `hdr-qwen3`·`nohdr-qwen3` 두 개만(bge-m3 없음). 서비스 인덱싱으로 `hdr-qwen3` 를
  먼저 완성하고 R1·R2·R3·R5 를 실행한 뒤 `nohdr-qwen3` 를 추가해 R4 를 실행한다
  `python -m scripts.build_profiles <CASE_ID> --profiles nohdr-qwen3 --progress ../data/progress/build.json`
  (프로젝트 루트에서, 임베딩 체크포인트로 중단 지점부터 이어 감, 진행 건수 JSON)
- `config.yaml`의 `case_id` 확인. `primary_subset: search`(주 분석 집합)는 실행 전에 고정하고 바꾸지 않는다
  (첫 `run` 이 `runs/primary_subset.json` 에 기록, 이후 값이 다르면 run·evaluate·failures 가 멈춘다)
- 질의: 감사인 표준 질문 45개(논문 부록 A, v4.3)를 그대로 쓴다 — `docs/v4/data/queries_45.jsonl` →
  `data/queries.jsonl`. 사건별 맞춤 질의·개발용(dev) 질의는 없다(모두 `split: test`).
  `{"qid": "q1.1", "text": "...", "area": "1", "support_class": "A1", "support_group": "search",
  "type": "semantic", "split": "test", "query_lang": "ko", "target_evidence_lang": "", "source_question": "1.1", "note": ""}`
  - `support_class`: A1·A2·C·D·B·E (v4.4 §3, 이전 A·A′). `support_group`: A1·A2 → `search`(17, A1 9·A2 8), C → `meta`(3), D·B·E → `nonsearch`(25)
  - `type`: `search` 집합만 exact 2·semantic 9·relation 6, 나머지는 `none`
  - `target_evidence_lang`: 빈칸. 증거 언어는 판정 후 등급 2 단위로 정한다
  - 입력 검증: 중복 qid, 분류 6개 밖, 분류↔집합 불일치, search 인데 유형 없음, 나머지인데 `none` 아님은
    오류로 멈춘다. 질의가 45개가 아니면 경고. `python -m ev4 queries` 로 T0 와 경고를 본다.

### 분석 집합 (v4.3 §3, 실험 전에 고정)

| 집합 | 질의 | 용도 |
|---|---|---|
| S17 (주 분석) | `support_group = search` 17개 | 사전 지정 비교 4개의 검정·Holm 보정(nDCG@10). T2, F1, 언어별 보조 값 |
| ALL45 | 45개 전체 | 기술 통계(평균·95% CI)만. 검정하지 않는다 |
| 집합별 | search 17 · meta 3 · nonsearch 25 | T8 (분류가 실제 검색 결과와 맞는지 확인) |

### 실행 5개와 사전 비교 4개 (v4.4 §2)

`config.yaml` 의 키는 하네스 번호(TREC 파일 이름)이고, 표·보고서에는 `labels` 의 논문 번호를 쓴다.

| 논문 | 하네스 | 색인 | 구성 |
|---|---|---|---|
| R1 | R2 | hdr-qwen3 | BM25 단독 |
| R2 | R4 | hdr-qwen3 | 밀집 검색 단독 |
| R3 | R6 | hdr-qwen3 | 하이브리드(RRF k=60) |
| R4 | R7 | nohdr-qwen3 | 하이브리드 + 재순위, 헤더 없음 |
| R5 (기본) | R8 | hdr-qwen3 | 하이브리드 + 재순위, 헤더 있음 |

사전 비교(Holm 단일 계열, S17 nDCG@10): R1→R3, R2→R3, R3→R5, R4→R5. 판정 풀은 실행 5개 상위 10개 합집합.
논문 표 번호: [표 12] = T2, [표 13] = T8, [표 14] = G1, [표 15] = F1.

### 질의 3유형 (논문 부록 [표 A-1], S17 에만 적용)

| `type` | 논문 이름 | 정의 | 주요 검색 신호 |
|---|---|---|---|
| `exact` | 정확 일치 | 인명, 업체명, 계약번호, 금액, 특정 문구처럼 본문이나 파일명에 같은 문자열로 나타나는 대상을 찾는 질의 | 문자열 일치(BM25) |
| `semantic` | 정황 | 본문에 직접 드러나지 않는 행위, 회피 표현, 사내 은어를 서술하여 찾는 질의 | 의미 유사도(밀집 검색) |
| `relation` | 관계 | 특정 인물이나 업체 사이 소통, 또는 날짜 선후를 조건으로 하는 질의 | 메타데이터 헤더와 의미 유사도 |

## 실행 순서 (`experiments_v4/` 에서)

```bash
python -m ev4 corpus                                  # T1
python -m ev4 queries                                 # T0 + 입력 검증 경고
python -m ev4 smoke                                   # 동작 점검: q1.1·q2.6·q7.2, 오류·응답 시간·GPU 메모리만 (설정 변경 없음)
ollama stop gemma4:12b                                # 검색 실험은 LLM 내린 상태 (V11)
python -m ev4 run --runs R2 R4 R6 R8                  # 논문 R1·R2·R3·R5 (hdr-qwen3), primary_subset 기록
python -m ev4 run --runs R7                           # 논문 R4 (nohdr-qwen3 완성 뒤)
python -m ev4 pool                                    # data/judgments/pool_A.csv(45개), pool_B.csv(11개), config c_queries 가 있으면 pool_C.csv(B 와 겹치지 않는 11개)
#   → 판정기준.md 를 보고 감사인 판정 (grade 0/1/2, annotator) · 풀 통계·B 질의 qid results/J1_pool.md
python -m ev4 judgments import data/judgments/pool_A.csv data/judgments/pool_B.csv
python -m ev4 judgments agreement                     # 불일치 → data/judgments/adjudication.csv
#   → adjudicated_grade 입력
python -m ev4 judgments finalize                      # data/qrels.txt, J4, T0b
python -m ev4 evaluate                                # T2·사전 비교·T8·언어별 보조 값 results/tables.md
python -m ev4 latency                                 # 응답 시간(R5 한 실행, 세 모델 적재, 45개 전체) + 답변 생성 → runs/gen/answers.jsonl
python -m ev4 judge gen                               # 답변 판정 시트(S17) data/judgments/gen_{A,B,C}.md·_citations.csv·_answers.csv
#   → 판정자 A·B(·C) 가 S17 전체 독립 판정 (인용별 supported 1/0, 근거 없는 주장 수, 무근거 응답 확인, 유용성 0/1/2)
python -m ev4 judge g1                                # 일치도 → data/judgments/gen_adjudication.csv final 협의 → 다시 실행 → G1
python -m ev4 failures                                # F1 (S17 만, category 는 사용자가 입력, 재실행해도 보존)
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
| `results/T0_queries.md` | T0([표 8]) 영역 1~7·계 × 분류 A1·A2·C·D·B·E 수·search 유형별 수, 집합별 질의 수, 입력 검증 경고 |
| `results/T0b_relevant.md` | 집합별(S17·meta·nonsearch·ALL45) 등급≥1·등급 2 단위 수 평균·중앙값·최대, 관련 0개·10개 초과 질의 수 |
| `results/smoke.md` | 동작 점검 (실행별 오류 수·유형, 응답 시간, GPU 메모리 최대 — 순위·점수 없음) |
| `results/T1_corpus.md` | 단위 통계 (색인 기준) — [표 7] 언어 행은 한국어·영어·현지 언어(합산)·계(세부·und 는 집계 메모), 4.1.1절 수치에 광고 메일 제외 `ads_removed` |
| `results/tables.md` | T2([표 12]) 실행 5개 구성별 성능(S17, 부가 ALL45; nDCG@10·P@10·Recall@10·Recall@10(등급 2)·MRR@10·Recall@100) · 사전 지정 비교 4개 · T8([표 13]) 집합별 R1·R2·R3·R5 nDCG@10·P@10 · 언어별 보조 값(본문용: 등급 2 현지 언어 S17 질의 수, R5 상위 10 포함 수, 질의별 R1·R2·R5 포함 여부). T3~T6 은 만들지 않는다 |
| `results/per_query.csv`, `comparisons.csv` | 질의별 지표(45개, support_group 열) / 비교별 nDCG@10 차이·짝지은 bootstrap 95% CI·fisher p·Holm p·교차 확인 p (S17) |
| `results/J1_pool.md` | 판정 풀 크기(질의당 평균·최소·최대), 번역·절단 단위 수, B(·C) 판정 질의 qid |
| `results/agreement.json`, `J3_agreement.md` | 일치도 A–B, A–C 따로 (완전 일치·관련성≥1·QWK n_levels=3, 불일치 수, 판정 질의 비율·집합별·qid) |
| `results/J4_qrels.md` | 최종 정답 통계 (등급별 단위 수, 등급 2 없는 질의) |
| `data/judgments/판정기준.md` | 등급 기준([표 9]), 집계·비교형 질문 안내([사용자 확정]), 질의 유형 정의 (pool 이 생성) |
| `results/T7_latency.md` | v4.4 §5 보고 값(검색 전체 p50/p95, 재순위 p50, 응답 생성 포함 질의당 전체 p50, GPU 최대 사용량 nvidia-smi·torch·ollama ps, 재순위 4,096토큰 절단 비율) + 참고용 단계별 시간·생성 시간 |
| `runs/gen/answers.jsonl` | 질문 45개 답변 기록(qid, 답변 전문, 컨텍스트 단위 ID 5개, 인용 번호, 인용 → 단위 매핑, 모델 태그) — P0 |
| `data/judgments/gen_{A,B,C}*` | 답변 판정 시트(S17, 질문 순서 seed 셔플, 단위 등급 숨김, 현지 언어 단위 번역 첨부) — P0 |
| `results/G1_generation.md`, `G1.json` | [표 14] 인용 정확도(전체 합산·질문별 평균, 인용 없는 답변 수), 결정적 증거 인용률(n), 근거 없는 주장, 무근거 응답(n), 유용성 평균·분포, 일치도(인용 kappa, 유용성 QWK, 근거 없는 주장 수 완전 일치율) |
| `results/F1_failures.csv`, `F1_summary.md` | S17 의 놓친 등급 2 단위 전체 시트(category_hint 자동·category 수동, 질의 문장 포함 → P0) / 분류별 단위 수·비율 |
| `results/E1_env.md`, `env.json` | 실험 환경 (primary_subset 포함) |
| `runs/pool_provenance.jsonl` | 판정 대상 단위의 출처 실행 (블라인드용 분리) |
| `runs/primary_subset.json` | 첫 실행 때 고정한 주 분석 집합 |

평가 규칙: 관련 단위(등급 ≥1)가 없는 질의는 평가에서 빼고 표에 수를 적는다.
`recall@10-l2` 와 언어별 보조 값은 등급 2 단위가 있는 질의만 쓴다. 현지 언어 = 한국어·영어 외 단위 언어.
P@10 은 상위 10개 중 등급 1 이상 단위 비율(ranx `precision@10`).
질의 5개 미만 그룹(예: S17 exact 2개)은 `†사례 수준` 표시를 붙인다. ranx fisher 는 |평균 차이|를 비교하는 양측 검정이다.
