# experiments_v4 — AuditRAG v4 실험 하네스

서비스의 `src.rag.retrieval.retrieve()`를 그대로 호출하는 실험 스크립트다(지시서 §6).
실데이터·실행 결과(`data/ runs/ results/ cache/`)는 커밋하지 않는다.

## 준비

- 케이스 색인: 서비스 인덱싱으로 `hdr-qwen3` 프로필 생성 → 실험 프로필 추가
  `python -m scripts.build_profiles <CASE_ID> --profiles nohdr-qwen3 hdr-bgem3` (프로젝트 루트에서)
- `config.yaml`의 `case_id`, `b_ratio` 확인
- `data/queries.jsonl` 작성 (감사인, 시스템 결과를 보기 전에 확정)
  `{"qid": "q001", "text": "...", "type": "exact|semantic|relation", "split": "test|dev", "query_lang": "ko", "note": ""}`

## 실행 순서 (`experiments_v4/` 에서)

```bash
python -m ev4 corpus                                  # T1
python -m ev4 run --split dev --runs R2 R4 R6 R8 R9   # 동작 점검 (모델 선정 아님)
ollama stop gemma4:12b                                # 검색 실험은 LLM 내린 상태 (V11)
python -m ev4 run --split test --runs all
python -m ev4 pool --split test                       # data/judgments/pool_A.csv, pool_B.csv
#   → 감사인 판정 (grade 0/1/2, annotator)
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
| `results/T1_corpus.md` | 단위 통계 (색인 기준) |
| `results/tables.md` | T2 사다리 · T3 헤더/비교(Holm) · T4 유형 · T5 증거 언어 · T6 모델 |
| `results/per_query.csv`, `comparisons.csv` | 질의별 지표, 비교 p값 |
| `results/agreement.json` | 판정자 일치도 (정확·관련성≥1·QWK) |
| `results/T7_latency.md` | 단계별 p50/p95, 재순위 절단 비율, 생성 시간, VRAM |
| `results/F1_failures.csv`, `F1_summary.md` | 실패 분석 시트와 분류 집계 |
| `results/E1_env.md`, `env.json` | 실험 환경 |
| `runs/pool_provenance.jsonl` | 판정 대상 단위의 출처 실행 (블라인드용 분리) |

평가 규칙: 관련 단위(등급 ≥1)가 없는 test 질의는 평가에서 빼고 표에 수를 적는다.
`recall@10-l2` 는 등급 2 단위가 있는 질의만 쓴다.
