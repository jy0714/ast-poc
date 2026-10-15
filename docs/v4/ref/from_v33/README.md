# v3.3 하네스에서 가져올 코드

어제(v3.3) 만든 실험 하네스(`AuditRAG_experiments_harness_v3.3.zip`)는 **그대로 적용하지 않는다.**
청크→증거 단위 매핑, 절삭·청킹 비교, 볼록 결합, Recall@2k 등 v4에서 없앤 실험을 전제로 만들어졌기 때문이다.
아래 함수만 `experiments_v4/` 로 옮기고, 나머지는 버린다.

| 파일 | 가져올 것 | 바꿀 것 |
|---|---|---|
| `stats.py` | `bootstrap_ci`, `holm`, `quadratic_weighted_kappa`, `percentile` | `quadratic_weighted_kappa(n_levels=3)` (등급 0/1/2). `paired_tests` 는 ranx fisher 로 대체하므로 선택 |
| `text.py` | `TokenCounter` (HF 캐시 없으면 근사로 떨어지고 `kind` 에 기록) | `hf_name` 기본값을 `Qwen/Qwen3-Embedding-4B` 로 바꾼다(v4.1 V3). `detect_lang`·`lang_bucket`·`KiwiTokenizer` 는 쓰지 않음(언어 감지는 lingua, 토큰화는 `script_tokenizer`) |
| `pooling.py` | `import_judgments`, `agreement` 의 로직(CSV utf-8-sig, 중재 열, 불일치 CSV) | 등급 범위 0~2, `relevant_agreement` 기준 ≥1, `schema`/`adapter` 의존 제거 |
