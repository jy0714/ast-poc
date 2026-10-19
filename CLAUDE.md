# CLAUDE.md - Claude Code 작업 가이드

## 프로젝트 개요

**AST (Audit Support Tool) PoC — v4 (AuditRAG)**
내부 이메일(PST·EML), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX/XLS/HTML/EML/MSG)를 통합 검색하는 로컬 RAG 시스템.
1TB PoC 규모 · GPU: NVIDIA GeForce RTX 3090 24GB (2026-10-16부터 개발·실험 PC) · 케이스 기반 운영

v4 구현 지시서: `docs/CLAUDE_CODE_v4_구현지시서.md` (v4.1) + 추가 지시 `docs/v4/CLAUDE_CODE_v4.2_논문반영_추가지시.md` (v4.2, 충돌 시 v4.2 우선) + `docs/v4/CLAUDE_CODE_v4.3_표준질문45_추가지시.md` (v4.3: 평가는 감사인 표준 질문 45개만, 충돌 시 v4.3 우선). 참조 코드: `docs/v4/ref/`.

## 아키텍처 v4 요약

### 케이스 라이프사이클
```
케이스 생성 → Phase A(인덱싱) → Phase B(분석) → 보관/폐기
                  ↑                    |
                  └── 추가 자료 유입 ──┘
```

상태 머신 (`src/cases/case_store.py` `_VALID_TRANSITIONS`):
`CREATED → INDEXING → READY → ARCHIVED`, 각 상태에서 `ERROR` 전이 가능. `ARCHIVED`는 종착 상태(전이 불가) — 다시 쓰려면 같은 원본으로 새 케이스를 만든다.

### GPU 자원 (RTX 3090 24GB)
- 서비스 조건: 임베딩(qwen3-embedding 4B q8) + 재순위(Qwen3-Reranker-4B bf16) + LLM(gemma4:12b) 동시 적재
- 검색 실험(R1~R9)은 LLM을 내린 상태에서 실행 (`ollama stop gemma4:12b`, 지시서 V11)

### 데이터 파이프라인 (Phase A)
```
PST/EML 파일 / 내부 문서
  → 파싱 (libpff, PyMuPDF, python-docx, xlrd, OCR 등) [멀티프로세싱, 파서 크래시 시 풀 재시작]
  → 증거 단위 생성 (src/units): 이메일 1통 / 채팅 세션 1개 / 문서·첨부 1쪽
     · 이메일 정제(다국어 인용·서명 제거), 중복 메일 제거, 첨부↔부모 연결
     · 제목이 "(광고)"·"[광고]"로 시작하는 메일은 첨부와 함께 제외 (EMAIL_DROP_ADS, 건수만 build_stats.ads_removed, units-v3)
     · 단위별 언어 감지(lingua), 한국어 레이블 헤더(문서·첨부는 작성자·최종 수정자 포함, units-v2), UNIT_MAX_PART_TOKENS(2048) 초과 시 파트 분할
  → 로컬 임베딩 (Ollama qwen3-embedding:4b-q8_0, 2560-dim, 헤더+본문)
  → 케이스 색인 data/index/{case_id}/ (units.jsonl + BM25 hdr/nohdr + profiles/{profile}/dense.npy)
     · 원자적 교체(임시 폴더 → rename), 변경 없는 파트는 기존 벡터 재사용
     · 임베딩 체크포인트 data/embed_cache/{case_id}/{profile}/shard_*.npz — 중단·재시작 시 재사용, 색인 저장 후 삭제
     · 파일 1,000개 이상이면 모든 파일을 ProcessPool로 파싱, 진행률·ETA는 바이트 기준, BM25 토큰화는 2만 파트 이상이면 프로세스 병렬
```

### RAG 질의 (Phase B) — `src/rag/retrieval.py` `retrieve()` (서비스·실험 공용)
```
질의 → BM25 상위 50 단위 + Dense 상위 50 단위 → RRF(k=60) → 재순위 상위 30 (Qwen3-Reranker) → 상위 100
     → 엔진: 상위 단위 컨텍스트 → LLM 응답 (보안 ON: Ollama gemma4:12b / OFF: 외부 API) → 스트리밍 + 출처
```
- 색인 프로필: `hdr-qwen3`(서비스 기본), `nohdr-qwen3`, `hdr-bgem3`(R9 비교 전용). 실험 프로필은 `scripts/build_profiles.py`로 추가
- 질의 파서(규칙 기반)의 필터는 `CaseIndex.filter_mask()`로 적용

### 보안 모드
- **ON**: 임베딩 + 색인 + 재순위 + LLM 전체 로컬. 데이터 외부 전송 차단
- **OFF**: 임베딩/색인/재순위는 로컬 유지. LLM만 외부 API로 라우팅. 검색된 단위만 외부 전송

## 영구 저장소 구조

### SQLite (`data/ast.db`)
- ORM: SQLAlchemy (sync + async via aiosqlite)
- 테이블: `cases`, `indexing_logs`, `chat_history`, `chat_sources`

### 케이스 색인 (`data/index/{case_id}/`)
- `manifest.json` — 스키마·토크나이저·단위 규칙 버전, 원본 파일 지문, `build_stats`(T1 원천)
- `units.jsonl` — 파트(UnitPart) 한 줄씩 (unit_id, header, text, n_tokens, meta)
- `bm25/hdr.pkl`, `bm25/nohdr.pkl` — BM25 (헤더 포함/제외)
- `profiles/{profile}/dense.npy` (fp16) + `meta.json` (임베딩 모델·차원)
- 버전이 다르면 `IndexVersionError` → 재색인 필요. 임베딩 모델·차원이 다르면 `EmbeddingDimensionError`

### Failed embeddings (`data/failed_embeddings/{case_id}.jsonl`)
- binary subdivide 후에도 실패한 파트를 보존. 색인(Dense·BM25 모두)에서 빠짐 (재처리 스크립트 미구현)

### v3 잔여 데이터
- `data/vectordb/`, `data/bm25_index/`, `data/processed/`는 더 이상 쓰지 않지만 **자동 삭제하지 않는다** (사용자 데이터)

### Error logs (`error_logs/`)
- `src/utils/logger.py`가 component별 RotatingFileHandler 부착, WARNING 이상만 기록
- `embeddings.log`, `llm.log`, `indexing.log` 등 (component = 모듈 경로 두 번째 세그먼트)

## 실험 하네스 (`experiments_v4/`)

- `cd experiments_v4 && python -m ev4 <명령>`: corpus(T1) · queries(T0) · smoke(동작 점검) · run · pool · judgments(import/agreement/finalize → T0b) · evaluate(T2~T6·T8) · latency(T7) · failures(F1) · env(E1)
- `config.yaml`: R1~R9 실행 목록·비교 쌍. k값은 RetrievalConfig 기본값 고정. `primary_subset: search`(S17, 검정·Holm·T2~T6·F1 기준)는 실행 전 고정 — 첫 run 이 `runs/primary_subset.json` 에 기록
- 질의: 표준 질문 45개 `docs/v4/data/queries_45.jsonl` → `experiments_v4/data/queries.jsonl` (dev 없음, search 17·meta 3·nonsearch 25)
- 실데이터·결과(`data/ runs/ results/ cache/`)는 gitignore
- `AST_FAKE_MODELS=1 python -m ev4 selftest` — GPU·실데이터 없이 전체 흐름 점검

## 핵심 규칙

### 조사 데이터 기밀 (v4.2 P0) — 최우선
- Claude Code는 조사 데이터 원문을 열지 않는다: `experiments_v4/{data,runs,cache}/**`, `experiments_v4/results/failures*`, `data/{processed,index,vectordb,bm25_index,unit_stats,failed_embeddings,embed_cache,ocr_cache,parity,benchmark,judgments,input,logs}/**`, `error_logs/**`, `D:\ast-file\**`(목록 조회 포함)
- 예외(v4.3 §1): 표준 질문 45개는 사건 정보 없는 논문 부록 내용이라 읽어도 된다 — 원본 `docs/v4/data/queries_45.jsonl` 을 읽는다 (`experiments_v4/data/` 사본은 훅이 막는다)
- 차단: `.claude/settings.json` permissions.deny(Read) + PreToolUse 훅 `.claude/hooks/p0_guard.py`(Bash·PowerShell, `grep -c`/`-l`만 허용). `.claude/`는 gitignore(이 PC 전용)
- 보고는 집계 표(T0~T7, F1 요약, E1)와 로그의 건수·시간·오류 유형만. 식별자는 `qid`, `unit_id`만. 본문·질의 문장·실명·업체명·파일명 금지
- 원문이 필요한 디버깅은 사용자에게 가명 조각을 요청하거나 `tests/fixtures/` 가명 자료로 재현
- 스크립트·백그라운드 대기 명령도 건수만 출력한다 (로그 꼬리 출력 금지)

### 보안
- 임베딩·색인·재순위는 **항상 로컬**에서만 동작
- `security_mode=on` 일 때 외부 API 호출 절대 금지 (판정 시트 번역도 로컬 LLM 전용)
- PST 데이터, 내부 문서, 판정·질의는 data/ 디렉토리에만 저장 (gitignore됨)

### 코드 스타일
- Python 3.11+ 타입 힌트 필수 (실행 환경 `.venv` = Python 3.12)
- ruff 린트 (`ruff check src/ tests/ experiments_v4/`)
- 모든 public 함수에 docstring 작성
- 한국어 주석 허용, 코드와 변수명은 영어
- 서비스와 실험은 같은 `src/` 코드를 쓴다 — 실험 전용 검색 로직을 따로 만들지 않는다

### 프로젝트 구조
```
ast-poc/
├── src/
│   ├── api/routes/          # health, cases, indexing, documents, settings (Admin) / chat, dashboard (Analyst)
│   ├── parsers/             # PST, 문서 파서 (OCR 포함)
│   ├── units/               # 증거 단위: schema, builder, header, email_clean, split, lang_detect, tokens, topics
│   ├── text/                # bm25_tokenizer (문자 체계별)
│   ├── index/               # case_index, bm25
│   ├── embeddings/          # Ollama 임베딩 (+ fake_embed)
│   ├── rag/                 # retrieval, reranker, engine, query_parser
│   ├── llm/                 # LLM 라우터
│   ├── cases/               # 케이스 메타데이터 + 라이프사이클
│   ├── indexing/            # 인덱싱 파이프라인 (Producer-Consumer)
│   ├── db/                  # SQLAlchemy
│   └── utils/               # 설정, 로깅
├── experiments_v4/          # 실험 하네스 (ev4/, config.yaml, tests/)
├── frontend/                # React + TypeScript + Vite
├── tests/{unit,integration}/
├── docs/                    # v4 지시서, 참조 코드, 개선 항목
├── scripts/                 # run_indexing, build_profiles, unit_stats, check_embed_parity, check_models, docker-init.sh
├── docker/ · docker-compose.yml   # backend + frontend(nginx) + ollama(GPU)
└── data/                    # gitignore됨 (ast.db, input/, index/, failed_embeddings/, ocr_cache/)
```

### 주요 의존성
- **백엔드**: FastAPI, SQLAlchemy + aiosqlite, langchain-ollama
- **검색**: numpy(dense 정확 탐색) + rank-bm25 + kiwipiepy + RRF
- **Reranker**: transformers + torch (Qwen3-Reranker-4B / 비교용 bge-reranker-v2-m3)
- **언어 감지**: lingua-language-detector
- **실험**: ranx, pyyaml
- **문서 파싱**: PyMuPDF, python-docx, python-pptx, openpyxl, xlrd, extract-msg, libpff(옵션)

## 자주 쓰는 명령어

```bash
uvicorn src.api.main:app --reload --port 8000         # 백엔드
cd frontend && npm run dev                            # 프론트엔드
ruff check src/ tests/ experiments_v4/ --fix          # 린트
pytest tests/ -q                                      # 테스트
pytest experiments_v4/tests -q                        # 하네스 테스트
python -m scripts.run_indexing <case_id>              # 인덱싱 CLI
python -m scripts.build_profiles <case_id> --profiles nohdr-qwen3 hdr-bgem3
python -m scripts.unit_stats <case_id>                # 단위 통계 (임베딩 없음, T1 초안)
python -m scripts.repro_kiwi --out experiments_v4/results/S6_kiwi_repro.md   # kiwipiepy 현상 재현 (합성 문자열)
python -m scripts.bench_embed --grid 16x2 16x8 --num-batch 0 2048   # 임베딩 처리량 (합성 문장, 숫자만 출력)
python -m scripts.langdetect_compare <case_id>        # langdetect↔lingua 대조 (건수만, langdetect 별도 설치)
docker compose up --build                             # Docker 전체 스택
```

## 현재 상태 (2026-10-19 기준)

- **브랜치**: `v4-simplify` (지시서 1~5단계 + v4.2·v4.3 추가 지시 코드 반영 완료). 다음: 코퍼스 확정 → 색인 → smoke → 6단계 실측
- **테스트**: tests + experiments_v4 합계 582
- **GPU**: NVIDIA GeForce RTX 3090 24GB, 드라이버 617.42(CUDA 13.4). 임베딩 일치 통과(cos 0.995), 세 모델 동시 적재 ≈24.1/24.6GB (`LLM_NUM_CTX=16384` 필수)
- **케이스**: 685850a87ee2 (메일함 1, CREATED·미색인 — 코퍼스 확정 후 결정), 9dec87cd8d6c (메일함 1 중 300파일 실규모 점검, units-v2 색인 READY). 구 케이스 b392e4f27760·8cdadf2ab3db는 ARCHIVED 그대로 둠
- **본 색인 (2026-10-19)**: 케이스 2b7885b9a43f (파일 140,517) 재시작 — 개선판(광고 제외·체크포인트·파싱 워커 낮은 우선순위) + OLLAMA_NUM_PARALLEL=8. 임베딩 실측 분당 약 350파트 (이전 134). 합성 벤치 상한 분당 약 490파트(732토큰), 약 6,000토큰/s
- **장시간 인덱싱 중에는 백엔드를 `--reload` 없이 띄운다** — src 수정 시 서버가 재시작되어 진행 중인 인덱싱이 사라진다. Claude 세션 종료와 함께 꺼지지 않도록 별도 창으로 띄운다 (`Start-Process .venv\Scripts\python.exe -ArgumentList "-m","uvicorn","src.api.main:app","--port","8000"`). 중단 후 `/start` 재호출 시 30분 stuck timeout 뒤 자동 복구, 임베딩 체크포인트 재사용
- **평가 코퍼스 폴더**: `D:\ast-file\corpus\` (사용자가 확정 자료를 넣으면 새 케이스로 색인)

### 환경 설정 파일
- `.env.prod.example` — RTX 3090 24GB (gemma4:12b, Qwen3-Reranker-4B bf16, EMBED_BATCH_SIZE=16)
- `.env.dev.example` — 구 개발 PC RTX 2060 12GB (gemma4:e4b, Reranker-0.6B fp16, EMBED_BATCH_SIZE=8)
- 설정은 모르는 키를 무시(`extra="ignore"`)하므로 v3 키가 남은 `.env`도 기동된다

> **대량 인덱싱 (Phase A)**: Ollama `OLLAMA_NUM_PARALLEL`(동시 처리 슬롯)과 `.env` `EMBED_MAX_CONCURRENT`를 함께 올려야 GPU가 찬다. `OLLAMA_NUM_PARALLEL`은 LLM에도 적용되므로(슬롯마다 KV 캐시) 서비스·LLM 동시 적재 전에는 1로 되돌린다. 값은 `scripts/bench_embed.py`로 정한다.

> **임베딩 배치 크기 주의**: 큰 배치는 Ollama timeout 캐스케이드를 유발할 수 있다. 안정 동작 확인 후 단계적으로 상향하고 로그를 모니터링한다.

### API 엔드포인트 목록

| 구분 | 경로 | 메서드 | 설명 |
|---|---|---|---|
| Admin | `/api/admin/cases/` | POST/GET | 케이스 생성 / 목록 |
| Admin | `/api/admin/cases/{id}` | GET/PATCH/DELETE | 상세 / 데이터 소스 추가 / 삭제(색인 포함) |
| Admin | `/api/admin/cases/{id}/archive` | POST | 케이스 보관 |
| Admin | `/api/admin/indexing/start` · `stop/{id}` · `progress/{id}` · `increment/{id}` | POST/GET | 인덱싱 |
| Admin | `/api/admin/documents/upload/{id}` | POST | 문서 업로드 |
| Admin | `/api/admin/settings/...` | GET/PUT | 보안 모드, 모델, reranker 설정 |
| Analyst | `/api/analyst/chat/` · `/chat/stream` · `/chat/cases` | POST/GET | RAG 질의 / SSE / 케이스 목록 |
| Analyst | `/api/analyst/dashboard/...` | GET | 커뮤니케이션 분석 (units.jsonl 메타 집계) |
| 공통 | `/health` | GET | 헬스체크 |
