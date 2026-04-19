# CLAUDE.md - Claude Code 작업 가이드

## 프로젝트 개요

**AST (Audit Support Tool) PoC**
내부 이메일(PST), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX/EML/MSG)를 통합 검색하는 로컬 RAG 시스템.
1TB PoC 규모 · 개발: NVIDIA RTX 3060 12GB · 운영: NVIDIA A5000 24GB · 케이스 기반 운영

## 아키텍처 v5 요약

### 케이스 라이프사이클
```
케이스 생성 → Phase A(인덱싱) → Phase B(분석) → 보관/폐기
                  ↑                    |
                  └── 추가 자료 유입 ──┘
```

상태 머신 (`src/cases/case_store.py` `_VALID_TRANSITIONS`):
`CREATED → INDEXING → READY → ARCHIVED`, 각 상태에서 `ERROR` 전이 가능, `ARCHIVED → CREATED` 복구 가능.

### GPU 자원 분배 (개발 3060 12GB / 운영 A5000 24GB — 순차 전용)
- Phase A: 임베딩 모델이 VRAM 전체 사용 (LLM 언로드)
- Phase B: LLM이 VRAM 전체 사용 (임베딩 언로드)
- 동시 사용 없음 — 각 Phase에서 GPU 100% 활용

### 데이터 파이프라인 (Phase A)
```
PST 파일 / 내부 문서
  → 파싱 (libpff, PyMuPDF, python-docx 등) [멀티프로세싱]
  → 자동 분류 (이메일 본문 / Teams 대화 / 첨부파일)
  → 스마트 청킹 (문서: 문자수, 채팅: 시간윈도우, 이메일: 스레드, 첨부: 타입별)
  → 메타데이터 부착 (participants, date_range, source_type, topics, case_id, file_name)
  → 로컬 임베딩 (Ollama bge-m3, 1024-dim, 8192 토큰)
  → 단일 공유 ChromaDB 컬렉션 + BM25 키워드 인덱스 (케이스별 pickle)
  → case_id 메타필터로 케이스 격리
```

### RAG 질의 (Phase B)
```
사용자 질의
  → 질의 파서 (의도 분석 + 필터 추출)
  → 하이브리드 검색 (벡터 유사도 + BM25 키워드 매칭)
  → RRF (Reciprocal Rank Fusion)
  → 선택적 Reranker (BAAI/bge-reranker-v2-m3)
  → LLM 응답 생성 (보안 ON: Ollama gemma4:e4b/gpt-oss:20b / OFF: 외부 API)
  → 스트리밍 응답 + 출처 표시
```

### 보안 모드
- **ON**: 임베딩 + 벡터DB + LLM 전체 로컬. 데이터 외부 전송 차단
- **OFF**: 임베딩/벡터DB는 로컬 유지. LLM만 외부 API로 라우팅 (OpenAI 등). 검색된 청크만 외부 전송

## 영구 저장소 구조

### SQLite (`data/ast.db`)
- ORM: SQLAlchemy (sync + async via aiosqlite, 향후 PostgreSQL 전환 대비)
- 테이블:
  - `cases` — 케이스 메타데이터 (이름, 상태, 생성일, 데이터소스 경로 등)
  - `indexing_logs` — 인덱싱 실행 이력 (시작/종료 시간, 처리 건수, 에러)
  - `chat_history` — 질의/응답 대화 이력 (케이스별, 사용자 질문 + LLM 응답)
  - `chat_sources` — 응답에 사용된 출처 청크 (chat_history FK, 문서명, 관련도 점수)

### ChromaDB (`data/vectordb/`)
- `persist_directory`로 디스크 영구 저장
- **단일 공유 컬렉션** (`ast_chunks`) + `case_id` 메타필터로 케이스 격리
- 벡터 유사도 검색용 (1024-dim bge-m3)

### BM25 (`data/bm25_index/`)
- pickle 직렬화로 키워드 인덱스 저장
- 케이스별 독립 인덱스 파일 (`{case_id}.pkl`)
- 하이브리드 검색의 키워드 매칭 담당 (kiwipiepy 한국어 형태소 토크나이저)

### Processed (`data/processed/`)
- 파싱/청킹 결과 JSONL 캐시
- 증분 인덱싱 시 재파싱 방지 (파일 해시 기반 중복 체크)

## 프론트엔드 구조

### Admin UI (케이스 관리 + 인덱싱 = Phase A)
- **케이스 관리**: 생성, 목록, 상태 확인, 삭제/보관
- **데이터 소스 설정**: PST 파일 경로 지정, 문서 폴더 지정
- **인덱싱 관리**: 인덱싱 실행, 진행률 모니터링, 증분 인덱싱 트리거
- **시스템 설정**: 청킹 파라미터, 임베딩 모델, 벡터DB 상태

### Analyst UI (RAG 질의 = Phase B)
- **채팅 인터페이스**: LLM 스타일 대화형 질의
- **보안 모드 토글**: ON/OFF 전환
- **검색 결과**: 출처 문서/이메일/채팅 참조 표시
- **필터**: 날짜 범위, 참여자, 소스 타입 등 수동 필터
- **케이스 선택**: 분석할 케이스 선택 (인덱싱 완료된 케이스만)
- **대시보드**: 커뮤니케이션 분석 (참여자/토픽/타임라인)

## 핵심 규칙

### 보안
- 임베딩과 벡터DB는 **항상 로컬**에서만 동작
- `security_mode=on` 일 때 외부 API 호출 절대 금지
- PST 데이터, 내부 문서는 data/ 디렉토리에만 저장 (gitignore됨)

### 코드 스타일
- Python 3.11+ 타입 힌트 필수
- ruff로 린트/포맷 (`ruff check`, `ruff format`)
- 모든 public 함수에 docstring 작성
- 한국어 주석 허용, 코드와 변수명은 영어

### 프로젝트 구조
```
ast-poc/
├── src/                        # 백엔드 소스
│   ├── api/                    # FastAPI 엔드포인트
│   │   ├── main.py
│   │   └── routes/
│   │       ├── health.py       # 헬스체크
│   │       ├── cases.py        # 케이스 CRUD (Admin)
│   │       ├── indexing.py     # 인덱싱 관리 (Admin)
│   │       ├── documents.py    # 문서 업로드 (Admin)
│   │       ├── settings.py     # 설정/모델 (Admin)
│   │       ├── chat.py         # RAG 질의 (Analyst)
│   │       └── dashboard.py    # 커뮤니케이션 분석 (Analyst)
│   ├── parsers/                # PST, PDF, DOCX 등 파서
│   ├── chunkers/               # 텍스트 청킹 + 메타데이터 보강
│   ├── embeddings/             # Ollama 임베딩 클라이언트
│   ├── vectorstore/            # ChromaDB + BM25 하이브리드 저장소
│   ├── rag/                    # 질의 파서, 검색, RAG 엔진, Reranker
│   ├── llm/                    # LLM 라우터 (로컬/외부 분기)
│   ├── cases/                  # 케이스 메타데이터 + 라이프사이클
│   ├── indexing/               # 인덱싱 파이프라인 오케스트레이터
│   ├── db/                     # SQLAlchemy 엔진/세션 (sync + async)
│   └── utils/                  # 설정, 로깅
├── frontend/                   # React + TypeScript + Vite
│   └── src/
│       ├── components/
│       │   ├── admin/          # CaseManager / DataSourceConfig / IndexingMonitor / SystemSettings
│       │   ├── analyst/        # ChatInterface / SecurityToggle / SearchResults / CaseSelector / Dashboard
│       │   └── shared/         # 공통 컴포넌트
│       ├── hooks/
│       ├── styles/
│       └── utils/
├── tests/
│   ├── unit/                   # 유닛 테스트 (~340)
│   └── integration/            # E2E 통합 테스트 (~14)
├── docs/                       # 아키텍처/가이드/개선 항목
├── scripts/                    # docker-init.sh, run_indexing.py, run_benchmark.py
├── docker/
│   ├── Dockerfile.backend      # Python 3.11 + uvicorn
│   ├── Dockerfile.frontend     # Node build → nginx 서빙
│   └── nginx.conf              # SPA + API 프록시
├── docker-compose.yml          # backend + frontend + chromadb + ollama
├── .dockerignore
└── data/                       # gitignore됨
    ├── ast.db                  # SQLite (cases, indexing_logs, chat_history, chat_sources)
    ├── input/{pst,documents}/  # 원본 데이터
    ├── processed/              # 파싱/청킹 JSONL 캐시
    ├── bm25_index/             # BM25 케이스별 pickle
    └── vectordb/               # ChromaDB 영구 저장
```

### 주요 의존성
- **백엔드**: Python 3.11+, FastAPI, ChromaDB, Ollama (langchain-ollama), SQLAlchemy + SQLite (aiosqlite)
- **프론트엔드**: React + TypeScript + Vite
- **검색**: ChromaDB (벡터) + rank-bm25 + kiwipiepy (한국어 형태소) + RRF
- **Reranker**: BAAI/bge-reranker-v2-m3 (FlagEmbedding) — 선택적
- **LLM 로컬**: 개발 `gemma4:e4b`, 운영 `gpt-oss:20b` (Ollama)
- **LLM 외부**: OpenAI (langchain-openai)
- **임베딩**: Ollama `bge-m3` (1024-dim, 8192 토큰)
- **PST 파싱**: libpff / pypff (옵션)
- **문서 파싱**: PyMuPDF, python-docx, python-pptx, openpyxl, extract-msg

## 자주 쓰는 명령어

```bash
# 백엔드 서버
uvicorn src.api.main:app --reload --port 8000

# 프론트엔드
cd frontend && npm run dev

# 린트 & 포맷
ruff check src/ --fix
ruff format src/

# 테스트
pytest tests/ -v

# 인덱싱 CLI (API 대신 직접 실행)
python -m scripts.run_indexing <case_id>

# Docker (전체 스택)
docker compose up --build

# Docker 초기 설정 (모델 다운로드 포함)
bash scripts/docker-init.sh

# Docker 개별 서비스
docker compose up backend       # 백엔드만
docker compose logs -f backend  # 로그 확인
```

## 구현 우선순위 (TODO)

### Phase 1 — 기반 구조 ✅
1. ~~프로젝트 구조 셋업~~ ✅
2. ~~PST 파서 구현 (pypff + libratom + Mock)~~ ✅
3. ~~v5 아키텍처 확정~~ ✅

### Phase 2 — 인덱싱 파이프라인 (Phase A) ✅
4. ~~문서 파서 구현 (PDF, DOCX, PPTX, XLSX, EML, MSG)~~ ✅
5. ~~스마트 청킹 엔진 (문서/채팅/이메일스레드/첨부)~~ ✅
6. ~~메타데이터 부착 + 토픽 태깅~~ ✅
7. ~~Ollama 임베딩 연동 (bge-m3)~~ ✅
8. ~~ChromaDB 단일 공유 컬렉션 + BM25 인덱스~~ ✅
9. ~~케이스 관리 API (CRUD + 라이프사이클)~~ ✅
10. ~~인덱싱 파이프라인 오케스트레이터~~ ✅

### Phase 3 — RAG 질의 (Phase B) ✅
11. ~~질의 파서 (의도 분석 + 필터 추출)~~ ✅ — 규칙 기반, 향후 LLM 기반으로 튜닝 예정
12. ~~하이브리드 검색 (벡터 + BM25 + RRF)~~ ✅
13. ~~Reranker (BAAI/bge-reranker-v2-m3, 선택적)~~ ✅
14. ~~LLM 라우터 (보안 모드 분기)~~ ✅
15. ~~스트리밍 응답 + 출처 표시~~ ✅

### Phase 4 — 프론트엔드 ✅
16. ~~Admin UI (케이스 관리, 데이터 소스 설정, 인덱싱 모니터, 시스템 설정)~~ ✅
17. ~~Analyst UI (채팅 인터페이스, 보안 토글, 검색 결과, 대시보드)~~ ✅

### Phase 5 — 통합 & 배포 ✅
18. ~~Docker 컨테이너화~~ ✅
19. ~~통합 테스트~~ ✅
20. 성능 튜닝 (1TB 데이터 기준) — PoC 이후 실데이터 투입 시 진행

## 현재 상태 (2026-04-19 기준)

- **최신 커밋**: `d50f7b0` (main) — Skip BM25 tokenization on hot path during bulk indexing
- **테스트**: ~356 (unit ~342 + integration ~14)
- **환경**: Python 3.11+, Windows 11, VS 2026 Community (C++ 빌드 도구 설치됨)
- **chroma-hnswlib**: 0.7.6 (C++ 빌드 완료 — 한글 Windows에서 DISTUTILS_USE_SDK=1 필요)
- **Phase 5 Docker**: 완료 (backend + frontend/nginx + ChromaDB + Ollama GPU)

### 환경 이슈 (chroma-hnswlib 빌드)

한글 Windows + VS 2026(v18)에서 `pip install chroma-hnswlib`가 실패함.
- 원인: `setuptools`의 `_get_vc_env()`가 `cmd /u` (UTF-16LE)로 vcvarsall.bat 출력을 파싱할 때 한글 인코딩 깨짐 → MSVC 못 찾음
- 해결: vcvars64.bat 환경 로드 후 `DISTUTILS_USE_SDK=1` 설정하여 빌드
```python
# 빌드 스크립트 (scripts/ 참고)
result = subprocess.run(
    ['cmd', '/c', vcvars64_path, '&&', 'set'],
    capture_output=True, text=True
)
env = {**os.environ}
for line in result.stdout.splitlines():
    if '=' in line:
        key, _, val = line.partition('=')
        env[key] = val
env['DISTUTILS_USE_SDK'] = '1'
subprocess.run(['pip', 'install', 'chroma-hnswlib', '--no-build-isolation'], env=env)
```

### 주요 구현 파일 요약

| 모듈 | 핵심 파일 | 설명 |
|---|---|---|
| 문서 파서 | `src/parsers/document_parser.py` | PDF/DOCX/PPTX/XLSX/EML/MSG 지원 |
| PST 파서 | `src/parsers/pst_parser.py` | PST 이메일/채팅/첨부 파싱 |
| 청킹 | `src/chunkers/chunker.py` | 문서/채팅/이메일/첨부 4종 청커 |
| 메타데이터 | `src/chunkers/metadata_enricher.py` | 토픽 추출 + case_id 부착 |
| 임베딩 | `src/embeddings/embedding_service.py` | Ollama bge-m3 (1024-dim) |
| 벡터 저장소 | `src/vectorstore/vector_store.py` | ChromaDB 단일 공유 컬렉션 + BM25 + RRF |
| 케이스 관리 | `src/cases/case_store.py` | SQLAlchemy CRUD + 라이프사이클 상태 머신 |
| DB 엔진 | `src/db/database.py`, `db/models.py` | SQLAlchemy sync + async (aiosqlite) |
| 인덱싱 | `src/indexing/pipeline.py` | 파이프라인 오케스트레이터 (백그라운드 + Producer-Consumer) |
| 질의 파서 | `src/rag/query_parser.py` | 규칙 기반 (소스타입/날짜/참여자/의도) |
| RAG 엔진 | `src/rag/engine.py` | 검색→컨텍스트→LLM 응답 생성 |
| Reranker | `src/rag/reranker.py` | BAAI/bge-reranker-v2-m3 (FlagEmbedding) |
| LLM 라우터 | `src/llm/router.py` | 보안 ON=Ollama, OFF=OpenAI |
| Admin API | `src/api/routes/cases.py`, `indexing.py`, `documents.py`, `settings.py` | 케이스 CRUD + 인덱싱 + 업로드 + 설정 |
| Analyst API | `src/api/routes/chat.py`, `dashboard.py` | RAG 질의 + 커뮤니케이션 분석 |
| Docker | `docker/Dockerfile.backend` | Python 3.11 + uvicorn |
| Docker | `docker/Dockerfile.frontend` | Node build → nginx SPA 서빙 |
| Docker | `docker/nginx.conf` | API 프록시 + SSE 스트리밍 + SPA fallback |
| Docker | `docker-compose.yml` | 4-서비스 스택 (backend/frontend/chromadb/ollama) |
| Docker | `scripts/docker-init.sh` | 초기 설정 (디렉토리 + 모델 다운로드) |

### API 엔드포인트 목록

| 구분 | 경로 | 메서드 | 설명 |
|---|---|---|---|
| Admin | `/api/admin/cases/` | POST | 케이스 생성 (201) |
| Admin | `/api/admin/cases/` | GET | 케이스 목록 (?status= 필터) |
| Admin | `/api/admin/cases/{id}` | GET | 케이스 상세 조회 |
| Admin | `/api/admin/cases/{id}` | PATCH | 데이터 소스 추가 |
| Admin | `/api/admin/cases/{id}` | DELETE | 케이스 삭제 (벡터DB 포함) |
| Admin | `/api/admin/cases/{id}/archive` | POST | 케이스 보관 |
| Admin | `/api/admin/indexing/start` | POST | 인덱싱 시작 (백그라운드) |
| Admin | `/api/admin/indexing/stop/{id}` | POST | 인덱싱 중단 |
| Admin | `/api/admin/indexing/progress/{id}` | GET | 진행률 조회 |
| Admin | `/api/admin/indexing/increment/{id}` | POST | 증분 인덱싱 |
| Admin | `/api/admin/documents/upload/{id}` | POST | 문서 업로드 |
| Admin | `/api/admin/settings/...` | GET/PATCH | 보안 모드, 모델, reranker 설정 |
| Analyst | `/api/analyst/chat/` | POST | RAG 동기 질의 |
| Analyst | `/api/analyst/chat/stream` | POST | SSE 스트리밍 질의 |
| Analyst | `/api/analyst/chat/cases` | GET | 분석 가능 케이스 목록 |
| Analyst | `/api/analyst/dashboard/...` | GET | 커뮤니케이션 분석 (참여자/토픽/타임라인) |
| 공통 | `/health` | GET | 헬스체크 |

### 테스트 구조

| 경로 | 설명 |
|---|---|
| `tests/unit/test_pst_parser.py` | PST 파서 |
| `tests/unit/test_document_parser.py` | 문서 파서 (PDF/DOCX/PPTX/XLSX/EML/MSG) |
| `tests/unit/test_chunker.py` | 4종 청커 |
| `tests/unit/test_metadata_enricher.py` | 메타데이터 + 토픽 추출 |
| `tests/unit/test_embedding_service.py` | Ollama 임베딩 |
| `tests/unit/test_vector_store.py` | ChromaDB + BM25 + RRF |
| `tests/unit/test_config.py` | 설정 |
| `tests/unit/test_case_store.py` | 케이스 CRUD + 라이프사이클 |
| `tests/unit/test_cases_api.py` | 케이스 API |
| `tests/unit/test_indexing_pipeline.py` | 인덱싱 파이프라인 |
| `tests/unit/test_indexing_api.py` | 인덱싱 API |
| `tests/unit/test_query_parser.py` | 질의 파서 |
| `tests/unit/test_llm_router.py` | LLM 라우터 |
| `tests/unit/test_rag_engine.py` | RAG 엔진 |
| `tests/unit/test_reranker.py` | Reranker |
| `tests/unit/test_chat_api.py` | 채팅 API |
| `tests/unit/test_dashboard_api.py` | 대시보드 API |
| `tests/integration/test_e2e_pipeline.py` | E2E 통합 (생성→인덱싱→질의→보관→삭제) |

### 환경 설정 파일

환경별 `.env` 예시 파일을 사용:
- `.env.dev.example` — 개발 환경 (3060 12GB, gemma4:e4b, EMBED_BATCH_SIZE=256, RERANK_ENABLED=false)
- `.env.prod.example` — 운영 환경 (A5000 24GB, gpt-oss:20b, EMBED_BATCH_SIZE=512, RERANK_ENABLED=true)

대상 환경 파일을 `.env`로 복사하여 사용. 개별 변수는 환경 변수로 오버라이드 가능.
