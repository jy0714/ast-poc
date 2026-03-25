# CLAUDE.md - Claude Code 작업 가이드

## 프로젝트 개요

**AST (Audit Support Tool) PoC**
내부 이메일(PST), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX)를 통합 검색하는 로컬 RAG 시스템.
1TB PoC 규모 · NVIDIA A5000 24GB · 케이스 기반 운영

## 아키텍처 v5 요약

### 케이스 라이프사이클
```
케이스 생성 → Phase A(인덱싱) → Phase B(분석) → 보관/폐기
                  ↑                    |
                  └── 추가 자료 유입 ──┘
```

### GPU 자원 분배 (A5000 24GB — 순차 전용)
- Phase A: 임베딩 모델이 VRAM 24GB 전체 사용 (LLM 언로드)
- Phase B: LLM이 VRAM 24GB 전체 사용 (임베딩 언로드)
- 동시 사용 없음 — 각 Phase에서 GPU 100% 활용

### 데이터 파이프라인 (Phase A)
```
PST 파일 / 내부 문서
  → 파싱 (libpff, PyMuPDF, python-docx 등) [멀티프로세싱]
  → 자동 분류 (이메일 본문 / Teams 대화 / 첨부파일)
  → 스마트 청킹 (문서: 문자수, 채팅: 시간윈도우, 이메일: 스레드, 첨부: 타입별)
  → 메타데이터 부착 (participants, date_range, source_type, topics, case_id, file_name)
  → 로컬 임베딩 (Ollama nomic-embed-text, A5000 전용)
  → 하이브리드 저장소 (ChromaDB 벡터 + BM25 키워드 인덱스)
  → 케이스별 독립 컬렉션 생성
```

### RAG 질의 (Phase B)
```
사용자 질의
  → 질의 파서 (의도 분석 + 필터 추출)
  → 하이브리드 검색 (벡터 유사도 + BM25 키워드 매칭)
  → Rank Fusion (결과 통합)
  → LLM 응답 생성 (보안 ON: Ollama gpt-oss:20b / OFF: 외부 API)
  → 스트리밍 응답 + 출처 표시
```

### 보안 모드
- **ON**: 임베딩 + 벡터DB + LLM 전체 로컬. 데이터 외부 전송 차단
- **OFF**: 임베딩/벡터DB는 로컬 유지. LLM만 외부 API로 라우팅 (ChatGPT 5.4 / Gemini / Opus 4.6 등 미정). 검색된 청크만 외부 전송

## 영구 저장소 구조

### SQLite (`data/ast.db`)
- ORM: SQLAlchemy (향후 PostgreSQL 전환 대비)
- 테이블:
  - `cases` — 케이스 메타데이터 (이름, 상태, 생성일, 데이터소스 경로 등)
  - `indexing_logs` — 인덱싱 실행 이력 (시작/종료 시간, 처리 건수, 에러)
  - `chat_history` — 질의/응답 대화 이력 (케이스별, 사용자 질문 + LLM 응답)
  - `chat_sources` — 응답에 사용된 출처 청크 (chat_history FK, 문서명, 관련도 점수)

### ChromaDB (`data/vectordb/`)
- `persist_directory`로 디스크 영구 저장
- 케이스별 독립 컬렉션 (`case_{id}`)
- 벡터 유사도 검색용

### BM25 (`data/bm25/`)
- pickle 직렬화로 키워드 인덱스 저장
- 케이스별 독립 인덱스 파일
- 하이브리드 검색의 키워드 매칭 담당

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
│   │   └── routes/
│   │       ├── health.py       # 헬스체크
│   │       ├── cases.py        # 케이스 CRUD (Admin)
│   │       ├── indexing.py     # 인덱싱 관리 (Admin)
│   │       ├── chat.py         # RAG 질의 (Analyst)
│   │       ├── documents.py    # 문서 업로드
│   │       └── settings.py     # 보안 모드 등 설정
│   ├── parsers/                # PST, PDF, DOCX 등 파서
│   ├── chunkers/               # 텍스트 청킹 (문서/채팅/이메일/첨부)
│   ├── embeddings/             # Ollama 임베딩
│   ├── vectorstore/            # ChromaDB + BM25 하이브리드 저장소
│   ├── rag/                    # RAG 엔진, 질의 파서, Rank Fusion
│   ├── llm/                    # LLM 라우터 (로컬/외부 분기)
│   └── utils/                  # 설정, 로깅, 공통 유틸
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── admin/          # Admin UI 컴포넌트
│       │   │   ├── CaseManager/
│       │   │   ├── DataSourceConfig/
│       │   │   ├── IndexingMonitor/
│       │   │   └── SystemSettings/
│       │   ├── analyst/        # Analyst UI 컴포넌트
│       │   │   ├── ChatInterface/
│       │   │   ├── SecurityToggle/
│       │   │   ├── SearchResults/
│       │   │   └── CaseSelector/
│       │   └── shared/         # 공통 컴포넌트
│       ├── hooks/
│       ├── styles/
│       └── utils/
├── tests/
├── docs/
├── scripts/
├── docker/
│   ├── Dockerfile.backend      # Python 3.11 + uvicorn
│   ├── Dockerfile.frontend     # Node build → nginx 서빙
│   └── nginx.conf              # SPA + API 프록시
├── docker-compose.yml          # backend + frontend + chromadb + ollama
├── .dockerignore
└── data/                       # gitignore됨
    ├── ast.db                  # SQLite (케이스/인덱싱로그/채팅이력)
    ├── input/{pst,documents}/
    ├── processed/              # 파싱/청킹 JSONL 캐시
    ├── bm25/                   # BM25 키워드 인덱스 (pickle)
    └── vectordb/               # ChromaDB 영구 저장
```

### 주요 의존성
- **백엔드**: Python 3.11+, FastAPI, LangChain, ChromaDB, Ollama, SQLAlchemy + SQLite
- **프론트엔드**: React + TypeScript
- **검색**: ChromaDB (벡터) + BM25 (키워드) + Rank Fusion
- **LLM 로컬**: Ollama gpt-oss:20b
- **LLM 외부**: ChatGPT 5.4 / Gemini / Opus 4.6 등 (미정)
- **임베딩**: Ollama nomic-embed-text
- **PST 파싱**: libpff / pypff / libratom

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
7. ~~Ollama 임베딩 연동~~ ✅
8. ~~ChromaDB 벡터 저장 + BM25 인덱스~~ ✅
9. ~~케이스 관리 API (CRUD + 라이프사이클)~~ ✅
10. ~~인덱싱 파이프라인 오케스트레이터~~ ✅

### Phase 3 — RAG 질의 (Phase B) ✅
11. ~~질의 파서 (의도 분석 + 필터 추출)~~ ✅ — 규칙 기반, 향후 LLM 기반으로 튜닝 예정
12. ~~하이브리드 검색 (벡터 + BM25 + Rank Fusion)~~ ✅
13. ~~LLM 라우터 (보안 모드 분기)~~ ✅
14. ~~스트리밍 응답 + 출처 표시~~ ✅

### Phase 4 — 프론트엔드 ✅
15. ~~Admin UI (케이스 관리, 데이터 소스 설정, 인덱싱 모니터)~~ ✅
16. ~~Analyst UI (채팅 인터페이스, 보안 토글, 검색 결과)~~ ✅

### Phase 5 — 통합 & 배포 ✅
17. ~~Docker 컨테이너화~~ ✅
18. ~~통합 테스트~~ ✅
19. 성능 튜닝 (1TB 데이터 기준) — PoC 이후 실데이터 투입 시 진행

## 현재 상태 (2026-03-24 기준)

- **최신 커밋**: `e588717` (main) — Phase 4 프론트엔드 + E2E 통합 테스트
- **테스트**: 291 passed (단위 277 + 통합 14), 0 skipped
- **환경**: Python 3.14.2, Windows 11, VS 2026 Community (C++ 빌드 도구 설치됨)
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
| 임베딩 | `src/embeddings/embedding_service.py` | Ollama nomic-embed-text |
| 벡터 저장소 | `src/vectorstore/vector_store.py` | ChromaDB + BM25 + RRF 하이브리드 |
| 케이스 관리 | `src/cases/case_store.py` | JSON 파일 기반 CRUD + 라이프사이클 |
| 인덱싱 | `src/indexing/pipeline.py` | 파이프라인 오케스트레이터 (백그라운드 실행) |
| 질의 파서 | `src/rag/query_parser.py` | 규칙 기반 (소스타입/날짜/참여자/의도) |
| RAG 엔진 | `src/rag/engine.py` | 검색→컨텍스트→LLM 응답 생성 |
| LLM 라우터 | `src/llm/router.py` | 보안 ON=Ollama, OFF=OpenAI |
| Admin API | `src/api/routes/cases.py`, `indexing.py` | 케이스 CRUD + 인덱싱 관리 |
| Analyst API | `src/api/routes/chat.py` | RAG 질의 (동기+스트리밍+케이스목록) |
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
| Analyst | `/api/analyst/chat/` | POST | RAG 동기 질의 |
| Analyst | `/api/analyst/chat/stream` | POST | SSE 스트리밍 질의 |
| Analyst | `/api/analyst/chat/cases` | GET | 분석 가능 케이스 목록 |
| 공통 | `/health` | GET | 헬스체크 |

### 테스트 구조

| 경로 | 테스트 수 | 설명 |
|---|---|---|
| `tests/unit/test_pst_parser.py` | 15 | PST 파서 |
| `tests/unit/test_document_parser.py` | 30 | 문서 파서 (PDF/DOCX/PPTX/XLSX/EML/MSG) |
| `tests/unit/test_chunker.py` | 21 | 4종 청커 |
| `tests/unit/test_metadata_enricher.py` | 16 | 메타데이터 + 토픽 추출 |
| `tests/unit/test_embedding_service.py` | 9 | Ollama 임베딩 |
| `tests/unit/test_vector_store.py` | 12 | ChromaDB + BM25 + RRF |
| `tests/unit/test_config.py` | 3 | 설정 |
| `tests/unit/test_case_store.py` | 30 | 케이스 CRUD + 라이프사이클 |
| `tests/unit/test_cases_api.py` | 16 | 케이스 API |
| `tests/unit/test_indexing_pipeline.py` | 25 | 인덱싱 파이프라인 |
| `tests/unit/test_indexing_api.py` | 9 | 인덱싱 API |
| `tests/unit/test_query_parser.py` | 30 | 질의 파서 |
| `tests/unit/test_llm_router.py` | 8 | LLM 라우터 |
| `tests/unit/test_rag_engine.py` | 11 | RAG 엔진 |
| `tests/unit/test_chat_api.py` | 11 | 채팅 API |
| `tests/integration/test_e2e_pipeline.py` | 14 | E2E 통합 (생성→인덱싱→질의→보관→삭제) |
