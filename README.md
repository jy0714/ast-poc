# AST (Audit Support Tool) - PoC

내부 이메일(PST), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX/EML/MSG)를 통합 검색할 수 있는 로컬 RAG 시스템.

## 주요 기능

- **PST 파싱**: Outlook 이메일 및 Teams 채팅 기록 자동 파싱
- **문서 파싱**: PDF, DOCX, PPTX, XLSX, EML, MSG 파일 지원 (OCR 옵션 포함)
- **스마트 청킹**: 문서는 문자수 기반, 채팅은 시간 윈도우, 이메일은 스레드, 첨부는 타입별 분할
- **메타데이터 태깅**: 참여자, 날짜, 소스 타입, 토픽, case_id 자동 부착
- **로컬 임베딩**: Ollama bge-m3 (1024차원, 외부 전송 없음)
- **하이브리드 검색**: ChromaDB 벡터 유사도 + BM25 키워드 + RRF (Reciprocal Rank Fusion)
- **케이스 격리**: 단일 공유 ChromaDB 컬렉션 + `case_id` 메타필터로 케이스별 격리
- **선택적 Reranker**: BAAI/bge-reranker-v2-m3 (전역 토글 + 요청별 오버라이드)
- **보안 모드**: ON(임베딩+벡터DB+LLM 전부 로컬) / OFF(LLM만 외부 API 라우팅) 토글
- **Admin/Analyst UI 분리**: 케이스 관리·인덱싱과 RAG 질의 분리

## 기술 스택

| 영역 | 기술 |
|------|------|
| 백엔드 | Python 3.11+, FastAPI, Uvicorn |
| 프론트엔드 | React + TypeScript + Vite |
| LLM (로컬) | Ollama — 개발: `gemma4:e4b`, 운영: `gpt-oss:20b` |
| LLM (외부) | OpenAI (보안 모드 OFF 시) |
| 임베딩 | Ollama `bge-m3` (1024-dim, 4096 토큰) — 배치 실패 시 binary subdivide 자동 재시도 |
| 벡터DB | ChromaDB (단일 공유 컬렉션 + 메타필터) |
| 키워드 검색 | rank-bm25 + kiwipiepy 한국어 형태소 분석 |
| Reranker | BAAI/bge-reranker-v2-m3 (FlagEmbedding) |
| 메타DB | SQLite (SQLAlchemy ORM, PostgreSQL 전환 대비) |
| 문서 파싱 | PyMuPDF, python-docx, python-pptx, openpyxl, extract-msg |
| PST 파싱 | libpff / pypff (옵션) |
| 컨테이너 | Docker + Docker Compose (Ollama GPU 패스스루) |

## 프로젝트 구조

```
ast-poc/
├── src/                        # 백엔드 소스
│   ├── api/routes/             # FastAPI 엔드포인트
│   │   ├── health.py
│   │   ├── cases.py            # 케이스 CRUD (Admin)
│   │   ├── indexing.py         # 인덱싱 관리 (Admin)
│   │   ├── documents.py        # 문서 업로드 (Admin)
│   │   ├── settings.py         # 설정/모델 (Admin)
│   │   ├── chat.py             # RAG 질의 (Analyst)
│   │   └── dashboard.py        # 커뮤니케이션 분석 (Analyst)
│   ├── parsers/                # PST, PDF, DOCX 등 파서
│   ├── chunkers/               # 텍스트 청킹 + 메타데이터 보강
│   ├── embeddings/             # Ollama 임베딩 클라이언트
│   ├── vectorstore/            # ChromaDB + BM25 하이브리드 저장소
│   ├── rag/                    # 질의 파서, 검색, RAG 엔진
│   ├── llm/                    # LLM 라우터 (로컬/외부 분기)
│   ├── cases/                  # 케이스 메타데이터 + 라이프사이클
│   ├── indexing/               # 인덱싱 파이프라인 오케스트레이터
│   ├── db/                     # SQLAlchemy 엔진/세션 (sync + async)
│   └── utils/                  # 설정, 로깅
├── frontend/                   # React + TypeScript + Vite
├── tests/
│   ├── unit/                   # 유닛 테스트 (~340)
│   └── integration/            # E2E 통합 테스트 (~14)
├── docs/                       # 아키텍처/가이드/개선 항목
├── scripts/                    # docker-init.sh, run_indexing.py, run_benchmark.py
├── docker/                     # Dockerfile.backend, Dockerfile.frontend, nginx.conf
├── docker-compose.yml
└── data/                       # gitignore됨
    ├── ast.db                  # SQLite (cases, indexing_logs, chat_history, chat_sources)
    ├── input/{pst,documents}/  # 원본 데이터
    ├── processed/              # 파싱/청킹 캐시
    ├── bm25_index/             # BM25 케이스별 pickle
    ├── failed_embeddings/      # 임베딩 영구 실패 청크 (case_id별 JSONL, 재처리용)
    └── vectordb/               # ChromaDB 영구 저장
└── error_logs/                 # WARNING+ 사후 분석용 로그 (gitignore)
    ├── embeddings.log          # 임베딩 timeout/실패
    ├── llm.log                 # LLM 응답/스트리밍 실패
    ├── vectorstore.log         # 벡터 저장 실패
    └── indexing.log            # 파이프라인/파일 처리 실패
```

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- Node.js 18+ (프론트엔드)
- Ollama (`bge-m3`, `gemma4:e4b` 또는 `gpt-oss:20b` 모델)
- Docker & Docker Compose (선택)

### 설치

```bash
# 저장소 클론
git clone https://github.com/<your-org>/ast-poc.git
cd ast-poc

# 백엔드 의존성 설치
pip install -e ".[dev]"

# 프론트엔드 의존성 설치
cd frontend && npm install && cd ..

# Ollama 모델 준비 (개발 환경 기준)
ollama pull bge-m3            # 임베딩 (~1.2GB)
ollama pull gemma4:e4b        # 개발 LLM (가벼움)
# 또는 운영 환경
ollama pull gpt-oss:20b       # 운영 LLM (~12GB)
```

### 환경 설정

환경별 `.env` 예시 파일을 제공합니다. 대상 환경에 맞는 파일을 `.env`로 복사하세요.

```bash
# 개발 환경 (RTX 3060 12GB)
cp .env.dev.example .env

# 운영 환경 (A5000 24GB)
cp .env.prod.example .env
```

| 설정 | 개발 (3060 12GB) | 운영 (A5000 24GB) |
|------|-----------------|-------------------|
| `OLLAMA_LLM_MODEL` | `gemma4:e4b` | `gpt-oss:20b` |
| `EMBED_BATCH_SIZE` | 8 | 16 |
| `INDEXING_STORE_BATCH_SIZE` | 64 | 128 |
| `RERANK_ENABLED` | false | true |
| `OLLAMA_BASE_URL` | localhost:11434 | ollama:11434 (Docker) |

> **임베딩 배치 크기 주의**: bge-m3 실제 컨텍스트 한도(4096 토큰)와 Ollama 큐 동작상, 큰 배치는 타임아웃 캐스케이드를 유발해 인덱싱이 멈출 수 있습니다. 위 기본값에서 안정 동작을 확인 후 단계적으로 상향하세요. 영구 실패한 청크는 `data/failed_embeddings/{case_id}.jsonl`에 보존되어 재처리 가능합니다.

필요에 따라 `.env` 값을 직접 수정하거나, 환경 변수로 개별 오버라이드할 수 있습니다.

### 실행

```bash
# 백엔드 서버
uvicorn src.api.main:app --reload --port 8000

# 프론트엔드 (별도 터미널)
cd frontend && npm run dev
```

### Docker로 실행

```bash
# 4-서비스 스택: backend + frontend(nginx) + chromadb + ollama(GPU)
docker compose up --build -d

# 모델 다운로드 + 디렉토리 초기화 (최초 1회)
bash scripts/docker-init.sh
```

## 개발 가이드

```bash
# 린트
ruff check src/

# 포맷팅
ruff format src/

# 테스트 (전체)
pytest tests/

# 단위 테스트만
pytest tests/unit/ -v

# 인덱싱 CLI (API 대신 직접 실행)
python -m scripts.run_indexing <case_id>
```

## 라이선스

Private — Internal Use Only
