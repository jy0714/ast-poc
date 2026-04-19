# AST PoC 아키텍처 문서 (v5)

## 시스템 개요

AST(Audit Support Tool)는 내부 커뮤니케이션(이메일, Teams 채팅)과 문서를 통합 검색하는 로컬 RAG 시스템.

- **규모**: 1TB PoC (본 프로젝트 4TB 대응 설계)
- **GPU**: 개발 NVIDIA RTX 3060 12GB / 운영 NVIDIA A5000 24GB
- **운영**: 케이스 단위 격리 · Phase A/B 순차 실행

## 케이스 라이프사이클

상태 머신 (`src/cases/case_store.py` `_VALID_TRANSITIONS`):

| 단계 | 상태 | GPU 사용 | 설명 |
|------|------|---------|------|
| 생성 | `created` | - | 케이스 생성 + 데이터 소스 지정 |
| 인덱싱 | `indexing` | 임베딩 전용 | 파싱 → 청킹 → 임베딩 → 저장 |
| 분석 준비 | `ready` | LLM (질의 시) | 인덱싱 완료, 분석 가능 |
| 보관 | `archived` | - | 읽기 전용 보관 (`created`로 복구 가능) |
| 에러 | `error` | - | 실패 (각 상태에서 전이 가능) |
| 폐기 | (삭제) | - | 컬렉션 + 인덱스 클렌징 |

전이: `created → indexing → ready → archived`, 각 상태에서 `error` 전이 가능, `archived → created` 복구 가능.

### 증분 인덱싱
- **모드 1 (순차)**: 분석 중단 → 추가분 인덱싱 (GPU 100%) → 분석 재개
- **모드 2 (백그라운드)**: 분석 중 CPU 기반 인덱싱 (속도 느림, 분석 유지)

## 사용자 역할별 UI

### Admin UI (Phase A 관리)
- 케이스 CRUD (생성/조회/삭제/보관)
- 데이터 소스 설정 (PST 경로, 문서 폴더 지정)
- 인덱싱 실행/중단/모니터링
- 증분 인덱싱 트리거
- 시스템 설정 (청킹 파라미터, 모델 설정, 보안 모드, reranker 토글)

### Analyst UI (Phase B 수행)
- 채팅 스타일 RAG 질의 (동기 + SSE 스트리밍)
- 보안 모드 토글 (ON: 로컬 / OFF: 외부 API)
- 검색 결과 + 출처 표시
- 날짜/참여자/소스타입 필터
- 분석 대상 케이스 선택
- 커뮤니케이션 분석 대시보드 (참여자/토픽/타임라인)

## API 구조

### Admin API (`/api/admin/`)
- `POST /cases/` — 케이스 생성
- `GET /cases/` — 케이스 목록 (?status= 필터)
- `GET /cases/{id}` — 케이스 상세
- `PATCH /cases/{id}` — 데이터 소스 추가
- `DELETE /cases/{id}` — 케이스 삭제 (벡터DB 포함)
- `POST /cases/{id}/archive` — 케이스 보관
- `POST /indexing/start` — 인덱싱 시작
- `POST /indexing/stop/{id}` — 인덱싱 중단
- `GET /indexing/progress/{id}` — 진행률 조회
- `POST /indexing/increment/{id}` — 증분 인덱싱
- `POST /documents/upload/{id}` — 문서 업로드
- `GET/PATCH /settings/...` — 보안 모드, 모델, reranker 설정

### Analyst API (`/api/analyst/`)
- `POST /chat/` — RAG 동기 질의
- `POST /chat/stream` — SSE 스트리밍 질의
- `GET /chat/cases` — 분석 가능 케이스 목록
- `GET /dashboard/...` — 커뮤니케이션 분석 (참여자/토픽/타임라인)

## 하이브리드 검색

| 엔진 | 용도 | 강점 |
|------|------|------|
| ChromaDB (벡터) | 의미 기반 유사도 검색 (bge-m3, 1024-dim) | "가격 관련 논의" |
| BM25 + kiwipiepy | 한국어 형태소 키워드 매칭 | "계약번호 ABC-123" |
| RRF (Reciprocal Rank Fusion) | 두 결과 통합 정렬 | 종합 관련성 최적화 |
| Reranker (선택) | BAAI/bge-reranker-v2-m3 | 상위 K개 정밀 재정렬 |

저장 구조: 단일 공유 ChromaDB 컬렉션 (`ast_chunks`) + `case_id` 메타필터로 케이스 격리. BM25는 케이스별 pickle 파일 (`data/bm25_index/{case_id}.pkl`).

## 보안 정책

| 구성 요소 | 보안 ON | 보안 OFF |
|----------|---------|---------|
| 임베딩 | 로컬 (Ollama bge-m3) | 로컬 (Ollama bge-m3) |
| 벡터DB | 로컬 (ChromaDB) | 로컬 (ChromaDB) |
| BM25 | 로컬 | 로컬 |
| Reranker | 로컬 (FlagEmbedding) | 로컬 (FlagEmbedding) |
| LLM | 로컬 (gemma4:e4b / gpt-oss:20b) | 외부 API (OpenAI 등) |
| 외부 전송 | 없음 | 검색된 청크만 |

## 기술 스택

- **백엔드**: Python 3.11+, FastAPI, Uvicorn
- **프론트엔드**: React + TypeScript + Vite
- **RAG**: 자체 구현 (langchain-ollama, langchain-openai 활용)
- **벡터DB**: ChromaDB (단일 공유 컬렉션 + case_id 메타필터)
- **키워드 검색**: rank-bm25 + kiwipiepy (한국어 형태소)
- **Reranker**: BAAI/bge-reranker-v2-m3 (FlagEmbedding) — 선택적
- **LLM (개발)**: Ollama `gemma4:e4b`
- **LLM (운영)**: Ollama `gpt-oss:20b`
- **LLM 외부**: OpenAI (보안 모드 OFF 시)
- **임베딩**: Ollama `bge-m3` (1024-dim, 8192 토큰)
- **메타DB**: SQLite (SQLAlchemy ORM, sync + async)
- **컨테이너**: Docker + Docker Compose (Ollama GPU 패스스루)
