# AST PoC 아키텍처 문서 (v5)

## 시스템 개요

AST(Audit Support Tool)는 내부 커뮤니케이션(이메일, Teams 채팅)과 문서를 통합 검색하는 로컬 RAG 시스템.

- **규모**: 1TB PoC (본 프로젝트 4TB 대응 설계)
- **GPU**: NVIDIA A5000 24GB VRAM
- **운영**: 케이스 단위 격리 · Phase A/B 순차 실행

## 케이스 라이프사이클

| 단계 | 상태 | GPU 사용 | 설명 |
|------|------|---------|------|
| 생성 | `created` | - | 케이스 생성 + 데이터 소스 지정 |
| 인덱싱 | `indexing` | 임베딩 전용 (24GB) | 파싱 → 청킹 → 임베딩 → 저장 |
| 분석 준비 | `ready` | - | 인덱싱 완료, 분석 대기 |
| 분석 중 | `analyzing` | LLM 전용 (24GB) | RAG 질의 응답 |
| 보관 | `archived` | - | 읽기 전용 보관 |
| 폐기 | (삭제) | - | 컬렉션 + 인덱스 클렌징 |

### 증분 인덱싱
- **모드 1 (순차)**: 분석 중단 → 추가분 인덱싱 (GPU 100%) → 분석 재개
- **모드 2 (백그라운드)**: 분석 중 CPU 기반 인덱싱 (속도 느림, 분석 유지)

## 사용자 역할별 UI

### Admin UI (Phase A 관리)
- 케이스 CRUD (생성/조회/삭제/보관)
- 데이터 소스 설정 (PST 경로, 문서 폴더 지정)
- 인덱싱 실행/중단/모니터링
- 증분 인덱싱 트리거
- 시스템 설정 (청킹 파라미터, 모델 설정)

### Analyst UI (Phase B 수행)
- 채팅 스타일 RAG 질의
- 보안 모드 토글 (ON: 로컬 / OFF: 외부 API)
- 검색 결과 + 출처 표시
- 날짜/참여자/소스타입 필터
- 분석 대상 케이스 선택

## API 구조

### Admin API (`/api/admin/`)
- `POST /cases/` — 케이스 생성
- `GET /cases/` — 케이스 목록
- `DELETE /cases/{id}` — 케이스 삭제
- `POST /indexing/start` — 인덱싱 시작
- `GET /indexing/progress/{id}` — 진행률 조회
- `POST /indexing/increment/{id}` — 증분 인덱싱

### Analyst API (`/api/analyst/`)
- `POST /chat/` — RAG 질의
- `POST /chat/stream` — 스트리밍 질의
- `GET /chat/cases` — 분석 가능 케이스 목록

## 하이브리드 검색

| 엔진 | 용도 | 강점 |
|------|------|------|
| ChromaDB (벡터) | 의미 기반 유사도 검색 | "가격 관련 논의" |
| BM25 (키워드) | 정확 키워드 매칭 | "계약번호 ABC-123" |
| Rank Fusion | 두 결과 통합 정렬 | 종합 관련성 최적화 |

## 보안 정책

| 구성 요소 | 보안 ON | 보안 OFF |
|----------|---------|---------|
| 임베딩 | 로컬 (Ollama) | 로컬 (Ollama) |
| 벡터DB | 로컬 (ChromaDB) | 로컬 (ChromaDB) |
| BM25 | 로컬 | 로컬 |
| LLM | 로컬 (gpt-oss:20b) | 외부 API (미정) |
| 외부 전송 | 없음 | 검색된 청크만 |

## 기술 스택

- **백엔드**: Python 3.11+, FastAPI
- **프론트엔드**: React + TypeScript
- **RAG**: LangChain
- **벡터DB**: ChromaDB (케이스별 독립 컬렉션)
- **키워드 검색**: BM25 (rank_bm25)
- **LLM 로컬**: Ollama gpt-oss:20b
- **LLM 외부**: ChatGPT 5.4 / Gemini / Opus 4.6 등 (미정)
- **임베딩**: Ollama nomic-embed-text
- **컨테이너**: Docker + Docker Compose
