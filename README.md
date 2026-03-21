# AST (Audit Support Tool) - PoC

내부 이메일(PST), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX)를 통합 검색할 수 있는 로컬 RAG 시스템

## 주요 기능

- **PST 파싱**: Outlook 이메일 및 Teams 채팅 기록 자동 파싱
- **문서 파싱**: PDF, DOCX, PPTX, XLSX 첨부파일 및 독립 문서 파싱
- **스마트 청킹**: 문서는 문자수 기반, 채팅은 시간 윈도우 기반 분할
- **메타데이터 태깅**: 참여자, 날짜, 소스 타입, 토픽 자동 태깅
- **로컬 임베딩**: Ollama nomic-embed-text (외부 전송 없음)
- **벡터 검색**: ChromaDB 기반 하이브리드 검색 (벡터 유사도 + 메타데이터 필터)
- **보안 모드**: ON(로컬 LLM만 사용) / OFF(외부 API 허용) 토글
- **채팅 UI**: LLM 스타일 웹 인터페이스

## 기술 스택

| 영역 | 기술 |
|------|------|
| 백엔드 | Python 3.11+, FastAPI |
| 프론트엔드 | React + TypeScript |
| LLM (로컬) | Ollama - gpt-oss:20b |
| LLM (외부) | OpenAI o3 (보안 모드 OFF 시) |
| 임베딩 | Ollama - nomic-embed-text |
| 벡터DB | ChromaDB |
| PST 파싱 | libpff / pypff |
| 문서 파싱 | PyMuPDF, python-docx, python-pptx, openpyxl |
| RAG 프레임워크 | LangChain |
| 컨테이너 | Docker + Docker Compose |

## 프로젝트 구조

```
ast-poc/
├── src/                    # 백엔드 소스 코드
│   ├── parsers/            # PST, PDF, DOCX 등 파서
│   ├── chunkers/           # 텍스트 청킹 엔진
│   ├── embeddings/         # 임베딩 모델 연동
│   ├── vectorstore/        # ChromaDB 연동
│   ├── rag/                # RAG 질의 엔진
│   ├── llm/                # LLM 라우팅 (로컬/외부)
│   ├── api/                # FastAPI 엔드포인트
│   └── utils/              # 공통 유틸리티
├── frontend/               # React 프론트엔드
│   ├── src/
│   │   ├── components/     # UI 컴포넌트
│   │   ├── hooks/          # 커스텀 훅
│   │   ├── styles/         # 스타일
│   │   └── utils/          # 프론트 유틸
│   └── public/
├── tests/                  # 테스트
│   ├── unit/
│   ├── integration/
│   └── fixtures/           # 테스트용 샘플 데이터
├── docs/                   # 문서화
│   ├── architecture/       # 아키텍처 문서
│   ├── guides/             # 사용 가이드
│   └── api-reference/      # API 레퍼런스
├── scripts/                # 유틸리티 스크립트
├── docker/                 # Docker 설정
├── data/                   # 데이터 디렉토리 (gitignore)
│   ├── input/              # 원본 파일 투입
│   │   ├── pst/
│   │   └── documents/
│   ├── processed/          # 처리된 중간 데이터
│   └── vectordb/           # ChromaDB 저장소
└── .github/workflows/      # CI/CD
```

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- Ollama (gpt-oss:20b, nomic-embed-text 모델)
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

# Ollama 모델 준비
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### 실행

```bash
# 백엔드 서버
uvicorn src.api.main:app --reload --port 8000

# 프론트엔드 (별도 터미널)
cd frontend && npm run dev
```

### Docker로 실행

```bash
docker compose up --build
```

## 개발 가이드

```bash
# 린트
ruff check src/

# 포맷팅
ruff format src/

# 테스트
pytest tests/
```

## 라이선스

Private - Internal Use Only
