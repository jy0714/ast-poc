# AST (Audit Support Tool) - PoC

내부 이메일(PST·EML), Teams 채팅(PST), 문서(PDF/DOCX/PPTX/XLSX/XLS/HTML/EML/MSG)를 통합 검색하는 로컬 RAG 시스템.

## 주요 기능 (v4)

- **증거 단위 색인**: 이메일 1통·채팅 세션 1개·문서/첨부 1쪽을 판정 단위로 만들고, 긴 단위만 파트로 분할 (`src/units/`)
- **메타데이터 헤더**: 단위마다 한국어 레이블 헤더(제목·참여자·날짜·파일명 등)를 붙여 검색 입력에 포함
- **이메일 정제·언어 감지**: 다국어 인용·서명 제거, lingua 기반 단위별 언어 판정
- **로컬 임베딩**: Ollama `qwen3-embedding:4b-q8_0` (2560차원, 외부 전송 없음)
- **하이브리드 검색**: BM25(문자 체계별 토크나이저) 50 + Dense 50 → RRF(k=60) → 재순위 30 → 상위 100
- **재순위**: `Qwen/Qwen3-Reranker-4B` (bf16). `bge-m3` + `bge-reranker-v2-m3` 는 비교 실험(R9) 전용
- **케이스 폴더 색인**: `data/index/{case_id}/` 에 단위·BM25·프로필별 dense 행렬 저장 (케이스 격리 자명, 원자적 교체)
- **보안 모드**: ON(전부 로컬) / OFF(LLM만 외부 API)
- **실험 하네스**: `experiments_v4/` — 서비스와 같은 `retrieve()`로 R1~R9 실행·판정·평가

## 기술 스택

| 영역 | 기술 |
|------|------|
| 백엔드 | Python 3.12, FastAPI, Uvicorn |
| 프론트엔드 | React + TypeScript + Vite |
| LLM (로컬) | Ollama `gemma4:12b` |
| LLM (외부) | OpenAI / Anthropic (보안 모드 OFF 시) |
| 임베딩 | Ollama `qwen3-embedding:4b-q8_0` (2560-dim) |
| 색인 | 케이스 폴더: `units.jsonl` + BM25 pickle(hdr/nohdr) + 프로필별 `dense.npy`(fp16, 정확 코사인 탐색) |
| 키워드 검색 | rank-bm25 + 문자 체계별 토크나이저 (한국어 kiwipiepy) |
| Reranker | Qwen3-Reranker-4B (transformers) |
| 메타DB | SQLite (SQLAlchemy ORM) |
| 문서 파싱 | PyMuPDF, python-docx, python-pptx, openpyxl, xlrd, extract-msg, OCR(Tesseract/Paddle) |
| 컨테이너 | Docker Compose (backend + frontend + ollama GPU) |

## 프로젝트 구조

```
ast-poc/
├── src/
│   ├── api/routes/             # health, cases, indexing, documents, settings, chat, dashboard
│   ├── parsers/                # PST, 문서 파서
│   ├── units/                  # 증거 단위: 스키마·빌더·헤더·정제·분할·언어 감지·토큰 수
│   ├── text/                   # BM25 토크나이저 (문자 체계별)
│   ├── index/                  # 케이스 폴더 색인 (case_index, bm25)
│   ├── embeddings/             # Ollama 임베딩 클라이언트 (+ 가짜 모델)
│   ├── rag/                    # retrieval(retrieve), reranker, engine, query_parser
│   ├── llm/                    # LLM 라우터 (로컬/외부 분기)
│   ├── cases/                  # 케이스 메타데이터 + 라이프사이클
│   ├── indexing/               # 인덱싱 파이프라인
│   ├── db/                     # SQLAlchemy 엔진/세션
│   └── utils/                  # 설정, 로깅
├── experiments_v4/             # 실험 하네스 (python -m ev4 …, README 참고)
├── frontend/
├── tests/{unit,integration}/
├── scripts/                    # run_indexing, build_profiles, unit_stats, check_embed_parity, check_models, docker-init.sh
├── docker/ · docker-compose.yml
└── data/                       # gitignore됨
    ├── ast.db                  # SQLite
    ├── input/{pst,documents}/  # 원본 데이터
    ├── index/{case_id}/        # v4 케이스 색인 (manifest.json, units.jsonl, bm25/, profiles/)
    ├── failed_embeddings/      # 임베딩 영구 실패 파트 (case_id별 JSONL)
    └── ocr_cache/
```

v3 데이터 폴더(`data/vectordb`, `data/bm25_index`, `data/processed`)는 더 이상 쓰지 않지만 자동 삭제하지 않습니다.

## 빠른 시작

### 사전 요구사항

- Python 3.12, Node.js 18+
- NVIDIA 드라이버: CUDA 12.x 지원 버전 (torch cu126·Ollama GPU 사용 조건)
- Ollama: `qwen3-embedding:4b-q8_0`, `gemma4:12b` (실험 R9용 `bge-m3`)
- HF 가중치: `Qwen/Qwen3-Reranker-4B` (실험 R9용 `BAAI/bge-reranker-v2-m3`)

### 설치

```bash
pip install -e ".[dev]"
cd frontend && npm install && cd ..
ollama pull qwen3-embedding:4b-q8_0
ollama pull gemma4:12b
```

### 환경 설정

```bash
cp .env.prod.example .env   # RTX 3090 24GB (gemma4:12b, Reranker-4B bf16, EMBED_BATCH_SIZE=16)
# cp .env.dev.example .env  # 구 개발 PC RTX 2060 12GB (gemma4:e4b, Reranker-0.6B fp16)
```

### 실행

```bash
uvicorn src.api.main:app --reload --port 8000
cd frontend && npm run dev
```

### Docker로 실행

```bash
# 3-서비스 스택: backend + frontend(nginx) + ollama(GPU)
docker compose up --build -d
bash scripts/docker-init.sh   # 최초 1회: 디렉토리 + 모델 다운로드
```

## OCR (스캔 PDF 텍스트 추출)

스캔된 PDF나 텍스트 레이어가 없는 PDF는 OCR로 텍스트를 추출합니다. 두 엔진을 지원하며
`settings.ocr_engine`으로 토글합니다.

### Tesseract (기본, 가벼움)

- **정확도**: 한국어 70~85%, 영어 90%+
- **의존성**: 시스템 `tesseract` 바이너리 + 언어 모델
- **설치**:
  - Linux/Docker: `apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng`
  - macOS: `brew install tesseract tesseract-lang`
  - Windows: [UB Mannheim 빌드](https://github.com/UB-Mannheim/tesseract/wiki) 설치 후 PATH 등록

### PaddleOCR (선택, 한국어 정확도 95%+)

- **정확도**: 한국어 95%+, GPU 활용 시 빠름
- **의존성**: paddleocr + paddlepaddle (~4GB)
- **설치**:
  ```bash
  # CPU
  pip install -e ".[ocr-paddle]"

  # GPU (CUDA)
  pip install paddleocr paddlepaddle-gpu
  ```
- 설치 후 `settings.ocr_engine="auto"`(기본)이면 자동으로 paddle 우선 사용. 미설치 시
  자동으로 tesseract fallback.

### 동작 옵션 (`.env` 또는 환경 변수)

| 키 | 기본 | 설명 |
|---|---|---|
| `ocr_engine` | `auto` | `auto` / `tesseract` / `paddle` |
| `ocr_languages` | `eng+kor+chi_sim` | tesseract 형식 (paddle은 첫 매칭 언어 자동 선택) |
| `ocr_dpi` | `200` | 페이지 → 이미지 렌더링 DPI (300이 표준이지만 200이 속도 1.5배) |
| `ocr_max_workers` | `4` | tesseract 페이지 병렬 워커 (paddle은 GPU 단일 컨텍스트라 1로 강제) |
| `ocr_preprocess` | `true` | deskew + denoise + contrast 보정 (OpenCV) |
| `ocr_cache_enabled` | `true` | 페이지 이미지 sha256 → 결과 캐시 (재인덱싱 시 즉시 반환) |
| `ocr_cache_dir` | `./data/ocr_cache` | 캐시 위치 (gitignored) |

### 동작이 안 될 때 확인 순서

1. 인덱싱 완료 로그에서 `스캔 PDF(OCR 미적용) N개`가 보이는지 — 보이면 OCR 엔진 미가용
2. 시작 시 로그에 `Tesseract OCR 가용` 또는 `PaddleOCR 가용`이 한 번 출력되는지
3. 안 보이면 위 설치 단계 확인

## 개발 가이드

```bash
# 린트
ruff check src/ tests/ experiments_v4/

# 테스트 (전체)
pytest tests/
pytest experiments_v4/tests        # 실험 하네스 (가짜 모델 selftest 포함)

# 단위 테스트만
pytest tests/unit/ -v

# 인덱싱 CLI (API 대신 직접 실행)
python -m scripts.run_indexing <case_id>
```

## 라이선스

Private — Internal Use Only
