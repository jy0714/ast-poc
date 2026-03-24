# AST PoC 실행 가이드

프론트엔드에 접속해서 시스템을 사용하기까지의 실행 방법입니다.
**두 가지 방식**으로 실행할 수 있습니다:

- **방법 A**: 로컬 직접 실행 (개발/디버깅에 추천)
- **방법 B**: Docker Compose (한 번에 전체 스택 실행)

---

## 사전 요구사항 (공통)

| 항목 | 최소 버전 | 확인 명령 |
|------|----------|----------|
| NVIDIA GPU | A5000 24GB 권장 | `nvidia-smi` |
| Ollama | 0.3+ | `ollama --version` |

Ollama 모델 다운로드 (최초 1회):

```powershell
ollama pull nomic-embed-text     # 임베딩 모델 (~274MB)
ollama pull gpt-oss:20b          # LLM 모델 (~12GB)
```

> GPU VRAM이 부족하면 가벼운 모델로 대체 가능:
> ```powershell
> ollama pull llama3.2:3b         # LLM 대체 (~2GB)
> ```
> `.env`에서 `OLLAMA_LLM_MODEL=llama3.2:3b`로 변경

---

## 방법 A: 로컬 직접 실행

### 추가 요구사항

| 항목 | 최소 버전 | 확인 명령 |
|------|----------|----------|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| pip | 최신 | `pip --version` |

### Step 1. 환경 설정

```powershell
# 프로젝트 루트에서
cd ast-poc

# Python 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -e ".[dev]"

# 환경변수 파일 생성
copy .env.example .env
```

`.env` 파일을 열어 필요시 수정:

```env
# Ollama 주소 (로컬이면 그대로, 원격이면 IP 변경)
OLLAMA_BASE_URL=http://localhost:11434

# 보안 모드 (on=로컬 전용, off=외부 API 허용)
SECURITY_MODE=on
```

### Step 2. 프론트엔드 의존성 설치 (최초 1회)

```powershell
cd frontend
npm install
cd ..
```

### Step 3. 실행 (터미널 2개 필요)

**터미널 1 — 백엔드:**

```powershell
cd ast-poc
.venv\Scripts\activate
uvicorn src.api.main:app --reload --port 8000
```

정상 실행 시:

```
INFO:     AST PoC 서버 시작
INFO:     보안 모드: ON (로컬 전용)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**터미널 2 — 프론트엔드:**

```powershell
cd ast-poc/frontend
npm run dev
```

정상 실행 시:

```
  VITE v8.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 4. 접속

브라우저에서 **http://localhost:5173** 을 열면 AST PoC 화면이 나타납니다.

> - 프론트엔드(Vite)가 `/api` 요청을 자동으로 백엔드(8000)로 프록시합니다.
> - API 문서: http://localhost:8000/docs
> - 헬스체크: http://localhost:8000/health

---

## 방법 B: Docker Compose 실행

### 추가 요구사항

| 항목 | 최소 버전 | 확인 명령 |
|------|----------|----------|
| Docker | 24+ | `docker --version` |
| Docker Compose | v2+ | `docker compose version` |
| NVIDIA Container Toolkit | — | `nvidia-ctk --version` |

> NVIDIA Container Toolkit이 없으면 Ollama가 GPU를 사용하지 못합니다.
> 설치: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Step 1. 환경변수 파일 생성

```powershell
cd ast-poc
copy .env.example .env
```

### Step 2. 데이터 디렉토리 생성

```powershell
mkdir data\input\pst data\input\documents data\processed data\vectordb data\cases data\bm25_index
```

### Step 3. Docker Compose 실행

```powershell
docker compose up --build -d
```

4개 컨테이너가 시작됩니다:

| 서비스 | 포트 | 설명 |
|--------|------|------|
| frontend | **80** | nginx (프론트엔드 + API 프록시) |
| backend | 8000 | FastAPI |
| chromadb | 8100 | ChromaDB 벡터DB |
| ollama | 11434 | Ollama LLM/임베딩 |

### Step 4. Ollama 모델 다운로드 (최초 1회)

```powershell
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gpt-oss:20b
```

### Step 5. 접속

브라우저에서 **http://localhost** 을 열면 AST PoC 화면이 나타납니다.

> - API 문서: http://localhost:8000/docs
> - 헬스체크: http://localhost/health

### Docker 관리 명령어

```powershell
# 로그 확인
docker compose logs -f backend     # 백엔드 로그
docker compose logs -f              # 전체 로그

# 중지
docker compose down

# 재시작 (코드 변경 후)
docker compose up --build -d

# 볼륨 포함 완전 삭제
docker compose down -v
```

### GPU 없는 환경에서 실행

```powershell
copy docker-compose.override.yml.example docker-compose.override.yml
docker compose up --build -d
```

> Ollama가 CPU 모드로 실행됩니다 (속도 매우 느림).

---

## 사용 방법

### 1. Admin — 케이스 생성 및 인덱싱

1. 상단 네비게이션에서 **Admin** 클릭
2. **새 케이스 생성** 섹션에서:
   - 케이스명 입력 (예: `2024년 감사`)
   - 문서 폴더 경로 입력 (예: `D:\data\audit-2024\documents`)
   - PST 파일 경로 입력 (예: `D:\data\audit-2024\mailbox.pst`)
3. **생성** 버튼 클릭
4. 케이스 목록에서 해당 케이스의 **인덱싱** 버튼 클릭
5. 프로그레스 바가 100%가 될 때까지 대기 (상태: `indexing` → `ready`)

```
[케이스 생성] → [인덱싱 시작] → [완료 대기] → [ready 상태]
```

### 2. Analyst — RAG 질의

1. 상단 네비게이션에서 **Analyst** 클릭
2. 상단 드롭다운에서 **케이스 선택** (ready 상태만 표시)
3. 보안 모드 확인 (기본: ON = 로컬 전용)
4. 채팅 입력창에 질문 입력 후 Enter

질문 예시:

```
김철수가 보낸 이메일 요약해줘
2024년 3월 회의록에서 예산 관련 내용 찾아줘
감사 보고서에서 위험 요소 정리해줘
```

> - 응답은 스트리밍으로 실시간 출력됩니다.
> - 응답 하단의 **출처** 카드를 클릭하면 원본 내용을 확인할 수 있습니다.
> - **보안 ON**: 모든 처리가 로컬 서버 내에서 수행 (데이터 외부 전송 없음)
> - **보안 OFF**: 검색은 로컬, LLM 응답 생성만 외부 API 사용

---

## 트러블슈팅

### 백엔드가 시작되지 않음

```powershell
# 포트 충돌 확인
netstat -ano | findstr :8000

# Ollama 연결 확인
curl http://localhost:11434/api/tags
# 또는 PowerShell:
Invoke-RestMethod http://localhost:11434/api/tags
```

### 프론트엔드에서 API 호출 실패 (CORS / 502)

- 로컬 실행: 백엔드가 8000번 포트에서 실행 중인지 확인
- Docker: `docker compose ps`로 backend 컨테이너 상태 확인
- Docker 로그: `docker compose logs backend`

### 인덱싱 시 "Ollama 연결 실패"

```powershell
# Ollama 서비스 확인
ollama list

# Docker 환경이면
docker compose exec ollama ollama list
```

### chroma-hnswlib 빌드 실패 (Windows)

한글 Windows에서 `pip install` 시 C++ 빌드 오류가 발생할 수 있습니다.

1. Visual Studio 2022+ 설치 → "C++를 사용한 데스크톱 개발" 워크로드 포함
2. 빌드:

```powershell
# Developer Command Prompt 또는 vcvars64.bat 실행 후
set DISTUTILS_USE_SDK=1
pip install chroma-hnswlib --no-build-isolation
```

자세한 내용은 `CLAUDE.md`의 "환경 이슈" 섹션 참고.

---

## 접속 URL 정리

| 환경 | 프론트엔드 | API 문서 | 헬스체크 |
|------|-----------|---------|---------|
| 로컬 (Vite) | http://localhost:5173 | http://localhost:8000/docs | http://localhost:8000/health |
| Docker | http://localhost | http://localhost:8000/docs | http://localhost/health |
