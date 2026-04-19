# Windows 개발 환경 셋업 가이드

AST PoC를 Windows 클라이언트에서 개발 및 테스트하기 위한 가이드입니다.
실제 LLM 서버와 분리된 개발 환경을 구성합니다.

---

## 1. 사전 준비

### 1-1. Python 설치

1. https://www.python.org/downloads/ 에서 Python 3.11+ 다운로드
2. 설치 시 **"Add python.exe to PATH"** 반드시 체크
3. 설치 확인:

```powershell
python --version
# Python 3.11.x 이상 확인
```

### 1-2. Node.js 설치

1. https://nodejs.org/ 에서 LTS 버전 다운로드 (18+)
2. 설치 확인:

```powershell
node --version
npm --version
```

### 1-3. Git 설치

1. https://git-scm.com/download/win 에서 다운로드
2. 설치 시 기본 설정 그대로 진행
3. 설치 확인:

```powershell
git --version
```

### 1-4. Visual Studio Build Tools (PST 파싱 라이브러리용)

일부 Python 패키지(libpff 등)가 C 확장을 빌드해야 할 수 있습니다.

1. https://visualstudio.microsoft.com/visual-cpp-build-tools/ 에서 다운로드
2. "C++를 사용한 데스크톱 개발" 워크로드 선택 후 설치

---

## 2. 프로젝트 클론 및 환경 구성

### 2-1. 저장소 클론

```powershell
git clone https://github.com/<your-org>/ast-poc.git
cd ast-poc
```

### 2-2. Python 가상환경 생성

```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate

# 활성화 확인 (프롬프트 앞에 (.venv) 표시)
```

> PowerShell에서 스크립트 실행이 차단되면:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2-3. 의존성 설치

```powershell
# pip 업그레이드
python -m pip install --upgrade pip

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

> libpff-python이 Windows에서 설치 실패하는 경우:
> ```powershell
> # libpff 없이 나머지 먼저 설치 (PST 파싱은 서버에서만 수행)
> pip install -e ".[dev]" --no-deps
> pip install fastapi uvicorn python-multipart websockets
> pip install langchain-ollama langchain-openai chromadb
> pip install ollama openai
> pip install PyMuPDF python-docx python-pptx openpyxl extract-msg
> pip install rank-bm25 kiwipiepy FlagEmbedding
> pip install sqlalchemy aiosqlite
> pip install pydantic pydantic-settings python-dotenv rich tqdm
> pip install pytest pytest-asyncio pytest-cov ruff mypy httpx
> ```

### 2-4. 환경변수 설정

```powershell
# 개발 환경 (3060 12GB 권장)
copy .env.dev.example .env

# 또는 운영 환경 (A5000 24GB)
copy .env.prod.example .env
```

`.env` 파일을 열어서 수정:

```env
# 로컬 Ollama가 없는 개발 PC에서는 서버 주소로 변경
OLLAMA_BASE_URL=http://<LLM서버IP>:11434

# 또는 개발 PC에 Ollama를 설치한 경우
OLLAMA_BASE_URL=http://localhost:11434

# 외부 API (테스트용)
OPENAI_API_KEY=sk-your-key-here

# 보안 모드 (개발 시 off로 해도 됨)
SECURITY_MODE=off
```

---

## 3. 개발 PC에서의 두 가지 모드

### 모드 A: 원격 Ollama 서버 연결 (추천)

LLM은 별도 서버에서 구동하고, 개발 PC에서는 API만 호출하는 구조입니다.

```
[Windows 개발 PC]              [LLM 서버]
  FastAPI 백엔드  ──HTTP──▶  Ollama (gemma4:e4b / gpt-oss:20b)
  React 프론트엔드            Ollama (bge-m3)
  ChromaDB (로컬)
```

설정 방법:

```env
# .env에서 LLM 서버 주소 지정
OLLAMA_BASE_URL=http://192.168.x.x:11434
```

LLM 서버에서 외부 접속 허용:

```bash
# LLM 서버의 Ollama 설정 (Linux)
sudo systemctl edit ollama

# [Service] 섹션에 추가:
Environment="OLLAMA_HOST=0.0.0.0:11434"

# 재시작
sudo systemctl restart ollama
```

### 모드 B: 개발 PC에 Ollama 직접 설치

GPU가 있는 Windows 개발 PC라면 로컬에서도 구동 가능합니다.

1. https://ollama.com/download/windows 에서 다운로드 및 설치
2. 모델 다운로드:

```powershell
# 임베딩 (필수, 1024-dim)
ollama pull bge-m3

# LLM (개발 환경)
ollama pull gemma4:e4b

# LLM (운영 환경, A5000 24GB)
ollama pull gpt-oss:20b
```

3. 동작 확인:

```powershell
ollama list
# bge-m3, gemma4:e4b 또는 gpt-oss:20b 확인

# 간단한 테스트
ollama run gemma4:e4b "안녕하세요"
```

> 개발 PC GPU VRAM이 부족하면 `gemma4:e4b` 또는 더 작은 모델로 대체 가능
> `.env`의 `OLLAMA_LLM_MODEL`로 변경

---

## 4. 개발 서버 실행

### 4-1. 백엔드 실행

```powershell
# 가상환경 활성화 확인
.venv\Scripts\activate

# FastAPI 개발 서버 (자동 리로드)
uvicorn src.api.main:app --reload --port 8000
```

정상 실행 시:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     AST PoC 서버 시작
INFO:     보안 모드: OFF (외부 API 허용)
```

API 문서 확인: http://localhost:8000/docs

### 4-2. 프론트엔드 실행 (별도 터미널)

```powershell
cd frontend
npm install    # 최초 1회
npm run dev
```

### 4-3. 헬스체크

```powershell
# PowerShell에서
Invoke-RestMethod http://localhost:8000/health

# 또는 브라우저에서
# http://localhost:8000/health
```

---

## 5. 개발 워크플로우

### 5-1. 코드 린트 & 포맷팅

```powershell
# 린트 체크
ruff check src/

# 린트 자동 수정
ruff check src/ --fix

# 코드 포맷팅
ruff format src/

# 타입 체크
mypy src/ --ignore-missing-imports
```

### 5-2. 테스트 실행

```powershell
# 전체 테스트
pytest tests/ -v

# 유닛 테스트만
pytest tests/unit/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
# htmlcov/index.html 열어서 확인
```

### 5-3. Claude Code 사용

```powershell
# Claude Code 설치 (npm 전역)
npm install -g @anthropic-ai/claude-code

# 프로젝트 디렉토리에서 실행
cd ast-poc
claude
```

Claude Code가 `CLAUDE.md`를 자동으로 읽고 프로젝트 컨텍스트를 이해합니다.
이후 자연어로 지시하면 됩니다:

```
> PST 파서에서 이메일 본문 추출 기능 구현해줘
> 채팅 청커의 시간 윈도우 로직 만들어줘
> ChromaDB 검색에 메타데이터 필터링 추가해줘
```

### 5-4. Git 작업 흐름

```powershell
# 브랜치 생성
git checkout -b feature/pst-parser

# 변경사항 확인
git status
git diff

# 커밋
git add .
git commit -m "feat: PST 이메일 파서 구현"

# 푸시
git push origin feature/pst-parser

# PR 생성 후 main에 머지
```

---

## 6. 트러블슈팅

### libpff-python 설치 실패

Windows에서 libpff 빌드가 어려울 수 있습니다.
개발 단계에서는 PST 파싱을 Mock으로 대체하고, 실제 파싱은 Linux 서버에서 수행하는 방식도 가능합니다.

```python
# tests/fixtures/ 에 샘플 파싱 결과 JSON을 두고 개발
# src/parsers/pst_parser.py에서 Mock 모드 지원
```

### ChromaDB sqlite3 버전 오류

```powershell
# 오류: chromadb requires sqlite3 >= 3.35.0
pip install pysqlite3-binary
```

그래도 안되면 `chromadb`를 클라이언트 모드로 사용:

```python
# Docker로 ChromaDB 서버 실행
docker run -p 8100:8000 chromadb/chroma

# .env 에서 ChromaDB 서버 주소 지정 (구현 시 반영)
```

### Ollama 연결 실패

```powershell
# Ollama 서비스 상태 확인
ollama list

# 원격 서버 연결 테스트
curl http://<서버IP>:11434/api/tags

# PowerShell에서:
Invoke-RestMethod http://<서버IP>:11434/api/tags
```

### PowerShell 스크립트 실행 정책

```powershell
# 현재 정책 확인
Get-ExecutionPolicy

# 개발용으로 변경
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 포트 충돌

```powershell
# 8000번 포트 사용 중인 프로세스 확인
netstat -ano | findstr :8000

# 해당 PID 프로세스 종료
taskkill /PID <PID> /F
```

---

## 7. 추천 IDE 설정

### VS Code 확장 프로그램

- **Python** (Microsoft)
- **Pylance** (타입 체크)
- **Ruff** (린터/포맷터)
- **ES7+ React/Redux/React-Native snippets**
- **Tailwind CSS IntelliSense**
- **Docker**
- **GitLens**

### VS Code settings.json

```json
{
    "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```
