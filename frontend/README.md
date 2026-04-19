# AST PoC Frontend

AST(Audit Support Tool) PoC의 React + TypeScript + Vite 프론트엔드.

## 구성

- **Admin UI** (`src/components/admin/`): 케이스 관리, 데이터 소스 설정, 인덱싱 모니터, 시스템 설정
- **Analyst UI** (`src/components/analyst/`): RAG 채팅, 보안 토글, 검색 결과, 케이스 선택, 대시보드
- **Shared** (`src/components/shared/`): 공통 컴포넌트

## 개발 실행

```bash
# 의존성 설치 (최초 1회)
npm install

# 개발 서버 (HMR)
npm run dev
```

기본 접속: http://localhost:5173

`/api` 요청은 백엔드(`http://localhost:8000`)로 자동 프록시됩니다 (`vite.config.ts` 참고).

## 빌드

```bash
# 프로덕션 빌드
npm run build

# 빌드 결과 미리보기
npm run preview
```

빌드 산출물은 `dist/`에 생성되며, Docker 환경에서는 nginx가 SPA로 서빙합니다 (`docker/nginx.conf`).

## 린트

```bash
npm run lint
```

## 백엔드 연동

- API 문서: http://localhost:8000/docs (FastAPI 자동 생성)
- 헬스체크: http://localhost:8000/health
- SSE 스트리밍: `/api/analyst/chat/stream` (RAG 응답 실시간 출력)

전체 시스템 실행 가이드는 프로젝트 루트의 `docs/guides/quickstart.md` 참고.
