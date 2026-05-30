#!/bin/bash
# AST PoC Docker 초기 설정 스크립트
# Ollama 모델 다운로드 + 데이터 디렉토리 생성

set -e

echo "=== AST PoC Docker 초기 설정 ==="

# 1. 데이터 디렉토리 생성
echo "[1/3] 데이터 디렉토리 생성..."
mkdir -p data/input/pst data/input/documents data/processed data/vectordb data/cases data/bm25_index

# 2. .env 파일 생성 (없는 경우)
if [ ! -f .env ]; then
    echo "[2/3] .env 파일 생성 (.env.prod.example 복사)..."
    cp .env.prod.example .env
    echo "  → .env 파일을 확인하고 필요시 수정하세요"
    echo "  → 개발 환경은: cp .env.dev.example .env"
else
    echo "[2/3] .env 파일 이미 존재"
fi

# 3. Docker Compose 실행 + Ollama 모델 다운로드
echo "[3/3] Docker 컨테이너 시작..."
docker compose up -d

echo ""
echo "컨테이너 시작 대기 (10초)..."
sleep 10

echo "Ollama 모델 다운로드..."
docker compose exec ollama ollama pull bge-m3
echo "  ✓ bge-m3 (임베딩, 1024-dim) 완료"

# 운영 LLM (기본: gpt-oss:20b). 개발 환경은 OLLAMA_LLM_MODEL=gemma4:e4b 로 변경.
LLM_MODEL="${OLLAMA_LLM_MODEL:-gpt-oss:20b}"
docker compose exec ollama ollama pull "$LLM_MODEL"
echo "  ✓ $LLM_MODEL (LLM) 완료"

echo ""
echo "=== 설정 완료 ==="
echo "  Frontend: http://localhost:80"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  ChromaDB: http://localhost:8100"
echo "  Ollama:   http://localhost:11434"
