# v4 모델 오프라인 반입 절차

사내망(A5000)에서 인터넷 없이 v4 모델을 쓰기 위한 절차다. 인터넷이 되는 PC에서 받아
복사한 뒤, 사내 PC에서는 오프라인 모드로만 연다. 조사 데이터는 이 과정에서 밖으로 나가지 않는다.

## 필요한 모델

| 용도 | 위치 | 이름 | 크기(약) |
|---|---|---|---|
| 임베딩(서비스) | Ollama | `qwen3-embedding:4b-q8_0` | 4.7GB |
| 응답 생성 | Ollama | `gemma4:12b` | 8GB |
| 재순위(서비스) | HF | `Qwen/Qwen3-Reranker-4B` | 8GB |
| 재순위(R9 비교) | HF | `BAAI/bge-reranker-v2-m3` | 2.2GB |
| 임베딩(R9 비교) | Ollama | `bge-m3` | 1.2GB |
| 임베딩 일치 확인 | HF | `Qwen/Qwen3-Embedding-4B` | 8GB (`check_embed_parity` 1회용) |
| 토큰 수 계산 | HF | `Qwen/Qwen3-Embedding-4B` 토크나이저만 | 16MB |

개발 PC(RTX 2060)는 재순위에 `Qwen/Qwen3-Reranker-0.6B`(1.2GB, fp16)를 쓴다.

## 1. 인터넷 PC에서 받기

```bash
# Ollama
ollama pull qwen3-embedding:4b-q8_0
ollama pull gemma4:12b
ollama pull bge-m3

# Hugging Face (venv에서)
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['Qwen/Qwen3-Reranker-4B','BAAI/bge-reranker-v2-m3','Qwen/Qwen3-Embedding-4B']]"
```

## 2. 복사

| 원본 (Windows 기본 경로) | 사내 PC의 같은 경로 |
|---|---|
| `%USERPROFILE%\.ollama\models\` 전체 | `%USERPROFILE%\.ollama\models\` (`OLLAMA_MODELS`를 바꿨다면 그 경로) |
| `%USERPROFILE%\.cache\huggingface\hub\models--*` | `%USERPROFILE%\.cache\huggingface\hub\` (`HF_HOME`을 바꿨다면 그 아래 `hub\`) |

Docker로 띄우는 경우 Ollama 볼륨에 `models\`를, 백엔드 컨테이너의 `HF_HOME`에 `hub\`를 넣는다.

## 3. 사내 PC 설정

`.env` 또는 환경 변수:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RERANK_DTYPE=bf16
OLLAMA_EMBED_MODEL=qwen3-embedding:4b-q8_0
```

## 4. 확인

```bash
ollama list                                   # 세 태그가 보이는지
python -m scripts.check_models                # 임베딩 차원 2560, 재순위 점수 방향, 시간·VRAM
python -m scripts.check_embed_parity <case_id> --dtype bf16   # 평균 코사인 ≥ 0.99 (A5000, 1회)
```

`HF_HUB_OFFLINE=1`에서 모델 로드가 실패하면 `hub\models--…\snapshots\` 아래에 가중치(`*.safetensors`)가
있는지 확인한다. Windows에서 심볼릭 링크 없이 받은 캐시도 그대로 복사하면 동작한다.
