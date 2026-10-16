"""AST PoC 전역 설정"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # v3 설정(CHROMA_*, BM25_INDEX_DIR 등)이 남은 기존 .env 도 기동되게 모르는 키는 무시한다
        extra="ignore",
    )

    # === Ollama ===
    # localhost는 Windows에서 IPv6(::1) 우선이라 IPv4로만 듣는 Ollama 요청마다 ~2초 지연
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "gemma4:12b"  # v4 (지시서 §5.9). 개발 PC .env는 gemma4:e4b
    ollama_embed_model: str = "qwen3-embedding:4b-q8_0"

    # === External LLM ===
    # 보안 모드 OFF 일 때 사용할 provider — "openai" | "anthropic"
    external_llm_provider: str = "anthropic"
    openai_api_key: str = ""
    openai_model: str = "o3"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # === LLM 생성 파라미터 ===
    # RAG에서 temperature가 높으면 할루시네이션 증가. 0.1로 결정적 답변 유도.
    llm_temperature: float = 0.1
    # 로컬 LLM(Ollama) num_ctx. 지정하지 않으면 Ollama가 모델 최대(gemma4 262k)로 KV 캐시를 잡아
    # 24GB에서 임베딩 모델을 밀어낸다. 출처 5단위 × 2,048토큰 + 프롬프트·답변 기준 16k.
    llm_num_ctx: int = 16384

    # === Data Paths ===
    data_input_pst_dir: str = "./data/input/pst"
    data_input_docs_dir: str = "./data/input/documents"
    data_processed_dir: str = "./data/processed"

    # === 이메일 정제 ===
    # 이메일 본문에서 인용 답장 체인, 서명, 법적 면책고지를 제거하여
    # 임베딩 벡터가 보일러플레이트에 오염되는 것을 방지 (src/units/email_clean.py).
    email_clean_enabled: bool = True

    # === v4 증거 단위 (src/units) ===
    # 이 토큰 수(Qwen3 토크나이저 기준, 헤더 제외)를 넘는 단위는 줄·문단 경계에서 파트로 나눈다.
    # XLSX 행 블록·DOCX 가상 페이지는 애초에 이 상한 이하로 만든다.
    unit_max_part_tokens: int = 2048
    # 같은 방에서 메시지 간격이 이 값(분)을 넘으면 새 채팅 세션
    chat_session_gap_minutes: int = 30
    # 토큰 수 기준 토크나이저 (HF 이름). 로컬 캐시·온라인 모두 실패하면 문자 수 근사로 떨어지고
    # 그 사실을 TokenCounter.kind에 기록한다.
    unit_token_hf_name: str = "Qwen/Qwen3-Embedding-4B"
    # 헤더에 표시할 시각의 시간대 (메타데이터 date는 원본 오프셋 그대로 ISO 8601로 저장)
    unit_header_tz: str = "Asia/Seoul"

    # === v4 케이스 색인 (src/index) ===
    # 케이스별 폴더 색인 (units.jsonl + BM25 + 프로필별 dense.npy). 기존 vectordb/bm25_index는 건드리지 않는다.
    index_dir: str = "./data/index"
    # 서비스 인덱싱이 만드는 프로필 (실험 프로필은 scripts/build_profiles.py로 추가)
    service_index_profile: str = "hdr-qwen3"

    # === Embedding ===
    embed_batch_size: int = 256  # VRAM에 따라 조정 (3060 12GB: 256, A5000 24GB: 512+)
    embed_max_concurrent: int = 2  # 동시 배치 요청 수 (Ollama OLLAMA_NUM_PARALLEL과 맞춰 조정)
    # 한 super-batch 전체에 허용되는 최대 시간(초). 초과 시 재시도/분할을 즉시 중단하고
    # 성공한 것만 반환 — "어떤 상황에서도 인덱싱이 N초 이상 같은 배치에 멈춰있으면 안 된다"
    # 원칙. 5일 멈춤 사고 이후 도입.
    embed_total_timeout_sec: int = 120
    # binary subdivide 최대 깊이. 초과하면 해당 배치를 통째로 failed_embeddings에 저장.
    # 256 → 128 → 64 → 32 (3단계, 32개 배치) 까지만 시도하고 포기.
    embed_max_subdivide_depth: int = 3
    # bge-m3 8192 토큰 한도를 바이트 기반으로 사전 차단. 한자/일본어/특수기호 혼재
    # 시 2500자 컷오프 내에서도 BPE 토큰이 8192 초과 가능 → utf-8 12000 byte로 안전망.
    embed_max_bytes: int = 12000
    # 첫 배치에서 성공한 최대 크기를 기억해서 이후 배치 재시도가 항상 큰 크기에서
    # 시작하지 않도록. False면 매 호출 settings.embed_batch_size로 시작.
    embed_adaptive_batch: bool = True
    # 매우 짧은 텍스트(N자 미만)를 길이가 큰 텍스트와 분리. attention softmax NaN /
    # padding overhead로 인한 5xx 예방.
    embed_short_text_threshold: int = 5
    # v4 (지시서 §5.3): Qwen3 임베딩은 질의에만 영어 지시문을 붙이고 문서에는 붙이지 않는다.
    embed_dim: int = 2560
    # Ollama 요청 옵션 num_ctx. 파트는 2,048토큰 이하라 잘리지 않아야 하고, 넘으면 경고 로그를 남긴다.
    embed_num_ctx: int = 8192
    embed_query_task: str = (
        "Given an internal audit investigation query, retrieve emails, chat messages, "
        "and documents in any language that contain evidence relevant to the query"
    )

    # === 가짜 모델 (AST_FAKE_MODELS=1) ===
    # 해시 기반 가짜 임베더와 토큰 겹침 기반 가짜 재순위기를 쓴다. GPU·Ollama 없이
    # 단위·통합 테스트와 실험 selftest를 돌리기 위한 것이다.
    ast_fake_models: bool = False

    # === OCR ===
    # 사용할 OCR 엔진. "tesseract" | "paddle" | "auto"(paddle 가능 시 우선, 아니면 tesseract).
    # paddle은 paddleocr + paddlepaddle 패키지가 설치되어 있어야 함 — 미설치 시 자동으로
    # tesseract fallback. 한국어 정확도: paddle ≈ 95%, tesseract ≈ 70~85%.
    ocr_engine: str = "auto"
    # OCR 언어 — tesseract 형식 ("eng+kor+chi_sim"). paddle 엔진은 내부적으로 매핑.
    ocr_languages: str = "eng+kor+chi_sim"
    # PDF 페이지 → 이미지 렌더링 DPI. 300이 표준이지만 200으로도 한국어 95% 이상 인식되며
    # 메모리/시간 ~50% 절약.
    ocr_dpi: int = 200
    # 한 PDF 안에서 여러 페이지를 동시에 OCR할 워커 수. tesseract는 CPU 바운드이므로 코어 수,
    # paddle은 GPU 단일 인스턴스라 1로 강제됨.
    ocr_max_workers: int = 4
    # 이미지 전처리 (deskew + contrast 보정 + denoise) 적용 여부. 약간의 비용 대비 정확도 향상.
    ocr_preprocess: bool = True
    # OCR 결과 캐싱 — 페이지 이미지 sha256 → 인식 텍스트. 재인덱싱 시 즉시 반환.
    ocr_cache_enabled: bool = True
    # 캐시 디렉토리 (gitignored). 케이스 삭제 시에도 보존됨 (다른 케이스에서도 같은 PDF가 들어올 수 있음).
    ocr_cache_dir: str = "./data/ocr_cache"

    # === Indexing Performance ===
    indexing_workers: int = 0  # 파싱/청킹 병렬 워커 수 (0=CPU 코어 수 자동)
    max_indexing_workers: int = 16  # 워커 수 상한 (Windows는 61 미만 필수)
    max_parsing_workers: int = 8  # 파싱 풀 워커 상한 (ProcessPool/ThreadPool 각각 cap; Windows 61 하드캡 자동 적용)
    indexing_store_batch_size: int = 2000  # 벡터 저장 배치 크기 (3060 12GB 기준)
    # cancel 후 워커 스레드 종료 대기 시간. timeout 초과 시 daemon 스레드로 두고 진행.
    indexing_shutdown_timeout: int = 30
    # 마지막 성공 임베딩 후 N초 경과하면 stalled 플래그 ON (progress API에 노출).
    indexing_stall_threshold_sec: int = 300
    # 케이스가 INDEXING 상태에서 N분 이상 updated_at 갱신 없이 멈춰있으면 비정상 종료로
    # 판정. start_indexing API가 자동으로 ERROR로 복구한 뒤 새 인덱싱 진행 허용.
    # 프로세스 kill / OOM / 정전 등으로 INDEXING 상태가 영구 고착되는 것 방지.
    indexing_stuck_timeout_min: int = 30

    # === Search ===
    # v4 retrieve(): 검색기별 상위 단위 수 (RRF 입력). 지시서 §5.6 고정값.
    search_k_bm25: int = 50
    search_k_dense: int = 50
    rrf_k: int = 60  # Reciprocal Rank Fusion 파라미터

    # === Reranker ===
    # cross-encoder reranker가 검색 품질에 가장 큰 영향을 주므로 기본 ON.
    # FlagEmbedding 미설치 시 런타임에 경고 로그 후 rerank 건너뜀 (graceful degradation).
    # 필요시 API 요청별 rerank=false 또는 환경변수 RERANK_ENABLED=false로 비활성화 가능.
    rerank_enabled: bool = True
    # qwen3 백엔드 모델. 개발 PC(Turing, BF16 미지원)는 Qwen/Qwen3-Reranker-0.6B + fp16
    rerank_model: str = "Qwen/Qwen3-Reranker-4B"
    # bge 백엔드 모델 (R9 비교 실험 전용)
    rerank_bge_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 5  # reranker가 최종 선별할 결과 수
    # reranker top_n 선별 후 이 점수 이하 결과는 제거 (관련 없는 결과가 LLM에
    # 들어가 억지 답변을 만드는 것 방지). normalize=True 기준 0~1 점수.
    # v4: Qwen3 P(yes) 척도에서 다시 정하기 전까지 끈다 (지시서 §5.7)
    rerank_min_score: float = 0.0
    # v4 (지시서 §5.4): 백엔드 qwen3(서비스 기본) | bge(R9 비교 실험 전용)
    rerank_backend: str = "qwen3"
    # 템플릿·질의·문서를 합친 토큰 상한. qwen3·bge 두 백엔드에 같은 값을 쓴다(R9 공정성).
    rerank_max_length: int = 4096
    rerank_dtype: str = "bf16"  # bf16 | fp16 | fp32
    rerank_batch_size: int = 4
    rerank_candidates: int = 30
    rerank_instruction: str = (
        "Judge whether this email, chat session, or document page contains evidence "
        "relevant to the internal audit investigation query. The document may be in any language."
    )

    # === 할루시네이션 제어 ===
    # LLM에 전달하기 전 content가 이 길이 미만인 소스 제거 (과도한 맥락 생성 방지).
    min_source_length: int = 20
    # 질의 키워드가 검색 결과에 하나도 없으면 프롬프트에 추가 경고 삽입.
    enable_relevance_check: bool = True

    # === Token Usage Logging ===
    # 질의 1건당 입력/출력 토큰 수를 TSV 파일에 기록 (외부 API 비용 산정 근거).
    token_log_enabled: bool = True
    token_log_file: str = "logs/token_usage.txt"
    # 입력 토큰 상세 분해(system/context/question)를 JSONL로 기록 (비용 분석용, 기본 OFF).
    token_log_detail: bool = False
    token_log_detail_file: str = "logs/token_usage_detail.jsonl"

    # === API ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # === Security ===
    security_mode: str = "on"  # "on" = local only, "off" = external API allowed

    @property
    def is_secure_mode(self) -> bool:
        return self.security_mode.lower() == "on"

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def get_data_path(self, sub: str) -> Path:
        path = self.project_root / sub
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
