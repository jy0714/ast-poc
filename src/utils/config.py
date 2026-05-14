"""AST PoC 전역 설정"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # === Ollama ===
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gemma4:e4b"
    ollama_embed_model: str = "bge-m3"

    # === External LLM ===
    openai_api_key: str = ""
    openai_model: str = "o3"

    # === ChromaDB ===
    chroma_persist_dir: str = "./data/vectordb"
    chroma_collection_name: str = "ast_documents"

    # === Data Paths ===
    data_input_pst_dir: str = "./data/input/pst"
    data_input_docs_dir: str = "./data/input/documents"
    data_processed_dir: str = "./data/processed"

    # === Chunking ===
    chunk_size_docs: int = 1000
    chunk_overlap_docs: int = 200
    chat_window_minutes: int = 30
    email_thread_max_chars: int = 2000  # 스레드 청크 최대 문자 수

    # === BM25 ===
    bm25_index_dir: str = "./data/bm25_index"

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
    search_top_k: int = 20  # 하이브리드 검색 초기 후보 수
    rrf_k: int = 60  # Reciprocal Rank Fusion 파라미터
    rrf_min_score: float = 0.0141  # RRF 최소 스코어 임계값 (한쪽만 10위 이하 필터링)

    # === Reranker ===
    rerank_enabled: bool = False  # 기본 OFF, API 요청별 또는 전역 토글 가능
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k_candidates: int = 50  # reranker에 넘길 초기 후보 수
    rerank_top_n: int = 5  # reranker가 최종 선별할 결과 수

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
