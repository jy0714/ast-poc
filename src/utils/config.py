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
    ollama_llm_model: str = "gpt-oss:20b"
    ollama_embed_model: str = "nomic-embed-text"

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
    embed_batch_size: int = 100

    # === Search ===
    search_top_k: int = 20  # 하이브리드 검색 초기 후보 수
    rrf_k: int = 60  # Reciprocal Rank Fusion 파라미터

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
