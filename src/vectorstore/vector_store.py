"""벡터 저장소 — ChromaDB + BM25 하이브리드 검색

ChromaDB: 벡터 유사도 검색 + 메타데이터 필터 (단일 컬렉션 + case_id 필터)
BM25: 케이스별 키워드 인덱스 (per-case pickle)
RRF (Reciprocal Rank Fusion): 두 검색 결과를 통합 순위로 결합

설계 변경 (2026-04-18):
- 케이스별 컬렉션(`case_{id}`) → 단일 공유 컬렉션 + 메타데이터 case_id 필터.
  이유: 케이스 수가 많아질 때 ChromaDB 컬렉션당 HNSW 인덱스 부담 누적 회피.
- BM25는 케이스별 파일 유지 (IDF 분리로 검색 품질 보존).
- 중복 체크: collection.upsert()로 위임 (인메모리 _known_ids 캐시 제거).
- BM25 토큰화: 신규 텍스트만 증분 토큰화 (전체 재토큰화 O(N²) 제거).
- get_stats(): 메타데이터 전체 로드 대신 source_type별 where 카운트.
- BM25 검색: 인메모리 corpus + metadata 캐시로 ChromaDB 조회 제거.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from src.chunkers.chunker import Chunk
from src.chunkers.metadata_enricher import (
    deserialize_metadata_from_chroma,
    serialize_metadata_for_chroma,
)
from src.embeddings.embedding_service import EmbeddingDimensionError, EmbeddingService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# get_stats()에서 카운트하는 known source_type — 신규 타입은 여기 추가
_KNOWN_SOURCE_TYPES = ("email", "teams_chat", "document", "attachment")

# HNSW 세그먼트가 정상이라면 모두 0바이트 초과여야 하는 필수 파일.
# 하나라도 누락되거나 0바이트면 ChromaDB가 'unsupported opcode \0' 등으로 deserialize 실패.
_CHROMA_REQUIRED_SEGMENT_FILES = (
    "data_level0.bin",
    "header.bin",
    "length.bin",
    "link_lists.bin",
)

# 무결성 체크는 같은 persist_dir에 대해 프로세스당 1회만 수행 (불필요한 fs 스캔 방지)
_integrity_checked: set[str] = set()


class ChromaIntegrityError(RuntimeError):
    """ChromaDB persist 디렉토리의 HNSW 세그먼트가 손상된 상태로 감지됨"""


def _is_chroma_segment_dir(name: str) -> bool:
    """ChromaDB 세그먼트 디렉토리 이름은 UUID4 형식 (8-4-4-4-12 hex)"""
    if len(name) != 36:
        return False
    parts = name.split("-")
    if tuple(len(p) for p in parts) != (8, 4, 4, 4, 12):
        return False
    return all(c in "0123456789abcdef-" for c in name.lower())


def _auto_quarantine_enabled() -> bool:
    """CHROMA_AUTO_QUARANTINE — 손상 감지 시 persist_dir 자동 격리 여부 (기본 ON)"""
    return os.environ.get("CHROMA_AUTO_QUARANTINE", "1").lower() in ("1", "true", "yes", "on")

# --- BM25 토크나이저 (kiwipiepy 우선, 정규식 fallback) ---

_USE_KIWI = False
_kiwi_instance = None

try:
    from kiwipiepy import Kiwi as _Kiwi

    _kiwi_instance = _Kiwi()
    _USE_KIWI = True
    logger.info("kiwipiepy 형태소 분석기 로드 완료, BM25 토크나이저로 사용")
except ImportError:
    logger.info("kiwipiepy 미설치, 정규식 기반 BM25 토크나이저 사용")

# 추출할 품사 태그 (명사/동사/형용사/부사 계열)
_KIWI_POS_TAGS = frozenset({
    "NNG",  # 일반명사
    "NNP",  # 고유명사
    "NNB",  # 의존명사
    "VV",   # 동사
    "VA",   # 형용사
    "MAG",  # 일반부사
    "SL",   # 외국어 (영어 등)
    "SH",   # 한자
    "SN",   # 숫자
})


def _kiwi_tokenize(text: str) -> list[str]:
    """kiwipiepy 또는 정규식 fallback으로 토큰 추출"""
    if _USE_KIWI and _kiwi_instance is not None:
        tokens: list[str] = []
        for token in _kiwi_instance.tokenize(text):
            if token.tag not in _KIWI_POS_TAGS:
                continue
            form = token.form
            # 외국어(SL)는 3글자 이상, 그 외는 2글자 이상
            min_len = 3 if token.tag == "SL" else 2
            if len(form) < min_len:
                continue
            if token.tag == "SL":
                form = form.lower()
            tokens.append(form)
        return tokens

    # fallback: 정규식 기반
    import re

    tokens = []
    tokens.extend(re.findall(r"[가-힣]{2,}", text))
    tokens.extend(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text))
    return tokens


class VectorStoreService:
    """ChromaDB + BM25 하이브리드 벡터 저장소

    두 가지 모드 지원:
        1) 멀티-케이스 모드 (case_id 지정): 단일 공유 컬렉션 + case_id 메타 필터
        2) 단일-컬렉션 모드 (collection_name 지정): 컬렉션 자체로 격리 (테스트/레거시)

    사용법 (운영):
        store = VectorStoreService(case_id="abc123")
        store.add_chunks(chunks)
        results = store.search("감사 보고서")  # 자동으로 case_id 필터 적용

    사용법 (테스트/레거시):
        store = VectorStoreService(collection_name="test_col", ephemeral=True)
        store.add_chunks(chunks)
    """

    def __init__(
        self,
        case_id: str | None = None,
        collection_name: str | None = None,
        persist_dir: str | None = None,
        ephemeral: bool = False,
    ) -> None:
        """VectorStoreService 초기화

        Args:
            case_id: 케이스 ID (지정 시 단일 공유 컬렉션 + 자동 필터 모드)
            collection_name: ChromaDB 컬렉션명 (case_id가 없을 때 사용; 기본=settings)
            persist_dir: ChromaDB 저장 경로 (기본: settings.chroma_persist_dir)
            ephemeral: True이면 인메모리 모드 (테스트용)
        """
        self.case_id = case_id
        if case_id:
            # 멀티-케이스 모드: 단일 공유 컬렉션
            self.collection_name = collection_name or settings.chroma_collection_name
            self._bm25_key = f"case_{case_id}"
        else:
            # 단일-컬렉션 모드 (테스트/레거시)
            self.collection_name = collection_name or settings.chroma_collection_name
            self._bm25_key = self.collection_name

        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.ephemeral = ephemeral
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._embedding_service = EmbeddingService()

        # BM25 인덱스 상태 (case_id 모드에서는 케이스별 corpus만 보유)
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_tokenized: list[list[str]] = []  # 신규 텍스트만 증분 토큰화
        # BM25 검색 결과를 ChromaDB 재조회 없이 반환하기 위한 메타데이터 캐시
        self._bm25_metadata: dict[str, dict[str, Any]] = {}

        # 차원 검증 1회만 수행 (대량 인덱싱 시 매 배치 collection.count()/peek() 회피)
        self._dim_validated: bool = False

    def _get_client(self) -> chromadb.ClientAPI:
        """ChromaDB 클라이언트 초기화 (지연 생성)"""
        if self._client is None:
            if self.ephemeral:
                self._client = chromadb.EphemeralClient()
            else:
                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._check_chroma_integrity()
                self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _check_chroma_integrity(self) -> None:
        """ChromaDB persist_dir의 HNSW 세그먼트 무결성 검증.

        강제 종료(process kill) 시 `link_lists.bin` 등이 0바이트로 남아
        다음 write 시 `unsupported opcode '\\0'` pickle deserialize 에러로
        모든 저장이 실패하는 케이스를 사전 차단.

        - 손상 감지 + `CHROMA_AUTO_QUARANTINE=1`(기본): persist_dir 전체를
          `vectordb.quarantine_<ts>`로 이동하고 빈 디렉토리 새로 생성.
          기존 데이터는 보존되어 사후 분석 가능.
        - 손상 감지 + `CHROMA_AUTO_QUARANTINE=0`: ChromaIntegrityError 발생.
        """
        persist_path = Path(self.persist_dir).resolve()
        key = str(persist_path)
        if key in _integrity_checked:
            return

        if not persist_path.exists():
            _integrity_checked.add(key)
            return

        corrupted: list[tuple[str, list[str]]] = []
        for entry in persist_path.iterdir():
            if not entry.is_dir() or not _is_chroma_segment_dir(entry.name):
                continue
            problems: list[str] = []
            for fname in _CHROMA_REQUIRED_SEGMENT_FILES:
                fpath = entry / fname
                if not fpath.exists():
                    problems.append(f"{fname} 누락")
                elif fpath.stat().st_size == 0:
                    problems.append(f"{fname} 0바이트")
            if problems:
                corrupted.append((entry.name, problems))

        if not corrupted:
            _integrity_checked.add(key)
            return

        details = "\n".join(
            f"  - {seg_id}: {', '.join(probs)}" for seg_id, probs in corrupted
        )
        diagnosis = (
            f"ChromaDB HNSW 세그먼트 손상 감지 ({len(corrupted)}개, "
            f"persist_dir={persist_path}):\n{details}\n"
            f"원인: 인덱싱 도중 강제 종료로 세그먼트 write가 끊겨 필수 파일이 "
            f"0바이트로 남음. 그대로 진행하면 모든 벡터 저장이 "
            f"'unsupported opcode \\0' 에러로 실패함."
        )

        if not _auto_quarantine_enabled():
            logger.error(diagnosis)
            raise ChromaIntegrityError(
                diagnosis
                + "\n복구: CHROMA_AUTO_QUARANTINE=1로 자동 격리하거나 "
                "persist_dir을 백업 후 삭제하고 케이스를 재인덱싱."
            )

        quarantine = self._quarantine_persist_dir(persist_path)
        logger.warning(
            f"{diagnosis}\n→ 자동 격리: {persist_path} → {quarantine}\n"
            f"주의: 이 케이스의 BM25 인덱스(data/bm25_index/)와 SQLite 케이스 상태도 "
            f"stale 상태이므로 케이스를 삭제 후 재생성하거나 status를 created로 되돌려 "
            f"재인덱싱 필요."
        )
        _integrity_checked.add(key)

    def _quarantine_persist_dir(self, persist_path: Path) -> Path:
        """persist_dir 전체를 형제 디렉토리 `vectordb.quarantine_<ts>`로 이동 후 빈 디렉토리 재생성"""
        ts = int(time.time())
        quarantine = persist_path.parent / f"{persist_path.name}.quarantine_{ts}"
        # 동일 timestamp 충돌 방지
        suffix = 0
        while quarantine.exists():
            suffix += 1
            quarantine = persist_path.parent / f"{persist_path.name}.quarantine_{ts}_{suffix}"
        shutil.move(str(persist_path), str(quarantine))
        persist_path.mkdir(parents=True, exist_ok=True)
        return quarantine

    def _get_collection(self) -> chromadb.Collection:
        """ChromaDB 컬렉션 가져오기/생성"""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _case_filter(self) -> dict[str, Any] | None:
        """case_id 모드일 때 자동 부착되는 ChromaDB where 필터"""
        if self.case_id:
            return {"case_id": self.case_id}
        return None

    def _merge_filter(self, user_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        """사용자 필터에 case_id 필터를 안전하게 병합"""
        case_filter = self._case_filter()
        if not case_filter:
            return user_filter or None
        if not user_filter:
            return case_filter
        # 사용자 필터가 case_id를 명시했더라도 케이스 격리 우선
        merged = {**user_filter, **case_filter}
        return merged

    def add_chunks(self, chunks: list[Chunk], rebuild_bm25: bool = True) -> int:
        """청크를 벡터 저장소에 추가 (collection.upsert로 멱등 보장)

        1. 빈 content 필터링
        2. EmbeddingService로 임베딩 생성
        3. ChromaDB upsert (중복 체크 자체를 DB에 위임)
        4. BM25 corpus/메타데이터 캐시 갱신
        5. BM25 인덱스 재구축 (rebuild_bm25=True일 때만)

        Args:
            chunks: 저장할 청크 리스트
            rebuild_bm25: BM25 인덱스 재구축 여부 (대량 배치 시 False로 두고 마지막에 1회 호출)

        Returns:
            업서트된 청크 수
        """
        if not chunks:
            return 0

        # 빈 content 청크 제거 (임베딩 400 에러 방지)
        chunks = [c for c in chunks if c.content and c.content.strip()]
        if not chunks:
            return 0

        collection = self._get_collection()

        # 차원 검증 — 인스턴스당 1회만 (단일 공유 컬렉션에서 매 배치 count()/peek() 누적 회피)
        if not self._dim_validated:
            if collection.count() > 0:
                self._validate_embedding_dimension(collection)
            self._dim_validated = True

        # 임베딩 생성
        texts = [c.content for c in chunks]
        embeddings = self._embedding_service.embed_texts(texts)

        # 영구 실패 청크는 retry 큐 파일에 저장하고 본 배치에서 제외 → 데이터 유실 방지
        failed_indices = set(self._embedding_service._last_failed_indices)
        if failed_indices:
            failed_chunks = [chunks[i] for i in sorted(failed_indices)]
            self._save_failed_chunks(failed_chunks)
            logger.warning(
                f"임베딩 실패 청크 {len(failed_chunks)}개 → retry 큐에 저장 "
                f"(case={self.case_id})"
            )
            kept_indices = [i for i in range(len(chunks)) if i not in failed_indices]
            chunks = [chunks[i] for i in kept_indices]
            texts = [texts[i] for i in kept_indices]
            embeddings = [embeddings[i] for i in kept_indices]

        if not chunks:
            return 0

        # 메타데이터 직렬화 + source_type / case_id 자동 부착
        ids = [c.chunk_id for c in chunks]
        metadatas = []
        for c in chunks:
            meta = serialize_metadata_for_chroma(c.metadata)
            meta["source_type"] = c.source_type
            if self.case_id:
                meta["case_id"] = self.case_id
            metadatas.append(meta)

        # upsert로 중복 체크/덮어쓰기 위임 (인메모리 _known_ids 불필요)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # BM25 캐시 갱신 + 인덱스 재구축은 rebuild_bm25=True일 때만.
        # 대량 인덱싱(rebuild_bm25=False)에서는 GPU 컨슈머 critical path에서 형태소 분석 회피 —
        # 마지막에 rebuild_bm25(corpus=..., ids=...) 호출로 일괄 토큰화.
        if rebuild_bm25:
            existing_id_set = set(self._bm25_ids)
            for cid, text, meta in zip(ids, texts, metadatas):
                if cid in existing_id_set:
                    continue
                self._bm25_corpus.append(text)
                self._bm25_ids.append(cid)
                self._bm25_tokenized.append(self._tokenize(text))
                self._bm25_metadata[cid] = meta
            self._rebuild_bm25_from_cache()

        logger.info(
            f"벡터 저장 완료: {len(chunks)}개 청크 upsert, 컬렉션={self.collection_name}"
            + (f", case={self.case_id}" if self.case_id else "")
        )
        return len(chunks)

    def add_chunks_with_embeddings(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        rebuild_bm25: bool = False,
    ) -> int:
        """이미 임베딩이 계산된 청크를 벡터 저장소에 저장 (임베딩 호출 생략)

        파이프라인의 임베딩 단계와 저장 단계를 분리할 때 사용.
        호출자가 EmbeddingService.embed_texts()를 직접 호출하고 실패 청크를
        retry 큐로 처리한 뒤, 성공한 (chunk, embedding) 쌍만 이 메서드에 전달한다.

        Args:
            chunks: 저장할 청크 리스트 (임베딩 성공한 것만)
            embeddings: chunks와 동일 순서의 임베딩 벡터 리스트
            rebuild_bm25: BM25 인덱스 즉시 재구축 여부 (대량 처리 시 False)

        Returns:
            업서트된 청크 수
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks({len(chunks)})와 embeddings({len(embeddings)}) 길이가 다릅니다"
            )

        collection = self._get_collection()

        # 차원 검증 — 인스턴스당 1회만
        if not self._dim_validated:
            if collection.count() > 0:
                self._validate_embedding_dimension(collection)
            self._dim_validated = True

        # 메타데이터 직렬화 + source_type / case_id 자동 부착
        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = []
        for c in chunks:
            meta = serialize_metadata_for_chroma(c.metadata)
            meta["source_type"] = c.source_type
            if self.case_id:
                meta["case_id"] = self.case_id
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if rebuild_bm25:
            existing_id_set = set(self._bm25_ids)
            for cid, text, meta in zip(ids, texts, metadatas):
                if cid in existing_id_set:
                    continue
                self._bm25_corpus.append(text)
                self._bm25_ids.append(cid)
                self._bm25_tokenized.append(self._tokenize(text))
                self._bm25_metadata[cid] = meta
            self._rebuild_bm25_from_cache()

        logger.info(
            f"벡터 저장 완료(pre-embedded): {len(chunks)}개 청크 upsert, "
            f"컬렉션={self.collection_name}"
            + (f", case={self.case_id}" if self.case_id else "")
        )
        return len(chunks)

    def rebuild_bm25(
        self,
        corpus: list[str] | None = None,
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """BM25 인덱스를 수동으로 재구축 (대량 인덱싱 후 1회 호출용)

        Args:
            corpus: 직접 전달할 문서 텍스트 리스트
            ids: corpus에 대응하는 chunk_id 리스트
            metadatas: corpus에 대응하는 메타데이터 리스트 (선택, BM25 검색 캐시용)
        """
        if corpus is not None and ids is not None:
            self._build_bm25_from_corpus(corpus, ids, metadatas)
        else:
            self._rebuild_bm25_from_chroma()
        self._save_bm25_index()

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
        search_method: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """하이브리드 검색 (벡터 유사도 + BM25 키워드)

        Args:
            query: 검색 쿼리
            n_results: 반환할 최종 결과 수
            filters: ChromaDB 메타데이터 필터 (case_id는 자동 병합됨)
            search_method: "hybrid" | "vector" | "bm25"
        """
        if search_method == "vector":
            return self._search_vector(query, n_results, filters)
        elif search_method == "bm25":
            return self._search_bm25(query, n_results)
        else:
            return self._search_hybrid(query, n_results, filters)

    def _search_vector(
        self,
        query: str,
        n_results: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """ChromaDB 벡터 유사도 검색 (case_id 필터 자동 적용)"""
        collection = self._get_collection()
        query_embedding = self._embedding_service.embed_text(query)

        merged = self._merge_filter(filters)
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if merged:
            query_params["where"] = self._build_chroma_filter(merged)

        results = collection.query(**query_params)
        return self._format_chroma_results(results, "vector")

    def _search_bm25(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """BM25 키워드 검색 — 인메모리 corpus/메타 캐시에서 직접 반환"""
        if not self._bm25_index:
            self._rebuild_bm25_from_chroma()
        if not self._bm25_index:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # 상위 후보 추출 (score > 0)
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        results: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            cid = self._bm25_ids[idx]
            content = self._bm25_corpus[idx]
            meta = self._bm25_metadata.get(cid, {})
            results.append({
                "content": content,
                "metadata": deserialize_metadata_from_chroma(meta),
                "score": float(score),
                "search_method": "bm25",
                "chunk_id": cid,
            })

        return results

    def _search_hybrid(
        self,
        query: str,
        n_results: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """RRF 기반 하이브리드 검색 (벡터 + BM25 결합)"""
        top_k = settings.search_top_k
        rrf_k = settings.rrf_k

        # 두 검색을 각각 실행
        vector_results = self._search_vector(query, top_k, filters)
        bm25_results = self._search_bm25(query, top_k)

        # case_id 모드: BM25는 already 케이스별 corpus, 벡터는 필터 적용됨
        # 단, BM25에 case_id 필터링이 없으므로 명시적으로 제거
        if self.case_id:
            bm25_results = [
                r for r in bm25_results
                if r["metadata"].get("case_id", self.case_id) == self.case_id
            ]

        # RRF (Reciprocal Rank Fusion) 스코어 계산
        rrf_scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(vector_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)
            result_map[cid] = result

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)
            if cid not in result_map:
                result_map[cid] = result

        # RRF 최소 스코어 임계값 필터링
        min_score = settings.rrf_min_score
        if min_score > 0:
            rrf_scores = {
                cid: score for cid, score in rrf_scores.items() if score >= min_score
            }

        # RRF 스코어 기준 정렬
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results: list[dict[str, Any]] = []
        for cid in sorted_ids[:n_results]:
            entry = result_map[cid]
            entry["score"] = rrf_scores[cid]
            entry["search_method"] = "hybrid"
            results.append(entry)

        filtered_count = len(result_map) - len(rrf_scores) if min_score > 0 else 0
        logger.info(
            f"하이브리드 검색: 벡터={len(vector_results)}건, "
            f"BM25={len(bm25_results)}건 → RRF={len(results)}건"
            + (f" (임계값 {min_score} 이하 {filtered_count}건 필터링)" if filtered_count else "")
        )
        return results

    def get_collection_dimension(self) -> int | None:
        """기존 컬렉션의 벡터 차원을 조회"""
        collection = self._get_collection()
        if collection.count() == 0:
            return None

        sample = collection.peek(limit=1)
        if sample is None:
            return None

        embeddings = sample.get("embeddings")
        if embeddings is None:
            return None

        try:
            if len(embeddings) == 0 or len(embeddings[0]) == 0:
                return None
        except (TypeError, IndexError):
            return None

        return len(embeddings[0])

    def _save_failed_chunks(self, failed_chunks: list[Chunk]) -> None:
        """임베딩이 영구 실패한 청크를 retry 큐 파일에 append

        파일 경로: data/failed_embeddings/{case_id or collection}.jsonl
        한 줄당 청크 1개 (chunk_id, content, metadata, source_type, ts).
        나중에 별도 재처리 스크립트로 다시 임베딩 가능.
        """
        try:
            base = Path(settings.bm25_index_dir).parent / "failed_embeddings"
            base.mkdir(parents=True, exist_ok=True)
            key = self.case_id or self.collection_name
            path = base / f"{key}.jsonl"
            ts = time.time()
            with path.open("a", encoding="utf-8") as f:
                for c in failed_chunks:
                    record = {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "metadata": c.metadata,
                        "source_type": c.source_type,
                        "ts": ts,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # JSONL 저장이 실패하면 청크가 영구 증발 — 호출측이 인지할 수 있도록
            # error 로그 + raise 로 데이터 손실을 가시화.
            logger.error(
                f"실패 청크 JSONL 저장 실패 (DATA LOSS, {len(failed_chunks)}개): {e}",
                exc_info=True,
            )
            raise

    def _validate_embedding_dimension(self, collection: chromadb.Collection) -> None:
        """현재 임베딩 모델과 기존 컬렉션의 벡터 차원이 일치하는지 검증

        Raises:
            EmbeddingDimensionError: 차원 불일치 시
        """
        sample = collection.peek(limit=1)
        if sample is None:
            return

        embeddings = sample.get("embeddings")
        if embeddings is None:
            return

        try:
            if len(embeddings) == 0 or len(embeddings[0]) == 0:
                return
        except (TypeError, IndexError):
            return

        existing_dim = len(embeddings[0])
        self._embedding_service.validate_dimension(existing_dim)

    def delete_case_data(self) -> int:
        """현재 case_id에 해당하는 청크만 삭제 (단일 컬렉션 모드)

        Returns:
            삭제된 청크 수
        """
        if not self.case_id:
            raise ValueError(
                "delete_case_data()는 case_id 모드에서만 사용 가능합니다. "
                "단일-컬렉션 모드는 delete_collection()을 사용하세요."
            )

        collection = self._get_collection()
        before = collection.count()
        collection.delete(where={"case_id": self.case_id})
        after = collection.count()

        # BM25 캐시도 비우기 (이 케이스의 in-memory 상태)
        self._bm25_index = None
        self._bm25_corpus = []
        self._bm25_ids = []
        self._bm25_tokenized = []
        self._bm25_metadata = {}

        # BM25 pickle 파일 삭제
        bm25_path = Path(settings.bm25_index_dir) / f"{self._bm25_key}.pkl"
        if bm25_path.exists():
            try:
                bm25_path.unlink()
            except OSError as e:
                logger.warning(f"BM25 인덱스 파일 삭제 실패: {bm25_path} — {e}")

        deleted = before - after
        logger.info(f"케이스 데이터 삭제: case_id={self.case_id}, {deleted}개 청크 제거")
        return deleted

    def delete_collection(self) -> None:
        """현재 컬렉션 전체 삭제 (단일-컬렉션 모드 전용)

        ⚠️ 멀티-케이스 모드(case_id 지정)에서는 delete_case_data()를 사용해야 함.
        """
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_ids = []
            self._bm25_tokenized = []
            self._bm25_metadata = {}
            logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
        except (ValueError, chromadb.errors.NotFoundError):
            logger.warning(f"컬렉션이 존재하지 않습니다: {self.collection_name}")

    def get_stats(self) -> dict[str, Any]:
        """저장소 통계 조회 — 메타데이터 전체 로드 대신 source_type별 카운트

        case_id 모드: 해당 케이스만 카운트.
        단일-컬렉션 모드: 컬렉션 전체 카운트.
        """
        collection = self._get_collection()

        # 전체 카운트 (case_id 모드는 where로 한정)
        case_filter = self._case_filter()
        if case_filter:
            total_result = collection.get(where=case_filter, include=[])
            total = len(total_result["ids"]) if total_result.get("ids") else 0
        else:
            total = collection.count()

        # source_type별 카운트 — 알려진 타입만 분할 카운트
        source_counts: dict[str, int] = {}
        if total > 0:
            for st in _KNOWN_SOURCE_TYPES:
                where: dict[str, Any] = {"source_type": st}
                if case_filter:
                    where = {"$and": [where, case_filter]}
                try:
                    res = collection.get(where=where, include=[])
                    n = len(res["ids"]) if res.get("ids") else 0
                except Exception as e:
                    logger.warning(f"source_type={st} 카운트 실패: {e}")
                    n = 0
                if n > 0:
                    source_counts[st] = n

        return {
            "collection_name": self.collection_name,
            "case_id": self.case_id,
            "total_chunks": total,
            "source_type_counts": source_counts,
        }

    # === BM25 인덱스 관리 ===

    def _rebuild_bm25_from_cache(self) -> None:
        """인메모리 _bm25_tokenized 캐시로부터 BM25 인덱스 재구축 (전체 재토큰화 없음)"""
        if not self._bm25_tokenized:
            self._bm25_index = None
            return
        self._bm25_index = BM25Okapi(self._bm25_tokenized)
        logger.info(f"BM25 인덱스 구축 완료: {len(self._bm25_tokenized)}개 문서 (캐시)")

    def _build_bm25_from_corpus(
        self,
        corpus: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """직접 전달받은 corpus로 BM25 인덱스 구축

        대량 인덱싱 파이프라인에서 이미 보유한 텍스트를 그대로 활용하여
        ChromaDB 전체 문서 로드를 건너뜀. 신규 텍스트만 증분 토큰화.
        """
        if not corpus:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_ids = []
            self._bm25_tokenized = []
            self._bm25_metadata = {}
            return

        # 기존 데이터가 있으면 신규만 추가 (증분)
        existing_id_set = set(self._bm25_ids)
        for i, (text, cid) in enumerate(zip(corpus, ids)):
            if cid in existing_id_set:
                continue
            self._bm25_corpus.append(text)
            self._bm25_ids.append(cid)
            self._bm25_tokenized.append(self._tokenize(text))
            if metadatas and i < len(metadatas):
                self._bm25_metadata[cid] = metadatas[i]

        self._bm25_index = BM25Okapi(self._bm25_tokenized)
        logger.info(
            f"BM25 인덱스 구축 완료: {len(self._bm25_corpus)}개 문서 "
            f"(신규 {len(self._bm25_corpus) - len(existing_id_set)}개)"
        )

    def _rebuild_bm25_from_chroma(self) -> None:
        """ChromaDB에서 케이스별 corpus를 로드하여 BM25 재구축

        case_id 모드: 해당 케이스만 로드. 단일-컬렉션 모드: 전체 로드.
        디스크 pickle 캐시가 있으면 우선 시도.
        """
        # 디스크 캐시 우선 시도
        if self._load_bm25_index():
            return

        collection = self._get_collection()
        case_filter = self._case_filter()

        if case_filter:
            res = collection.get(
                where=case_filter,
                include=["documents", "metadatas"],
            )
        else:
            res = collection.get(include=["documents", "metadatas"])

        ids = res.get("ids") or []
        if not ids:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_ids = []
            self._bm25_tokenized = []
            self._bm25_metadata = {}
            return

        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []

        self._bm25_ids = list(ids)
        self._bm25_corpus = list(documents)
        self._bm25_tokenized = [self._tokenize(doc) for doc in self._bm25_corpus]
        self._bm25_metadata = {
            cid: (meta or {}) for cid, meta in zip(ids, metadatas)
        }
        self._bm25_index = BM25Okapi(self._bm25_tokenized)

        logger.info(f"BM25 인덱스 구축 완료: {len(ids)}개 문서 (ChromaDB 로드)")

    def _save_bm25_index(self) -> None:
        """BM25 인덱스를 디스크에 저장 (케이스별 별도 파일)"""
        bm25_dir = Path(settings.bm25_index_dir)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        index_path = bm25_dir / f"{self._bm25_key}.pkl"
        data = {
            "bm25_index": self._bm25_index,
            "corpus": self._bm25_corpus,
            "ids": self._bm25_ids,
            "tokenized": self._bm25_tokenized,
            "metadata": self._bm25_metadata,
        }
        with open(index_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 인덱스 저장: {index_path}")

    def _load_bm25_index(self) -> bool:
        """디스크에서 BM25 인덱스 로드"""
        index_path = Path(settings.bm25_index_dir) / f"{self._bm25_key}.pkl"
        if not index_path.exists():
            return False

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._bm25_index = data["bm25_index"]
            self._bm25_corpus = data["corpus"]
            self._bm25_ids = data["ids"]
            self._bm25_tokenized = data.get("tokenized") or [
                self._tokenize(doc) for doc in self._bm25_corpus
            ]
            self._bm25_metadata = data.get("metadata") or {}
            logger.info(f"BM25 인덱스 로드: {index_path}")
            return True
        except Exception as e:
            logger.warning(f"BM25 인덱스 로드 실패: {e}")
            return False

    # === 유틸리티 ===

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """텍스트를 토큰으로 분할 (BM25용)"""
        return _kiwi_tokenize(text)

    @staticmethod
    def _build_chroma_filter(filters: dict[str, Any]) -> dict[str, Any]:
        """사용자 필터를 ChromaDB where 절로 변환

        이미 $and/$or 등 ChromaDB 연산자가 들어있으면 그대로 반환.
        """
        if not filters:
            return {}

        # 이미 ChromaDB 연산자 형태면 그대로 사용
        if any(k.startswith("$") for k in filters.keys()):
            return filters

        conditions: list[dict] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _format_chroma_results(
        results: dict[str, Any], search_method: str
    ) -> list[dict[str, Any]]:
        """ChromaDB 검색 결과를 표준 형식으로 변환"""
        formatted: list[dict[str, Any]] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for i, doc_id in enumerate(ids):
            # ChromaDB 코사인 거리 → 유사도 스코어 변환
            score = 1.0 - distances[i] if distances[i] else 0.0
            metadata = deserialize_metadata_from_chroma(metadatas[i]) if metadatas[i] else {}

            formatted.append({
                "content": documents[i],
                "metadata": metadata,
                "score": score,
                "search_method": search_method,
                "chunk_id": doc_id,
            })

        return formatted
