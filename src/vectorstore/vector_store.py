"""벡터 저장소 — ChromaDB + BM25 하이브리드 검색

ChromaDB: 벡터 유사도 검색 + 메타데이터 필터
BM25: 키워드 매칭 검색
RRF (Reciprocal Rank Fusion): 두 검색 결과를 통합 순위로 결합
"""

from __future__ import annotations

import json
import pickle
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

    사용법:
        store = VectorStoreService(collection_name="case_001")
        store.add_chunks(chunks)

        results = store.search("감사 보고서 비용 분석")
        # [{"content": ..., "metadata": ..., "score": ..., "search_method": "hybrid"}, ...]
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
        ephemeral: bool = False,
    ) -> None:
        """VectorStoreService 초기화

        Args:
            collection_name: ChromaDB 컬렉션명 (기본: settings.chroma_collection_name)
            persist_dir: ChromaDB 저장 경로 (기본: settings.chroma_persist_dir)
            ephemeral: True이면 인메모리 모드 (테스트용)
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.ephemeral = ephemeral
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._embedding_service = EmbeddingService()

        # BM25 인덱스 상태
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []  # 원본 텍스트
        self._bm25_ids: list[str] = []  # chunk_id 매핑

    def _get_client(self) -> chromadb.ClientAPI:
        """ChromaDB 클라이언트 초기화 (지연 생성)"""
        if self._client is None:
            if self.ephemeral:
                self._client = chromadb.EphemeralClient()
            else:
                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        """ChromaDB 컬렉션 가져오기/생성"""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], rebuild_bm25: bool = True) -> int:
        """청크를 벡터 저장소에 추가

        1. EmbeddingService로 임베딩 생성
        2. ChromaDB에 벡터 + 메타데이터 저장
        3. BM25 인덱스 갱신 (rebuild_bm25=True일 때만)

        Args:
            chunks: 저장할 청크 리스트
            rebuild_bm25: BM25 인덱스 재구축 여부 (대량 배치 시 False로 두고 마지막에 1회 호출)

        Returns:
            추가된 청크 수
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        # 기존 컬렉션에 데이터가 있으면 벡터 차원 호환성 검증
        if collection.count() > 0:
            self._validate_embedding_dimension(collection)

        # 기존 ID 확인 → 중복 제거
        chunk_ids = [c.chunk_id for c in chunks]
        existing = set()
        try:
            result = collection.get(ids=chunk_ids)
            if result and result["ids"]:
                existing = set(result["ids"])
        except Exception:
            pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            logger.info("모든 청크가 이미 저장되어 있습니다.")
            return 0

        # 임베딩 생성
        texts = [c.content for c in new_chunks]
        embeddings = self._embedding_service.embed_texts(texts)

        # ChromaDB에 저장
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [serialize_metadata_for_chroma(c.metadata) for c in new_chunks]
        # source_type을 메타데이터에 포함
        for i, chunk in enumerate(new_chunks):
            metadatas[i]["source_type"] = chunk.source_type

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # BM25 인덱스 갱신
        if rebuild_bm25:
            self._rebuild_bm25_index()

        logger.info(
            f"벡터 저장 완료: {len(new_chunks)}개 청크 추가 "
            f"(중복 {len(existing)}개 스킵), 컬렉션={self.collection_name}"
        )
        return len(new_chunks)

    def rebuild_bm25(self) -> None:
        """BM25 인덱스를 수동으로 재구축 (대량 인덱싱 후 1회 호출용)"""
        self._rebuild_bm25_index()
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
            filters: ChromaDB 메타데이터 필터
            search_method: "hybrid" | "vector" | "bm25"

        Returns:
            [{"content": str, "metadata": dict, "score": float,
              "search_method": str, "chunk_id": str}, ...]
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
        """ChromaDB 벡터 유사도 검색"""
        collection = self._get_collection()
        query_embedding = self._embedding_service.embed_text(query)

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = self._build_chroma_filter(filters)

        results = collection.query(**query_params)

        return self._format_chroma_results(results, "vector")

    def _search_bm25(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """BM25 키워드 검색"""
        if not self._bm25_index:
            self._rebuild_bm25_index()
        if not self._bm25_index:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # 상위 n_results 추출
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        # ChromaDB에서 해당 청크 정보 조회
        collection = self._get_collection()
        result_ids = [self._bm25_ids[idx] for idx, score in scored_indices if score > 0]

        if not result_ids:
            return []

        chroma_results = collection.get(
            ids=result_ids[:n_results],
            include=["documents", "metadatas"],
        )

        # 스코어 매핑
        score_map = {
            self._bm25_ids[idx]: score for idx, score in scored_indices if score > 0
        }

        results: list[dict[str, Any]] = []
        for i, doc_id in enumerate(chroma_results["ids"]):
            metadata = chroma_results["metadatas"][i] if chroma_results["metadatas"] else {}
            results.append({
                "content": chroma_results["documents"][i] if chroma_results["documents"] else "",
                "metadata": deserialize_metadata_from_chroma(metadata),
                "score": score_map.get(doc_id, 0.0),
                "search_method": "bm25",
                "chunk_id": doc_id,
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
        """기존 컬렉션의 벡터 차원을 조회

        Returns:
            벡터 차원 수, 비어 있으면 None
        """
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

        # ChromaDB peek()은 numpy array를 반환할 수 있음
        try:
            if len(embeddings) == 0 or len(embeddings[0]) == 0:
                return
        except (TypeError, IndexError):
            return

        existing_dim = len(embeddings[0])
        self._embedding_service.validate_dimension(existing_dim)

    def delete_collection(self) -> None:
        """현재 컬렉션 삭제"""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_ids = []
            logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
        except ValueError:
            logger.warning(f"컬렉션이 존재하지 않습니다: {self.collection_name}")

    def get_stats(self) -> dict[str, Any]:
        """저장소 통계 조회

        Returns:
            {"collection_name": str, "total_chunks": int,
             "source_type_counts": dict, ...}
        """
        collection = self._get_collection()
        total = collection.count()

        # source_type별 카운트
        source_counts: dict[str, int] = {}
        if total > 0:
            all_meta = collection.get(include=["metadatas"])
            if all_meta["metadatas"]:
                for meta in all_meta["metadatas"]:
                    st = meta.get("source_type", "unknown")
                    source_counts[st] = source_counts.get(st, 0) + 1

        return {
            "collection_name": self.collection_name,
            "total_chunks": total,
            "source_type_counts": source_counts,
        }

    # === BM25 인덱스 관리 ===

    def _rebuild_bm25_index(self) -> None:
        """ChromaDB 전체 문서로 BM25 인덱스 재구축"""
        collection = self._get_collection()
        total = collection.count()
        if total == 0:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_ids = []
            return

        all_docs = collection.get(include=["documents"])
        self._bm25_ids = all_docs["ids"]
        self._bm25_corpus = all_docs["documents"] or []

        tokenized_corpus = [self._tokenize(doc) for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 인덱스 구축 완료: {total}개 문서")

    def _save_bm25_index(self) -> None:
        """BM25 인덱스를 디스크에 저장"""
        bm25_dir = Path(settings.bm25_index_dir)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        index_path = bm25_dir / f"{self.collection_name}.pkl"
        data = {
            "bm25_index": self._bm25_index,
            "corpus": self._bm25_corpus,
            "ids": self._bm25_ids,
        }
        with open(index_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 인덱스 저장: {index_path}")

    def _load_bm25_index(self) -> bool:
        """디스크에서 BM25 인덱스 로드

        Returns:
            로드 성공 여부
        """
        index_path = Path(settings.bm25_index_dir) / f"{self.collection_name}.pkl"
        if not index_path.exists():
            return False

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._bm25_index = data["bm25_index"]
            self._bm25_corpus = data["corpus"]
            self._bm25_ids = data["ids"]
            logger.info(f"BM25 인덱스 로드: {index_path}")
            return True
        except Exception as e:
            logger.warning(f"BM25 인덱스 로드 실패: {e}")
            return False

    # === 유틸리티 ===

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """텍스트를 토큰으로 분할 (BM25용)

        kiwipiepy 형태소 분석기를 사용하여 명사/동사/형용사 등
        의미 있는 형태소를 추출. kiwipiepy가 없으면 정규식 fallback.
        """
        return _kiwi_tokenize(text)

    @staticmethod
    def _build_chroma_filter(filters: dict[str, Any]) -> dict[str, Any]:
        """사용자 필터를 ChromaDB where 절로 변환

        Args:
            filters: {"source_type": "email", "case_id": "C001", ...}

        Returns:
            ChromaDB where 딕셔너리
        """
        if not filters:
            return {}

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
