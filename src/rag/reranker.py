"""Reranker — 하이브리드 검색 결과를 Cross-Encoder로 재정렬

FlagEmbedding의 FlagReranker(bge-reranker-v2-m3)를 사용하여
query-document 쌍의 관련성을 정밀 평가한 뒤 top_n개만 선별.

FlagEmbedding 미설치 환경에서도 import 에러 없이 동작하며,
실제 rerank 호출 시에만 의존성을 확인.
"""

from __future__ import annotations

import time
from typing import Any

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 모듈 레벨 캐시 — 모델 중복 로딩 방지
_reranker_instance: Reranker | None = None


class RerankerLoadError(Exception):
    """Reranker 모델 로드 실패"""


class Reranker:
    """Cross-Encoder 기반 Reranker

    사용법:
        reranker = get_reranker()
        reranked = reranker.rerank(query="질의", documents=[...], top_n=5)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Reranker 초기화 (모델은 첫 호출 시 지연 로딩)

        Args:
            model_name: reranker 모델명 (기본: settings.rerank_model)
        """
        self.model_name = model_name or settings.rerank_model
        self._model = None

    def _load_model(self) -> None:
        """FlagReranker 모델 지연 로딩"""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise RerankerLoadError(
                "FlagEmbedding 패키지가 설치되지 않았습니다. "
                "pip install FlagEmbedding 으로 설치해 주세요."
            ) from e

        logger.info(f"Reranker 모델 로딩: {self.model_name}")
        t0 = time.time()
        try:
            self._model = FlagReranker(self.model_name, use_fp16=True)
        except Exception as e:
            raise RerankerLoadError(
                f"Reranker 모델 로드 실패 ({self.model_name}): {e}"
            ) from e
        elapsed = time.time() - t0
        logger.info(f"Reranker 모델 로딩 완료: {elapsed:.1f}s")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """검색 결과를 query와의 관련성으로 재정렬

        Args:
            query: 사용자 질의
            documents: vector_store.search 반환 형식의 dict 리스트
                       (각 dict에 "content" 키 필수)
            top_n: 반환할 최대 결과 수 (기본: settings.rerank_top_n)

        Returns:
            rerank_score가 추가된 dict 리스트 (점수 내림차순, top_n개)
        """
        if not documents:
            return []

        top_n = top_n or settings.rerank_top_n

        self._load_model()

        # query-document 쌍 구성
        pairs = [[query, doc.get("content", "")] for doc in documents]

        logger.info(f"Reranker 실행: {len(documents)}개 후보 -> top {top_n}")
        t0 = time.time()

        scores = self._model.compute_score(pairs, normalize=True)

        # 단일 문서인 경우 float 반환 → 리스트로 통일
        if isinstance(scores, (int, float)):
            scores = [scores]

        elapsed = time.time() - t0
        logger.info(f"Reranker 완료: {elapsed:.2f}s")

        # 점수 부착 + 내림차순 정렬
        scored_docs = []
        for doc, score in zip(documents, scores):
            enriched = dict(doc)
            enriched["rerank_score"] = float(score)
            scored_docs.append(enriched)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)

        return scored_docs[:top_n]


def get_reranker(model_name: str | None = None) -> Reranker:
    """싱글톤 Reranker 인스턴스 반환

    Args:
        model_name: 모델명 (None이면 settings 기본값)

    Returns:
        Reranker 인스턴스 (캐싱됨)
    """
    global _reranker_instance

    target_model = model_name or settings.rerank_model

    if _reranker_instance is not None and _reranker_instance.model_name == target_model:
        return _reranker_instance

    _reranker_instance = Reranker(model_name=target_model)
    return _reranker_instance
