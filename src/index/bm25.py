"""BM25 색인 — rank_bm25.BM25Okapi 래퍼 (토크나이저 버전 기록)

색인과 질의에 같은 `src.text.bm25_tokenizer.tokenize`를 쓴다. 저장 파일에 토크나이저 버전을
함께 기록하고, 로드 시 현재 버전과 다르면 재색인을 요구한다(지시서 §5.2).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.text.bm25_tokenizer import (
    PARALLEL_MIN_TEXTS,
    TOKENIZER_VERSION,
    tokenize,
    tokenize_many,
    tokenize_many_parallel,
)


class TokenizerVersionError(Exception):
    """색인의 토크나이저 버전이 현재 코드와 다름 — 재색인 필요"""


class BM25Index:
    """파트 행 순서와 같은 순서의 BM25 색인

    Attributes:
        tokenizer_version: 색인을 만든 토크나이저 버전
        n_docs: 문서(파트) 수
    """

    def __init__(self, corpus_tokens: list[list[str]], tokenizer_version: str = TOKENIZER_VERSION) -> None:
        """토큰화된 코퍼스로 색인 생성

        Args:
            corpus_tokens: 파트별 토큰 리스트 (행 순서 = units.jsonl 순서)
            tokenizer_version: 토크나이저 버전
        """
        self.tokenizer_version = tokenizer_version
        self.n_docs = len(corpus_tokens)
        # rank_bm25는 빈 코퍼스·빈 문서에서 0 나눗셈이 나므로 빈 문서에 자리표시 토큰을 둔다
        safe = [toks if toks else ["\x00"] for toks in corpus_tokens]
        self._bm25: Any = BM25Okapi(safe) if safe else None

    @classmethod
    def from_texts(cls, texts: list[str]) -> BM25Index:
        """원문 텍스트를 토큰화해 색인 생성 (많으면 프로세스 병렬 토큰화, 결과는 같다)"""
        if len(texts) >= PARALLEL_MIN_TEXTS:
            return cls(tokenize_many_parallel(texts))
        return cls(tokenize_many(texts))

    def scores(self, query: str) -> np.ndarray:
        """모든 파트의 BM25 점수 (질의 토큰이 없으면 0 벡터)"""
        if self._bm25 is None:
            return np.zeros(0, dtype=np.float64)
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.n_docs, dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def save(self, path: Path) -> None:
        """pickle 저장 (호출자가 임시 경로 → 교체로 원자성 보장)"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"tokenizer_version": self.tokenizer_version, "n_docs": self.n_docs, "bm25": self._bm25},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """pickle 로드 + 토크나이저 버전 검사

        Raises:
            TokenizerVersionError: 저장된 버전이 현재 TOKENIZER_VERSION과 다를 때
        """
        with path.open("rb") as f:
            data = pickle.load(f)  # noqa: S301 — 로컬에서 직접 만든 색인 파일만 읽는다
        version = data.get("tokenizer_version")
        if version != TOKENIZER_VERSION:
            raise TokenizerVersionError(
                f"BM25 색인 토크나이저 버전 {version} ≠ 현재 {TOKENIZER_VERSION}. 재색인이 필요합니다."
            )
        obj = cls.__new__(cls)
        obj.tokenizer_version = version
        obj.n_docs = data["n_docs"]
        obj._bm25 = data["bm25"]
        return obj
