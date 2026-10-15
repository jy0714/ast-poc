"""Qwen3 재순위 백엔드 테스트 — 모델을 모킹해 입력 형식·절단 집계·점수 순서를 확인 (지시서 §5.10)"""

from unittest.mock import MagicMock

import pytest
import torch

from src.rag.reranker import (
    RERANK_PREFIX,
    RERANK_SUFFIX,
    Qwen3RerankBackend,
    RerankerLoadError,
    format_rerank_pair,
)
from src.utils.config import settings

_YES, _NO = 1, 2


class _CharTokenizer:
    """문자 하나 = 토큰 하나인 가짜 토크나이저 (yes/no는 고정 ID)"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, tok: str) -> int:
        return {"yes": _YES, "no": _NO}[tok]

    def pad(self, enc: dict, padding: bool, return_tensors: str) -> "_Batch":
        ids = enc["input_ids"]
        width = max(len(x) for x in ids)
        # 왼쪽 패딩 (padding_side="left")
        rows = [[0] * (width - len(x)) + x for x in ids]
        return _Batch(torch.tensor(rows))


class _Batch(dict):
    def __init__(self, input_ids: torch.Tensor) -> None:
        super().__init__(input_ids=input_ids)

    def to(self, device: str) -> "_Batch":
        return self


def _backend(max_length: int = 4096, batch_size: int = 4) -> Qwen3RerankBackend:
    """모델 적재 없이 쓸 수 있게 토크나이저·모델을 끼운 백엔드"""
    b = Qwen3RerankBackend(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        max_length=max_length,
        batch_size=batch_size,
        dtype="fp16",
        instruction="INST",
    )
    b.tok = _CharTokenizer()
    b.yes_id, b.no_id = _YES, _NO
    b.prefix_ids = b.tok.encode(RERANK_PREFIX)
    b.suffix_ids = b.tok.encode(RERANK_SUFFIX)
    b.model = MagicMock()
    return b


def _logits_for(yes_minus_no: list[float], vocab: int = 8) -> MagicMock:
    """마지막 위치 로짓: yes 로짓 = 값, no 로짓 = 0"""
    out = torch.zeros(len(yes_minus_no), 1, vocab)
    for i, v in enumerate(yes_minus_no):
        out[i, 0, _YES] = v
    return MagicMock(logits=out)


class TestFormat:
    def test_pair_format(self):
        """<Instruct>/<Query>/<Document> 형식"""
        assert format_rerank_pair("Q", "D", "I") == "<Instruct>: I\n<Query>: Q\n<Document>: D"

    def test_default_instruction_from_settings(self, monkeypatch):
        """지시문 기본값은 RERANK_INSTRUCTION"""
        monkeypatch.setattr(settings, "rerank_instruction", "custom inst")
        assert format_rerank_pair("q", "d").startswith("<Instruct>: custom inst\n")

    def test_template_matches_model_card(self):
        """system 프롬프트·빈 think 블록이 모델 카드 형식과 같다"""
        assert RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether the Document")
        assert RERANK_PREFIX.endswith("<|im_start|>user\n")
        assert RERANK_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class TestEncode:
    def test_encode_layout(self):
        """prefix + 지시문·질의·문서 + suffix 순서"""
        b = _backend()
        ids = b.encode("Q", "DOC")
        text = "".join(chr(i) for i in ids)
        assert text == RERANK_PREFIX + format_rerank_pair("Q", "DOC", "INST") + RERANK_SUFFIX
        assert b.n_scored == 1 and b.n_truncated == 0

    def test_truncation_cuts_document_tail_and_counts(self):
        """상한을 넘으면 문서 끝만 자르고 n_truncated를 올린다"""
        b = _backend()
        fixed = len(b.prefix_ids) + len(b.suffix_ids) + len(b.tok.encode(format_rerank_pair("Q", "", "INST")))
        b.max_length = fixed + 5
        ids = b.encode("Q", "ABCDEFGHIJ")
        assert len(ids) == b.max_length
        text = "".join(chr(i) for i in ids)
        assert "<Document>: ABCDE" + RERANK_SUFFIX in text
        assert b.n_truncated == 1


class TestScore:
    def test_scores_are_p_yes_in_input_order(self):
        """점수 = softmax([no, yes])[yes], 입력 순서 유지"""
        b = _backend(batch_size=8)
        b.model.side_effect = lambda **kw: _logits_for([2.0, -1.0, 0.0])
        scores = b.score("q", ["a", "b", "c"])
        assert scores[0] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6)
        assert scores[2] == pytest.approx(0.5)
        assert scores[0] > scores[2] > scores[1]

    def test_logits_to_keep_one(self):
        """마지막 위치 로짓만 계산하도록 logits_to_keep=1을 넘긴다"""
        b = _backend()
        b.model.side_effect = lambda **kw: _logits_for([0.0])
        b.score("q", ["a"])
        assert b.model.call_args.kwargs["logits_to_keep"] == 1

    def test_batching(self):
        """batch_size 단위로 모델을 호출한다"""
        b = _backend(batch_size=2)
        b.model.side_effect = lambda **kw: _logits_for([0.0] * kw["input_ids"].shape[0])
        assert len(b.score("q", ["a", "b", "c", "d", "e"])) == 5
        assert b.model.call_count == 3

    def test_empty(self):
        """빈 입력은 모델을 적재하지 않는다"""
        b = Qwen3RerankBackend(model_name="Qwen/Qwen3-Reranker-0.6B")
        assert b.score("q", []) == []
        assert b.model is None


class TestLoadGuards:
    def test_rejects_non_qwen3_model(self):
        """qwen3 백엔드에 bge 모델 이름 → 명확한 로드 오류"""
        b = Qwen3RerankBackend(model_name="BAAI/bge-reranker-v2-m3")
        with pytest.raises(RerankerLoadError, match="RERANK_MODEL"):
            b._load()

    def test_rejects_bad_dtype(self):
        """지원하지 않는 dtype"""
        b = Qwen3RerankBackend(model_name="Qwen/Qwen3-Reranker-0.6B", dtype="int4")
        with pytest.raises(RerankerLoadError, match="RERANK_DTYPE"):
            b._load()
