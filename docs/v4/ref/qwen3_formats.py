"""Qwen3-Embedding / Qwen3-Reranker 입력 형식 — AuditRAG v4 참조 구현.

형식은 Hugging Face 모델 카드(Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Reranker-4B)의 예제를 따른다.
실제 모델로 동작 확인은 Claude Code 가 A5000 환경에서 수행한다(§5.3, §5.4 완료 기준).

- 임베딩: 질의에만 지시문을 붙이고, 문서(증거 단위)에는 붙이지 않는다. 지시문은 영어로 쓴다.
- 재순위: system/user/assistant 템플릿에 <Instruct>/<Query>/<Document> 를 넣고,
  마지막 위치의 'yes'·'no' 토큰 로짓으로 P(yes) 를 점수로 쓴다.
"""

from __future__ import annotations

from typing import Sequence

# 사용자 확인 후 확정(지시서 §5.3). 논문 3.4절 [확인 필요: 지시문 원문]에 그대로 옮긴다.
EMBED_TASK = (
    "Given an internal audit investigation query, retrieve emails, chat messages, "
    "and documents in any language that contain evidence relevant to the query"
)
RERANK_INSTRUCTION = (
    "Judge whether this email, chat session, or document page contains evidence "
    "relevant to the internal audit investigation query. The document may be in any language."
)


def format_embed_query(query: str, task: str = EMBED_TASK) -> str:
    return f"Instruct: {task}\nQuery:{query}"


RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_rerank_pair(query: str, doc: str, instruction: str = RERANK_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class Qwen3Reranker:
    """transformers 기반 Qwen3-Reranker. torch·transformers 는 지연 로드한다.

    max_length 는 템플릿·질의·문서를 합친 토큰 수 상한이다. 문서가 길면 문서 쪽 끝을 자르고,
    잘린 건수를 self.n_truncated 에 누적한다(논문 4.6절 '절단 청크 비율' 근거).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", max_length: int = 4096,
                 batch_size: int = 4, device: str = "cuda", attn_implementation: str = "sdpa"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(RERANK_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(RERANK_SUFFIX, add_special_tokens=False)
        self.n_scored = 0
        self.n_truncated = 0

    def _encode(self, query: str, doc: str, instruction: str) -> list[int]:
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        head = self.tok.encode(format_rerank_pair(query, "", instruction), add_special_tokens=False)
        doc_ids = self.tok.encode(doc, add_special_tokens=False)
        room = max(0, budget - len(head))
        if len(doc_ids) > room:
            doc_ids = doc_ids[:room]
            self.n_truncated += 1
        self.n_scored += 1
        return self.prefix_ids + head + doc_ids + self.suffix_ids

    def score(self, query: str, docs: Sequence[str], instruction: str = RERANK_INSTRUCTION) -> list[float]:
        torch = self.torch
        scores: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            ids = [self._encode(query, d, instruction) for d in docs[i : i + self.batch_size]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # logits_to_keep=1: 마지막 위치의 로짓만 계산한다. 생략하면 배치×길이×어휘(약 15만)
                # 크기의 로짓을 만들어, 4,096토큰 × 배치 4 에서만 수 GB 를 더 쓴다.
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            scores.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
        return scores
