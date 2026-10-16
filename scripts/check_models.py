"""모델 동작·시간·VRAM 점검 — 설정된 임베딩(Ollama)·재순위(transformers) (지시서 §7 3단계)

실데이터를 쓰지 않는다. 내장 합성 예문(한·영·중·체코어)으로 다음을 확인한다.
1. 임베딩: 차원, L2 정규화, 질의 지시문 적용, 같은 뜻의 다국어 문장이 더 가깝게 나오는지
2. 재순위: 관련 문서 점수가 무관 문서보다 높은지, 후보 30개 × 약 2,000토큰 시간, 절단 수
3. VRAM: torch.cuda.max_memory_allocated(재순위 프로세스), nvidia-smi 사용량

사용:
    python -m scripts.check_models [--skip-embed] [--skip-rerank] [--backend qwen3|bge]
        [--candidates 30] [--doc-tokens 2000] [--out result.json]

개발 PC(RTX 2060): RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B RERANK_DTYPE=fp16
RTX 3090: RERANK_MODEL=Qwen/Qwen3-Reranker-4B RERANK_DTYPE=bf16 (세 모델 동시 적재는 ev4 latency에서)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_service import EmbeddingService  # noqa: E402
from src.rag.reranker import Reranker  # noqa: E402
from src.utils.config import settings  # noqa: E402

QUERY = "3월 정산에서 결재 없이 선급금을 지급한 정황"
RELEVANT = [
    "[이메일] 2024-03-05 14:22 | 보낸이: 홍길동 | 제목: RE: 3월 정산 건\n\n"
    "팀장님 결재 전이지만 협력업체 요청으로 선급금을 먼저 송금했습니다. 사후 결재 올리겠습니다.",
    "[이메일] 2024-03-06 09:10 | 보낸이: John Kim | 제목: March settlement\n\n"
    "We already paid the advance to the supplier before the approval was signed. Please backdate the form.",
    "[첨부] 付款说明.docx | 1/1부\n\n三月结算的预付款在审批之前已经支付给供应商，审批单将在事后补签。",
    "[이메일] 2024-03-07 11:00 | 보낸이: Petr Novák | 제목: Záloha\n\n"
    "Zálohu jsme dodavateli zaplatili ještě před schválením. Schválení doplníme dodatečně.",
]
IRRELEVANT = [
    "[이메일] 2024-03-05 12:00 | 보낸이: 총무팀 | 제목: 점심 메뉴 안내\n\n이번 주 구내식당 메뉴는 비빔밥과 된장찌개입니다.",
    "[문서] 안전교육_안내.pdf | 1/2쪽\n\nAll employees must complete the annual fire safety training by the end of the month.",
    "[이메일] 2024-03-08 16:30 | 보낸이: IT | 제목: 비밀번호 변경\n\n시스템 점검으로 이번 주 금요일 저녁 메일 접속이 제한됩니다.",
]


def _nvidia_smi_used() -> int | None:
    """현재 GPU 메모리 사용량(MiB). nvidia-smi가 없으면 None"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def _dot(a: list[float], b: list[float]) -> float:
    """내적 (정규화 벡터면 코사인)"""
    return sum(x * y for x, y in zip(a, b, strict=True))


def check_embedding() -> dict[str, Any]:
    """임베딩 차원·정규화·다국어 관련/무관 분리 확인"""
    svc = EmbeddingService()
    t0 = time.time()
    q = svc.embed_query(QUERY)
    t_query = time.time() - t0
    t0 = time.time()
    docs = svc.embed_texts(RELEVANT + IRRELEVANT)
    t_docs = time.time() - t0
    sims = [_dot(q, d) for d in docs]
    rel, irr = sims[: len(RELEVANT)], sims[len(RELEVANT) :]
    return {
        "model": svc.model,
        "dim": len(q),
        "dim_expected": settings.embed_dim,
        "query_norm": round(sum(x * x for x in q) ** 0.5, 6),
        "query_has_instruction": svc.format_query("x") != "x",
        "sim_relevant": [round(s, 4) for s in rel],
        "sim_irrelevant": [round(s, 4) for s in irr],
        "separated": min(rel) > max(irr),
        "seconds": {"query": round(t_query, 3), "docs": round(t_docs, 3)},
        "vram_used_mib": _nvidia_smi_used(),
    }


def _long_doc(n_tokens: int, i: int) -> str:
    """약 n_tokens 토큰의 합성 문서 (한·영 혼합 문장 반복)"""
    base = (
        f"[이메일] 2024-03-{(i % 28) + 1:02d} | 보낸이: 담당자{i} | 제목: 업무 보고 {i}\n\n"
    )
    sent = "이번 분기 협력업체 계약 진행 상황을 공유드립니다. The vendor review meeting is scheduled next week. "
    # 한 문장 약 30토큰으로 잡고 반복
    return base + sent * max(1, n_tokens // 30)


def check_rerank(backend: str | None, candidates: int, doc_tokens: int) -> dict[str, Any]:
    """재순위 점수 방향·시간·VRAM·절단 확인"""
    import torch

    r = Reranker(backend=backend)
    cuda = torch.cuda.is_available()
    t0 = time.time()
    scores = r.score(QUERY, RELEVANT + IRRELEVANT)  # 첫 호출 = 적재 포함
    t_first = time.time() - t0
    rel, irr = scores[: len(RELEVANT)], scores[len(RELEVANT) :]

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    docs = [_long_doc(doc_tokens, i) for i in range(candidates)]
    trunc_before = r.n_truncated
    timings = []
    for _ in range(3):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        r.score(QUERY, docs)
        if cuda:
            torch.cuda.synchronize()
        timings.append(time.time() - t0)
    from src.units.tokens import get_token_counter

    counter = get_token_counter()
    return {
        "backend": r.backend_name,
        "model": r.model_name,
        "dtype": getattr(r.backend, "dtype", "fp16"),
        "max_length": getattr(r.backend, "max_length", None),
        "batch_size": getattr(r.backend, "batch_size", None),
        "device": getattr(r.backend, "device", "cuda" if cuda else "cpu"),
        "score_relevant": [round(s, 4) for s in rel],
        "score_irrelevant": [round(s, 4) for s in irr],
        "separated": min(rel) > max(irr),
        "seconds_first_call_incl_load": round(t_first, 2),
        "candidates": candidates,
        "doc_tokens_qwen3": counter.count(docs[0]),
        "seconds_per_query": [round(t, 2) for t in timings],
        "truncated_in_timing_runs": r.n_truncated - trunc_before,
        "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "vram_used_mib": _nvidia_smi_used(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--backend", choices=["qwen3", "bge"])
    ap.add_argument("--candidates", type=int, default=settings.rerank_candidates)
    ap.add_argument("--doc-tokens", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    result: dict[str, Any] = {"fake_models": settings.ast_fake_models}
    if not args.skip_embed:
        result["embedding"] = check_embedding()
    if not args.skip_rerank:
        result["rerank"] = check_rerank(args.backend, args.candidates, args.doc_tokens)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
