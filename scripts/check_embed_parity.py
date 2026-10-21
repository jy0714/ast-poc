"""임베딩 일치 확인 — Ollama 양자화 모델 vs transformers 원본 (지시서 §5.3)

샘플 파트 30개(헤더 포함, 지시문 없음)와 질의 10개(지시문 포함)에 대해 Ollama 임베딩과
transformers `Qwen/Qwen3-Embedding-4B`(last-token 풀링, L2 정규화) 임베딩의 코사인
유사도를 계산한다. 평균 0.99 이상이면 통과다. RTX 3090에서 실행한다(BF16).

실데이터 본문은 출력하지 않는다. 수치(코사인, 토큰 수, 언어)만 출력한다.

사용:
    python -m scripts.check_embed_parity <case_id>
        [--units PATH] [--n-units 30] [--queries FILE] [--hf-model Qwen/Qwen3-Embedding-4B]
        [--ollama-model qwen3-embedding:4b-q8_0] [--dtype bf16] [--out results.json]

--units를 주지 않으면 data/index/<case_id>/units.jsonl, 없으면 data/unit_stats/<case_id>/units.jsonl.
--queries는 한 줄에 질의 하나(또는 queries.jsonl의 text). 없으면 내장 샘플 질의 10개.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_service import EmbeddingService, format_embed_query  # noqa: E402
from src.units.schema import UnitPart  # noqa: E402
from src.utils.config import settings  # noqa: E402

PASS_THRESHOLD = 0.99
SEED = 20261015

# 실데이터와 무관한 형식 점검용 질의 (평가 질의 아님)
SAMPLE_QUERIES = [
    "3월 정산 건에서 결재를 건너뛴 정황",
    "협력업체에 선급금을 지급한 근거 메일",
    "공장 방문 초청장 발급 요청",
    "invoice approved without purchase order",
    "who requested the price change for the supplier contract",
    "견적서 금액이 계약서와 다른 경우",
    "访问工厂的邀请函",
    "objednávka byla schválena bez podpisu",
    "FDB21-000963 관련 회신",
    "회의록에서 리베이트를 언급한 부분",
]


def _load_units(path: Path, n: int) -> list[UnitPart]:
    """units.jsonl에서 파트 n개를 무작위 표본 추출 (seed 고정, 유형·길이 섞임)"""
    parts = [
        UnitPart.from_dict(json.loads(line)) for line in path.open(encoding="utf-8") if line.strip()
    ]
    rng = random.Random(SEED)
    return rng.sample(parts, min(n, len(parts)))


def _load_queries(path: Path | None) -> list[str]:
    """질의 파일(텍스트 또는 queries.jsonl) 또는 내장 샘플"""
    if path is None:
        return SAMPLE_QUERIES
    out: list[str] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line)["text"] if line.startswith("{") else line)
    return out[:10]


def _ollama_embed(model: str, inputs: list[str]) -> list[list[float]]:
    """Ollama 임베딩 (서비스와 같은 EmbeddingService 경로, 문서는 지시문 없음)"""
    svc = EmbeddingService(model=model)
    vecs = svc.embed_texts(inputs)
    if svc._last_failed_indices:
        raise RuntimeError(f"Ollama 임베딩 실패 {len(svc._last_failed_indices)}건")
    return vecs


def _ollama_unload(model: str) -> None:
    """VRAM 확보를 위해 Ollama 모델 내림"""
    import httpx

    with contextlib.suppress(Exception):
        httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": model, "input": [], "keep_alive": 0},
            timeout=30,
        )


def _hf_embed(
    model_name: str, dtype: str, inputs: list[str], max_length: int
) -> tuple[list[list[float]], int]:
    """transformers 원본 임베딩 — last-token 풀링(왼쪽 패딩) + L2 정규화 (모델 카드 방식)

    Returns:
        (벡터 목록, 잘린 입력 수)
    """
    import torch
    from torch.nn.functional import normalize
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = (
        AutoModel.from_pretrained(model_name, dtype=torch_dtype, attn_implementation="sdpa")
        .to(device)
        .eval()
    )
    vecs: list[list[float]] = []
    n_trunc = 0
    for text in inputs:  # 1개씩 — 패딩 영향 배제
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if len(ids) > max_length:
            n_trunc += 1
        batch = tok(
            [text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**batch).last_hidden_state[:, -1]
        vecs.append(normalize(out.float(), p=2, dim=1)[0].cpu().tolist())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return vecs, n_trunc


def _cos(a: list[float], b: list[float]) -> float:
    """코사인 (두 벡터 모두 정규화되어 있다고 보지 않고 계산)"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _summary(values: list[float]) -> dict[str, float]:
    """평균·최소·중앙값"""
    return {
        "mean": round(statistics.fmean(values), 5),
        "min": round(min(values), 5),
        "median": round(statistics.median(values), 5),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점. 통과하면 0, 미달이면 1"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case_id")
    ap.add_argument("--units", type=Path)
    ap.add_argument("--n-units", type=int, default=30)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--hf-model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--ollama-model", default=settings.ollama_embed_model)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    units_path = args.units
    if units_path is None:
        for cand in (
            PROJECT_ROOT / "data" / "index" / args.case_id / "units.jsonl",
            PROJECT_ROOT / "data" / "unit_stats" / args.case_id / "units.jsonl",
        ):
            if cand.exists():
                units_path = cand
                break
    if units_path is None or not units_path.exists():
        print(f"units.jsonl을 찾지 못했습니다: {args.case_id}", file=sys.stderr)
        return 2

    parts = _load_units(units_path, args.n_units)
    queries = _load_queries(args.queries)
    doc_inputs = [p.full_text(with_header=True) for p in parts]
    query_inputs = [format_embed_query(q) for q in queries]

    t0 = time.time()
    oll_docs = _ollama_embed(args.ollama_model, doc_inputs)
    # 질의는 이미 지시문을 붙였으므로 embed_texts(지시문 없음 경로)로 같은 문자열을 보낸다
    oll_queries = _ollama_embed(args.ollama_model, query_inputs)
    t_ollama = time.time() - t0
    _ollama_unload(args.ollama_model)

    t0 = time.time()
    hf_all, n_trunc = _hf_embed(
        args.hf_model, args.dtype, doc_inputs + query_inputs, settings.embed_num_ctx
    )
    t_hf = time.time() - t0
    hf_docs, hf_queries = hf_all[: len(doc_inputs)], hf_all[len(doc_inputs) :]

    if len(oll_docs[0]) != len(hf_docs[0]):
        print(f"차원 불일치: Ollama {len(oll_docs[0])} vs HF {len(hf_docs[0])}", file=sys.stderr)
        return 1

    doc_cos = [_cos(a, b) for a, b in zip(oll_docs, hf_docs, strict=True)]
    q_cos = [_cos(a, b) for a, b in zip(oll_queries, hf_queries, strict=True)]
    # 순위 일치: 질의별 문서 순위 상위 5가 얼마나 겹치는지 (참고 지표)
    overlap = []
    for qo, qh in zip(oll_queries, hf_queries, strict=True):
        ro = sorted(range(len(parts)), key=lambda i: -_cos(qo, oll_docs[i]))[:5]
        rh = sorted(range(len(parts)), key=lambda i: -_cos(qh, hf_docs[i]))[:5]
        overlap.append(len(set(ro) & set(rh)) / 5)

    all_cos = doc_cos + q_cos
    result: dict[str, Any] = {
        "ollama_model": args.ollama_model,
        "hf_model": args.hf_model,
        "dtype": args.dtype,
        "dim": len(oll_docs[0]),
        "n_units": len(parts),
        "n_queries": len(queries),
        "units_source": str(
            units_path.relative_to(PROJECT_ROOT)
            if units_path.is_relative_to(PROJECT_ROOT)
            else units_path
        ),
        "unit_tokens": _summary([float(p.n_tokens) for p in parts]),
        "unit_langs": sorted({p.meta.get("lang", "und") for p in parts}),
        "cos_docs": _summary(doc_cos),
        "cos_queries": _summary(q_cos),
        "cos_all_mean": round(statistics.fmean(all_cos), 5),
        "top5_overlap_mean": round(statistics.fmean(overlap), 3),
        "hf_truncated": n_trunc,
        "seconds": {"ollama": round(t_ollama, 1), "hf": round(t_hf, 1)},
        "threshold": PASS_THRESHOLD,
        "passed": statistics.fmean(all_cos) >= PASS_THRESHOLD,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
