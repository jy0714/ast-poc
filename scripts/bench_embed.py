"""임베딩 처리량 벤치마크 — 합성 문장으로 배치 크기·동시 요청 수 조합별 파트/분을 잰다

조사 데이터를 쓰지 않는다. 일반 업무 어휘로 만든 한·영 혼합 문장을 쓰고, 출력은 숫자만이다.
인덱싱과 같은 경로(EmbeddingService.embed_texts)를 쓰므로 결과가 실제 인덱싱 속도의 상한이다.

사용법 (인덱싱을 멈추고, Ollama를 원하는 OLLAMA_NUM_PARALLEL 로 다시 띄운 뒤):
    python -m scripts.bench_embed --tokens 700 --n 256 --grid 16x2 16x4 16x8 32x8
"""

from __future__ import annotations

import argparse
import random
import time

from src.embeddings.embedding_service import EmbeddingService
from src.units.tokens import get_token_counter
from src.utils.config import settings

_KO = ["검토", "결과", "보고", "일정", "회의", "자료", "첨부", "요청", "확인", "계약", "금액", "지급", "승인", "담당자", "부서", "거래처", "견적", "발주", "납품", "세금계산서", "송금", "정산", "비용", "예산", "집행", "내역", "변경", "사항", "공유", "드립니다", "부탁드립니다", "감사합니다", "관련하여", "아래와", "같이", "다음", "주", "이번", "달", "진행", "상황", "문의", "회신", "전달", "검토", "바랍니다", "수정본", "최종", "확정", "계획", "일정", "조정", "협의", "완료", "예정", "처리", "누락", "오류"]
_EN = ["please", "review", "the", "attached", "invoice", "payment", "approval", "contract", "amount", "vendor", "schedule", "meeting", "report", "budget", "update", "status", "follow", "up", "regards", "thanks", "confirm", "revised", "final", "draft", "order", "delivery", "account", "transfer", "reconciliation", "pending", "issue", "request", "team"]


def synthetic_texts(n: int, target_tokens: int, seed: int = 0) -> list[str]:
    """한·영 업무 어휘로 대략 target_tokens 토큰인 문장 n개"""
    rng = random.Random(seed)
    counter = get_token_counter()
    out: list[str] = []
    for _ in range(n):
        words: list[str] = []
        text = ""
        while True:
            words.extend(rng.choice(_KO if rng.random() < 0.7 else _EN) for _ in range(40))
            text = " ".join(words)
            if counter.count(text) >= target_tokens:
                break
        out.append(text)
    return out


def main() -> None:
    """조합별 처리량 출력"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=700, help="문장당 목표 토큰 수")
    ap.add_argument("--n", type=int, default=256, help="조합당 문장 수")
    ap.add_argument("--grid", nargs="+", default=["16x2", "16x4", "16x8"], help="배치x동시요청")
    ap.add_argument("--num-batch", type=int, nargs="+", default=[0], help="Ollama num_batch 값들 (0=기본)")
    args = ap.parse_args()

    texts = synthetic_texts(args.n, args.tokens)
    n_tok = sum(get_token_counter().count(t) for t in texts)
    svc = EmbeddingService(model=settings.ollama_embed_model)
    svc.embed_texts(texts[:4])  # 모델 적재·워밍업

    print(f"model={svc.model} num_ctx={settings.embed_num_ctx} texts={len(texts)} avg_tokens={n_tok / len(texts):.0f}")
    print(f"{'nbatch':>6} {'batch':>5} {'conc':>4} {'sec':>7} {'parts/min':>9} {'tok/s':>7} {'failed':>6}")
    for nb in args.num_batch:
        settings.embed_num_batch = nb
        svc.embed_texts(texts[:2])  # num_batch가 바뀌면 Ollama가 모델을 다시 올린다 — 측정에서 뺀다
        for combo in args.grid:
            b, c = (int(x) for x in combo.lower().split("x"))
            settings.embed_batch_size, settings.embed_max_concurrent = b, c
            svc._adaptive_batch_size = None
            t0 = time.monotonic()
            # 인덱싱처럼 super-batch(b*c) 단위로 나눠 보낸다
            failed = 0
            for i in range(0, len(texts), b * c):
                svc.embed_texts(texts[i : i + b * c])
                failed += len(svc._last_failed_indices)
            sec = time.monotonic() - t0
            print(f"{nb:>6} {b:>5} {c:>4} {sec:>7.1f} {len(texts) * 60 / sec:>9.0f} "
                  f"{n_tok / sec:>7.0f} {failed:>6}")


if __name__ == "__main__":
    main()
