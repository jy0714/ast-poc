"""kiwipiepy 동작 재현 (v4.2 S6-2, 논문 3.4절) — 실데이터 없이 합성 문자열만 쓴다

v3.x BM25 토크나이저(`vector_store._kiwi_tokenize`, 체크포인트 3에서 삭제)가 겪은 세 현상을
설치된 kiwipiepy로 다시 확인하고, v4 `src.text.bm25_tokenizer.tokenize` 결과와 나란히 보인다.

1. 문장 끝 마침표가 붙은 라틴 단어 분리: 'payment.' → 'pa', 'yment.'
2. 띄어쓰기 없는 중국어 구절이 한 토큰으로 처리됨
3. 품사 필터 적용 시 'INV-2024-0312'에서 'inv'만 남음

사용:
    python -m scripts.repro_kiwi [--out results.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# v3.x `_KIWI_POS_TAGS` (git e257abd^:src/vectorstore/vector_store.py) — 재현용 사본
V3_POS_TAGS = frozenset({"NNG", "NNP", "NNB", "VV", "VA", "MAG", "SL", "SH", "SN"})

CASES: list[dict[str, str]] = [
    {
        "id": "1",
        "name": "마침표가 붙은 라틴 단어 분리 ('payment.' → 'pa', 'yment.')",
        "text": "Please approve the invoice before payment.",
    },
    {"id": "2", "name": "띄어쓰기 없는 중국어 구절이 한 토큰", "text": "请确认付款金额并回复"},
    {
        "id": "3",
        "name": "품사 필터 적용 시 'INV-2024-0312' → 'inv'만 남음",
        "text": "INV-2024-0312 송장 확인 바랍니다",
    },
]


def v3_tokenize(kiwi: Any, text: str) -> list[str]:
    """v3.x `_kiwi_tokenize` 재현: 품사 필터 + 최소 길이(SL 3, 그 외 2) + SL 소문자"""
    out: list[str] = []
    for tok in kiwi.tokenize(text):
        if tok.tag not in V3_POS_TAGS:
            continue
        form = tok.form
        if len(form) < (3 if tok.tag == "SL" else 2):
            continue
        out.append(form.lower() if tok.tag == "SL" else form)
    return out


def reproduced(case_id: str, raw: list[tuple[str, str]], v3: list[str], text: str) -> bool:
    """현상이 재현되었는지 판정"""
    forms = [f for f, _t in raw]
    if case_id == "1":
        return "payment" not in forms and any(f in ("pa", "yment", "yment.") for f in forms)
    if case_id == "2":
        return any(len(f) == len(text) for f in forms)
    if case_id == "3":
        code_toks = [t for t in v3 if "inv" in t or "2024" in t or "0312" in t]
        return code_toks == ["inv"]
    return False


def run() -> tuple[str, list[dict[str, Any]]]:
    """세 현상 재현 → (kiwipiepy 버전, 결과 행)"""
    import kiwipiepy
    from kiwipiepy import Kiwi

    from src.text.bm25_tokenizer import tokenize

    kiwi = Kiwi()
    rows: list[dict[str, Any]] = []
    for case in CASES:
        raw = [(t.form, t.tag) for t in kiwi.tokenize(case["text"])]
        v3 = v3_tokenize(kiwi, case["text"])
        rows.append(
            {
                **case,
                "raw": raw,
                "v3": v3,
                "v4": tokenize(case["text"]),
                "reproduced": reproduced(case["id"], raw, v3, case["text"]),
            }
        )
    return kiwipiepy.__version__, rows


def render(version: str, rows: list[dict[str, Any]]) -> str:
    """마크다운 표"""
    lines = [
        f"# kiwipiepy 동작 재현 — kiwipiepy {version}",
        "",
        "| # | 현상 | 재현 | Kiwi 원 토큰 (형태/태그) | v3 필터 결과 | v4 bm25_tokenizer |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        raw = ", ".join(f"{f}/{t}" for f, t in r["raw"])
        lines.append(
            f"| {r['id']} | {r['name']} | {'예' if r['reproduced'] else '아니오'} | `{raw}` | "
            f"`{', '.join(r['v3'])}` | `{', '.join(r['v4'])}` |"
        )
    lines += [
        "",
        "입력은 합성 문자열이다: " + " / ".join(f"({r['id']}) `{r['text']}`" for r in rows),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI 진입점"""
    ap = argparse.ArgumentParser(description="kiwipiepy 동작 재현 (합성 문자열)")
    ap.add_argument("--out", default="", help="결과 마크다운 저장 경로")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    md = render(*run())
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
