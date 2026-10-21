"""langdetect ↔ lingua 대조 (v4.2 S6-1, 논문 4.1.1절) — 건수만 출력한다

케이스 색인의 이메일 단위마다 단위 언어(meta.lang, lingua)와 langdetect 판정을 비교해
"lingua는 ko인데 langdetect가 vi·et로 판정한 수"와 판정 쌍 분포를 낸다. 본문은 출력하지 않는다(P0).

langdetect 는 v4 의존성에서 빠졌다. 실행 전에 한 번만 설치한다: `pip install langdetect==1.0.9`

사용:
    python -m scripts.langdetect_compare <case_id> [--all-types] [--out S6_langdetect.md]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_WRONG = ("vi", "et")


def langdetect_lang(text: str) -> str:
    """langdetect 판정 (결정적: seed 0). 실패하면 'error'"""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "error"


def compare(units: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """{unit_id: (lingua_lang, text)} → 대조 집계 (건수만)"""
    pairs: Counter = Counter()
    for lingua_lang, text in units.values():
        pairs[(lingua_lang, langdetect_lang(text))] += 1
    ko_total = sum(n for (lg, _ld), n in pairs.items() if lg == "ko")
    wrong = {w: pairs.get(("ko", w), 0) for w in TARGET_WRONG}
    return {
        "n_units": len(units),
        "lingua_ko": ko_total,
        "ko_as": wrong,
        "ko_as_vi_or_et": sum(wrong.values()),
        "pairs": {f"{a}→{b}": n for (a, b), n in pairs.most_common()},
    }


def render(case_id: str, scope: str, r: dict[str, Any], version: str) -> str:
    """마크다운 (건수만)"""
    lines = [
        f"# langdetect ↔ lingua 대조 — 케이스 {case_id} ({scope})",
        "",
        f"- langdetect {version} (seed 0) · 대상 단위 {r['n_units']:,} · lingua 한국어 {r['lingua_ko']:,}",
        f"- **lingua ko 인데 langdetect vi·et: {r['ko_as_vi_or_et']:,}** "
        f"(vi {r['ko_as']['vi']:,} · et {r['ko_as']['et']:,})",
        "",
        "| lingua → langdetect | 단위 |",
        "|---|---:|",
    ]
    for k, n in list(r["pairs"].items())[:30]:
        lines.append(f"| {k} | {n:,} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI 진입점"""
    ap = argparse.ArgumentParser(description="langdetect ↔ lingua 대조 (건수만)")
    ap.add_argument("case_id")
    ap.add_argument("--all-types", action="store_true", help="이메일 외 단위도 포함")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        import langdetect
    except ImportError:
        print("langdetect 미설치: pip install langdetect==1.0.9 후 다시 실행하세요")
        sys.exit(2)
    from src.index import case_index

    idx = case_index.load(args.case_id)
    texts: dict[str, list[str]] = {}
    langs: dict[str, str] = {}
    for p in idx.parts:
        if not args.all_types and p.unit_type != "email":
            continue
        texts.setdefault(p.unit_id, []).append(p.text)
        langs.setdefault(p.unit_id, str(p.meta.get("lang") or "und"))
    units = {u: (langs[u], "\n".join(t)) for u, t in texts.items()}
    r = compare(units)
    md = render(
        args.case_id,
        "전체 단위" if args.all_types else "이메일 단위",
        r,
        getattr(langdetect, "__version__", "1.0.9"),
    )
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
