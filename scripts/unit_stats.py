"""실데이터 단위 통계 (임베딩 없이) — v4 T1 초안

케이스의 데이터 소스에서 파일을 모아 증거 단위만 생성하고(src/units), 단위 통계와
정제 전후 샘플을 로컬(data/, gitignore)에 저장한다.

사용:
    python -m scripts.unit_stats <case_id> [--workers 12] [--limit N] [--no-ocr] [--samples 20]

출력 (data/unit_stats/<case_id>/):
    T1_corpus.md      — 단위 통계 표 (지시서 §6.8 T1)
    stats.json        — 원 수치 (집계만, 원문·파일명 없음)
    units.jsonl       — 생성된 파트 (UnitPart 한 줄씩)
    clean_samples/    — 정제 전후 샘플 (언어별, 전달 메일 포함). 실데이터이므로 커밋 금지
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.units.builder import BuildStats, UnitCollector  # noqa: E402
from src.units.schema import UNIT_RULES_VERSION, EvidenceUnit, UnitPart  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "experiments_v4"))
from ev4.corpus import CorpusStopError, render_t1, stats_from_parts  # noqa: E402

# ─────────────────────────── 워커 ───────────────────────────

_builder: Any = None
_ocr_clock = {"seconds": 0.0, "pages": 0, "calls": 0}


def _init_worker(case_id: str, no_ocr: bool) -> None:
    """워커 프로세스 초기화: 로그 소음 줄이고 빌더·OCR 계측 설치"""
    logging.disable(logging.WARNING)
    global _builder
    from src.parsers.document_parser import DocumentParser
    from src.units.builder import UnitBuilder

    parser = DocumentParser(detect_language=False)
    if no_ocr:

        class _NoOcr:
            name = "disabled"

            def is_available(self) -> bool:
                return False

        parser._get_ocr_engine = lambda: _NoOcr()  # type: ignore[method-assign]
    else:
        original = parser._ocr_sparse_pages

        def timed(doc: Any, page_nums: list[int], *a: Any, **kw: Any) -> dict[int, str]:
            t0 = time.perf_counter()
            try:
                return original(doc, page_nums, *a, **kw)
            finally:
                _ocr_clock["seconds"] += time.perf_counter() - t0
                _ocr_clock["pages"] += len(page_nums)
                _ocr_clock["calls"] += 1

        parser._ocr_sparse_pages = timed  # type: ignore[method-assign]
    _builder = UnitBuilder(case_id, doc_parser=parser)


def _process(path: str, rel: str) -> dict[str, Any]:
    """파일 1개 → 단위 (dict 직렬화) + 통계 증분"""
    _builder.stats = BuildStats()
    before = dict(_ocr_clock)
    t0 = time.perf_counter()
    error = ""
    units: list[EvidenceUnit] = []
    try:
        units = _builder.build_file(path, rel)
    except Exception as e:  # 파일 하나 실패가 전체를 멈추지 않게
        error = f"{type(e).__name__}: {e}"[:200]
    return {
        "path": path,
        "units": [
            {
                "unit_id": u.unit_id,
                "unit_type": u.unit_type,
                "header": u.header,
                "text": u.text,
                "meta": u.meta,
                "parts": [p.to_dict() for p in u.parts],
            }
            for u in units
        ],
        "stats": _builder.stats.to_dict(),
        "seconds": time.perf_counter() - t0,
        "ocr_seconds": _ocr_clock["seconds"] - before["seconds"],
        "ocr_pages": _ocr_clock["pages"] - before["pages"],
        "error": error,
    }


def _unit_from_dict(d: dict[str, Any]) -> EvidenceUnit:
    return EvidenceUnit(
        d["unit_id"],
        d["unit_type"],
        d["header"],
        d["text"],
        d["meta"],
        [UnitPart.from_dict(p) for p in d["parts"]],
    )


def _stats_from_dict(d: dict[str, Any]) -> BuildStats:
    return BuildStats.from_dict(d)


# ─────────────────────────── 집계 ───────────────────────────


def _collect_source_files(case_id: str) -> list[tuple[str, str]]:
    """케이스 데이터 소스 → [(절대 경로, 케이스 기준 상대 경로)] (인덱싱과 같은 규칙)"""
    from src.cases.case_store import CaseStore
    from src.indexing.pipeline import case_source_files

    files = case_source_files(CaseStore().get(case_id))
    return sorted(((str(f), rel) for f, rel in files), key=lambda x: x[1])


def _write_samples(
    units: list[EvidenceUnit], paths_by_unit: dict[str, str], out_dir: Path, n: int, seed: int
) -> list[str]:
    """정제 전후 샘플: 언어별 층화 + 전달 메일 포함. 원문 재파싱으로 '전'을 만든다"""
    from src.parsers.document_parser import DocumentParser
    from src.units.builder import _email_body_from_parsed

    emails = [u for u in units if u.unit_type == "email" and u.unit_id in paths_by_unit]
    rng = random.Random(seed)
    by_lang: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for u in emails:
        by_lang[u.meta["lang"]].append(u)
    forwarded = [u for u in emails if u.meta.get("forwarded")]
    changed = [u for u in emails if not u.meta.get("forwarded")]

    picked: list[EvidenceUnit] = []
    quota = {"forwarded": max(3, n // 6)}
    picked += rng.sample(forwarded, min(quota["forwarded"], len(forwarded)))
    # 한국어·영어 외 언어는 있는 만큼 (최대 4건씩)
    for lang in sorted(by_lang):
        if lang in ("ko", "en", "und"):
            continue
        pool = [u for u in by_lang[lang] if u not in picked]
        picked += rng.sample(pool, min(4, len(pool)))
    rest = n - len(picked)
    for lang, share in (("ko", 0.6), ("en", 0.4)):
        pool = [u for u in by_lang.get(lang, []) if u not in picked and u in changed]
        picked += rng.sample(pool, min(int(round(rest * share)), len(pool)))
    picked = picked[:n]

    sample_dir = out_dir / "clean_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("*.md"):
        old.unlink()
    parser = DocumentParser()
    written: list[str] = []
    for i, u in enumerate(picked, 1):
        parsed = parser.parse(paths_by_unit[u.unit_id])
        before = _email_body_from_parsed(parsed).strip()
        tag = "fwd" if u.meta.get("forwarded") else u.meta["lang"]
        name = f"{i:02d}_{tag}_{u.unit_id.replace(':', '_')}.md"
        (sample_dir / name).write_text(
            f"# {u.unit_id}\n\n- 언어: {u.meta['lang']} (혼용 {u.meta['lang_mixed']}, "
            f"두 번째 {u.meta['lang_second'] or '-'})\n- 전달 메일: {u.meta.get('forwarded')}\n"
            f"- 원본: {u.meta['source_file']}\n- 문자 수: {len(before)} → {len(u.text)}\n\n"
            f"## 헤더\n\n```\n{u.header}\n```\n\n## 정제 후 (단위 본문)\n\n```\n{u.text}\n```\n\n"
            f"## 정제 전 (원문 본문)\n\n```\n{before}\n```\n",
            encoding="utf-8",
        )
        written.append(name)
    return written


def _render_extra(s: dict[str, Any]) -> str:
    """이 스크립트에만 있는 처리 수치 (T1 본표는 ev4.corpus.render_t1)"""
    lines = [
        "",
        "## 부록: 단위 생성 처리 (unit_stats 초안 전용)",
        "",
        f"- 입력 파일 {s['files']['total']:,}개 (실패 {s['files']['failed']:,}) · "
        f"처리 시간 {s['elapsed_sec']:,.0f}초 (워커 {s['workers']}) · 파트 상한 {s['max_part_tokens']}",
        f"- 빈 페이지(단위 없음): PDF {s['counts'].get('empty_pages:pdf', 0):,}쪽 · "
        f"PPTX {s['counts'].get('empty_pages:pptx', 0):,}장 · 첨부된 메일 "
        f"{s['counts'].get('nested_email_attachments', 0):,}건",
        f"- OCR: {s['ocr']['pages']:,}쪽 시도 · 인식 {s['counts'].get('ocr_pages', 0):,}쪽 · "
        f"OCR 시간 합계 {s['ocr']['seconds']:,.0f}초 (쪽당 {s['ocr']['sec_per_page']:.1f}초, 워커 합산)",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────── 메인 ───────────────────────────


def main() -> None:
    """CLI 진입점"""
    ap = argparse.ArgumentParser(description="실데이터 단위 통계 (임베딩 없음)")
    ap.add_argument("case_id")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 파일만 (점검용)")
    ap.add_argument("--no-ocr", action="store_true", help="스캔 페이지 OCR 생략")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    sys.stdout.reconfigure(encoding="utf-8")
    from src.units.tokens import get_token_counter
    from src.utils.config import settings

    out_dir = Path(args.out_dir or PROJECT_ROOT / "data" / "unit_stats" / args.case_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_source_files(args.case_id)
    if args.limit:
        files = files[: args.limit]
    print(f"파일 {len(files):,}개 · 워커 {args.workers} · OCR {'끔' if args.no_ocr else '켬'}")

    collector = UnitCollector()
    stats = BuildStats()
    paths_by_unit: dict[str, str] = {}
    failed: list[tuple[str, str]] = []
    ocr_seconds = 0.0
    ocr_pages = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(args.case_id, args.no_ocr)
    ) as pool:
        futures = [pool.submit(_process, p, rel) for p, rel in files]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            units = [_unit_from_dict(d) for d in r["units"]]
            collector.add(units)
            for u in units:
                paths_by_unit.setdefault(u.unit_id, r["path"])
            stats.merge(_stats_from_dict(r["stats"]))
            ocr_seconds += r["ocr_seconds"]
            ocr_pages += r["ocr_pages"]
            if r["error"]:
                failed.append((r["path"], r["error"]))
            if i % 500 == 0 or i == len(futures):
                print(
                    f"  {i:,}/{len(futures):,} 파일 · {time.perf_counter() - t0:,.0f}초", flush=True
                )
    elapsed = time.perf_counter() - t0

    units = list(collector.units.values())
    counter = get_token_counter()
    build_stats = stats.to_dict()
    build_stats["duplicates_removed"] = dict(collector.duplicates)
    try:
        summary = stats_from_parts(
            collector.parts(), build_stats, counter, build_counter_kind=counter.kind
        )
    except CorpusStopError as e:
        print(f"중단: {e}")
        sys.exit(2)
    summary.update(
        {
            "case_id": args.case_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "unit_rules_version": UNIT_RULES_VERSION,
            "max_part_tokens": settings.unit_max_part_tokens,
            "workers": args.workers,
            "elapsed_sec": elapsed,
            "files": {"total": len(files), "failed": len(failed)},
            "failed_errors": dict(Counter(err.split(":", 1)[0] for _p, err in failed)),
            "counts": dict(stats.counts),
            "ocr": {
                "pages": ocr_pages,
                "seconds": ocr_seconds,
                "sec_per_page": ocr_seconds / ocr_pages if ocr_pages else 0.0,
            },
        }
    )

    with open(out_dir / "units.jsonl", "w", encoding="utf-8") as f:
        for p in collector.parts():
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    samples = _write_samples(units, paths_by_unit, out_dir, args.samples, args.seed)
    summary["clean_samples"] = samples
    (out_dir / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "T1_corpus.md").write_text(
        render_t1(summary) + _render_extra(summary), encoding="utf-8"
    )
    print(
        f"완료: {out_dir} (단위 {len(units):,} · 파트 {len(collector.parts()):,} · {elapsed:,.0f}초)"
    )


if __name__ == "__main__":
    main()
