"""단위 통계 → results/T1_corpus.md (지시서 §6.8 T1)

케이스 색인(`data/index/{case}/`)의 파트와 manifest `build_stats` 만으로 만든다. 재파싱하지 않으므로
표는 실험에 실제로 쓰인 색인과 정확히 같다. 원본 파일 단위 처리 시간·OCR 시간 등은
`scripts/unit_stats.py` 의 T1 초안에 있다.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from ev4.io import ExperimentConfig
from ev4.stats import percentile
from src.index import case_index

UNIT_TYPES = ("email", "chat", "document", "attachment")


def _dist(values: list[int]) -> dict[str, float]:
    if not values:
        return {"n": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "n": len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def corpus_stats(case_id: str) -> dict[str, Any]:
    """색인에서 T1 수치를 계산한다"""
    idx = case_index.load(case_id)
    bs = idx.manifest.get("build_stats", {}) or {}
    counts = bs.get("counts", {}) or {}

    units_by_type: Counter = Counter()
    parts_by_type: Counter = Counter()
    split_by_type: Counter = Counter()
    unit_tokens: dict[str, list[int]] = defaultdict(list)
    part_tokens: dict[str, list[int]] = defaultdict(list)
    lang_by_type: dict[str, Counter] = defaultdict(Counter)
    mixed_by_type: Counter = Counter()
    for rows in idx.unit_rows.values():
        parts = [idx.parts[i] for i in rows]
        t = parts[0].unit_type
        meta = parts[0].meta
        units_by_type[t] += 1
        parts_by_type[t] += len(parts)
        split_by_type[t] += int(len(parts) > 1)
        unit_tokens[t].append(sum(p.n_tokens for p in parts))
        for p in parts:
            part_tokens[t].append(p.n_tokens)
        lang_by_type[t][str(meta.get("lang") or "und")] += 1
        mixed_by_type[t] += int(bool(meta.get("lang_mixed")))

    lang_totals: Counter = Counter()
    for c in lang_by_type.values():
        lang_totals.update(c)
    return {
        "case_id": case_id,
        "created_at": idx.manifest.get("created_at", ""),
        "unit_rules_version": idx.manifest.get("unit_rules_version", ""),
        "tokenizer_version": idx.manifest.get("tokenizer_version", ""),
        "n_units": idx.n_units,
        "n_parts": idx.n_parts,
        "profiles": idx.profiles(),
        "units_by_type": dict(units_by_type),
        "parts_by_type": dict(parts_by_type),
        "split_units": dict(split_by_type),
        "duplicates_removed": dict(bs.get("duplicates_removed", {}) or {}),
        "unit_tokens": {t: _dist(v) for t, v in unit_tokens.items()},
        "part_tokens": {t: _dist(v) for t, v in part_tokens.items()},
        "lang_columns": [k for k, _ in lang_totals.most_common()],
        "lang_by_type": {t: dict(c) for t, c in lang_by_type.items()},
        "mixed_by_type": dict(mixed_by_type),
        "clean_chars": dict(bs.get("clean_chars", {}) or {}),
        "orphan_attachments": int(counts.get("orphan_attachments", 0)),
        "attachments_total": int(counts.get("attachments_total", 0)),
        "forwarded_emails": int(counts.get("forwarded_emails", 0)),
        "embedding": bs.get("embedding", {}),
    }


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def render_t1(s: dict[str, Any]) -> str:
    """T1 마크다운"""
    types = [t for t in UNIT_TYPES if s["units_by_type"].get(t) or s["duplicates_removed"].get(t)]
    lines = [
        f"# T1 단위 통계 — 케이스 {s['case_id']}",
        "",
        f"- 색인 생성 {s['created_at']} · 규칙 {s['unit_rules_version']} · BM25 토크나이저 "
        f"{s['tokenizer_version']} · 프로필 {', '.join(s['profiles']) or '-'}",
        f"- 단위 {s['n_units']:,} · 파트 {s['n_parts']:,}",
        "",
        "## 유형별 단위·파트",
        "",
        "| 유형 | 단위 | 파트 | 분할된 단위 | 중복 제거 |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in types:
        lines.append(
            f"| {t} | {s['units_by_type'].get(t, 0):,} | {s['parts_by_type'].get(t, 0):,} | "
            f"{s['split_units'].get(t, 0):,} | {s['duplicates_removed'].get(t, 0):,} |"
        )
    lines += [
        f"| **합계** | **{sum(s['units_by_type'].values()):,}** | **{sum(s['parts_by_type'].values()):,}** | "
        f"{sum(s['split_units'].values()):,} | {sum(s['duplicates_removed'].values()):,} |",
        "",
        "## 토큰 길이 (본문, 헤더 제외)",
        "",
        "| 구분 | n | 중앙값 | p95 | p99 | 최대 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, key in (("단위", "unit_tokens"), ("파트", "part_tokens")):
        for t in types:
            d = s[key].get(t)
            if d:
                lines.append(f"| {kind}:{t} | {d['n']:,} | {_fmt(d['median'])} | {_fmt(d['p95'])} | "
                             f"{_fmt(d['p99'])} | {_fmt(d['max'])} |")
    cols = s["lang_columns"]
    lines += [
        "",
        "## 언어 분포 (단위 기준)",
        "",
        "| 유형 | " + " | ".join(cols) + " | 혼용 |",
        "|---|" + "---:|" * (len(cols) + 1),
    ]
    for t in types:
        row = s["lang_by_type"].get(t, {})
        lines.append(f"| {t} | " + " | ".join(f"{row.get(c, 0):,}" for c in cols)
                     + f" | {s['mixed_by_type'].get(t, 0):,} |")
    lines += [
        "",
        "## 이메일 정제 (언어별 문자 수)",
        "",
        "| 언어 | 정제 전 | 정제 후 | 줄어든 비율 |",
        "|---|---:|---:|---:|",
    ]
    for lang, (b, a) in sorted(s["clean_chars"].items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {lang} | {b:,} | {a:,} | {(1 - a / b) if b else 0:.1%} |")
    lines += [
        "",
        f"- 부모 없는 첨부 {s['orphan_attachments']:,}건 (첨부 총 {s['attachments_total']:,}건) · "
        f"전달 메일 {s['forwarded_emails']:,}건",
    ]
    return "\n".join(lines) + "\n"


def build_corpus(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """T1 생성 → results/T1_corpus.md, results/corpus.json"""
    s = corpus_stats(cfg.case_id)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "corpus.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.results_dir / "T1_corpus.md").write_text(render_t1(s), encoding="utf-8")
    log(f"[corpus] 단위 {s['n_units']:,} · 파트 {s['n_parts']:,} → {cfg.results_dir / 'T1_corpus.md'}")
    return s
