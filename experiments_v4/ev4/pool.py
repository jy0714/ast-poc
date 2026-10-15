"""판정 시트 — pool_runs 상위 pool_depth 합집합 → pool_A.csv / pool_B.csv (지시서 §6.3)

- 블라인드: 시트에는 어느 실행이 찾았는지 넣지 않는다. 출처는 runs/pool_provenance.jsonl.
- 질의 안에서 단위 순서를 seed 고정으로 섞는다.
- A 는 전체 질의, B 는 질의 유형별 층화로 b_ratio 만큼 고른 질의의 전체 단위를 판정한다.
- CSV 는 Excel 호환 UTF-8 BOM.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from ev4.io import ExperimentConfig, Query, read_run, write_jsonl
from ev4.translate import NO_TRANSLATE_LANGS, Translator
from src.index import case_index

POOL_COLUMNS = ["qid", "query_text", "query_type", "unit_id", "unit_type", "date", "lang", "header",
                "text", "translation_ko", "grade", "annotator", "note"]
TRUNC_NOTE = "…(이하 생략, 전체: unit_id로 조회)"


def select_b_queries(queries: list[Query], ratio: float, seed: int) -> list[str]:
    """유형별 층화 표집 — 유형마다 ceil(n × ratio)개 (최소 1개, 유형에 질의가 있으면)"""
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_type[q.type].append(q.qid)
    rng = random.Random(seed)
    chosen: list[str] = []
    for t in sorted(by_type):
        qids = sorted(by_type[t])
        k = min(len(qids), max(1, math.ceil(len(qids) * ratio))) if ratio > 0 else 0
        chosen += rng.sample(qids, k)
    return sorted(chosen)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def build_pool(
    cfg: ExperimentConfig,
    queries: list[Query],
    split: str = "test",
    translate: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """판정 시트 생성

    Returns:
        {rows_A, rows_B, queries_B, translated}
    """
    idx = case_index.load(cfg.case_id)
    runs = {name: read_run(cfg.run_path(name, split)) for name in cfg.pool_runs}
    translator = Translator(cfg.cache_dir / "translations.jsonl") if translate else None
    b_qids = set(select_b_queries(queries, cfg.b_ratio, cfg.seed))

    rows_a: list[dict] = []
    provenance: list[dict] = []
    for q in queries:
        found: dict[str, dict[str, int]] = defaultdict(dict)
        for name, run in runs.items():
            for rank, uid in enumerate(run.get(q.qid, [])[: cfg.pool_depth], 1):
                found[uid][name] = rank
        units = sorted(found)
        random.Random(f"{cfg.seed}:{q.qid}").shuffle(units)
        for uid in units:
            parts = idx.unit_parts(uid)
            if not parts:
                raise KeyError(f"색인에 없는 단위 {uid} (실행 파일과 색인이 다름 — 재실행 필요)")
            meta = parts[0].meta
            text = idx.unit_text(uid)
            if len(text) > cfg.text_max_chars:
                text = text[: cfg.text_max_chars] + TRUNC_NOTE
            lang = str(meta.get("lang", ""))
            ko = translator.translate(text) if translator and lang not in NO_TRANSLATE_LANGS else ""
            rows_a.append({
                "qid": q.qid, "query_text": q.text, "query_type": q.type,
                "unit_id": uid, "unit_type": parts[0].unit_type, "date": meta.get("date", ""),
                "lang": lang, "header": parts[0].header, "text": text, "translation_ko": ko,
                "grade": "", "annotator": "", "note": "",
            })
            provenance.append({"qid": q.qid, "unit_id": uid, "found_by": found[uid]})
        log(f"  {q.qid}: 판정 대상 {len(units)}건" + (" (B)" if q.qid in b_qids else ""))
    if translator:
        translator.save()
    rows_b = [r for r in rows_a if r["qid"] in b_qids]
    _write_csv(cfg.judgments_dir / "pool_A.csv", rows_a)
    _write_csv(cfg.judgments_dir / "pool_B.csv", rows_b)
    write_jsonl(cfg.runs_dir / "pool_provenance.jsonl", provenance)
    n_tr = translator.n_new if translator else 0
    log(f"[pool] A {len(rows_a)}행 · B {len(rows_b)}행 (질의 {len(b_qids)}건) · 새 번역 {n_tr}건 "
        f"→ {cfg.judgments_dir}")
    return {"rows_A": len(rows_a), "rows_B": len(rows_b), "queries_B": len(b_qids), "translated": n_tr}
