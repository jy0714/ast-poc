"""정답 구축: 풀링(pooling) → 감사관 블라인드 판정(CSV) → qrels 변환 → 판정자 간 일치도.

TREC 방식: 여러 시스템의 상위 depth 결과를 증거 단위로 합쳐(union) 판정 대상을 만든다.
CSV 에는 어떤 시스템이 찾았는지 표시하지 않는다(블라인드). 출처는 pool_provenance.jsonl 에 따로 남긴다.
CSV 는 Excel 에서 한글이 깨지지 않도록 UTF-8 BOM(utf-8-sig) 으로 저장한다.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .adapter import BaseAdapter
from .config import ExperimentConfig, SystemConfig
from .pipeline import run_query
from .relevance import load_qrels
from .runner import make_sparse
from .schema import Item, Query, read_jsonl, write_jsonl
from .stats import quadratic_weighted_kappa

POOL_COLUMNS = ["qid", "query_type", "query_text", "filters", "unit_id", "modality", "timestamp",
                "participants", "parent_id", "preview", "grade", "annotator", "note"]


def _chat_context(items: dict[str, Item], width: int = 2) -> dict[str, str]:
    """채팅 메시지는 앞뒤 width 개 메시지를 함께 보여준다 (대상은 ▶ 표시)."""
    sessions: dict[str, list[Item]] = defaultdict(list)
    for it in items.values():
        if it.modality == "chat":
            sessions[it.session_id or it.channel_id or "?"].append(it)
    ctx = {}
    for msgs in sessions.values():
        msgs.sort(key=lambda m: (m.timestamp is None, m.timestamp, m.unit_id))
        for i, m in enumerate(msgs):
            lines = []
            for j in range(max(0, i - width), min(len(msgs), i + width + 1)):
                o = msgs[j]
                who = o.participants[0] if o.participants else "?"
                ts = o.timestamp.strftime("%m-%d %H:%M") if o.timestamp else ""
                lines.append(f"{'▶ ' if j == i else '   '}[{ts}] {who}: {o.text.strip()[:300]}")
            ctx[m.unit_id] = "\n".join(lines)
    return ctx


def build_pool(cfg: ExperimentConfig, adapter: BaseAdapter, systems: list[SystemConfig], queries: list[Query],
               out_csv: Path, depth: int = 20, exclude_qrels: Optional[Path] = None,
               preview_chars: int = 1200, log=print) -> int:
    sparse = make_sparse(cfg, adapter, log)
    seed = int(cfg.settings.get("seed", 0))
    rng = random.Random(seed)
    judged: dict[str, set] = defaultdict(set)
    if exclude_qrels and Path(exclude_qrels).exists():
        for r in read_jsonl(exclude_qrels):
            judged[str(r["qid"])].add(str(r["unit_id"]))

    items_cache: dict[str, dict[str, Item]] = {}
    ctx_cache: dict[str, dict[str, str]] = {}
    rows, provenance = [], []
    for q in queries:
        if q.case_id not in items_cache:
            items_cache[q.case_id] = {it.unit_id: it for it in adapter.iter_items(q.case_id)}
            ctx_cache[q.case_id] = _chat_context(items_cache[q.case_id])
        items = items_cache[q.case_id]
        found: dict[str, dict[str, int]] = defaultdict(dict)
        for s in systems:
            res = run_query(adapter, sparse, s, q)
            for rank, h in enumerate(res.hits[:depth], 1):
                for u in h.source_ids:
                    found[u].setdefault(s.name, rank)
        units = [u for u in found if u not in judged[q.qid]]
        rng.shuffle(units)
        for u in units:
            it = items.get(u)
            preview = ctx_cache[q.case_id].get(u) or (it.text if it else "(iter_items 에 없음)")
            rows.append({
                "qid": q.qid, "query_type": q.qtype, "query_text": q.text,
                "filters": q.filters.key() if q.filters else "",
                "unit_id": u, "modality": it.modality if it else "",
                "timestamp": it.timestamp.isoformat(sep=" ") if it and it.timestamp else "",
                "participants": "; ".join(it.participants) if it else "",
                "parent_id": (it.parent_id or "") if it else "",
                "preview": preview[:preview_chars], "grade": "", "annotator": "", "note": "",
            })
            provenance.append({"qid": q.qid, "unit_id": u, "found_by": found[u]})
        log(f"  {q.qid}: 판정 대상 {len(units)}건")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    write_jsonl(out_csv.with_name(out_csv.stem + "_provenance.jsonl"), provenance)
    log(f"[pool] {len(rows)}행 → {out_csv}")
    return len(rows)


def import_judgments(csv_paths: list[Path], out_qrels: Path, default_annotator: str = "",
                     append: bool = True, log=print) -> int:
    """판정 CSV → qrels.jsonl. 'adjudicated_grade' 열이 채워진 행은 중재 결과로 기록한다."""
    existing = read_jsonl(out_qrels) if (append and out_qrels.exists()) else []
    keyed = {(str(r["qid"]), str(r["unit_id"]), r.get("annotator", ""), bool(r.get("adjudicated"))): r
             for r in existing}
    n_new = 0
    for p in csv_paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for ln, row in enumerate(csv.DictReader(f), 2):
                adj = (row.get("adjudicated_grade") or "").strip()
                g = adj or (row.get("grade") or "").strip()
                if not g:
                    continue
                if g not in ("0", "1", "2", "3"):
                    raise ValueError(f"{p}:{ln}: grade 는 0~3 정수여야 함 (입력값 {g!r})")
                ann = "adjudicator" if adj else ((row.get("annotator") or "").strip() or default_annotator)
                if not adj and not ann:
                    raise ValueError(f"{p}:{ln}: annotator 가 비어 있음 (--annotator 로 기본값 지정 가능)")
                rec = {"qid": row["qid"].strip(), "unit_id": row["unit_id"].strip(), "grade": int(g),
                       "annotator": ann, "note": (row.get("note") or "").strip()}
                if adj:
                    rec["adjudicated"] = True
                keyed[(rec["qid"], rec["unit_id"], ann, bool(adj))] = rec
                n_new += 1
    out_qrels.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_qrels, list(keyed.values()))
    log(f"[import] {n_new}행 반영 → {out_qrels} (총 {len(keyed)}행)")
    return n_new


def agreement(qrels_path: Path, out_csv: Optional[Path] = None, log=print) -> dict:
    """판정자 2인 이상이 판정한 단위의 일치도 + 불일치 목록(중재용 CSV)."""
    by_unit: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for r in read_jsonl(qrels_path):
        if r.get("adjudicated"):
            continue
        by_unit[(str(r["qid"]), str(r["unit_id"]))][r.get("annotator", "")] = int(r["grade"])
    annotators = sorted({a for d in by_unit.values() for a in d})
    shared = {k: d for k, d in by_unit.items() if len(d) >= 2}
    res = {"annotators": annotators, "units_total": len(by_unit), "units_shared": len(shared)}
    if len(annotators) < 2 or not shared:
        log("[agreement] 2인 이상 판정한 단위가 없음")
        return res
    a1, a2 = annotators[0], annotators[1]
    pairs = [(d[a1], d[a2]) for d in shared.values() if a1 in d and a2 in d]
    xa, xb = [p[0] for p in pairs], [p[1] for p in pairs]
    res.update({
        "pair": [a1, a2], "n": len(pairs),
        "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
        "relevant_agreement": sum((x >= 2) == (y >= 2) for x, y in pairs) / len(pairs),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(xa, xb),
    })
    log(f"[agreement] {a1} vs {a2}: n={len(pairs)}  정확일치={res['exact_agreement']:.3f}  "
        f"관련성(≥2)일치={res['relevant_agreement']:.3f}  QWK={res['quadratic_weighted_kappa']:.3f}")
    if out_csv:
        dis = [(k, d) for k, d in shared.items() if len(set(d.values())) > 1]
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "unit_id", *annotators, "adjudicated_grade", "note"])
            for (qid, uid), d in sorted(dis):
                w.writerow([qid, uid, *[d.get(a, "") for a in annotators], "", ""])
        log(f"  불일치 {len(dis)}건 → {out_csv} (adjudicated_grade 채운 뒤 import-judgments)")
    return res
