"""판정 CSV → 판정 기록 → 일치도 → 중재 → 최종 qrels (지시서 §6.3, v3.3 pooling 이식)

- `import`: pool_A.csv·pool_B.csv(·adjudication.csv) → data/judgments/judgments.jsonl
  annotator 칸이 비면 파일 이름(pool_A → "A")을 쓴다. `adjudicated_grade` 가 채워진 행은 중재 기록이다.
- `agreement`: 두 판정자가 모두 판정한 단위의 일치도(정확 일치·관련성(≥1) 일치·2차 가중 카파)
  → results/agreement.json, 불일치 단위 → data/judgments/adjudication.csv
- `finalize`: 최종 등급 = 중재 등급 > 판정자 일치 등급 > (불일치 미중재면 오류) → data/qrels.txt
  B 만 판정한 단위는 없다(B 는 A 질의의 부분집합). A 한 명만 판정한 단위는 A 등급.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import GRADES, ExperimentConfig, Qrels, read_jsonl, write_jsonl, write_qrels
from ev4.stats import quadratic_weighted_kappa

ADJUDICATOR = "adjudicator"
PRIMARY = "A"


class JudgmentError(ValueError):
    """판정 CSV 형식 오류·미중재 불일치"""


def judgments_path(cfg: ExperimentConfig) -> Path:
    """판정 기록 JSONL"""
    return cfg.judgments_dir / "judgments.jsonl"


def adjudication_path(cfg: ExperimentConfig) -> Path:
    """중재 시트"""
    return cfg.judgments_dir / "adjudication.csv"


def _default_annotator(path: Path) -> str:
    """pool_A.csv → "A" (그 외 파일은 빈 문자열)"""
    stem = path.stem
    return stem.split("_", 1)[1] if stem.startswith("pool_") and "_" in stem else ""


def import_judgments(cfg: ExperimentConfig, csv_paths: list[Path], log: Callable[[str], None] = print) -> int:
    """판정 CSV 를 판정 기록에 반영한다 (같은 (qid, unit_id, 판정자) 는 덮어씀)

    Returns:
        반영한 행 수

    Raises:
        JudgmentError: 등급이 0~2 가 아니거나 판정자를 알 수 없을 때
    """
    out = judgments_path(cfg)
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if out.exists():
        for r in read_jsonl(out):
            keyed[(r["qid"], r["unit_id"], r["annotator"])] = r
    n = 0
    blank = 0
    for p in csv_paths:
        p = Path(p)
        default_ann = _default_annotator(p)
        with open(p, encoding="utf-8-sig", newline="") as f:
            for ln, row in enumerate(csv.DictReader(f), 2):
                adj = (row.get("adjudicated_grade") or "").strip()
                g = adj or (row.get("grade") or "").strip()
                if not g:
                    blank += int("grade" in row)
                    continue
                if g not in {str(x) for x in GRADES}:
                    raise JudgmentError(f"{p}:{ln}: 등급은 0/1/2 (입력 {g!r})")
                ann = ADJUDICATOR if adj else ((row.get("annotator") or "").strip() or default_ann)
                if not ann:
                    raise JudgmentError(f"{p}:{ln}: annotator 가 비어 있습니다")
                rec = {"qid": row["qid"].strip(), "unit_id": row["unit_id"].strip(), "grade": int(g),
                       "annotator": ann, "note": (row.get("note") or "").strip()}
                keyed[(rec["qid"], rec["unit_id"], ann)] = rec
                n += 1
    write_jsonl(out, keyed.values())
    log(f"[judgments] {n}행 반영 (등급 빈칸 {blank}행) → {out} (총 {len(keyed)}행)")
    return n


def _by_unit(cfg: ExperimentConfig) -> dict[tuple[str, str], dict[str, int]]:
    path = judgments_path(cfg)
    if not path.exists():
        raise JudgmentError(f"판정 기록이 없습니다: {path} (먼저 judgments import)")
    by_unit: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for r in read_jsonl(path):
        by_unit[(r["qid"], r["unit_id"])][r["annotator"]] = int(r["grade"])
    return by_unit


def agreement(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """판정자 간 일치도와 중재 시트

    Returns:
        {annotators, pair, n, exact_agreement, relevant_agreement, quadratic_weighted_kappa,
         n_disagree, by_type}
    """
    by_unit = _by_unit(cfg)
    annotators = sorted({a for d in by_unit.values() for a in d if a != ADJUDICATOR})
    res: dict[str, Any] = {"annotators": annotators, "units_total": len(by_unit)}
    if len(annotators) < 2:
        log("[agreement] 판정자가 2명 미만입니다")
        return res
    a1 = PRIMARY if PRIMARY in annotators else annotators[0]
    a2 = next(a for a in annotators if a != a1)
    shared = {k: d for k, d in by_unit.items() if a1 in d and a2 in d}
    pairs = [(d[a1], d[a2]) for d in shared.values()]
    if not pairs:
        log("[agreement] 두 판정자가 함께 판정한 단위가 없습니다")
        return res
    res.update({
        "pair": [a1, a2],
        "n": len(pairs),
        "n_queries": len({q for q, _u in shared}),
        "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
        "relevant_agreement": sum((x >= 1) == (y >= 1) for x, y in pairs) / len(pairs),
        "quadratic_weighted_kappa": quadratic_weighted_kappa([x for x, _ in pairs], [y for _, y in pairs], 3),
        "confusion": {f"{x}{y}": sum(1 for p in pairs if p == (x, y)) for x in GRADES for y in GRADES},
    })
    dis = sorted(k for k, d in shared.items() if d[a1] != d[a2])
    res["n_disagree"] = len(dis)

    # 중재 시트 — 이미 중재한 등급은 보존
    adj_path = adjudication_path(cfg)
    prev = _read_adjudication(adj_path)
    queries = _query_texts(cfg)
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(adj_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query_text", "unit_id", f"grade_{a1}", f"grade_{a2}", "adjudicated_grade", "note"])
        for qid, uid in dis:
            d = shared[(qid, uid)]
            pg, pn = prev.get((qid, uid), ("", ""))
            if not pg and ADJUDICATOR in by_unit[(qid, uid)]:
                pg = str(by_unit[(qid, uid)][ADJUDICATOR])
            w.writerow([qid, queries.get(qid, ""), uid, d[a1], d[a2], pg, pn])
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[agreement] {a1} vs {a2}: n={res['n']} (질의 {res['n_queries']}) 정확 일치 {res['exact_agreement']:.3f} · "
        f"관련성(≥1) 일치 {res['relevant_agreement']:.3f} · QWK {res['quadratic_weighted_kappa']:.3f} · "
        f"불일치 {len(dis)}건 → {adj_path}")
    return res


def _read_adjudication(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(r["qid"], r["unit_id"]): ((r.get("adjudicated_grade") or "").strip(), r.get("note") or "")
                for r in csv.DictReader(f)}


def _query_texts(cfg: ExperimentConfig) -> dict[str, str]:
    if not cfg.queries_path.exists():
        return {}
    return {str(r["qid"]): str(r["text"]) for r in read_jsonl(cfg.queries_path)}


def finalize(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> Qrels:
    """최종 qrels 를 쓴다 (adjudication.csv 의 중재 등급을 읽어 반영)

    Raises:
        JudgmentError: 중재되지 않은 불일치가 남아 있을 때
    """
    adj = _read_adjudication(adjudication_path(cfg))
    if any(g for g, _n in adj.values()):
        import_judgments(cfg, [adjudication_path(cfg)], log=lambda _m: None)
    by_unit = _by_unit(cfg)
    qrels: Qrels = {}
    pending: list[tuple[str, str]] = []
    n_adj = 0
    for (qid, uid), d in by_unit.items():
        if ADJUDICATOR in d:
            grade = d[ADJUDICATOR]
            n_adj += 1
        else:
            grades = set(d.values())
            if len(grades) > 1:
                pending.append((qid, uid))
                continue
            grade = grades.pop()
        qrels.setdefault(qid, {})[uid] = grade
    if pending:
        raise JudgmentError(
            f"중재되지 않은 불일치 {len(pending)}건 (예: {pending[:3]}). "
            f"judgments agreement → {adjudication_path(cfg)} 의 adjudicated_grade 를 채운 뒤 다시 실행하세요"
        )
    write_qrels(cfg.qrels_path, qrels)
    n = sum(len(v) for v in qrels.values())
    dist = {g: sum(1 for v in qrels.values() for x in v.values() if x == g) for g in GRADES}
    log(f"[finalize] 질의 {len(qrels)} · 단위 {n} (등급 0/1/2 = {dist[0]}/{dist[1]}/{dist[2]}, 중재 {n_adj}) "
        f"→ {cfg.qrels_path}")
    return qrels
