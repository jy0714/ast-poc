"""판정 CSV → 판정 기록 → 일치도 → 중재 → 최종 qrels (지시서 §6.3, v3.3 pooling 이식)

- `import`: pool_A.csv·pool_B.csv(·adjudication.csv) → data/judgments/judgments.jsonl
  annotator 칸이 비면 파일 이름(pool_A → "A")을 쓴다. `adjudicated_grade` 가 채워진 행은 중재 기록이다.
- `agreement`: 두 판정자가 모두 판정한 단위의 일치도(정확 일치·관련성(≥1) 일치·2차 가중 카파),
  A–B 와 A–C(판정자 C 가 있으면, v4.4 §9)를 따로 낸다.
  → results/agreement.json, 불일치 단위 → data/judgments/adjudication.csv
- `finalize`: 최종 등급 = 중재 등급 > 판정자 일치 등급 > (불일치 미중재면 오류) → data/qrels.txt
  B·C 만 판정한 단위는 없다(B·C 는 A 질의의 부분집합). A 한 명만 판정한 단위는 A 등급.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import (
    GRADES,
    SUPPORT_GROUPS,
    ExperimentConfig,
    Qrels,
    read_jsonl,
    read_queries,
    write_jsonl,
    write_qrels,
)
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


def _pair_stats(by_unit: dict[tuple[str, str], dict[str, int]], a1: str, a2: str,
                qgroups: dict[str, str]) -> dict[str, Any] | None:
    """두 판정자의 일치도 (함께 판정한 단위가 없으면 None)"""
    shared = {k: d for k, d in by_unit.items() if a1 in d and a2 in d}
    pairs = [(d[a1], d[a2]) for d in shared.values()]
    if not pairs:
        return None
    q_a = {q for (q, _u), d in by_unit.items() if a1 in d}
    q_o = {q for (q, _u), d in by_unit.items() if a2 in d}
    return {
        "pair": [a1, a2],
        "n": len(pairs),
        "n_queries": len({q for q, _u in shared}),
        "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
        "relevant_agreement": sum((x >= 1) == (y >= 1) for x, y in pairs) / len(pairs),
        "quadratic_weighted_kappa": quadratic_weighted_kappa([x for x, _ in pairs], [y for _, y in pairs], 3),
        "confusion": {f"{x}{y}": sum(1 for p in pairs if p == (x, y)) for x in GRADES for y in GRADES},
        "n_disagree": sum(1 for x, y in pairs if x != y),
        "disagree": sorted(k for k, d in shared.items() if d[a1] != d[a2]),
        # 두 번째 판정자 질의 비율·집합별 분포 (v4.2 J3, v4.3 §5)
        "n_queries_A": len(q_a),
        "n_queries_B": len(q_o),
        "b_query_ratio": len(q_o) / len(q_a) if q_a else float("nan"),
        "a_queries_by_group": {g: sum(1 for q in q_a if qgroups.get(q) == g) for g in SUPPORT_GROUPS},
        "b_queries_by_group": {g: sum(1 for q in q_o if qgroups.get(q) == g) for g in SUPPORT_GROUPS},
        "qids_B": sorted(q_o),
    }


def agreement(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """판정자 간 일치도와 중재 시트 — A–B, A–C(있으면) 를 따로 낸다 (v4.4 §9)

    Returns:
        {annotators, units_total, pairs: {"A-B": {...}, "A-C": {...}}, n_disagree,
         + 첫 쌍(A–B)의 값: pair, n, exact_agreement, relevant_agreement, quadratic_weighted_kappa, ...}
    """
    by_unit = _by_unit(cfg)
    annotators = sorted({a for d in by_unit.values() for a in d if a != ADJUDICATOR})
    res: dict[str, Any] = {"annotators": annotators, "units_total": len(by_unit)}
    if len(annotators) < 2:
        log("[agreement] 판정자가 2명 미만입니다")
        return res
    a1 = PRIMARY if PRIMARY in annotators else annotators[0]
    qgroups = _query_groups(cfg)
    pairs: dict[str, dict[str, Any]] = {}
    for a2 in (a for a in annotators if a != a1):
        st = _pair_stats(by_unit, a1, a2, qgroups)
        if st is not None:
            pairs[f"{a1}-{a2}"] = st
    if not pairs:
        log("[agreement] 두 판정자가 함께 판정한 단위가 없습니다")
        return res
    first = next(iter(pairs.values()))
    res.update({k: v for k, v in first.items() if k != "disagree"})
    res["pairs"] = {k: {kk: vv for kk, vv in v.items() if kk != "disagree"} for k, v in pairs.items()}
    dis = sorted({k for st in pairs.values() for k in st["disagree"]})
    res["n_disagree"] = len(dis)

    # 중재 시트 — 이미 중재한 등급은 보존. 판정자마다 등급 열 하나
    adj_path = adjudication_path(cfg)
    prev = _read_adjudication(adj_path)
    queries = _query_texts(cfg)
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(adj_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query_text", "unit_id", *[f"grade_{a}" for a in annotators], "adjudicated_grade", "note"])
        for qid, uid in dis:
            d = by_unit[(qid, uid)]
            pg, pn = prev.get((qid, uid), ("", ""))
            if not pg and ADJUDICATOR in d:
                pg = str(d[ADJUDICATOR])
            w.writerow([qid, queries.get(qid, ""), uid, *[d.get(a, "") for a in annotators], pg, pn])
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.results_dir / "J3_agreement.md").write_text(_agreement_markdown(res), encoding="utf-8")
    for name, st in res["pairs"].items():
        log(f"[agreement] {name}: n={st['n']} (질의 {st['n_queries']}) 정확 일치 {st['exact_agreement']:.3f} · "
            f"관련성(≥1) 일치 {st['relevant_agreement']:.3f} · QWK {st['quadratic_weighted_kappa']:.3f} · "
            f"불일치 {st['n_disagree']}건")
    log(f"[agreement] 중재 대상 {len(dis)}건 → {adj_path}")
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


def disputed_units(cfg: ExperimentConfig) -> set[tuple[str, str]]:
    """판정 이견 (qid, unit_id) — 판정자 등급이 달랐거나 중재 등급이 1차(A) 등급과 다름 (v4.2 F2)

    판정 기록이 없으면 빈 집합.
    """
    if not judgments_path(cfg).exists():
        return set()
    out: set[tuple[str, str]] = set()
    for key, d in _by_unit(cfg).items():
        human = {a: g for a, g in d.items() if a != ADJUDICATOR}
        first = human.get(PRIMARY, next(iter(human.values()), None))
        if len(set(human.values())) > 1 or (ADJUDICATOR in d and d[ADJUDICATOR] != first):
            out.add(key)
    return out


def _query_groups(cfg: ExperimentConfig) -> dict[str, str]:
    if not cfg.queries_path.exists():
        return {}
    return {str(r["qid"]): str(r.get("support_group", "")) for r in read_jsonl(cfg.queries_path)}


def _agreement_markdown(res: dict[str, Any]) -> str:
    """J3 일치도 표 (건수·비율만) — 판정자 쌍마다 한 절"""
    lines = ["# J3 판정자 간 일치도", "",
             "- 2차 가중 카파: 등급 3단계(0/1/2, n_levels=3) · 관련 여부 일치: 등급 1 이상 기준",
             f"- 중재 대상 단위(어느 쌍이든 불일치) {res['n_disagree']}건", ""]
    for name, st in res["pairs"].items():
        a1, a2 = st["pair"]
        lines += [
            f"## {name} — 함께 판정한 단위 {st['n']}건 (질의 {st['n_queries']}건)", "",
            "| 항목 | 값 |", "|---|---:|",
            f"| 등급 완전 일치율 | {st['exact_agreement']:.3f} |",
            f"| 관련 여부(≥1) 일치율 | {st['relevant_agreement']:.3f} |",
            f"| 2차 가중 카파 | {st['quadratic_weighted_kappa']:.3f} |",
            f"| 불일치 단위 수 | {st['n_disagree']} |",
            f"| {a2} 판정 질의 비율 | {st['n_queries_B']}/{st['n_queries_A']} = {st['b_query_ratio']:.3f} |",
            "", f"| 집합 | {a1} 판정 질의 | {a2} 판정 질의 |", "|---|---:|---:|",
        ]
        for g in SUPPORT_GROUPS:
            lines.append(f"| {g} | {st['a_queries_by_group'][g]} | {st['b_queries_by_group'][g]} |")
        lines += ["", f"{a2} 판정 질의 (qid): " + (", ".join(st.get("qids_B", [])) or "없음")]
        lines += ["", f"| {a1}\\{a2} | 0 | 1 | 2 |", "|---|---:|---:|---:|"]
        for x in GRADES:
            lines.append(f"| {x} | " + " | ".join(str(st["confusion"][f"{x}{y}"]) for y in GRADES) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def qrels_stats(qrels: Qrels, test_qids: set[str] | None = None) -> dict[str, Any]:
    """최종 정답 통계 (v4.2 J4) — 등급별 단위 수, 등급 2 단위가 없는 질의

    Args:
        qrels: 최종 qrels
        test_qids: 이 질의만 센다 (None 이면 qrels 전체)
    """
    qs = {q: v for q, v in qrels.items() if test_qids is None or q in test_qids}
    no_l2 = sorted(q for q, v in qs.items() if not any(g == 2 for g in v.values()))
    missing = sorted((test_qids or set()) - set(qs))  # 판정 기록 자체가 없는 질의도 등급 2 없음
    return {
        "n_queries": len(qs) + len(missing),
        "grade2_units": sum(1 for v in qs.values() for g in v.values() if g == 2),
        "grade1_units": sum(1 for v in qs.values() for g in v.values() if g == 1),
        "grade0_units": sum(1 for v in qs.values() for g in v.values() if g == 0),
        "queries_without_grade2": sorted(no_l2 + missing),
    }


def _qrels_markdown(st: dict[str, Any], n_adj: int) -> str:
    lines = [
        "# J4 최종 정답 통계 (평가용 질의)", "",
        "| 항목 | 값 |", "|---|---:|",
        f"| 질의 수 | {st['n_queries']} |",
        f"| 등급 2 단위 수 | {st['grade2_units']} |",
        f"| 등급 1 단위 수 | {st['grade1_units']} |",
        f"| 등급 0 단위 수 | {st['grade0_units']} |",
        f"| 등급 2 단위가 없는 질의 수 | {len(st['queries_without_grade2'])} |",
        f"| 중재로 정한 단위 수 | {n_adj} |",
        "",
        "등급 2 단위가 없는 질의는 Recall@10(등급 2)와 언어별 보조 값에서 뺀다.",
        "해당 qid: " + (", ".join(st["queries_without_grade2"]) or "없음"),
    ]
    return "\n".join(lines) + "\n"


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
    queries = read_queries(cfg.queries_path, split="test") if cfg.queries_path.exists() else []
    st = qrels_stats(qrels, {q.qid for q in queries} if queries else None)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "J4_qrels.md").write_text(_qrels_markdown(st, n_adj), encoding="utf-8")
    if queries:
        from ev4.querystats import write_t0b

        write_t0b(cfg, queries, qrels)  # v4.3 §4
    log(f"[finalize] 평가용: 등급 2 {st['grade2_units']} · 등급 1 {st['grade1_units']} · "
        f"등급 2 없는 질의 {len(st['queries_without_grade2'])}건")
    return qrels
