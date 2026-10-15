"""평가 — ranx 지표·bootstrap CI·fisher 검정(Holm)·분해 → results/ (지시서 §6.6, §6.8 T2~T6)

- 평가 대상: test 질의 중 qrels 에 등급 ≥1 단위가 하나라도 있는 질의(관련 단위 없는 질의는
  recall 이 정의되지 않으므로 제외하고 수를 표에 적는다).
- `recall@10-l2` 는 등급 2 단위가 있는 질의만으로 계산한다(ranx 는 없는 질의를 0 으로 센다).
- nDCG 는 ranx `ndcg`(선형 이득 1·2).
- 유의성: ranx.compare(stat_test="fisher", n_permutations=10000), 주 지표 nDCG@10,
  config `comparisons` 전체를 한 계열로 Holm 보정.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from ev4.io import ExperimentConfig, Qrels, Run, read_qrels, read_queries, read_run
from ev4.stats import bootstrap_ci, holm

METRICS = ["ndcg@10", "recall@10", "mrr@10", "recall@10-l2", "recall@100"]
PRIMARY = "ndcg@10"
L2_METRIC = "recall@10-l2"
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000
DEFAULT_BREAKDOWN_RUNS = ["R2", "R4", "R6", "R8", "R9"]
_EMPTY = "__no_hit__"


# ─────────────────────────── 증거 언어 ───────────────────────────


def lang_bucket(lang: str) -> str:
    """단위 언어 → ko / en / other"""
    return lang if lang in ("ko", "en") else "other"


def evidence_lang(grade2_langs: list[str]) -> str:
    """질의의 증거 언어 (지시서 §6.6)

    등급 2 단위 언어로 정한다. other 가 하나라도 있으면 other, 아니면 다수결(동수면 ko).
    등급 2 단위가 없으면 "n/a".
    """
    if not grade2_langs:
        return "n/a"
    buckets = [lang_bucket(x) for x in grade2_langs]
    if "other" in buckets:
        return "other"
    c = Counter(buckets)
    return "ko" if c["ko"] >= c["en"] else "en"


# ─────────────────────────── 지표 ───────────────────────────


def _ranx_run(run: Run, qids: list[str], name: str) -> Any:
    from ranx import Run as RanxRun

    d: dict[str, dict[str, float]] = {}
    for qid in qids:
        ranked = run.get(qid, [])
        # 결과 없는 질의도 ranx 가 0 으로 세도록 자리표시 한 건을 넣는다
        d[qid] = {u: float(len(ranked) - i) for i, u in enumerate(ranked)} or {_EMPTY: 0.0}
    return RanxRun(d, name=name)


def _ranx_qrels(qrels: Qrels, qids: list[str]) -> Any:
    from ranx import Qrels as RanxQrels

    return RanxQrels({q: {u: g for u, g in qrels[q].items() if g > 0} for q in qids})


def per_query_scores(qrels: Qrels, run: Run, qids: list[str], l2_qids: list[str], name: str) -> dict[str, dict[str, float]]:
    """{metric: {qid: 값}} — recall@10-l2 는 l2_qids 만"""
    from ranx import evaluate

    rq, rr = _ranx_qrels(qrels, qids), _ranx_run(run, qids, name)
    base = [m for m in METRICS if m != L2_METRIC]
    scores = evaluate(rq, rr, base, return_mean=False, make_comparable=True)
    out = {m: dict(zip(qids, map(float, scores[m]), strict=True)) for m in base}
    out[L2_METRIC] = {}
    if l2_qids:
        s = evaluate(_ranx_qrels(qrels, l2_qids), _ranx_run(run, l2_qids, name), [L2_METRIC],
                     return_mean=False, make_comparable=True)
        vals = s[L2_METRIC] if isinstance(s, dict) else s
        out[L2_METRIC] = dict(zip(l2_qids, map(float, vals), strict=True))
    return out


def fisher_p(qrels: Qrels, run_a: Run, run_b: Run, qids: list[str], seed: int) -> float:
    """두 실행의 nDCG@10 fisher 무작위화 검정 p (양측)"""
    from ranx import compare

    rep = compare(_ranx_qrels(qrels, qids), [_ranx_run(run_a, qids, "A"), _ranx_run(run_b, qids, "B")],
                  metrics=[PRIMARY], stat_test="fisher", n_permutations=N_PERMUTATIONS, max_p=1.0,
                  random_seed=seed, make_comparable=True)
    return float(rep.to_dict()["A"]["comparisons"]["B"][PRIMARY])


# ─────────────────────────── 표 ───────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def _f(x: float, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return "–" if math.isnan(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _p(p: float) -> str:
    if p is None or math.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def evaluate_all(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """T2~T6·per_query.csv·comparisons.csv 생성

    Returns:
        {n_queries, n_l2, runs: {run: {metric: (mean, lo, hi)}}, comparisons: [...]}
    """
    from src.index import case_index

    queries = {q.qid: q for q in read_queries(cfg.queries_path, split="test")}
    qrels_all = read_qrels(cfg.qrels_path)
    qrels = {q: v for q, v in qrels_all.items() if q in queries}
    qids = sorted(q for q, v in qrels.items() if any(g > 0 for g in v.values()))
    no_rel = sorted(set(queries) - set(qids))
    l2_qids = [q for q in qids if any(g == 2 for g in qrels[q].values())]
    if not qids:
        raise ValueError("관련 단위(등급 ≥1)가 있는 test 질의가 없습니다 — qrels 를 확인하세요")

    idx = case_index.load(cfg.case_id)

    def _lang(uid: str) -> str:
        parts = idx.unit_parts(uid)
        return str(parts[0].meta.get("lang", "und")) if parts else "und"

    ev_lang = {q: evidence_lang([_lang(u) for u, g in qrels[q].items() if g == 2]) for q in qids}
    cross = {q: ("n/a" if ev_lang[q] == "n/a" else ("yes" if lang_bucket(queries[q].query_lang) != ev_lang[q] else "no"))
             for q in qids}

    run_names = list(cfg.runs)
    runs: dict[str, Run] = {}
    for name in run_names:
        path = cfg.run_path(name, "test")
        if path.exists():
            runs[name] = read_run(path)
        else:
            log(f"[evaluate] {name} 실행 파일 없음 — 건너뜀 ({path})")
    scores = {name: per_query_scores(qrels, run, qids, l2_qids, name) for name, run in runs.items()}
    summary = {
        name: {m: bootstrap_ci(list(s[m].values()), n=N_BOOTSTRAP, seed=cfg.seed) for m in METRICS}
        for name, s in scores.items()
    }

    # 비교·Holm
    comps: list[dict[str, Any]] = []
    for fam, a, b in cfg.comparison_pairs():
        if a not in runs or b not in runs:
            continue
        ma, mb = _mean(list(scores[a][PRIMARY].values())), _mean(list(scores[b][PRIMARY].values()))
        comps.append({"family": fam, "A": a, "B": b, "metric": PRIMARY, "mean_A": ma, "mean_B": mb,
                      "diff": mb - ma, "p": fisher_p(qrels, runs[a], runs[b], qids, cfg.seed)})
    adj = holm({f"{c['A']}-{c['B']}-{i}": c["p"] for i, c in enumerate(comps)})
    for i, c in enumerate(comps):
        c["p_holm"] = adj[f"{c['A']}-{c['B']}-{i}"]

    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_per_query(res_dir / "per_query.csv", scores, queries, qids, ev_lang, cross)
    with open(res_dir / "comparisons.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "A", "B", "metric", "mean_A", "mean_B", "diff", "p", "p_holm"])
        w.writeheader()
        w.writerows(comps)

    bd_runs = [r for r in cfg.extra.get("breakdown_runs", DEFAULT_BREAKDOWN_RUNS) if r in runs]
    md = [
        "# 평가 결과",
        "",
        f"- test 질의 {len(queries)}건 중 평가 {len(qids)}건 (관련 단위 없는 질의 {len(no_rel)}건 제외"
        + (f": {', '.join(no_rel)}" if no_rel else "") + f") · 등급 2 단위 있는 질의 {len(l2_qids)}건",
        f"- 실행 깊이 {cfg.depth} · bootstrap {N_BOOTSTRAP}회 95% CI · fisher {N_PERMUTATIONS}회 · "
        f"Holm 보정 {len(comps)}개 비교(주 지표 {PRIMARY})",
        "",
    ]
    md += _t2(summary, run_names, len(qids), len(l2_qids))
    md += _t3(comps)
    md += _t4(scores, queries, qids, bd_runs)
    md += _t5(scores, qids, ev_lang, cross, bd_runs)
    md += _t6(scores, summary, comps, qids, ev_lang)
    (res_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out = {
        "n_queries": len(qids), "n_l2": len(l2_qids), "excluded": no_rel,
        "runs": {r: {m: list(v) for m, v in s.items()} for r, s in summary.items()},
        "comparisons": comps,
    }
    (res_dir / "evaluation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[evaluate] 질의 {len(qids)}건 · 실행 {len(runs)}개 · 비교 {len(comps)}개 → {res_dir / 'tables.md'}")
    return out


def _write_per_query(path, scores, queries, qids, ev_lang, cross) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "qid", "type", "query_lang", "evidence_lang", "cross_lingual", *METRICS])
        for run, s in scores.items():
            for q in qids:
                w.writerow([run, q, queries[q].type, queries[q].query_lang, ev_lang[q], cross[q],
                            *[_f(s[m].get(q, math.nan), 4) if q in s[m] else "" for m in METRICS]])


def _t2(summary, run_names, n, n_l2) -> list[str]:
    lines = ["## T2 사다리 표 (평균 [95% CI])", "",
             "| 실행 | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for r in run_names:
        if r in summary:
            lines.append(f"| {r} | " + " | ".join(_ci(summary[r][m]) for m in METRICS) + " |")
    lines += ["", f"n = {n} (recall@10-l2: n = {n_l2})", ""]
    return lines


def _t3(comps) -> list[str]:
    lines = ["## T3 헤더 효과 및 비교 전체 (nDCG@10, B − A)", "",
             "| 계열 | A | B | A 평균 | B 평균 | 차이 | p | p(Holm) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for c in sorted(comps, key=lambda c: (c["family"] != "header", c["family"])):
        lines.append(f"| {c['family']} | {c['A']} | {c['B']} | {_f(c['mean_A'])} | {_f(c['mean_B'])} | "
                     f"{c['diff']:+.3f} | {_p(c['p'])} | {_p(c['p_holm'])} |")
    return [*lines, ""]


def _group_table(title, scores, groups: dict[str, list[str]], runs, metrics=("ndcg@10", "recall@10")) -> list[str]:
    lines = [title, "", "| 그룹 | n | " + " | ".join(f"{r} {m}" for r in runs for m in metrics) + " |",
             "|---|---:|" + "---:|" * (len(runs) * len(metrics))]
    for g, qs in groups.items():
        if not qs:
            continue
        cells = [_f(_mean([scores[r][m][q] for q in qs])) for r in runs for m in metrics]
        lines.append(f"| {g} | {len(qs)} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _t4(scores, queries, qids, runs) -> list[str]:
    from ev4.io import QUERY_TYPES

    groups = {t: [q for q in qids if queries[q].type == t] for t in QUERY_TYPES}
    return _group_table("## T4 질의 유형별", scores, groups, runs)


def _t5(scores, qids, ev_lang, cross, runs) -> list[str]:
    groups = {f"증거 {lg}": [q for q in qids if ev_lang[q] == lg] for lg in ("ko", "en", "other")}
    groups |= {"교차 언어": [q for q in qids if cross[q] == "yes"],
               "같은 언어": [q for q in qids if cross[q] == "no"]}
    na = sum(1 for q in qids if ev_lang[q] == "n/a")
    out = _group_table("## T5 증거 언어별·교차 언어", scores, groups, runs)
    if na:
        out.insert(-1, f"(등급 2 단위가 없어 증거 언어를 정할 수 없는 질의 {na}건은 이 표에서 빠짐)")
    return out


def _t6(scores, summary, comps, qids, ev_lang) -> list[str]:
    if "R8" not in scores or "R9" not in scores:
        return ["## T6 모델 비교", "", "(R8·R9 실행이 없음)", ""]
    lines = ["## T6 모델 비교 R8(Qwen3) vs R9(bge)", "",
             "| 지표 | R8 | R9 | R9 − R8 |", "|---|---|---|---:|"]
    for m in METRICS:
        a, b = summary["R8"][m], summary["R9"][m]
        lines.append(f"| {m} | {_ci(a)} | {_ci(b)} | {b[0] - a[0]:+.3f} |")
    c = next((c for c in comps if {c["A"], c["B"]} == {"R8", "R9"}), None)
    if c:
        lines += ["", f"nDCG@10 fisher p = {_p(c['p'])}, Holm p = {_p(c['p_holm'])}"]
    groups = {f"증거 {lg}": [q for q in qids if ev_lang[q] == lg] for lg in ("ko", "en", "other")}
    lines += [""] + _group_table("### 증거 언어별", scores, groups, ["R8", "R9"])
    return lines
