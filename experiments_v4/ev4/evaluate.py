"""평가 — ranx 지표·bootstrap CI·fisher 검정(Holm) → results/ (지시서 §6.6, v4.2 E1~E8, v4.3, v4.4 §2)

- 실행(v4.4 §2): 5개. 표·보고서의 실행 이름은 config `labels` 의 논문 번호(R1~R5)다.
- 분석 집합(v4.3 §3): 주 분석 = config `primary_subset`(search, S17). 사전 지정 비교 4개의 검정·Holm 보정과
  T2 는 S17 로 한다. ALL45 는 평균·95% CI 만 낸다(T2 부가). T8 은 집합별 기술 통계(R1·R2·R3·R5).
  T3~T6 은 만들지 않는다(v4.4 §2). 언어별 보조 값은 표가 아니라 본문용 값이다.
- 평가 대상: qrels 에 등급 ≥1 단위가 하나라도 있는 질의(관련 단위 없는 질의는
  recall 이 정의되지 않으므로 제외하고 수를 표에 적는다).
- P@10(v4.3 §4): 상위 10개 중 등급 1 이상 단위 비율(ranx `precision@10`).
- `recall@10-l2` 는 등급 2 단위가 있는 질의만으로 계산한다(ranx 는 없는 질의를 0 으로 센다).
- nDCG 는 ranx `ndcg`(선형 이득 1·2).
- 유의성: ranx.compare(stat_test="fisher", n_permutations=10000), 주 지표 nDCG@10,
  config `comparisons` 전체(4개)를 한 계열로 Holm 보정.
  ranx 0.3.21 fisher 는 |평균 차이| 로 비교하는 양측 검정이다
  (ranx/statistical_tests/fisher_randomization_test.py 31·39행). 교차 확인용으로 단일 스레드
  양측 무작위화 검정(stats.paired_randomization_p)을 함께 낸다(p_check).
- 차이 CI: 같은 질의 재표집에서 두 실행의 차이를 계산하는 짝지은 bootstrap 95% (v4.2 E2).
- 질의 5개 미만 그룹은 값을 내되 `†사례 수준` 표시를 붙인다 (v4.2 E8).
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from ev4.io import (
    SUPPORT_GROUPS,
    ExperimentConfig,
    Qrels,
    Run,
    read_qrels,
    read_queries,
    read_run,
)
from ev4.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_randomization_p

METRICS = ["ndcg@10", "precision@10", "recall@10", "mrr@10", "recall@10-l2", "recall@100"]
PRIMARY = "ndcg@10"
L2_METRIC = "recall@10-l2"
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000
SMALL_N = 5
SMALL_MARK = "†사례 수준"
METRIC_LABELS = {
    "ndcg@10": "nDCG@10",
    "precision@10": "P@10",
    "recall@10": "Recall@10",
    "recall@10-l2": "Recall@10(등급 2)",
    "mrr@10": "MRR@10",
    "recall@100": "Recall@100",
}
T2_METRICS = [
    "ndcg@10",
    "precision@10",
    "recall@10",
    "recall@10-l2",
    "mrr@10",
    "recall@100",
]  # v4.3 §4 순서
T8_LABELS = ["R1", "R2", "R3", "R5"]  # 논문 번호 (v4.4 §2)
T8_METRICS = ["ndcg@10", "precision@10"]
SUBSET_LABELS = {"search": "S17 (search)", "meta": "meta", "nonsearch": "nonsearch"}
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


def per_query_scores(
    qrels: Qrels, run: Run, qids: list[str], l2_qids: list[str], name: str
) -> dict[str, dict[str, float]]:
    """{metric: {qid: 값}} — recall@10-l2 는 l2_qids 만"""
    from ranx import evaluate

    rq, rr = _ranx_qrels(qrels, qids), _ranx_run(run, qids, name)
    base = [m for m in METRICS if m != L2_METRIC]
    scores = evaluate(rq, rr, base, return_mean=False, make_comparable=True)
    out = {m: dict(zip(qids, map(float, scores[m]), strict=True)) for m in base}
    out[L2_METRIC] = {}
    if l2_qids:
        s = evaluate(
            _ranx_qrels(qrels, l2_qids),
            _ranx_run(run, l2_qids, name),
            [L2_METRIC],
            return_mean=False,
            make_comparable=True,
        )
        vals = s[L2_METRIC] if isinstance(s, dict) else s
        out[L2_METRIC] = dict(zip(l2_qids, map(float, vals), strict=True))
    return out


def fisher_p(qrels: Qrels, run_a: Run, run_b: Run, qids: list[str], seed: int) -> float:
    """두 실행의 nDCG@10 fisher 무작위화 검정 p (ranx, 양측)"""
    from ranx import compare

    rep = compare(
        _ranx_qrels(qrels, qids),
        [_ranx_run(run_a, qids, "A"), _ranx_run(run_b, qids, "B")],
        metrics=[PRIMARY],
        stat_test="fisher",
        n_permutations=N_PERMUTATIONS,
        max_p=1.0,
        random_seed=seed,
        make_comparable=True,
    )
    return float(rep.to_dict()["A"]["comparisons"]["B"][PRIMARY])


# ─────────────────────────── 형식 ───────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def _f(x: float | None, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return "–" if math.isnan(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _dci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return "–" if math.isnan(m) else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _p(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _n(n: int) -> str:
    """질의 수 (5개 미만이면 †사례 수준 표시)"""
    return f"{n} {SMALL_MARK}" if 0 < n < SMALL_N else str(n)


def _run_label(cfg: ExperimentConfig, run: str) -> str:
    spec = cfg.runs.get(run, {})
    parts = [x for x, on in (("BM25", spec.get("use_bm25")), ("밀집", spec.get("use_dense"))) if on]
    label = "+".join(parts) if len(parts) < 2 else "혼합(RRF)"
    return label + (" + 재순위" if spec.get("rerank") else "")


# ─────────────────────────── 실행 ───────────────────────────


def _summarize(
    scores: dict[str, dict[str, dict[str, float]]], qs: list[str], l2_qs: list[str], seed: int
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """{run: {metric: (mean, lo, hi)}} — qs 집합만 (recall@10-l2 는 l2_qs 만)"""
    return {
        name: {
            m: bootstrap_ci(
                [s[m][q] for q in (l2_qs if m == L2_METRIC else qs)], n=N_BOOTSTRAP, seed=seed
            )
            for m in METRICS
        }
        for name, s in scores.items()
    }


def evaluate_all(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """T0b·T2·T8·언어별 보조 값·per_query.csv·comparisons.csv 생성 (주 분석 집합 = config primary_subset)

    Returns:
        {primary_subset, n_queries, n_l2, n_queries_all, n_l2_all, excluded, labels,
         runs: {run: {metric: (mean, lo, hi)}}, runs_all: {...}, comparisons: [...], language_aux: {...}}
    """
    from ev4.querystats import write_t0b
    from src.index import case_index

    cfg.check_subset_lock()
    all_queries = read_queries(cfg.queries_path, split="test")
    queries = {q.qid: q for q in all_queries}
    qrels_all = read_qrels(cfg.qrels_path)
    qrels = {q: v for q, v in qrels_all.items() if q in queries}
    write_t0b(cfg, all_queries, qrels)
    qids_all = sorted(q for q, v in qrels.items() if any(g > 0 for g in v.values()))
    no_rel = sorted(set(queries) - set(qids_all))
    l2_all = [q for q in qids_all if any(g == 2 for g in qrels[q].values())]
    primary = cfg.primary_subset
    qids = [q for q in qids_all if queries[q].support_group == primary]
    l2_qids = [q for q in l2_all if queries[q].support_group == primary]
    if not qids:
        raise ValueError(
            f"관련 단위(등급 ≥1)가 있는 {primary} 질의가 없습니다 — qrels 를 확인하세요"
        )

    idx = case_index.load(cfg.case_id)

    def _lang(uid: str) -> str:
        parts = idx.unit_parts(uid)
        return str(parts[0].meta.get("lang", "und")) if parts else "und"

    ev_lang = {
        q: evidence_lang([_lang(u) for u, g in qrels[q].items() if g == 2]) for q in qids_all
    }
    cross = {
        q: (
            "n/a"
            if ev_lang[q] == "n/a"
            else ("yes" if lang_bucket(queries[q].query_lang) != ev_lang[q] else "no")
        )
        for q in qids_all
    }

    runs: dict[str, Run] = {}
    for name in cfg.runs:
        path = cfg.run_path(name, "test")
        if path.exists():
            runs[name] = read_run(path)
        else:
            log(f"[evaluate] {name} 실행 파일 없음 — 건너뜀 ({path})")
    scores = {
        name: per_query_scores(qrels, run, qids_all, l2_all, name) for name, run in runs.items()
    }
    summary = _summarize(scores, qids, l2_qids, cfg.seed)
    summary_all = _summarize(scores, qids_all, l2_all, cfg.seed)

    # 사전 지정 비교 (E2·E3) — 주 분석 집합만, 차이 = B − A, Holm 은 전체 한 계열
    comps: list[dict[str, Any]] = []
    for fam, a, b in cfg.comparison_pairs():
        if a not in runs or b not in runs:
            continue
        va = [scores[a][PRIMARY][q] for q in qids]
        vb = [scores[b][PRIMARY][q] for q in qids]
        d, lo, hi = paired_bootstrap_ci(va, vb, n=N_BOOTSTRAP, seed=cfg.seed)
        comps.append(
            {
                "family": fam,
                "A": a,
                "B": b,
                "comparison": f"{a}→{b}",
                "label_A": cfg.label(a),
                "label_B": cfg.label(b),
                "comparison_label": f"{cfg.label(a)}→{cfg.label(b)}",
                "metric": PRIMARY,
                "subset": primary,
                "n": len(qids),
                "mean_A": _mean(va),
                "mean_B": _mean(vb),
                "diff": d,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": fisher_p(qrels, runs[a], runs[b], qids, cfg.seed),
                "p_check": paired_randomization_p(va, vb, n=N_PERMUTATIONS, seed=cfg.seed),
            }
        )
    for key in ("p", "p_check"):
        adj = holm({str(i): c[key] for i, c in enumerate(comps)})
        for i, c in enumerate(comps):
            c[f"{key}_holm"] = adj[str(i)]

    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_per_query(res_dir / "per_query.csv", scores, queries, qids_all, ev_lang, cross)
    comp_cols = [
        "family",
        "comparison_label",
        "comparison",
        "label_A",
        "label_B",
        "A",
        "B",
        "metric",
        "subset",
        "n",
        "mean_A",
        "mean_B",
        "diff",
        "ci_lo",
        "ci_hi",
        "p",
        "p_holm",
        "p_check",
        "p_check_holm",
    ]
    with open(res_dir / "comparisons.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comp_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(comps)

    plabel = SUBSET_LABELS.get(primary, primary)
    n_primary = sum(1 for q in all_queries if q.support_group == primary)
    aux = language_aux(cfg, qrels, runs, l2_qids, _lang)
    md = [
        "# 평가 결과",
        "",
        f"- 주 분석 집합 {plabel} (`primary_subset: {primary}`): 질의 {n_primary}건 중 평가 {len(qids)}건 · "
        f"등급 2 단위 있는 질의 {len(l2_qids)}건. 검정·Holm 보정·T2 는 이 집합으로 한다",
        f"- ALL45: 질의 {len(queries)}건 중 평가 {len(qids_all)}건 (관련 단위 없는 질의 {len(no_rel)}건 제외"
        + (f": {', '.join(no_rel)}" if no_rel else "")
        + "). 기술 통계(평균·95% CI)만, 검정하지 않는다",
        f"- 실행 깊이 {cfg.depth} · bootstrap {N_BOOTSTRAP}회 95% CI(차이는 짝지은 bootstrap) · "
        f"fisher {N_PERMUTATIONS}회 양측 · Holm 보정 {len(comps)}개 비교(주 지표 {METRIC_LABELS[PRIMARY]})",
        f"- {SMALL_MARK}: 질의 {SMALL_N}개 미만 그룹 (값은 내되 일반화하지 않는다)",
        "",
    ]
    md += _t2(
        cfg, summary, len(qids), len(l2_qids), f"T2 검색 구성별 성능 — {plabel} (평균 [95% CI])"
    )
    md += _t2(
        cfg, summary_all, len(qids_all), len(l2_all), "T2 부가 — ALL45 (평균 [95% CI], 검정 없음)"
    )
    md += _comparisons_table(cfg, comps)
    md += _t8(cfg, scores, queries, qids_all, no_rel)
    md += _language_aux_md(aux, plabel)
    (res_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out = {
        "primary_subset": primary,
        "n_queries": len(qids),
        "n_l2": len(l2_qids),
        "n_queries_all": len(qids_all),
        "n_l2_all": len(l2_all),
        "excluded": no_rel,
        "runs": {r: {m: list(v) for m, v in s.items()} for r, s in summary.items()},
        "runs_all": {r: {m: list(v) for m, v in s.items()} for r, s in summary_all.items()},
        "comparisons": comps,
        "labels": {r: cfg.label(r) for r in runs},
        "language_aux": aux,
    }
    (res_dir / "evaluation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"[evaluate] {plabel} {len(qids)}건 · ALL45 {len(qids_all)}건 · 실행 {len(runs)}개 · 비교 {len(comps)}개 "
        f"→ {res_dir / 'tables.md'}"
    )
    return out


def _write_per_query(path, scores, queries, qids, ev_lang, cross) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "qid",
                "support_group",
                "support_class",
                "type",
                "query_lang",
                "evidence_lang",
                "cross_lingual",
                *METRICS,
            ]
        )
        for run, s in scores.items():
            for q in qids:
                qq = queries[q]
                w.writerow(
                    [
                        run,
                        q,
                        qq.support_group,
                        qq.support_class,
                        qq.type,
                        qq.query_lang,
                        ev_lang[q],
                        cross[q],
                        *[_f(s[m].get(q, math.nan), 4) if q in s[m] else "" for m in METRICS],
                    ]
                )


# ─────────────────────────── 표 ───────────────────────────


def _t2(cfg, summary, n, n_l2, title: str) -> list[str]:
    """T2 (논문 [표 12]) — 실행 5개(논문 번호 순), 평균 [95% CI]"""
    heads = [METRIC_LABELS[m] + (f" (n={n_l2})" if m == L2_METRIC else "") for m in T2_METRICS]
    lines = [
        f"## {title}",
        "",
        "| 실행 | 하네스 | 색인 | 구성 | " + " | ".join(heads) + " |",
        "|---|---|---|---|" + "---|" * len(heads),
    ]
    for r in cfg.runs_by_label([r for r in cfg.runs if r in summary]):
        prof = cfg.runs[r].get("profile", "")
        lines.append(
            f"| {cfg.label(r)} | {r} | {prof} | {_run_label(cfg, r)} | "
            + " | ".join(_ci(summary[r][m]) for m in T2_METRICS)
            + " |"
        )
    lines += ["", f"n = {n} (Recall@10(등급 2): 등급 2 단위가 있는 질의 n = {n_l2})", ""]
    return lines


def _comparisons_table(cfg, comps) -> list[str]:
    """사전 지정 비교 4개 (E2) — comparisons.csv 와 같은 값, 실행 이름은 논문 번호"""
    lines = [
        "## 사전 지정 비교 (nDCG@10, 차이 = 대상 − 기준)",
        "",
        "| 기준→대상 | 하네스 | 기준 | 대상 | 차이 [95% CI] | p(fisher) | p(Holm) | p(교차 확인) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for c in comps:
        lines.append(
            f"| {cfg.label(c['A'])}→{cfg.label(c['B'])} | {c['comparison']} | {_f(c['mean_A'])} | "
            f"{_f(c['mean_B'])} | {_dci((c['diff'], c['ci_lo'], c['ci_hi']))} | {_p(c['p'])} | "
            f"{_p(c['p_holm'])} | {_p(c['p_check'])} |"
        )
    return [*lines, ""]


def _run_by_label(cfg: ExperimentConfig, label: str) -> str | None:
    """논문 번호 → 하네스 실행 이름"""
    return next((r for r in cfg.runs if cfg.label(r) == label), None)


def language_aux(
    cfg: ExperimentConfig,
    qrels: Qrels,
    runs: dict[str, Run],
    qids: list[str],
    unit_lang: Callable[[str], str],
) -> dict[str, Any]:
    """언어별 보조 값 (v4.4 §2, 표 아님 — 본문용)

    현지 언어 = ko·en 이 아닌 단위 언어. 주 분석 집합 질의 가운데 등급 2 단위에 현지 언어 단위가 있는
    질의를 고르고, 그 질의마다 논문 R5·R1·R2 가 현지 언어 등급 2 단위를 상위 10개에 넣었는지 센다.

    Returns:
        {n_local_queries, hits: {R5: n, R1: n, R2: n}, per_query: [{qid, R5, R1, R2}], runs: {R5: 하네스 이름}}
    """
    names = {lab: _run_by_label(cfg, lab) for lab in ("R5", "R1", "R2")}
    local_q = {
        q: {
            u
            for u, g in qrels.get(q, {}).items()
            if g == 2 and lang_bucket(unit_lang(u)) == "other"
        }
        for q in qids
    }
    local_q = {q: us for q, us in local_q.items() if us}
    per_query: list[dict[str, Any]] = []
    for q in sorted(local_q):
        row: dict[str, Any] = {"qid": q}
        for lab, name in names.items():
            top = set(runs[name].get(q, [])[:10]) if name in runs else None
            row[lab] = None if top is None else bool(local_q[q] & top)
        per_query.append(row)
    hits = {lab: sum(1 for r in per_query if r[lab]) for lab in names}
    return {
        "n_local_queries": len(local_q),
        "hits": hits,
        "per_query": per_query,
        "runs": {lab: n for lab, n in names.items() if n},
    }


def _language_aux_md(aux: dict[str, Any], plabel: str) -> list[str]:
    n = aux["n_local_queries"]
    lines = [
        "## 언어별 보조 값 (본문용, 표 아님)",
        "",
        f"- 등급 2 단위에 현지 언어(한국어·영어 외) 단위가 있는 {plabel} 질의: {n}건",
        f"- 그중 R5 가 현지 언어 등급 2 단위를 상위 10개에 넣은 질의: {aux['hits']['R5']}건"
        + (f" / {n}" if n else ""),
        "",
    ]
    if aux["per_query"]:
        lines += ["| qid | R1 상위 10 | R2 상위 10 | R5 상위 10 |", "|---|---|---|---|"]
        mark = {True: "포함", False: "없음", None: "–"}
        lines += [
            f"| {r['qid']} | {mark[r['R1']]} | {mark[r['R2']]} | {mark[r['R5']]} |"
            for r in aux["per_query"]
        ]
        lines.append("")
    return lines


def _t8(cfg, scores, queries, qids_all, no_rel) -> list[str]:
    """T8 (논문 [표 13]) — 집합별(search·meta·nonsearch·전체) R1·R2·R3·R5 nDCG@10·P@10, 기술 통계"""
    runs = [r for r in (_run_by_label(cfg, lab) for lab in T8_LABELS) if r and r in scores]
    head = " | ".join(f"{cfg.label(r)} {METRIC_LABELS[m]}" for r in runs for m in T8_METRICS)
    groups = {
        SUBSET_LABELS[g]: [q for q in qids_all if queries[q].support_group == g]
        for g in SUPPORT_GROUPS
    }
    groups["전체 (ALL45)"] = qids_all
    lines = [
        "## T8 분석 집합별 성능 (표준 질문 분류 확인)",
        "",
        f"| 집합 | n | {head} |",
        "|---|---:|" + "---:|" * (len(runs) * len(T8_METRICS)),
    ]
    for label, qs in groups.items():
        cells = [
            _f(_mean([scores[r][m][q] for q in qs])) if qs else "–"
            for r in runs
            for m in T8_METRICS
        ]
        lines.append(f"| {label} | {_n(len(qs))} | " + " | ".join(cells) + " |")
    excl = {
        SUBSET_LABELS[g]: sum(1 for q in no_rel if queries[q].support_group == g)
        for g in SUPPORT_GROUPS
    }
    lines += [
        "",
        "n 은 관련 단위(등급 ≥1)가 있는 질의 수. 제외된 질의: "
        + ", ".join(f"{k} {v}건" for k, v in excl.items())
        + ". 검정하지 않은 기술 통계다.",
        "",
    ]
    return lines
