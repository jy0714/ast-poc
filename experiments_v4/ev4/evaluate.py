"""평가 — ranx 지표·bootstrap CI·fisher 검정(Holm)·분해 → results/ (지시서 §6.6, §6.8 T2~T6, v4.2 E1~E8, v4.3)

- 분석 집합(v4.3 §3): 주 분석 = config `primary_subset`(search, S17). 사전 지정 비교 11개의 검정·Holm 보정과
  T2~T6 은 S17 로 한다. ALL45 는 평균·95% CI 만 낸다(T2 부가, T6 부가 행). T8 은 집합별 기술 통계.
- 평가 대상: qrels 에 등급 ≥1 단위가 하나라도 있는 질의(관련 단위 없는 질의는
  recall 이 정의되지 않으므로 제외하고 수를 표에 적는다).
- P@10(v4.3 §4): 상위 10개 중 등급 1 이상 단위 비율(ranx `precision@10`).
- `recall@10-l2` 는 등급 2 단위가 있는 질의만으로 계산한다(ranx 는 없는 질의를 0 으로 센다).
  증거 언어 분해(T5·T6 언어 행)도 등급 2 단위가 있는 질의만 쓴다(v4.2 J4).
- nDCG 는 ranx `ndcg`(선형 이득 1·2).
- 유의성: ranx.compare(stat_test="fisher", n_permutations=10000), 주 지표 nDCG@10,
  config `comparisons` 전체(11개)를 한 계열로 Holm 보정.
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
    QUERY_TYPE_NAMES,
    QUERY_TYPES,
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
LANG_METRIC = "recall@10"  # T5·T6 언어 행 지표 (v4.2 E6)
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000
SMALL_N = 5
SMALL_MARK = "†사례 수준"
NOHDR_RUNS = ["R1", "R3", "R5", "R7"]
HDR_RUNS = ["R2", "R4", "R6", "R8"]
BREAKDOWN_RUNS = ["R2", "R4", "R6", "R8"]
EXTRA_RUN = "R9"
METRIC_LABELS = {"ndcg@10": "nDCG@10", "precision@10": "P@10", "recall@10": "Recall@10",
                 "recall@10-l2": "Recall@10(등급 2)", "mrr@10": "MRR@10", "recall@100": "Recall@100"}
T2_METRICS = ["ndcg@10", "precision@10", "recall@10", "recall@10-l2", "mrr@10", "recall@100"]  # v4.3 §4 순서
T8_RUNS = ["R2", "R4", "R6", "R8"]
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
    """두 실행의 nDCG@10 fisher 무작위화 검정 p (ranx, 양측)"""
    from ranx import compare

    rep = compare(_ranx_qrels(qrels, qids), [_ranx_run(run_a, qids, "A"), _ranx_run(run_b, qids, "B")],
                  metrics=[PRIMARY], stat_test="fisher", n_permutations=N_PERMUTATIONS, max_p=1.0,
                  random_seed=seed, make_comparable=True)
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


def _summarize(scores: dict[str, dict[str, dict[str, float]]], qs: list[str], l2_qs: list[str],
               seed: int) -> dict[str, dict[str, tuple[float, float, float]]]:
    """{run: {metric: (mean, lo, hi)}} — qs 집합만 (recall@10-l2 는 l2_qs 만)"""
    return {
        name: {m: bootstrap_ci([s[m][q] for q in (l2_qs if m == L2_METRIC else qs)], n=N_BOOTSTRAP, seed=seed)
               for m in METRICS}
        for name, s in scores.items()
    }


def evaluate_all(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """T0b·T2~T6·T8·per_query.csv·comparisons.csv 생성 (주 분석 집합 = config primary_subset, v4.3 §3)

    Returns:
        {primary_subset, n_queries, n_l2, n_queries_all, n_l2_all, excluded,
         runs: {run: {metric: (mean, lo, hi)}}, runs_all: {...}, comparisons: [...]}
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
        raise ValueError(f"관련 단위(등급 ≥1)가 있는 {primary} 질의가 없습니다 — qrels 를 확인하세요")

    idx = case_index.load(cfg.case_id)

    def _lang(uid: str) -> str:
        parts = idx.unit_parts(uid)
        return str(parts[0].meta.get("lang", "und")) if parts else "und"

    ev_lang = {q: evidence_lang([_lang(u) for u, g in qrels[q].items() if g == 2]) for q in qids_all}
    cross = {q: ("n/a" if ev_lang[q] == "n/a" else ("yes" if lang_bucket(queries[q].query_lang) != ev_lang[q] else "no"))
             for q in qids_all}

    runs: dict[str, Run] = {}
    for name in cfg.runs:
        path = cfg.run_path(name, "test")
        if path.exists():
            runs[name] = read_run(path)
        else:
            log(f"[evaluate] {name} 실행 파일 없음 — 건너뜀 ({path})")
    scores = {name: per_query_scores(qrels, run, qids_all, l2_all, name) for name, run in runs.items()}
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
        comps.append({"family": fam, "A": a, "B": b, "comparison": f"{a}→{b}", "metric": PRIMARY,
                      "subset": primary, "n": len(qids), "mean_A": _mean(va), "mean_B": _mean(vb),
                      "diff": d, "ci_lo": lo, "ci_hi": hi,
                      "p": fisher_p(qrels, runs[a], runs[b], qids, cfg.seed),
                      "p_check": paired_randomization_p(va, vb, n=N_PERMUTATIONS, seed=cfg.seed)})
    for key in ("p", "p_check"):
        adj = holm({str(i): c[key] for i, c in enumerate(comps)})
        for i, c in enumerate(comps):
            c[f"{key}_holm"] = adj[str(i)]

    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_per_query(res_dir / "per_query.csv", scores, queries, qids_all, ev_lang, cross)
    comp_cols = ["family", "comparison", "A", "B", "metric", "subset", "n", "mean_A", "mean_B", "diff", "ci_lo",
                 "ci_hi", "p", "p_holm", "p_check", "p_check_holm"]
    with open(res_dir / "comparisons.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comp_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(comps)

    plabel = SUBSET_LABELS.get(primary, primary)
    n_primary = sum(1 for q in all_queries if q.support_group == primary)
    groups_type = {t: [q for q in qids if queries[q].type == t] for t in QUERY_TYPES}
    md = [
        "# 평가 결과",
        "",
        f"- 주 분석 집합 {plabel} (`primary_subset: {primary}`): 질의 {n_primary}건 중 평가 {len(qids)}건 · "
        f"등급 2 단위 있는 질의 {len(l2_qids)}건. 검정·Holm 보정·T2~T6 은 이 집합으로 한다",
        f"- ALL45: 질의 {len(queries)}건 중 평가 {len(qids_all)}건 (관련 단위 없는 질의 {len(no_rel)}건 제외"
        + (f": {', '.join(no_rel)}" if no_rel else "") + "). 기술 통계(평균·95% CI)만, 검정하지 않는다",
        f"- 실행 깊이 {cfg.depth} · bootstrap {N_BOOTSTRAP}회 95% CI(차이는 짝지은 bootstrap) · "
        f"fisher {N_PERMUTATIONS}회 양측 · Holm 보정 {len(comps)}개 비교(주 지표 {METRIC_LABELS[PRIMARY]})",
        f"- {SMALL_MARK}: 질의 {SMALL_N}개 미만 그룹 (값은 내되 일반화하지 않는다)",
        "",
    ]
    md += _t2(cfg, summary, len(qids), len(l2_qids), f"T2 검색 구성별 성능 — {plabel} (평균 [95% CI])")
    md += _t2(cfg, summary_all, len(qids_all), len(l2_all), "T2 부가 — ALL45 (평균 [95% CI], 검정 없음)")
    md += _t3(cfg, comps, scores, groups_type)
    md += _comparisons_table(comps)
    md += _t4(scores, qids, groups_type)
    md += _t5(scores, l2_qids, ev_lang, cross)
    md += _t6(scores, comps, qids, l2_qids, ev_lang, cross, qids_all)
    md += _t8(scores, queries, qids_all, no_rel)
    (res_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out = {
        "primary_subset": primary, "n_queries": len(qids), "n_l2": len(l2_qids),
        "n_queries_all": len(qids_all), "n_l2_all": len(l2_all), "excluded": no_rel,
        "runs": {r: {m: list(v) for m, v in s.items()} for r, s in summary.items()},
        "runs_all": {r: {m: list(v) for m, v in s.items()} for r, s in summary_all.items()},
        "comparisons": comps,
    }
    (res_dir / "evaluation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[evaluate] {plabel} {len(qids)}건 · ALL45 {len(qids_all)}건 · 실행 {len(runs)}개 · 비교 {len(comps)}개 "
        f"→ {res_dir / 'tables.md'}")
    return out


def _write_per_query(path, scores, queries, qids, ev_lang, cross) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "qid", "support_group", "support_class", "type", "query_lang", "evidence_lang",
                    "cross_lingual", *METRICS])
        for run, s in scores.items():
            for q in qids:
                qq = queries[q]
                w.writerow([run, q, qq.support_group, qq.support_class, qq.type, qq.query_lang, ev_lang[q], cross[q],
                            *[_f(s[m].get(q, math.nan), 4) if q in s[m] else "" for m in METRICS]])


# ─────────────────────────── 표 ───────────────────────────


def _t2(cfg, summary, n, n_l2, title: str) -> list[str]:
    """T2 (논문 [표 12]) — 헤더 없음 R1·R3·R5·R7 / 헤더 있음 R2·R4·R6·R8, 평균 [95% CI]"""
    heads = [METRIC_LABELS[m] + (f" (n={n_l2})" if m == L2_METRIC else "") for m in T2_METRICS]
    lines = [f"## {title}", "",
             "| 헤더 | 실행 | 구성 | " + " | ".join(heads) + " |", "|---|---|---|" + "---|" * len(heads)]
    for hdr, rs in (("없음", NOHDR_RUNS), ("있음", HDR_RUNS)):
        for r in rs:
            if r in summary:
                lines.append(f"| {hdr} | {r} | {_run_label(cfg, r)} | "
                             + " | ".join(_ci(summary[r][m]) for m in T2_METRICS) + " |")
    lines += ["", f"n = {n} (Recall@10(등급 2): 등급 2 단위가 있는 질의 n = {n_l2})", ""]
    return lines


def _type_diff(scores, a: str, b: str, qs: list[str], metric: str = PRIMARY) -> float:
    return _mean([scores[b][metric][q] - scores[a][metric][q] for q in qs])


def _t3(cfg, comps, scores, groups_type) -> list[str]:
    """T3 (논문 [표 15]) — 헤더 효과 R1→R2 … R7→R8, 유형별 차이는 기술 통계(검정 없음)"""
    pairs = cfg.comparisons.get("header") or list(zip(NOHDR_RUNS, HDR_RUNS, strict=True))
    tcols = [f"{t} 차이 (n={_n(len(groups_type[t]))})" for t in QUERY_TYPES]
    lines = ["## T3 헤더 효과 (nDCG@10, 헤더 있음 − 헤더 없음)", "",
             "| 비교 | 헤더 없음 | 헤더 있음 | 차이 [95% CI] | p(Holm) | " + " | ".join(tcols) + " |",
             "|---|---:|---:|---|---:|" + "---:|" * len(tcols)]
    for a, b in pairs:
        c = next((c for c in comps if c["A"] == a and c["B"] == b), None)
        if c is None:
            continue
        cells = [f"{_type_diff(scores, a, b, groups_type[t]):+.3f}" if groups_type[t] else "–" for t in QUERY_TYPES]
        lines.append(f"| {a}→{b} | {_f(c['mean_A'])} | {_f(c['mean_B'])} | "
                     f"{_dci((c['diff'], c['ci_lo'], c['ci_hi']))} | {_p(c['p_holm'])} | " + " | ".join(cells) + " |")
    lines += ["", "유형별 n: " + ", ".join(f"{t}({QUERY_TYPE_NAMES[t]}) {_n(len(groups_type[t]))}" for t in QUERY_TYPES),
              "유형별 차이는 검정하지 않은 기술 통계다.", ""]
    return lines


def _comparisons_table(comps) -> list[str]:
    """사전 지정 비교 전체 (E2) — comparisons.csv 와 같은 값"""
    lines = ["## 사전 지정 비교 (nDCG@10, 차이 = 대상 − 기준)", "",
             "| 계열 | 기준→대상 | 기준 | 대상 | 차이 [95% CI] | p(fisher) | p(Holm) | p(교차 확인) |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for c in comps:
        lines.append(f"| {c['family']} | {c['comparison']} | {_f(c['mean_A'])} | {_f(c['mean_B'])} | "
                     f"{_dci((c['diff'], c['ci_lo'], c['ci_hi']))} | {_p(c['p'])} | {_p(c['p_holm'])} | "
                     f"{_p(c['p_check'])} |")
    return [*lines, ""]


def _group_rows(scores, groups: dict[str, list[str]], runs: list[str], metric: str) -> list[str]:
    rows = []
    for g, qs in groups.items():
        cells = [_f(_mean([scores[r][metric][q] for q in qs])) if qs else "–" for r in runs]
        rows.append(f"| {g} | {_n(len(qs))} | " + " | ".join(cells) + " |")
    return rows


def _t4(scores, qids, groups_type) -> list[str]:
    """T4 (논문 [표 16]) — 유형 3개 + 전체, R2·R4·R6·R8 nDCG@10 (R9 부가)"""
    runs = [r for r in BREAKDOWN_RUNS if r in scores] + ([EXTRA_RUN] if EXTRA_RUN in scores else [])
    head = " | ".join(f"{r}{' (부가)' if r == EXTRA_RUN else ''}" for r in runs)
    groups = {f"{QUERY_TYPE_NAMES[t]} ({t})": groups_type[t] for t in QUERY_TYPES} | {"전체": qids}
    return ["## T4 질의 유형별 nDCG@10", "", f"| 유형 | n | {head} |", "|---|---:|" + "---:|" * len(runs),
            *_group_rows(scores, groups, runs, PRIMARY), ""]


def _lang_groups(l2_qids, ev_lang, cross) -> dict[str, list[str]]:
    groups = {f"증거 {name}": [q for q in l2_qids if ev_lang[q] == lg]
              for lg, name in (("ko", "한국어"), ("en", "영어"), ("other", "기타"))}
    groups["같은 언어 질의"] = [q for q in l2_qids if cross[q] == "no"]
    groups["교차 언어 질의"] = [q for q in l2_qids if cross[q] == "yes"]
    return groups


def _t5(scores, l2_qids, ev_lang, cross) -> list[str]:
    """T5 (논문 [표 17]) — 증거 언어·교차 언어별 Recall@10, 등급 2 단위가 없는 질의는 뺀다"""
    runs = [r for r in BREAKDOWN_RUNS if r in scores]
    return ["## T5 증거 언어별 Recall@10", "", "| 그룹 | n | " + " | ".join(runs) + " |",
            "|---|---:|" + "---:|" * len(runs),
            *_group_rows(scores, _lang_groups(l2_qids, ev_lang, cross), runs, LANG_METRIC),
            "", f"등급 2 단위가 있는 질의 {len(l2_qids)}건만 쓴다(증거 언어 = 등급 2 단위 언어, 목표 언어 지정 없음).", ""]


def _t6(scores, comps, qids, l2_qids, ev_lang, cross, qids_all) -> list[str]:
    """T6 (논문 [표 18]) — 초기 모델(R9) vs 재설계(R8), ALL45 행은 부가(검정 없음)"""
    if "R8" not in scores or "R9" not in scores:
        return ["## T6 모델 비교", "", "(R8·R9 실행이 없음)", ""]
    c = next((c for c in comps if (c["A"], c["B"]) == ("R9", "R8")), None)
    c = c or next((c for c in comps if {c["A"], c["B"]} == {"R8", "R9"}), None)
    lines = ["## T6 모델 비교 R9(초기) vs R8(재설계)", "",
             "| 지표 | n | R9 | R8 | 차이 (R8 − R9) |", "|---|---:|---:|---:|---|"]

    def _row(label: str, metric: str, qs: list[str], with_test: bool = False) -> str:
        r9 = _mean([scores["R9"][metric][q] for q in qs])
        r8 = _mean([scores["R8"][metric][q] for q in qs])
        diff = f"{r8 - r9:+.3f}" if qs else "–"
        if with_test and c is not None:
            sign = 1 if (c["A"], c["B"]) == ("R9", "R8") else -1
            ci = (sign * c["diff"], *sorted((sign * c["ci_lo"], sign * c["ci_hi"])))
            diff = f"{_dci(ci)} · p(Holm) {_p(c['p_holm'])}"
        return f"| {label} | {_n(len(qs))} | {_f(r9) if qs else '–'} | {_f(r8) if qs else '–'} | {diff} |"

    g = _lang_groups(l2_qids, ev_lang, cross)
    lines += [
        _row("nDCG@10 전체", PRIMARY, qids, with_test=True),
        _row("Recall@10 전체", LANG_METRIC, qids),
        _row("Recall@10 한국어 증거", LANG_METRIC, g["증거 한국어"]),
        _row("Recall@10 영어 증거", LANG_METRIC, g["증거 영어"]),
        _row("Recall@10 기타 언어 증거", LANG_METRIC, g["증거 기타"]),
        _row("Recall@10 교차 언어 질의", LANG_METRIC, g["교차 언어 질의"]),
        _row("nDCG@10 ALL45 (부가)", PRIMARY, qids_all),
        _row("Recall@10 ALL45 (부가)", LANG_METRIC, qids_all),
        "",
    ]
    return lines


def _t8(scores, queries, qids_all, no_rel) -> list[str]:
    """T8 (v4.3 신설) — 집합별(search·meta·nonsearch·전체) nDCG@10·P@10, 기술 통계"""
    runs = [r for r in T8_RUNS if r in scores]
    head = " | ".join(f"{r} {METRIC_LABELS[m]}" for r in runs for m in T8_METRICS)
    groups = {SUBSET_LABELS[g]: [q for q in qids_all if queries[q].support_group == g] for g in SUPPORT_GROUPS}
    groups["전체 (ALL45)"] = qids_all
    lines = ["## T8 분석 집합별 성능 (표준 질문 분류 확인)", "", f"| 집합 | n | {head} |",
             "|---|---:|" + "---:|" * (len(runs) * len(T8_METRICS))]
    for label, qs in groups.items():
        cells = [_f(_mean([scores[r][m][q] for q in qs])) if qs else "–" for r in runs for m in T8_METRICS]
        lines.append(f"| {label} | {_n(len(qs))} | " + " | ".join(cells) + " |")
    excl = {SUBSET_LABELS[g]: sum(1 for q in no_rel if queries[q].support_group == g) for g in SUPPORT_GROUPS}
    lines += ["", "n 은 관련 단위(등급 ≥1)가 있는 질의 수. 제외된 질의: "
              + ", ".join(f"{k} {v}건" for k, v in excl.items()) + ". 검정하지 않은 기술 통계다.", ""]
    return lines
