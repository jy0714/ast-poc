"""T0 질의 구성 표 → results/T0_queries.md, T0b 질의별 관련 단위 분포 → results/T0b_relevant.md (v4.3 §4, §6)

- T0([표 8]): 행 영역 1~7·계 / 열 부록 A 분류(A1·A2·C·D·B·E) 수, search 집합의 유형별 수.
  부록 A 표와 같은 값이 나와야 한다(검증용).
- T0b(판정 후): 집합별(S17·meta·nonsearch) 등급 1 이상·등급 2 단위 수의 평균·중앙값·최대,
  관련 단위가 0개인 질의 수, 10개를 넘는 질의 수. Recall@10 해석용.
- 질의 문장은 넣지 않는다(P0).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from pathlib import Path

from ev4.io import (
    AREAS,
    QUERY_TYPES,
    SUPPORT_CLASSES,
    SUPPORT_GROUPS,
    ExperimentConfig,
    Qrels,
    Query,
    query_warnings,
)

GROUP_LABELS = {"search": "S17 (search)", "meta": "meta", "nonsearch": "nonsearch"}
MANY_RELEVANT = 10  # 관련 단위가 이 수를 넘으면 Recall@10 이 1 에 닿을 수 없다


def t0_lines(queries: list[Query]) -> list[str]:
    """T0 마크다운 줄 (영역 × 분류·유형 건수)"""
    head = (
        "| 영역 | "
        + " | ".join(SUPPORT_CLASSES)
        + " | 계 | "
        + " | ".join(f"search {t}" for t in QUERY_TYPES)
        + " |"
    )
    sep = "|---|" + "---:|" * (len(SUPPORT_CLASSES) + 1 + len(QUERY_TYPES))
    lines = ["## T0 질의 구성 (부록 A 영역 × 분류)", "", head, sep]

    def _row(label: str, qs: list[Query]) -> str:
        cls = [sum(1 for q in qs if q.support_class == c) for c in SUPPORT_CLASSES]
        typ = [
            sum(1 for q in qs if q.support_group == "search" and q.type == t) for t in QUERY_TYPES
        ]
        return (
            f"| {label} | "
            + " | ".join(map(str, cls))
            + f" | {len(qs)} | "
            + " | ".join(map(str, typ))
            + " |"
        )

    areas = list(AREAS) + sorted({q.area for q in queries} - set(AREAS))
    for a in areas:
        lines.append(_row(a or "(없음)", [q for q in queries if q.area == a]))
    lines.append(_row("계", queries))

    lines += ["", "| 분석 집합 | 질의 수 |", "|---|---:|"]
    lines += [
        f"| {GROUP_LABELS[g]} | {sum(1 for q in queries if q.support_group == g)} |"
        for g in SUPPORT_GROUPS
    ]
    lines.append(f"| 전체 (ALL45) | {len(queries)} |")

    warns = query_warnings(queries)
    lines += ["", "### 입력 검증 경고", ""] + ([f"- {w}" for w in warns] or ["- 없음"])
    return [*lines, ""]


def write_t0(
    cfg: ExperimentConfig, queries: list[Query], log: Callable[[str], None] = print
) -> Path:
    """T0 를 results/T0_queries.md 에 쓰고 입력 검증 경고를 출력한다"""
    for w in query_warnings(queries):
        log(f"[경고] {w}")
    path = cfg.results_dir / "T0_queries.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["# 질의 구성", "", *t0_lines(queries)]) + "\n", encoding="utf-8")
    counts = " · ".join(
        f"{g} {sum(1 for q in queries if q.support_group == g)}" for g in SUPPORT_GROUPS
    )
    log(f"[queries] 질의 {len(queries)}건 ({counts}) → {path}")
    return path


def _dist(values: list[int]) -> str:
    if not values:
        return "– | – | –"
    return f"{statistics.mean(values):.1f} | {statistics.median(values):g} | {max(values)}"


def t0b_lines(queries: list[Query], qrels: Qrels) -> list[str]:
    """T0b 마크다운 줄 — 판정 기록이 없는 질의는 관련 단위 0개로 센다"""
    lines = [
        "## T0b 질의별 관련 단위 분포 (판정 후)",
        "",
        "| 집합 | n | 등급≥1 평균 | 중앙값 | 최대 | 등급 2 평균 | 중앙값 | 최대 | 관련 0개 질의 | "
        f"관련 {MANY_RELEVANT}개 초과 질의 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [
        (GROUP_LABELS[g], [q for q in queries if q.support_group == g]) for g in SUPPORT_GROUPS
    ]
    for label, qs in [*groups, ("전체 (ALL45)", queries)]:
        rel = [sum(1 for g in qrels.get(q.qid, {}).values() if g >= 1) for q in qs]
        l2 = [sum(1 for g in qrels.get(q.qid, {}).values() if g == 2) for q in qs]
        lines.append(
            f"| {label} | {len(qs)} | {_dist(rel)} | {_dist(l2)} | {sum(1 for n in rel if n == 0)} | "
            f"{sum(1 for n in rel if n > MANY_RELEVANT)} |"
        )
    lines += [
        "",
        f"관련 단위가 {MANY_RELEVANT}개를 넘는 질의는 Recall@10 이 1 에 닿을 수 없다 — P@10 과 함께 읽는다.",
        "관련 단위가 0개인 질의는 평가(T2~T8)에서 뺀다.",
        "",
    ]
    return lines


def write_t0b(cfg: ExperimentConfig, queries: list[Query], qrels: Qrels) -> Path:
    """T0b 를 results/T0b_relevant.md 에 쓴다"""
    path = cfg.results_dir / "T0b_relevant.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(["# 질의별 관련 단위 분포", "", *t0b_lines(queries, qrels)]) + "\n",
        encoding="utf-8",
    )
    return path
