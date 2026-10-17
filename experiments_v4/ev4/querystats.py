"""T0 질의 구성 표 → results/T0_queries.md (v4.2 Q2·Q3, 논문 [표 8])

- 행: 정확 일치·정황·관계·계 / 열: 평가용 수, 개발용 수, 현지어 증거 질의 수
- 현지어 증거 질의 수는 판정 전에는 `target_evidence_lang` 기준(예상), 판정 후에는 정답 기준
  (등급 2 단위 언어가 ko·en 이 아님, 평가용만)으로 한 열씩 낸다.
- 부가 표: query_lang 분포, source_question 별 질의 수. 질의 문장은 넣지 않는다(P0).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ev4.io import QUERY_TYPE_NAMES, QUERY_TYPES, ExperimentConfig, Query, query_warnings


def t0_lines(queries: list[Query], judged_local: dict[str, bool] | None = None) -> list[str]:
    """T0 마크다운 줄

    Args:
        queries: 전체 질의 (test·dev)
        judged_local: 판정 후 {test qid: 현지어 증거 여부} — 등급 2 단위가 있는 질의만. None 이면 판정 전
    """
    test = [q for q in queries if q.split == "test"]
    dev = [q for q in queries if q.split == "dev"]
    judged = judged_local is not None
    head = "| 유형 | 평가용 | 개발용 | 현지어 증거(예상) |" + (" 현지어 증거(정답) |" if judged else "")
    sep = "|---|---:|---:|---:|" + ("---:|" if judged else "")
    lines = ["## T0 질의 구성", "", head, sep]

    def _row(label: str, ts: list[Query], ds: list[Query]) -> str:
        exp = sum(1 for q in ts if q.expects_local_evidence)
        row = f"| {label} | {len(ts)} | {len(ds)} | {exp} |"
        if judged:
            row += f" {sum(1 for q in ts if judged_local.get(q.qid))} |"
        return row

    for t in QUERY_TYPES:
        lines.append(_row(f"{QUERY_TYPE_NAMES[t]} ({t})", [q for q in test if q.type == t],
                          [q for q in dev if q.type == t]))
    lines.append(_row("계", test, dev))
    lines += ["", "- 현지어 증거(예상): `target_evidence_lang` 이 ko·en 이 아닌 평가용 질의"]
    if judged:
        lines.append(f"- 현지어 증거(정답): 등급 2 단위 언어로 정한 증거 언어가 ko·en 이 아닌 평가용 질의 "
                     f"(등급 2 단위가 있는 질의 {len(judged_local)}건 기준)")

    lines += ["", "### 질의 언어(query_lang) 분포", "", "| query_lang | 평가용 | 개발용 |", "|---|---:|---:|"]
    ct, cd = Counter(q.query_lang for q in test), Counter(q.query_lang for q in dev)
    for lang in sorted(set(ct) | set(cd)):
        lines.append(f"| {lang} | {ct.get(lang, 0)} | {cd.get(lang, 0)} |")

    lines += ["", "### 감사인 질문(source_question)별 질의 수", "", "| source_question | 평가용 | 개발용 |",
              "|---|---:|---:|"]
    st = Counter(q.source_question or "(없음)" for q in test)
    sd = Counter(q.source_question or "(없음)" for q in dev)
    for k in sorted(set(st) | set(sd), key=_question_key):
        lines.append(f"| {k} | {st.get(k, 0)} | {sd.get(k, 0)} |")

    warns = query_warnings(queries)
    lines += ["", "### 입력 검증 경고", ""] + ([f"- {w}" for w in warns] or ["- 없음"])
    return [*lines, ""]


def _question_key(k: str) -> tuple:
    """'1.3' < '1.10' < '2.1' < '(없음)' 순 정렬"""
    if k == "(없음)":
        return (1, ())
    try:
        return (0, tuple(int(x) for x in k.split(".")))
    except ValueError:
        return (0, (float("inf"), k))


def write_t0(
    cfg: ExperimentConfig,
    queries: list[Query],
    judged_local: dict[str, bool] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """T0 를 results/T0_queries.md 에 쓰고 입력 검증 경고를 출력한다"""
    for w in query_warnings(queries):
        log(f"[경고] {w}")
    path = cfg.results_dir / "T0_queries.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    basis = "판정 후" if judged_local is not None else "판정 전"
    path.write_text("\n".join([f"# 질의 구성 ({basis})", "", *t0_lines(queries, judged_local)]) + "\n",
                    encoding="utf-8")
    log(f"[queries] test {sum(q.split == 'test' for q in queries)} · dev {sum(q.split == 'dev' for q in queries)}"
        f" → {path}")
    return path
