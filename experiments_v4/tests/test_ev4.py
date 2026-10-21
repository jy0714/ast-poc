"""experiments_v4 하네스 테스트 — 가짜 모델 selftest + 순수 함수

실행: `pytest experiments_v4/tests -q` (프로젝트 루트에서)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ev4.evaluate import evidence_lang  # noqa: E402
from ev4.io import ConfigError, Query, load_config, read_qrels, read_run, write_qrels, write_run  # noqa: E402
from ev4.pool import select_b_queries  # noqa: E402
from ev4.stats import bootstrap_ci, holm, quadratic_weighted_kappa  # noqa: E402


def test_selftest_full_flow(tmp_path):
    from ev4.selftest import EXPECTED, selftest

    out = selftest(root=tmp_path / "st", keep=True, log=lambda _m: None)
    assert set(EXPECTED) <= set(out["results"])
    assert out["failures"]["n_rows"] >= 1
    assert out["evaluation"]["n_queries"] == 8 and out["evaluation"]["n_queries_all"] == 13


def test_config_rejects_k_override(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "case_id: c\nruns:\n  R1: {profile: hdr-qwen3, use_bm25: true, use_dense: false, rerank: false, k_bm25: 10}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="k값"):
        load_config(root=tmp_path)


def test_config_rejects_unknown_run_in_comparisons(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "case_id: c\nruns:\n  R1: {profile: hdr-qwen3}\ncomparisons:\n  x: [[R1, R2]]\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="R2"):
        load_config(root=tmp_path)


def test_qrels_and_run_roundtrip(tmp_path):
    q = {"q1": {"a": 2, "b": 0}, "q2": {"c": 1}}
    write_qrels(tmp_path / "qrels.txt", q)
    assert read_qrels(tmp_path / "qrels.txt") == q
    write_run(tmp_path / "R1.trec", "R1", {"q1": ["b", "a", "x"]}, depth=2)
    assert read_run(tmp_path / "R1.trec") == {"q1": ["b", "a"]}


def test_b_queries_stratified():
    groups = ["search"] * 17 + ["meta"] * 3 + ["nonsearch"] * 25
    qs = [Query(f"q{i:02d}", "t", "none", "test", support_group=g) for i, g in enumerate(groups)]
    quota = {"search": 4, "meta": 1, "nonsearch": 6}
    chosen = select_b_queries(qs, quota, seed=1)
    by_group = {g: sum(1 for q in qs if q.qid in chosen and q.support_group == g) for g in quota}
    assert by_group == quota
    assert chosen == select_b_queries(qs, quota, seed=1)
    assert select_b_queries(qs[17:20], {"meta": 5}, seed=1) == ["q17", "q18", "q19"]  # 집합보다 많으면 전부


@pytest.mark.parametrize(("langs", "expected"), [
    ([], "n/a"), (["ko"], "ko"), (["ko", "en", "en"], "en"), (["ko", "en"], "ko"), (["en", "zh"], "other"),
])
def test_evidence_lang(langs, expected):
    assert evidence_lang(langs) == expected


def test_stats():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert quadratic_weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    m, lo, hi = bootstrap_ci([1.0, 0.0, 1.0, 1.0], n=200, seed=0)
    assert m == pytest.approx(0.75) and lo <= m <= hi
    assert math.isnan(bootstrap_ci([])[0])


def test_paired_stats():
    from ev4.stats import paired_bootstrap_ci, paired_randomization_p

    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    b = [x + 0.2 for x in a]
    d, lo, hi = paired_bootstrap_ci(a, b, n=500, seed=1)
    assert d == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    assert paired_randomization_p(a, a, n=200, seed=0) == pytest.approx(1.0)  # 차이 0 → p = 1
    # 모든 질의에서 같은 방향 차이 6개 → 양측 p ≈ 2/64
    assert paired_randomization_p(a, b, n=20000, seed=0) == pytest.approx(2 / 64, abs=0.01)
    assert paired_randomization_p(b, a, n=20000, seed=0) == pytest.approx(paired_randomization_p(a, b, n=20000, seed=0))


def test_ranx_fisher_is_two_sided():
    """E3 — ranx fisher 가 방향과 관계없이 같은 p 를 낸다 (|평균 차이| 비교 = 양측)"""
    import numpy as np
    from ranx.statistical_tests import fisher_randomization_test

    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3])
    b = a + np.array([0.3, 0.1, 0.2, 0.0, 0.1, 0.2, -0.05, 0.1])
    p_ab, _ = fisher_randomization_test(a, b, 4000, 1.0, 7)
    p_ba, _ = fisher_randomization_test(b, a, 4000, 1.0, 7)
    assert p_ab == pytest.approx(p_ba, abs=0.03)


def _qrow(qid: str, cls: str, group: str, qtype: str, **kw) -> dict:
    return {"qid": qid, "text": "t", "area": "1", "support_class": cls, "support_group": group, "type": qtype,
            "split": "test", **kw}


def test_query_fields_and_warnings(tmp_path):
    """v4.3 §1 입력 검증 — 분류 6개, 집합↔분류, search 만 유형, 나머지 none, 45개 경고"""
    from ev4.io import query_warnings, read_queries, write_jsonl

    rows = [_qrow("q1.1", "A1", "search", "semantic", source_question="1.1"), _qrow("q1.2", "B", "nonsearch", "none"),
            _qrow("q1.8", "C", "meta", "none"), _qrow("q2.1", "A2", "search", "relation")]
    write_jsonl(tmp_path / "q.jsonl", rows)
    qs = read_queries(tmp_path / "q.jsonl")
    assert [q.support_group for q in qs] == ["search", "nonsearch", "meta", "search"]
    assert qs[0].source_question == "1.1" and qs[0].area == "1"
    assert query_warnings(qs) == ["질의 4개 — 표준 질문 45개가 아님"]

    bad = [
        (_qrow("x", "F", "search", "exact"), "support_class"),
        (_qrow("x", "C", "search", "exact"), "support_group"),
        (_qrow("x", "A1", "search", "none"), "type"),
        (_qrow("x", "D", "nonsearch", "semantic"), "type"),
        (_qrow("x", "A1", "search", "exact", split="dev"), "split"),
    ]
    for row, frag in bad:
        write_jsonl(tmp_path / "bad.jsonl", [row])
        with pytest.raises(ConfigError, match=frag):
            read_queries(tmp_path / "bad.jsonl")
    write_jsonl(tmp_path / "dup.jsonl", [rows[0], rows[0]])
    with pytest.raises(ConfigError, match="중복"):
        read_queries(tmp_path / "dup.jsonl")


def test_t0_lines_has_no_query_text():
    from ev4.querystats import t0_lines

    qs = [Query("q1.1", "비밀 질의 문장", "exact", "test", area="1", support_class="A1", support_group="search"),
          Query("q1.2", "다른 질의", "none", "test", area="1", support_class="B", support_group="nonsearch"),
          Query("q2.1", "세 번째", "relation", "test", area="2", support_class="A2", support_group="search")]
    text = "\n".join(t0_lines(qs))
    assert "비밀 질의 문장" not in text and "다른 질의" not in text
    assert "| 1 | 1 | 0 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 0 |" in text
    assert "| 계 | 1 | 1 | 0 | 0 | 1 | 0 | 3 | 1 | 0 | 1 |" in text


def test_t0b_distribution():
    from ev4.querystats import t0b_lines

    qs = [Query("a", "t", "exact", "test", support_group="search"),
          Query("b", "t", "semantic", "test", support_group="search"),
          Query("c", "t", "none", "test", support_group="meta")]
    qrels = {"a": {f"u{i}": 2 if i < 3 else 1 for i in range(12)} | {"z": 0}, "b": {"u": 1}}
    text = "\n".join(t0b_lines(qs, qrels))
    # search: 등급≥1 [12, 1] → 평균 6.5 중앙값 6.5 최대 12, 등급 2 [3, 0], 0개 질의 0, 10개 초과 1
    assert "| S17 (search) | 2 | 6.5 | 6.5 | 12 | 1.5 | 1.5 | 3 | 0 | 1 |" in text
    assert "| meta | 1 | 0.0 | 0 | 0 | 0.0 | 0 | 0 | 1 | 0 |" in text


def test_subset_lock(tmp_path):
    (tmp_path / "config.yaml").write_text("case_id: c\nprimary_subset: search\nruns:\n  R1: {profile: hdr-qwen3}\n",
                                          encoding="utf-8")
    cfg = load_config(root=tmp_path)
    cfg.check_subset_lock()  # 기록 없음 → 통과, 기록하지 않음
    assert not cfg.subset_lock_path.exists()
    cfg.check_subset_lock(write=True)
    cfg.primary_subset = "meta"
    with pytest.raises(ConfigError, match="primary_subset"):
        cfg.check_subset_lock()
    (tmp_path / "config.yaml").write_text("case_id: c\nprimary_subset: all\nruns:\n  R1: {profile: hdr-qwen3}\n",
                                          encoding="utf-8")
    with pytest.raises(ConfigError, match="primary_subset"):
        load_config(root=tmp_path)


def test_standard_queries_file_passes_validation():
    """docs/v4/data/queries_45.jsonl (사건 정보 없는 표준 질문, P0 예외) — 45개·분류·유형 수"""
    from collections import Counter

    from ev4.io import query_warnings, read_queries

    path = Path(__file__).resolve().parents[2] / "docs" / "v4" / "data" / "queries_45.jsonl"
    if not path.exists():
        pytest.skip("표준 질문 파일 없음")
    qs = read_queries(path)
    assert query_warnings(qs) == []
    assert Counter(q.support_group for q in qs) == {"search": 17, "meta": 3, "nonsearch": 25}
    assert Counter(q.type for q in qs if q.support_group == "search") == {"exact": 2, "semantic": 9, "relation": 6}
    assert all(q.split == "test" and not q.target_evidence_lang for q in qs)
    assert Counter(q.support_class for q in qs if q.support_group == "search") == {"A1": 9, "A2": 8}  # v4.4 §3


def test_category_hints():
    from ev4.failures import category_hints

    h = category_hints({"R4": ">100", "R6": 3, "R7": 8, "R8": 15}, "ko", "en", 2, 100, 500.0, True)
    assert h == ["재순위 강등", "헤더 영향", "교차 언어", "긴 단위 희석", "판정 이견"]
    assert category_hints({"R4": 2, "R6": 20, "R7": 30, "R8": 11}, "ko", "ko", 1, 100, 500.0, False) == []
    assert category_hints({"R4": 50, "R8": 12}, "ko", "und", 1, 600, 500.0, False) == ["긴 단위 희석"]


def test_qrels_stats_counts_missing_queries():
    from ev4.judgments import qrels_stats

    st = qrels_stats({"q1": {"a": 2, "b": 1}, "q2": {"c": 1, "d": 0}}, {"q1", "q2", "q3"})
    assert (st["grade2_units"], st["grade1_units"], st["grade0_units"]) == (1, 2, 1)
    assert st["queries_without_grade2"] == ["q2", "q3"]


def test_criteria_doc():
    from ev4.pool import criteria_markdown

    md = criteria_markdown()
    assert "등급은 원문 기준으로 매긴다." in md and "번역은 참고용이다." in md
    assert "| 2 | 결정적 |" in md and "| `relation` | 관계 |" in md


def test_run_calls_only_service_retrieve(tmp_path, monkeypatch):
    """T1 — ev4.run 은 서비스의 src.rag.retrieval.retrieve 만 호출한다"""
    import ast

    import ev4.run as run_mod
    from ev4.io import load_config

    import src.rag.retrieval as retrieval

    assert run_mod.retrieve is retrieval.retrieve
    # run.py 가 src.rag 에서 가져오는 이름은 retrieve(검색)와 release_rerankers(VRAM 정리)뿐
    tree = ast.parse(Path(run_mod.__file__).read_text(encoding="utf-8"))
    imported = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                and n.module.startswith("src.rag") for a in n.names}
    assert imported == {("src.rag.retrieval", "retrieve"), ("src.rag.reranker", "release_rerankers")}

    calls = []

    class _Res:
        hits: list = []
        timings_ms = {"total": 1.0}
        n_rerank_truncated = 0

    def fake_retrieve(case_id, query, rc, filters, index=None):
        calls.append((case_id, query, type(rc).__name__, filters))
        return _Res()

    monkeypatch.setattr(run_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(run_mod.case_index, "load", lambda _c: object())
    monkeypatch.setattr(run_mod.ollama, "is_loaded", lambda _m: False)
    (tmp_path / "config.yaml").write_text(
        "case_id: c1\nruns:\n  R2: {profile: hdr-qwen3, use_bm25: true, use_dense: false, rerank: false}\n",
        encoding="utf-8")
    cfg = load_config(root=tmp_path)
    qs = [Query("q1", "질의1", "exact", "test"), Query("q2", "질의2", "semantic", "test")]
    run_mod.run_all(cfg, qs, ["R2"], "test", log=lambda _m: None)
    assert calls == [("c1", "질의1", "RetrievalConfig", None), ("c1", "질의2", "RetrievalConfig", None)]


def test_repo_config_has_five_runs_and_four_comparisons():
    """v4.4 §2: 실행 5개(논문 R1~R5), 사전 비교 4개, 풀 5개, hdr-bgem3 없음"""
    cfg = load_config()
    assert sorted(cfg.runs) == ["R2", "R4", "R6", "R7", "R8"]
    assert [cfg.label(r) for r in cfg.runs_by_label()] == ["R1", "R2", "R3", "R4", "R5"]
    assert [(cfg.label(a), cfg.label(b)) for _f, a, b in cfg.comparison_pairs()] == \
        [("R1", "R3"), ("R2", "R3"), ("R3", "R5"), ("R4", "R5")]
    assert sorted(cfg.pool_runs) == sorted(cfg.runs)
    assert {s["profile"] for s in cfg.runs.values()} == {"hdr-qwen3", "nohdr-qwen3"}
    assert cfg.latency_runs == ["R8"]


def test_config_rejects_label_for_unknown_run(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "case_id: c\nruns:\n  R2: {profile: hdr-qwen3, use_bm25: true, use_dense: false, rerank: false}\n"
        "labels: {R9: R5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="labels"):
        load_config(root=tmp_path)


def test_language_aux_counts_local_grade2_in_top10(tmp_path):
    from ev4.evaluate import language_aux

    (tmp_path / "config.yaml").write_text(
        "case_id: c\nruns:\n"
        "  R2: {profile: hdr-qwen3, use_bm25: true, use_dense: false, rerank: false}\n"
        "  R4: {profile: hdr-qwen3, use_bm25: false, use_dense: true, rerank: false}\n"
        "  R8: {profile: hdr-qwen3, use_bm25: true, use_dense: true, rerank: true}\n"
        "labels: {R2: R1, R4: R2, R8: R5}\n",
        encoding="utf-8",
    )
    cfg = load_config(root=tmp_path)
    lang = {"zh1": "zh", "ko1": "ko", "vi1": "vi"}.get
    qrels = {"q1": {"zh1": 2, "ko1": 2}, "q2": {"ko1": 2}, "q3": {"vi1": 2}}
    runs = {"R8": {"q1": ["zh1"], "q3": ["x"] * 10 + ["vi1"]}, "R2": {"q1": ["ko1"]}, "R4": {"q3": ["vi1"]}}
    aux = language_aux(cfg, qrels, runs, ["q1", "q2", "q3"], lambda u: lang(u) or "und")
    assert aux["n_local_queries"] == 2  # q1·q3 (q2 는 한국어만)
    assert aux["hits"] == {"R5": 1, "R1": 0, "R2": 1}
    assert {r["qid"]: (r["R1"], r["R2"], r["R5"]) for r in aux["per_query"]} == \
        {"q1": (False, False, True), "q3": (False, True, False)}


def test_gen_sentence_and_citation_rows():
    from ev4.gen import citation_rows, split_sentences

    ans = "첫 문장입니다 [출처 1][출처 2]. 둘째 [출처 2] 문장 [출처 2]!\n셋째 줄 인용 없음"
    assert split_sentences(ans) == ["첫 문장입니다 [출처 1][출처 2].", "둘째 [출처 2] 문장 [출처 2]!", "셋째 줄 인용 없음"]
    assert citation_rows(ans) == [(1, 1), (1, 2), (2, 2)]  # 같은 문장 안 같은 인용은 한 번
    assert citation_rows("") == []


def test_cohen_kappa():
    from ev4.stats import cohen_kappa

    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert math.isclose(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
    assert math.isnan(cohen_kappa([1, 1], [1, 1]))  # 기대 일치 1 → 정의되지 않음
