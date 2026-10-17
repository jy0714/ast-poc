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
    assert out["evaluation"]["n_queries"] == 10


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
    qs = [Query(f"q{i:02d}", "t", t, "test") for i, t in enumerate(["exact"] * 8 + ["semantic"] * 4 + ["relation"] * 1)]
    chosen = select_b_queries(qs, 0.25, seed=1)
    by_type = {t: sum(1 for q in qs if q.qid in chosen and q.type == t) for t in ("exact", "semantic", "relation")}
    assert by_type == {"exact": 2, "semantic": 1, "relation": 1}
    assert chosen == select_b_queries(qs, 0.25, seed=1)


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


def test_query_optional_fields_and_warnings(tmp_path):
    from ev4.io import query_warnings, read_queries, write_jsonl

    rows = [{"qid": f"q{i}", "text": "t", "type": "exact", "split": "test",
             "target_evidence_lang": "zh" if i < 2 else "ko", "source_question": "1.3"} for i in range(3)]
    rows.append({"qid": "d1", "text": "t", "type": "semantic", "split": "dev"})
    write_jsonl(tmp_path / "q.jsonl", rows)
    qs = read_queries(tmp_path / "q.jsonl")
    assert qs[0].target_evidence_lang == "zh" and qs[0].source_question == "1.3" and qs[0].expects_local_evidence
    assert qs[3].target_evidence_lang == "" and not qs[3].expects_local_evidence
    w = " / ".join(query_warnings(qs))
    for frag in ("test 질의 3개", "dev 질의 1개", "semantic", "relation", "현지어 증거 예상"):
        assert frag in w
    assert "test exact" in w  # 3개 < 20


def test_t0_lines_has_no_query_text():
    from ev4.querystats import t0_lines

    qs = [Query("q1", "비밀 질의 문장", "exact", "test", target_evidence_lang="zh", source_question="1.8"),
          Query("q2", "다른 질의", "relation", "test"), Query("d1", "개발 질의", "semantic", "dev")]
    text = "\n".join(t0_lines(qs, {"q1": True, "q2": False}))
    assert "비밀 질의 문장" not in text and "개발 질의" not in text
    assert "| 계 | 2 | 1 | 1 | 1 |" in text
    assert "| 1.8 | 1 | 0 |" in text


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
