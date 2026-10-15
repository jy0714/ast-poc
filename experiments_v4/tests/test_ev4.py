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
