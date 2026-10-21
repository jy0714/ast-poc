"""retrieve() 테스트 — 가짜 검색기로 RRF 값, 파트→단위 합치기, 재순위 후 꼬리, 단일 검색기 depth (지시서 §5.10)"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.index.case_index import CaseIndex
from src.rag import retrieval
from src.rag.retrieval import RetrievalConfig, retrieve, service_config
from src.units.schema import UnitPart
from src.utils.config import settings


def _idx(unit_of_row: list[str]) -> CaseIndex:
    """행별 unit_id만 의미 있는 색인 (점수는 테스트마다 주입)"""
    counts: dict[str, int] = {}
    parts = []
    for uid in unit_of_row:
        counts[uid] = counts.get(uid, 0) + 1
        parts.append(
            UnitPart(f"{uid}#{counts[uid]}", uid, "email", f"H {uid}", f"text {uid}", 3, {})
        )
    idx = CaseIndex("c", None, {}, parts)  # type: ignore[arg-type]
    idx.check_embedding = MagicMock()  # type: ignore[method-assign]
    return idx


def _inject(
    idx: CaseIndex, bm25: list[float] | None = None, dense: list[float] | None = None
) -> None:
    if bm25 is not None:
        idx.bm25_scores = MagicMock(return_value=np.array(bm25, dtype=float))  # type: ignore[method-assign]
    if dense is not None:
        idx.dense_scores = MagicMock(return_value=np.array(dense, dtype=np.float32))  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    emb = MagicMock()
    emb.get_dimension.return_value = 4
    emb.embed_query.return_value = [0.0, 0.0, 0.0, 1.0]
    monkeypatch.setattr(retrieval, "get_embedder", lambda model: emb)


def _fake_reranker(monkeypatch, scores_by_unit: dict[str, float]):
    rr = MagicMock()
    rr.n_truncated = 0

    def score(query, docs):
        rr.n_truncated += 1  # 호출 1회당 1건 잘렸다고 가정
        return [scores_by_unit[d.split()[1]] for d in docs]  # 입력 = "H {uid}\n\ntext {uid}"

    rr.score.side_effect = score
    import src.rag.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda backend=None, model_name=None: rr)
    return rr


class TestDefaults:
    def test_fixed_defaults(self):
        cfg = RetrievalConfig()
        assert (cfg.k_bm25, cfg.k_dense, cfg.rrf_k, cfg.rerank_candidates, cfg.depth) == (
            50,
            50,
            60,
            30,
            100,
        )
        assert cfg.profile == "hdr-qwen3" and cfg.reranker == "qwen3"

    def test_service_config_reads_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_backend", "bge")
        monkeypatch.setattr(settings, "service_index_profile", "nohdr-qwen3")
        cfg = service_config(rerank=False)
        assert cfg.reranker == "bge" and cfg.profile == "nohdr-qwen3" and cfg.rerank is False

    def test_both_off_rejected(self):
        with pytest.raises(ValueError):
            retrieve("c", "q", RetrievalConfig(use_bm25=False, use_dense=False), index=_idx(["a"]))


class TestPartsToUnits:
    def test_first_part_represents_unit_and_zero_bm25_dropped(self):
        idx = _idx(["a", "a", "b", "c"])
        _inject(idx, bm25=[0.5, 0.9, 0.7, 0.0])
        res = retrieve("c", "q", RetrievalConfig(use_dense=False, rerank=False), index=idx)
        assert [h.unit_id for h in res.hits] == ["a", "b"]  # c는 점수 0
        assert res.hits[0].best_part_id == "a#2"  # 단위 a의 첫 등장 파트 = 최고 점수 파트
        assert res.hits[0].stage_ranks == {"bm25": 1}

    def test_single_retriever_returns_depth_not_k(self):
        n = 120
        idx = _idx([f"u{i:03d}" for i in range(n)])
        _inject(idx, dense=[float(n - i) for i in range(n)])
        res = retrieve(
            "c", "q", RetrievalConfig(use_bm25=False, rerank=False, k_dense=5, depth=100), index=idx
        )
        assert len(res.hits) == 100
        assert res.hits[-1].unit_id == "u099"

    def test_filter_mask_applied(self):
        idx = _idx(["a", "b"])
        idx.parts[0].unit_type = "chat"
        idx._types = np.array(["chat", "email"], dtype=object)
        _inject(idx, bm25=[1.0, 0.5])
        res = retrieve(
            "c",
            "q",
            RetrievalConfig(use_dense=False, rerank=False),
            filters={"source_type": "email"},
            index=idx,
        )
        assert [h.unit_id for h in res.hits] == ["b"]


class TestRRF:
    def test_rrf_scores_and_order(self):
        # BM25 순위: a, b, c (d는 0점이라 제외) / Dense 순위: c, a, d, b (코사인은 음수도 순위에 든다)
        idx = _idx(["a", "b", "c", "d"])
        _inject(idx, bm25=[3.0, 2.0, 1.0, 0.0], dense=[0.8, 0.1, 0.9, 0.5])
        res = retrieve("c", "q", RetrievalConfig(rerank=False), index=idx)
        got = {h.unit_id: h.score for h in res.hits}
        assert got["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert got["c"] == pytest.approx(1 / 63 + 1 / 61)
        assert got["b"] == pytest.approx(1 / 62 + 1 / 64)
        assert got["d"] == pytest.approx(1 / 63)
        assert [h.unit_id for h in res.hits] == ["a", "c", "b", "d"]
        assert res.hits[1].stage_ranks == {"bm25": 3, "dense": 1, "rrf": 2}

    def test_tie_broken_by_best_rank_then_unit_id(self):
        # a: bm25 1위만 / b: dense 1위만 → 같은 점수, 같은 최고 순위 → unit_id
        idx = _idx(["b", "a"])
        _inject(idx, bm25=[0.0, 1.0], dense=[0.9, -0.5])
        res = retrieve("c", "q", RetrievalConfig(rerank=False, k_dense=1), index=idx)
        assert [h.unit_id for h in res.hits] == ["a", "b"]
        assert res.hits[0].score == res.hits[1].score

    def test_symmetric_rank_pairs_tie_exactly(self):
        # x: (1, 3), y: (3, 1) → 비트 단위로 같은 점수, 최고 순위 같음 → unit_id 순
        idx = _idx(["y", "m", "x"])
        _inject(idx, bm25=[1.0, 2.0, 3.0], dense=[3.0, 2.0, 1.0])
        res = retrieve("c", "q", RetrievalConfig(rerank=False), index=idx)
        assert res.hits[0].score == res.hits[1].score
        assert [h.unit_id for h in res.hits[:2]] == ["x", "y"]

    def test_representative_part_from_stronger_retriever(self):
        # 단위 a: BM25 대표 a#1(1위), Dense 대표 a#2(2위) → BM25 쪽 기여가 커서 a#1
        idx = _idx(["a", "a", "b"])
        _inject(idx, bm25=[5.0, 0.0, 1.0], dense=[0.0, 0.5, 0.9])
        res = retrieve("c", "q", RetrievalConfig(rerank=False), index=idx)
        assert res.hits[0].unit_id == "a" and res.hits[0].best_part_id == "a#1"

    def test_k_limits_fusion_input(self):
        idx = _idx([f"u{i}" for i in range(10)])
        _inject(idx, bm25=[float(10 - i) for i in range(10)], dense=[float(i) for i in range(10)])
        res = retrieve("c", "q", RetrievalConfig(rerank=False, k_bm25=2, k_dense=2), index=idx)
        assert {h.unit_id for h in res.hits} == {"u0", "u1", "u8", "u9"}


class TestRerank:
    def test_rerank_head_then_tail_in_rrf_order(self, monkeypatch):
        units = [f"u{i}" for i in range(6)]
        idx = _idx(units)
        _inject(idx, bm25=[6, 5, 4, 3, 2, 1], dense=[6, 5, 4, 3, 2, 1])
        rr = _fake_reranker(
            monkeypatch, {"u0": 0.1, "u1": 0.9, "u2": 0.5, "u3": 1.0, "u4": 1.0, "u5": 1.0}
        )
        res = retrieve("c", "q", RetrievalConfig(rerank_candidates=3), index=idx)
        # 상위 3(u0,u1,u2)만 재순위 → u1, u2, u0 / 꼬리 u3,u4,u5는 RRF 순서 그대로
        assert [h.unit_id for h in res.hits] == ["u1", "u2", "u0", "u3", "u4", "u5"]
        assert res.hits[0].stage_ranks["rerank"] == 1 and res.hits[0].stage_ranks["rrf"] == 2
        assert "rerank" not in res.hits[3].stage_ranks
        assert res.hits[0].score == pytest.approx(0.9)
        assert res.n_rerank_truncated == 1
        assert rr.score.call_args.args[0] == "q"
        assert "rerank" in res.timings_ms and "total" in res.timings_ms

    def test_rerank_tie_keeps_rrf_order(self, monkeypatch):
        idx = _idx(["a", "b", "c"])
        _inject(idx, bm25=[3, 2, 1], dense=[3, 2, 1])
        _fake_reranker(monkeypatch, {"a": 0.5, "b": 0.5, "c": 0.7})
        res = retrieve("c", "q", RetrievalConfig(), index=idx)
        assert [h.unit_id for h in res.hits] == ["c", "a", "b"]

    def test_rerank_input_uses_header_per_profile(self, monkeypatch):
        idx = _idx(["a"])
        _inject(idx, bm25=[1.0], dense=[1.0])
        seen = []
        rr = MagicMock(n_truncated=0)
        rr.score.side_effect = lambda q, docs: seen.extend(docs) or [0.5] * len(docs)
        import src.rag.reranker as reranker_mod

        monkeypatch.setattr(reranker_mod, "get_reranker", lambda backend=None, model_name=None: rr)
        retrieve("c", "q", RetrievalConfig(profile="hdr-qwen3"), index=idx)
        retrieve("c", "q", RetrievalConfig(profile="nohdr-qwen3"), index=idx)
        assert seen == ["H a\n\ntext a", "text a"]


class TestEndToEndFake:
    def test_fake_models_over_written_index(self, tmp_path, monkeypatch):
        """가짜 임베더·재순위기 + 실제 색인 파일로 전체 경로"""
        from src.embeddings.embedding_service import fake_embed
        from src.index import case_index
        from src.index.case_index import ProfileVectors, embed_input
        from src.rag import reranker as reranker_mod

        monkeypatch.setattr(settings, "index_dir", str(tmp_path))
        monkeypatch.setattr(settings, "ast_fake_models", True)
        monkeypatch.setattr(settings, "embed_dim", 64)
        monkeypatch.setattr(settings, "ollama_embed_model", "qwen3-embedding:4b-q8_0")
        monkeypatch.setattr(retrieval, "get_embedder", _real_get_embedder())
        reranker_mod.release_rerankers()
        case_index._cache.clear()
        texts = {
            "em:1": "선급금 송금 결재 누락",
            "em:2": "점심 메뉴 안내",
            "em:3": "supplier invoice payment",
        }
        parts = [UnitPart(f"{u}#1", u, "email", "[이메일]", t, 3, {}) for u, t in texts.items()]
        vecs = np.array([fake_embed(embed_input(p, True), 64) for p in parts])
        case_index.write_index(
            "c9", parts, {"hdr-qwen3": ProfileVectors(vecs, "qwen3-embedding:4b-q8_0")}
        )

        res = retrieve("c9", "선급금 결재")
        assert res.hits[0].unit_id == "em:1"
        assert res.hits[0].stage_ranks.get("rerank") == 1
        assert set(res.timings_ms) >= {"bm25", "embed_query", "dense", "rrf", "rerank", "total"}
        case_index._cache.clear()


def _real_get_embedder():
    """autouse 가짜 임베더를 되돌리는 실제 함수"""
    from src.embeddings.embedding_service import EmbeddingService

    cache: dict = {}

    def get(model: str):
        if model not in cache:
            cache[model] = EmbeddingService(model=model)
        return cache[model]

    return get
