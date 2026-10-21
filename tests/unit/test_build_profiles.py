"""scripts/build_profiles.py — 실험 프로필 추가 (가짜 모델)"""

import json

import numpy as np
import pytest

from scripts.build_profiles import ProfileBuildError, build_profile
from src.embeddings.embedding_service import EmbeddingService, fake_embed
from src.index import case_index
from src.index.case_index import ProfileVectors, embed_input
from src.units.schema import UnitPart
from src.utils.config import settings

DIM = 16


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ast_fake_models", True)
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "index_dir", str(tmp_path / "index"))
    monkeypatch.setattr(settings, "ollama_embed_model", "qwen3-embedding:4b-q8_0")
    case_index._cache.clear()
    parts = [UnitPart(f"em:{i}#1", f"em:{i}", "email", f"[이메일] 제목{i}", f"본문 {i} 내용", 3, {})
             for i in range(5)]
    vecs = np.array([fake_embed(embed_input(p, True), DIM) for p in parts])
    case_index.write_index("c1", parts, {"hdr-qwen3": ProfileVectors(vecs, "qwen3-embedding:4b-q8_0")})
    yield
    case_index._cache.clear()


def test_adds_nohdr_and_bge_profiles():
    info = build_profile("c1", "nohdr-qwen3", batch=2)
    assert info == {**info, "n_parts": 5, "embedded": 5, "reused": 0}
    build_profile("c1", "hdr-bgem3", batch=2)
    idx = case_index.load("c1")
    assert idx.profiles() == ["hdr-bgem3", "hdr-qwen3", "nohdr-qwen3"]
    assert idx.profile_meta("hdr-bgem3")["embed_model"] == "bge-m3"
    # nohdr는 본문만 임베딩
    expect = fake_embed(idx.parts[0].text, DIM)
    assert np.allclose(idx.dense("nohdr-qwen3")[0].astype(np.float32), expect, atol=1e-2)


def test_rebuild_reuses_existing_vectors():
    build_profile("c1", "nohdr-qwen3", batch=2)
    info = build_profile("c1", "nohdr-qwen3", batch=2)
    assert info["embedded"] == 0 and info["reused"] == 5


def test_failure_does_not_write_profile():
    class Failing(EmbeddingService):
        def embed_texts(self, texts, cancel_event=None):
            out = super().embed_texts(texts, cancel_event=cancel_event)
            self._last_failed_indices = [0]
            return out

    with pytest.raises(ProfileBuildError):
        build_profile("c1", "nohdr-qwen3", batch=10, embedder=Failing(model="qwen3-embedding:4b-q8_0"))
    assert case_index.load("c1").profiles() == ["hdr-qwen3"]


def test_failure_keeps_checkpoint_and_retry_reuses_it(tmp_path):
    """실패해도 성공한 벡터는 체크포인트에 남고, 재실행은 실패 파트만 임베딩한 뒤 체크포인트를 지운다"""
    from src.index.embed_cache import cache_dir

    class Failing(EmbeddingService):
        def embed_texts(self, texts, cancel_event=None):
            out = super().embed_texts(texts, cancel_event=cancel_event)
            self._last_failed_indices = [0]
            return out

    with pytest.raises(ProfileBuildError):
        build_profile("c1", "nohdr-qwen3", batch=10, embedder=Failing(model="qwen3-embedding:4b-q8_0"))
    assert list(cache_dir("c1", "nohdr-qwen3").glob("shard_*.npz"))

    progress = tmp_path / "p.json"
    info = build_profile("c1", "nohdr-qwen3", batch=10, progress_path=progress)
    assert info["embedded"] == 1 and info["reused"] == 4
    assert not cache_dir("c1", "nohdr-qwen3").exists()
    assert json.loads(progress.read_text(encoding="utf-8"))["done"] == 1


def test_empty_body_gets_zero_vector():
    """본문 없는 메일(헤더만) → nohdr 입력이 빈 문자열 → 실패가 아니라 영벡터 행"""
    parts = [UnitPart("em:a#1", "em:a", "email", "[이메일] 제목a", "", 0, {}),
             UnitPart("em:b#1", "em:b", "email", "[이메일] 제목b", "본문 b", 2, {})]
    vecs = np.array([fake_embed(embed_input(p, True), DIM) for p in parts])
    case_index.write_index("c2", parts, {"hdr-qwen3": ProfileVectors(vecs, "qwen3-embedding:4b-q8_0")})
    info = build_profile("c2", "nohdr-qwen3", batch=4)
    assert info["empty_input"] == 1 and info["embedded"] == 1
    mat = case_index.load("c2").dense("nohdr-qwen3").astype(np.float32)
    assert not mat[0].any() and mat[1].any()
