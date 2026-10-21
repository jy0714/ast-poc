"""설정 API — 임베딩 모델 변경 시 재인덱싱 필요 케이스 판정 (케이스 프로필 색인 기준)"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.api.routes.settings import _cases_needing_reindex
from src.cases.case_store import CaseStatus, CaseStore
from src.index import case_index
from src.index.case_index import ProfileVectors
from src.units.schema import UnitPart
from src.utils.config import settings


@pytest.fixture
def store(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        from src.db import database

        database.reset_globals()
        st = CaseStore(db_url=f"sqlite:///{Path(d) / 'test.db'}")
        monkeypatch.setattr("src.cases.case_store.CaseStore", lambda *a, **k: st)
        monkeypatch.setattr(settings, "index_dir", str(Path(d) / "index"))
        monkeypatch.setattr(settings, "service_index_profile", "hdr-qwen3")
        case_index._cache.clear()
        yield st
        case_index._cache.clear()
        database.reset_globals()


def _ready(store: CaseStore, name: str, model: str | None, dim: int = 4) -> str:
    cid = store.create(name).case_id
    store.update_status(cid, CaseStatus.INDEXING)
    store.update_status(cid, CaseStatus.READY)
    if model is not None:
        part = UnitPart("doc:a:p1#1", "doc:a:p1", "document", "[문서]", "본문", 1, {})
        case_index.write_index(
            cid, [part], {"hdr-qwen3": ProfileVectors(np.zeros((1, dim)), model)}
        )
    return cid


def test_reindex_detection(store):
    same = _ready(store, "같은 모델", "qwen3-embedding:4b-q8_0", 4)
    other = _ready(store, "다른 모델", "bge-m3", 4)
    v3 = _ready(store, "v3 케이스", None)
    store.create("인덱싱 전")  # CREATED는 대상 아님
    got = _cases_needing_reindex("qwen3-embedding:4b-q8_0", 4)
    assert same not in got
    assert set(got) == {other, v3}
    assert same in _cases_needing_reindex("qwen3-embedding:4b-q8_0", 2560)
