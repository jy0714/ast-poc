"""케이스 색인 테스트 — 저장·로드·검색·필터·삭제·차원 불일치·토크나이저 버전 불일치·원자적 쓰기 (지시서 §5.10)"""

import json

import numpy as np
import pytest

from src.embeddings.embedding_service import EmbeddingDimensionError, fake_embed
from src.index import case_index
from src.index.case_index import (
    IndexNotFoundError,
    IndexVersionError,
    ProfileVectors,
    embed_input,
    input_hash,
)
from src.units.schema import UnitPart
from src.utils.config import settings

DIM = 32


def _part(uid: str, k: int, utype: str, text: str, **meta) -> UnitPart:
    return UnitPart(
        part_id=f"{uid}#{k}",
        unit_id=uid,
        unit_type=utype,
        header=f"[{utype}] {meta.get('subject', '')}",
        text=text,
        n_tokens=len(text.split()),
        meta={"case_id": "c1", **meta},
    )


def sample_parts() -> list[UnitPart]:
    """이메일 2(하나는 파트 2개)·채팅 1·문서 1"""
    return [
        _part(
            "em:a",
            1,
            "email",
            "3월 정산 결재 누락 payment advance",
            date="2024-03-05T14:22:00+09:00",
            participants=["hong@x.com", "kim@x.com"],
            sender="hong@x.com",
            sender_name="홍길동",
            subject="정산",
        ),
        _part(
            "em:a",
            2,
            "email",
            "두 번째 파트 supplier invoice",
            date="2024-03-05T14:22:00+09:00",
            participants=["hong@x.com", "kim@x.com"],
            sender="hong@x.com",
            sender_name="홍길동",
            subject="정산",
        ),
        _part(
            "em:b",
            1,
            "email",
            "점심 메뉴 lunch menu",
            date="2024-05-01T12:00:00+09:00",
            participants=["admin@x.com"],
            sender="admin@x.com",
            subject="메뉴",
        ),
        _part(
            "ch:c:1",
            1,
            "chat",
            "10:00 이영희: 선급금 송금 완료",
            date="2024-03-01T10:00:00+09:00",
            date_end="2024-03-10T11:00:00+09:00",
            participants=["이영희", "김철수"],
        ),
        _part(
            "doc:d:p1",
            1,
            "document",
            "안전 교육 safety training",
            author="박민수",
            filename="교육.pdf",
        ),
    ]


def fake_vectors(parts: list[UnitPart], header: bool = True) -> np.ndarray:
    return np.array([fake_embed(embed_input(p, header), DIM) for p in parts], dtype=np.float32)


@pytest.fixture(autouse=True)
def _tmp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", str(tmp_path / "index"))
    monkeypatch.setattr(settings, "ast_fake_models", True)
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr(settings, "ollama_embed_model", "qwen3-embedding:4b-q8_0")
    case_index._cache.clear()
    yield
    case_index._cache.clear()


def _write(parts=None, profiles=("hdr-qwen3",)):
    parts = parts or sample_parts()
    pv = {
        p: ProfileVectors(
            fake_vectors(parts, case_index.profile_spec(p).header),
            case_index.profile_spec(p).embed_model,
        )
        for p in profiles
    }
    return case_index.write_index("c1", parts, pv, build_stats={"counts": {"email": 2}})


class TestWriteLoad:
    def test_layout_and_manifest(self):
        root = _write()
        assert (root / "units.jsonl").exists()
        assert (root / "bm25" / "hdr.pkl").exists() and (root / "bm25" / "nohdr.pkl").exists()
        meta = json.loads(
            (root / "profiles" / "hdr-qwen3" / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["header"] is True and meta["embed_dim"] == DIM and meta["normalized"] is True
        dense = np.load(root / "profiles" / "hdr-qwen3" / "dense.npy")
        assert dense.dtype == np.float16 and dense.shape == (5, DIM)
        man = case_index.read_manifest("c1")
        assert man["n_units"] == 4 and man["n_parts"] == 5
        assert man["build_stats"] == {"counts": {"email": 2}}

    def test_load_roundtrip(self):
        _write()
        idx = case_index.load("c1")
        assert idx.n_parts == 5 and idx.n_units == 4
        assert idx.unit_rows["em:a"] == [0, 1]
        assert idx.part("em:a#2").text.startswith("두 번째")
        assert "supplier" in idx.unit_text("em:a")
        assert idx.profiles() == ["hdr-qwen3"]

    def test_cache_and_invalidation_on_write(self):
        _write()
        a = case_index.load("c1")
        assert case_index.load("c1") is a
        _write(sample_parts()[:3])
        b = case_index.load("c1")
        assert b is not a and b.n_parts == 3

    def test_missing_index(self):
        with pytest.raises(IndexNotFoundError):
            case_index.load("nope")
        assert case_index.exists("nope") is False

    def test_row_count_mismatch_rejected(self):
        parts = sample_parts()
        with pytest.raises(ValueError):
            case_index.write_index(
                "c1", parts, {"hdr-qwen3": ProfileVectors(np.zeros((2, DIM)), "m")}
            )
        # 실패한 쓰기는 임시 폴더를 남기지 않는다
        assert not list(case_index.index_root().glob(".tmp_*"))


class TestAtomicity:
    def test_failed_write_keeps_previous(self, monkeypatch):
        _write()

        def boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(case_index, "_write_profile_dir", boom)
        with pytest.raises(RuntimeError):
            _write(sample_parts()[:2])
        case_index._cache.clear()
        assert case_index.load("c1").n_parts == 5
        assert not list(case_index.index_root().glob(".tmp_*"))

    def test_recover_old_after_interrupted_swap(self):
        root = _write()
        root.rename(root.parent / "c1.old")  # 교체 도중 종료된 상태
        assert case_index.exists("c1")
        assert case_index.load("c1").n_parts == 5


class TestVersions:
    def test_tokenizer_version_mismatch(self):
        root = _write()
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        man["tokenizer_version"] = "script-v0"
        (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        case_index._cache.clear()
        with pytest.raises(IndexVersionError, match="재색인"):
            case_index.load("c1")

    def test_bm25_file_version_mismatch(self, monkeypatch):
        _write()
        from src.index import bm25 as bm25_mod

        monkeypatch.setattr(bm25_mod, "TOKENIZER_VERSION", "script-v9")
        idx = case_index.load("c1")
        with pytest.raises(IndexVersionError):
            idx.bm25(True)

    def test_embedding_model_or_dim_mismatch(self):
        _write()
        idx = case_index.load("c1")
        idx.check_embedding("hdr-qwen3", "qwen3-embedding:4b-q8_0", DIM)
        with pytest.raises(EmbeddingDimensionError):
            idx.check_embedding("hdr-qwen3", "bge-m3", DIM)
        with pytest.raises(EmbeddingDimensionError):
            idx.check_embedding("hdr-qwen3", "qwen3-embedding:4b-q8_0", 1024)
        with pytest.raises(EmbeddingDimensionError):
            idx.dense_scores(np.ones(DIM + 1), "hdr-qwen3")


class TestSearch:
    def test_bm25_scores_header_variants(self):
        _write()
        idx = case_index.load("c1")
        s = idx.bm25_scores("payment", header=True)
        assert s.shape == (5,) and s[0] > 0 and s[2] == 0
        # 헤더에만 있는 단어는 nohdr 색인에서 0
        assert idx.bm25_scores("메뉴", header=True)[2] > 0
        assert idx.bm25_scores("[email]", header=False).sum() == 0

    def test_dense_scores_cosine(self):
        _write()
        idx = case_index.load("c1")
        q = fake_embed(embed_input(idx.parts[2], True), DIM)
        s = idx.dense_scores(q, "hdr-qwen3")
        assert int(np.argmax(s)) == 2
        assert s[2] == pytest.approx(1.0, abs=1e-2)  # float16 저장 오차


class TestFilters:
    def test_source_type_and_teams_chat_alias(self):
        _write()
        idx = case_index.load("c1")
        assert idx.filter_mask({"source_type": "teams_chat"}).tolist() == [
            False,
            False,
            False,
            True,
            False,
        ]
        assert idx.filter_mask({"source_type": ["email", "document"]}).sum() == 4
        assert idx.filter_mask(None) is None

    def test_date_filters(self):
        _write()
        idx = case_index.load("c1")
        m = idx.filter_mask({"_date_gte": "2024-03-01", "_date_lte": "2024-03-31"})
        assert m.tolist() == [True, True, False, True, False]  # 날짜 없는 문서는 제외
        # 채팅 세션 구간 [03-01, 03-10]이 03-08을 포함
        assert idx.filter_mask({"_date_eq": "2024-03-08"}).tolist() == [
            False,
            False,
            False,
            True,
            False,
        ]

    def test_participants_partial_name_or_address(self):
        _write()
        idx = case_index.load("c1")
        assert idx.filter_mask({"participants": ["홍길동"]}).tolist() == [
            True,
            True,
            False,
            False,
            False,
        ]
        assert idx.filter_mask({"participants": ["KIM@x.com", "이영희"]}).tolist() == [
            True,
            True,
            False,
            True,
            False,
        ]

    def test_author_and_generic(self):
        _write()
        idx = case_index.load("c1")
        assert idx.filter_mask({"author": "박민수"}).tolist() == [False] * 4 + [True]
        assert idx.filter_mask({"filename": "교육.pdf"}).sum() == 1


class TestProfilesAndReuse:
    def test_write_profile_adds(self):
        _write()
        parts = case_index.load("c1").parts
        case_index.write_profile(
            "c1",
            "nohdr-qwen3",
            ProfileVectors(fake_vectors(parts, False), "qwen3-embedding:4b-q8_0"),
        )
        idx = case_index.load("c1")
        assert idx.profiles() == ["hdr-qwen3", "nohdr-qwen3"]
        assert idx.profile_meta("nohdr-qwen3")["header"] is False

    def test_reusable_vectors_keyed_by_input(self):
        _write()
        reuse = case_index.reusable_vectors("c1", "hdr-qwen3")
        p = sample_parts()[0]
        assert (p.part_id, input_hash(embed_input(p, True))) in reuse
        changed = sample_parts()[0]
        changed.text += " 수정"
        assert (changed.part_id, input_hash(embed_input(changed, True))) not in reuse

    def test_rewrite_carries_experiment_profile_when_parts_unchanged(self):
        _write(profiles=("hdr-qwen3", "nohdr-qwen3"))
        _write(profiles=("hdr-qwen3",))  # 서비스 재인덱싱 (같은 파트)
        assert case_index.load("c1").profiles() == ["hdr-qwen3", "nohdr-qwen3"]
        _write(sample_parts() + [_part("em:z", 1, "email", "new mail")], profiles=("hdr-qwen3",))
        assert case_index.load("c1").profiles() == ["hdr-qwen3"]  # 새 파트가 있으면 버림


class TestDelete:
    def test_delete(self):
        _write()
        case_index.load("c1")
        assert case_index.delete("c1") is True
        assert not case_index.exists("c1")
        assert case_index.delete("c1") is False
