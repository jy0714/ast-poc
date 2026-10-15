"""케이스 폴더 색인 — 저장·로드·검색·필터·삭제·검증 (지시서 §5.5)

```
data/index/{case_id}/
├─ manifest.json            # schema_version, tokenizer_version, unit_rules_version, n_units, n_parts,
│                           # created_at, source_files, build_stats
├─ units.jsonl              # UnitPart 한 줄씩 (행 순서 = 행렬 행 순서)
├─ bm25/hdr.pkl, nohdr.pkl  # BM25 색인 + tokenizer_version
└─ profiles/{profile}/
   ├─ meta.json             # profile, header, embed_model, embed_dim, normalized, created_at
   └─ dense.npy             # float16 [n_parts, dim]
```

- ChromaDB 단일 공유 컬렉션 + case_id 필터를 대체한다. 케이스마다 폴더가 따로라 격리가 자명하다.
- Dense 검색은 정확 탐색(코사인 = 정규화 벡터 내적)이다. 3만 파트 × 2,560 × 2바이트 ≈ 150MB.
- 쓰기는 임시 폴더에 전부 쓴 뒤 폴더를 교체한다. 강제 종료돼도 반쯤 쓰인 색인이 남지 않는다
  (교체 도중 종료되면 `{case_id}.old`가 남고, 다음 로드 때 되돌린다).
- 로드한 색인은 최근 2개 케이스까지 캐시한다. 쓰기·삭제 시 해당 케이스 캐시를 비운다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedding_service import EmbeddingDimensionError
from src.index.bm25 import BM25Index, TokenizerVersionError
from src.text.bm25_tokenizer import TOKENIZER_VERSION
from src.units.schema import UNIT_RULES_VERSION, UnitPart
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "case-index-v1"
PROFILE_NAMES = ("hdr-qwen3", "nohdr-qwen3", "hdr-bgem3")
BGE_M3_MODEL = "bge-m3"

# 프론트엔드·질의 파서가 쓰는 source_type 이름 → 단위 유형
_SOURCE_TYPE_TO_UNIT = {"teams_chat": "chat", "chat": "chat", "email": "email",
                        "document": "document", "attachment": "attachment"}
_DENSE_BLOCK_ROWS = 8192
_CACHE_SIZE = 2


class IndexNotFoundError(Exception):
    """케이스 색인이 없음 (인덱싱 전이거나 v4 이전 Chroma 색인만 있음)"""


class IndexVersionError(Exception):
    """색인 스키마·토크나이저 버전이 현재 코드와 다름 — 재색인 필요"""


@dataclass(frozen=True)
class ProfileSpec:
    """색인 프로필 — 헤더 사용 여부와 임베딩 모델"""

    name: str
    header: bool
    embed_model: str


def profile_spec(name: str) -> ProfileSpec:
    """프로필 이름 → 사양

    qwen3 프로필은 서비스 임베딩 모델(OLLAMA_EMBED_MODEL)을, hdr-bgem3는 bge-m3를 쓴다.

    Raises:
        ValueError: 알 수 없는 프로필
    """
    if name == "hdr-qwen3":
        return ProfileSpec(name, True, settings.ollama_embed_model)
    if name == "nohdr-qwen3":
        return ProfileSpec(name, False, settings.ollama_embed_model)
    if name == "hdr-bgem3":
        return ProfileSpec(name, True, BGE_M3_MODEL)
    raise ValueError(f"알 수 없는 색인 프로필: {name} (가능: {', '.join(PROFILE_NAMES)})")


def index_root() -> Path:
    """색인 루트 (INDEX_DIR, 상대 경로면 프로젝트 루트 기준)"""
    root = Path(settings.index_dir)
    return root if root.is_absolute() else settings.project_root / root


def case_dir(case_id: str) -> Path:
    """케이스 색인 폴더"""
    return index_root() / case_id


def embed_input(part: UnitPart, header: bool) -> str:
    """임베딩·재순위 입력 텍스트 (헤더 on: header + 빈 줄 + text)"""
    return part.full_text(with_header=header)


def input_hash(text: str) -> str:
    """임베딩 입력 해시 — 증분 색인에서 같은 입력의 벡터를 재사용할 때 키로 쓴다"""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


# ─────────────────────────── 로드된 색인 ───────────────────────────


class CaseIndex:
    """로드된 케이스 색인 (읽기 전용)

    Attributes:
        case_id: 케이스 ID
        root: 색인 폴더
        manifest: manifest.json 내용
        parts: 파트 목록 (행 순서)
        unit_ids: 행별 unit_id
    """

    def __init__(self, case_id: str, root: Path, manifest: dict[str, Any], parts: list[UnitPart]) -> None:
        """색인 객체 구성 (CaseIndex.load로 만든다)"""
        self.case_id = case_id
        self.root = root
        self.manifest = manifest
        self.parts = parts
        self.unit_ids = [p.unit_id for p in parts]
        self.unit_rows: dict[str, list[int]] = {}
        for i, uid in enumerate(self.unit_ids):
            self.unit_rows.setdefault(uid, []).append(i)
        self._types = np.array([p.unit_type for p in parts], dtype=object)
        starts = [str(p.meta.get("date") or "")[:10] for p in parts]
        ends = [str(p.meta.get("date_end") or p.meta.get("date") or "")[:10] for p in parts]
        self._date_start = np.array(starts, dtype="<U10")
        self._date_end = np.array(ends, dtype="<U10")
        self._has_date = self._date_start != ""
        self._people: list[str] | None = None
        self._part_rows: dict[str, int] | None = None
        self._bm25: dict[bool, BM25Index] = {}
        self._dense: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # --- 기본 정보 ---

    @property
    def n_parts(self) -> int:
        """파트 수"""
        return len(self.parts)

    @property
    def n_units(self) -> int:
        """단위 수"""
        return len(self.unit_rows)

    def profiles(self) -> list[str]:
        """만들어진 프로필 목록"""
        pdir = self.root / "profiles"
        if not pdir.is_dir():
            return []
        return sorted(
            d.name for d in pdir.iterdir() if (d / "meta.json").exists() and (d / "dense.npy").exists()
        )

    def profile_meta(self, profile: str) -> dict[str, Any]:
        """프로필 meta.json

        Raises:
            IndexNotFoundError: 프로필이 없을 때
        """
        path = self.root / "profiles" / profile / "meta.json"
        if not path.exists():
            raise IndexNotFoundError(f"케이스 {self.case_id}에 프로필 {profile} 색인이 없습니다")
        return json.loads(path.read_text(encoding="utf-8"))

    def check_embedding(self, profile: str, embed_model: str, embed_dim: int) -> None:
        """질의 임베딩 모델·차원이 프로필과 같은지 검증

        Raises:
            EmbeddingDimensionError: 모델 또는 차원이 다를 때 (재색인 필요)
        """
        meta = self.profile_meta(profile)
        if meta.get("embed_model") != embed_model or int(meta.get("embed_dim", -1)) != embed_dim:
            raise EmbeddingDimensionError(
                f"케이스 {self.case_id} 프로필 {profile} 색인은 {meta.get('embed_model')}"
                f"({meta.get('embed_dim')}차원)로 만들어졌습니다. 현재 {embed_model}({embed_dim}차원)과 "
                f"다르므로 재인덱싱해야 합니다."
            )

    # --- 단위 조회 ---

    def unit_parts(self, unit_id: str) -> list[UnitPart]:
        """단위의 파트들 (파트 순서)"""
        return [self.parts[i] for i in self.unit_rows.get(unit_id, [])]

    def part(self, part_id: str) -> UnitPart:
        """part_id로 파트 조회

        Raises:
            KeyError: 없는 part_id
        """
        if self._part_rows is None:
            self._part_rows = {p.part_id: i for i, p in enumerate(self.parts)}
        return self.parts[self._part_rows[part_id]]

    def unit_text(self, unit_id: str) -> str:
        """단위 본문 전체 (파트 이어붙임, 헤더 제외)"""
        return "\n".join(p.text for p in self.unit_parts(unit_id))

    def iter_unit_meta(self) -> Iterable[tuple[str, str, dict[str, Any]]]:
        """(unit_id, unit_type, meta) — 단위마다 한 번 (대시보드·통계용)"""
        for uid, rows in self.unit_rows.items():
            p = self.parts[rows[0]]
            yield uid, p.unit_type, p.meta

    # --- 검색 재료 ---

    def bm25(self, header: bool) -> BM25Index:
        """BM25 색인 (hdr/nohdr), 처음 쓸 때 로드"""
        with self._lock:
            if header not in self._bm25:
                path = self.root / "bm25" / ("hdr.pkl" if header else "nohdr.pkl")
                if not path.exists():
                    raise IndexNotFoundError(f"BM25 색인이 없습니다: {path}")
                try:
                    self._bm25[header] = BM25Index.load(path)
                except TokenizerVersionError as e:
                    raise IndexVersionError(str(e)) from e
            return self._bm25[header]

    def dense(self, profile: str) -> np.ndarray:
        """프로필 행렬 float16 [n_parts, dim], 처음 쓸 때 로드"""
        with self._lock:
            if profile not in self._dense:
                path = self.root / "profiles" / profile / "dense.npy"
                if not path.exists():
                    raise IndexNotFoundError(f"케이스 {self.case_id}에 프로필 {profile} 색인이 없습니다")
                mat = np.load(path)
                if mat.shape[0] != self.n_parts:
                    raise IndexVersionError(
                        f"프로필 {profile} 행 수 {mat.shape[0]} ≠ 파트 수 {self.n_parts}. 재색인이 필요합니다."
                    )
                self._dense[profile] = mat
            return self._dense[profile]

    def bm25_scores(self, query: str, header: bool) -> np.ndarray:
        """모든 파트의 BM25 점수"""
        return self.bm25(header).scores(query)

    def dense_scores(self, query_vec: list[float] | np.ndarray, profile: str) -> np.ndarray:
        """모든 파트의 코사인 점수 (float16 행렬을 블록 단위로 float32로 바꿔 계산)"""
        mat = self.dense(profile)
        q = np.asarray(query_vec, dtype=np.float32)
        if mat.shape[1] != q.shape[0]:
            raise EmbeddingDimensionError(
                f"질의 벡터 {q.shape[0]}차원 ≠ 색인 {mat.shape[1]}차원 (프로필 {profile}). 재인덱싱이 필요합니다."
            )
        out = np.empty(mat.shape[0], dtype=np.float32)
        for s in range(0, mat.shape[0], _DENSE_BLOCK_ROWS):
            out[s : s + _DENSE_BLOCK_ROWS] = mat[s : s + _DENSE_BLOCK_ROWS].astype(np.float32) @ q
        return out

    # --- 필터 ---

    def _people_text(self) -> list[str]:
        """행별 참여자 검색 문자열 (소문자, 이름·주소 합침)"""
        if self._people is None:
            keys = ("participants", "sender", "sender_name", "recipients", "recipient_names",
                    "cc", "cc_names", "author", "last_modified_by", "parent_sender")
            rows = []
            for p in self.parts:
                vals: list[str] = []
                for k in keys:
                    v = p.meta.get(k)
                    if isinstance(v, list):
                        vals.extend(str(x) for x in v)
                    elif v:
                        vals.append(str(v))
                rows.append("\n".join(vals).lower())
            self._people = rows
        return self._people

    def filter_mask(self, filters: dict[str, Any] | None) -> np.ndarray | None:
        """서비스 필터 → 파트 불리언 마스크 (필터가 없으면 None)

        지원 키 (질의 파서·API 공통):
        - source_type / unit_type: 문자열 또는 목록 (teams_chat은 chat으로 본다)
        - _date_gte / _date_lte / _date_eq: ISO 날짜. 날짜 없는 단위는 날짜 필터에서 빠진다.
          채팅 세션은 [date, date_end] 구간이 조건과 겹치면 통과한다.
        - participants: 목록 — 하나라도 참여자·보낸이·받는이·참조(이름·주소)에 부분 일치하면 통과
        - author / last_modified_by: 부분 일치 (문자열 또는 목록)
        - 그 밖의 키: 메타 값 같음(목록이면 포함)
        """
        if not filters:
            return None
        n = self.n_parts
        mask = np.ones(n, dtype=bool)
        for key, value in filters.items():
            if value is None or value == "" or value == []:
                continue
            if key.startswith("$"):
                logger.warning(f"지원하지 않는 필터 연산자 무시: {key}")
                continue
            if key in ("source_type", "unit_type"):
                vals = value if isinstance(value, list) else [value]
                types = {_SOURCE_TYPE_TO_UNIT.get(str(v), str(v)) for v in vals}
                mask &= np.isin(self._types, list(types))
            elif key == "_date_gte":
                mask &= self._has_date & (self._date_end >= str(value)[:10])
            elif key == "_date_lte":
                mask &= self._has_date & (self._date_start <= str(value)[:10])
            elif key == "_date_eq":
                v = str(value)[:10]
                mask &= self._has_date & (self._date_start <= v) & (self._date_end >= v)
            elif key == "participants":
                names = [str(x).lower() for x in (value if isinstance(value, list) else [value]) if str(x).strip()]
                people = self._people_text()
                mask &= np.fromiter((any(nm in people[i] for nm in names) for i in range(n)), bool, n)
            elif key in ("author", "last_modified_by"):
                names = [str(x).lower() for x in (value if isinstance(value, list) else [value])]
                col = [str(p.meta.get(key) or "").lower() for p in self.parts]
                mask &= np.fromiter((any(nm in col[i] for nm in names) for i in range(n)), bool, n)
            else:
                vals = value if isinstance(value, list) else [value]
                mask &= np.fromiter(
                    (_meta_match(p.meta.get(key), vals) for p in self.parts), bool, n
                )
        return mask


def _meta_match(actual: Any, wanted: list[Any]) -> bool:
    """메타 값이 원하는 값 중 하나와 같은지 (메타가 목록이면 교집합)"""
    if isinstance(actual, list):
        return any(a in wanted for a in actual)
    return actual in wanted


# ─────────────────────────── 로드·캐시·삭제 ───────────────────────────

_cache: OrderedDict[str, tuple[float, CaseIndex]] = OrderedDict()
_cache_lock = threading.Lock()
_write_lock = threading.Lock()


def _recover(case_id: str) -> None:
    """폴더 교체 도중 종료로 남은 .old를 되돌린다"""
    final = case_dir(case_id)
    old = index_root() / f"{case_id}.old"
    if not final.exists() and old.exists():
        logger.warning(f"색인 교체 중단 흔적 복구: {old} → {final}")
        old.rename(final)


def exists(case_id: str) -> bool:
    """케이스 색인(manifest)이 있는지"""
    _recover(case_id)
    return (case_dir(case_id) / "manifest.json").exists()


def read_manifest(case_id: str) -> dict[str, Any]:
    """manifest.json

    Raises:
        IndexNotFoundError: 색인이 없을 때
    """
    path = case_dir(case_id) / "manifest.json"
    if not path.exists():
        raise IndexNotFoundError(f"케이스 색인이 없습니다: {case_id} (v4 재인덱싱 필요)")
    return json.loads(path.read_text(encoding="utf-8"))


def load(case_id: str) -> CaseIndex:
    """케이스 색인 로드 (최근 2개 케이스 캐시)

    Raises:
        IndexNotFoundError: 색인이 없을 때
        IndexVersionError: 스키마·토크나이저 버전이 다를 때 (재색인 필요)
    """
    _recover(case_id)
    root = case_dir(case_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise IndexNotFoundError(f"케이스 색인이 없습니다: {case_id} (v4 재인덱싱 필요)")
    stamp = manifest_path.stat().st_mtime
    with _cache_lock:
        hit = _cache.get(case_id)
        if hit is not None and hit[0] == stamp:
            _cache.move_to_end(case_id)
            return hit[1]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IndexVersionError(
            f"색인 스키마 {manifest.get('schema_version')} ≠ 현재 {SCHEMA_VERSION}. 재색인이 필요합니다."
        )
    if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
        raise IndexVersionError(
            f"색인 토크나이저 {manifest.get('tokenizer_version')} ≠ 현재 {TOKENIZER_VERSION}. "
            f"재색인이 필요합니다."
        )
    t0 = time.monotonic()
    parts = [
        UnitPart.from_dict(json.loads(line))
        for line in (root / "units.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    idx = CaseIndex(case_id, root, manifest, parts)
    logger.info(f"케이스 색인 로드: {case_id} ({idx.n_units}단위/{idx.n_parts}파트, {time.monotonic() - t0:.1f}s)")
    with _cache_lock:
        _cache[case_id] = (stamp, idx)
        _cache.move_to_end(case_id)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return idx


def evict(case_id: str) -> None:
    """캐시에서 제거"""
    with _cache_lock:
        _cache.pop(case_id, None)


def delete(case_id: str) -> bool:
    """케이스 색인 폴더 삭제

    Returns:
        지운 폴더가 있었으면 True
    """
    evict(case_id)
    removed = False
    for path in (case_dir(case_id), index_root() / f"{case_id}.old"):
        if path.exists():
            shutil.rmtree(path)
            removed = True
    for tmp in index_root().glob(f".tmp_{case_id}_*") if index_root().exists() else []:
        shutil.rmtree(tmp, ignore_errors=True)
    if removed:
        logger.info(f"케이스 색인 삭제: {case_id}")
    return removed


# ─────────────────────────── 쓰기 ───────────────────────────


@dataclass
class ProfileVectors:
    """프로필 하나의 행렬과 모델 정보"""

    vectors: np.ndarray  # [n_parts, dim], 행 순서 = parts 순서
    embed_model: str


def reusable_vectors(case_id: str, profile: str) -> dict[tuple[str, str], np.ndarray]:
    """기존 색인에서 재사용할 수 있는 벡터 — {(part_id, 입력 해시): 벡터}

    같은 파트 ID에 같은 임베딩 입력이면 다시 임베딩하지 않는다(증분 색인).
    모델이 다르거나 색인이 없으면 빈 dict.
    """
    try:
        idx = load(case_id)
        spec = profile_spec(profile)
        meta = idx.profile_meta(profile)
        if meta.get("embed_model") != spec.embed_model:
            return {}
        mat = idx.dense(profile)
    except (IndexNotFoundError, IndexVersionError, ValueError):
        return {}
    except Exception as e:
        logger.warning(f"기존 색인 벡터 재사용 불가 ({case_id}/{profile}): {e}")
        return {}
    header = bool(meta.get("header"))
    return {
        (p.part_id, input_hash(embed_input(p, header))): mat[i]
        for i, p in enumerate(idx.parts)
    }


def _write_profile_dir(pdir: Path, profile: str, pv: ProfileVectors) -> None:
    """profiles/{profile}/ 내용 쓰기"""
    spec = profile_spec(profile)
    pdir.mkdir(parents=True, exist_ok=True)
    mat = np.ascontiguousarray(pv.vectors, dtype=np.float16)
    np.save(pdir / "dense.npy", mat)
    meta = {
        "profile": profile,
        "header": spec.header,
        "embed_model": pv.embed_model,
        "embed_dim": int(mat.shape[1]) if mat.ndim == 2 else 0,
        "normalized": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (pdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _swap_dir(tmp: Path, final: Path) -> None:
    """임시 폴더를 최종 폴더로 교체 (기존 → .old → 삭제)"""
    old = final.parent / f"{final.name}.old"
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def write_index(
    case_id: str,
    parts: list[UnitPart],
    profiles: dict[str, ProfileVectors],
    *,
    source_files: list[dict[str, Any]] | None = None,
    build_stats: dict[str, Any] | None = None,
    keep_profiles: bool = True,
) -> Path:
    """케이스 색인 전체를 새로 쓰고 원자적으로 교체

    Args:
        case_id: 케이스 ID
        parts: 파트 목록 (행 순서)
        profiles: 새로 쓸 프로필 행렬 (행 순서 = parts)
        source_files: 원본 파일 지문 목록 ({path, size, mtime})
        build_stats: 단위 생성 통계 (BuildStats.to_dict, 중복 제거 수 포함)
        keep_profiles: True면 이번에 쓰지 않은 기존 프로필 중 같은 파트 입력의 벡터를
            모두 재사용할 수 있는 것은 새 행 순서로 옮겨 유지한다(실험 프로필 보존)

    Returns:
        최종 색인 폴더
    """
    with _write_lock:
        root = index_root()
        root.mkdir(parents=True, exist_ok=True)
        _recover(case_id)
        final = case_dir(case_id)
        tmp = root / f".tmp_{case_id}_{int(time.time() * 1000)}"
        tmp.mkdir(parents=True)
        try:
            t0 = time.monotonic()
            with (tmp / "units.jsonl").open("w", encoding="utf-8") as f:
                for p in parts:
                    f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

            for header, name in ((True, "hdr.pkl"), (False, "nohdr.pkl")):
                BM25Index.from_texts([embed_input(p, header) for p in parts]).save(tmp / "bm25" / name)
            t_bm25 = time.monotonic() - t0

            for profile, pv in profiles.items():
                if pv.vectors.shape[0] != len(parts):
                    raise ValueError(f"프로필 {profile} 행 수 {pv.vectors.shape[0]} ≠ 파트 수 {len(parts)}")
                _write_profile_dir(tmp / "profiles" / profile, profile, pv)

            if keep_profiles and final.exists():
                for profile in PROFILE_NAMES:
                    if profile in profiles:
                        continue
                    carried = _carry_profile(case_id, profile, parts)
                    if carried is not None:
                        _write_profile_dir(tmp / "profiles" / profile, profile, carried)

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "unit_rules_version": UNIT_RULES_VERSION,
                "case_id": case_id,
                "n_units": len({p.unit_id for p in parts}),
                "n_parts": len(parts),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_files": source_files or [],
                "build_stats": build_stats or {},
            }
            (tmp / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            evict(case_id)
            _swap_dir(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        logger.info(
            f"케이스 색인 저장: {case_id} ({manifest['n_units']}단위/{len(parts)}파트, "
            f"프로필 {sorted(profiles)}, BM25 {t_bm25:.1f}s)"
        )
        return final


def _carry_profile(case_id: str, profile: str, parts: list[UnitPart]) -> ProfileVectors | None:
    """기존 프로필 벡터를 새 파트 순서로 옮긴다. 하나라도 없으면 None (그 프로필은 버림)"""
    reuse = reusable_vectors(case_id, profile)
    if not reuse:
        return None
    header = profile_spec(profile).header
    rows = []
    for p in parts:
        v = reuse.get((p.part_id, input_hash(embed_input(p, header))))
        if v is None:
            logger.info(f"프로필 {profile}: 새 파트가 있어 유지하지 않음 (build_profiles로 다시 생성)")
            return None
        rows.append(v)
    meta = load(case_id).profile_meta(profile)
    return ProfileVectors(np.stack(rows) if rows else np.zeros((0, 0), np.float16), meta["embed_model"])


def write_profile(case_id: str, profile: str, pv: ProfileVectors) -> None:
    """기존 색인에 프로필 하나를 추가·교체 (scripts/build_profiles.py)

    Raises:
        IndexNotFoundError: 케이스 색인이 없을 때
        ValueError: 행 수가 파트 수와 다를 때
    """
    with _write_lock:
        idx = load(case_id)
        if pv.vectors.shape[0] != idx.n_parts:
            raise ValueError(f"프로필 {profile} 행 수 {pv.vectors.shape[0]} ≠ 파트 수 {idx.n_parts}")
        pdir = idx.root / "profiles"
        tmp = pdir / f".tmp_{profile}_{int(time.time() * 1000)}"
        try:
            _write_profile_dir(tmp, profile, pv)
            evict(case_id)
            _swap_dir(tmp, pdir / profile)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        # manifest mtime을 갱신해 다른 프로세스 캐시도 다시 읽게 한다
        (idx.root / "manifest.json").touch()
        logger.info(f"프로필 저장: {case_id}/{profile} ({pv.vectors.shape})")
