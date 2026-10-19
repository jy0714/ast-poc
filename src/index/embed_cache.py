"""임베딩 체크포인트 — 인덱싱 도중 만든 벡터를 조각 파일로 바로 저장한다

케이스 색인은 인덱싱이 끝날 때 한 번만 쓰이므로, 수십 시간 걸리는 1TB 인덱싱이 중간에
멈추면(취소·정전·서버 재시작) 그때까지의 임베딩을 모두 잃는다. 이 모듈은 새로 만든 벡터를
`data/embed_cache/{case_id}/{profile}/shard_*.npz` 에 모아 두고, 다음 실행이 같은
(part_id, 입력 해시) 벡터를 다시 쓰게 한다. 색인 저장이 성공하면 지운다.

조각 파일에는 part_id·입력 해시·벡터만 있고 본문은 없다. 그래도 조사 데이터에서 나온 것이므로
P0 규칙상 Claude Code가 여는 대상에서 제외한다 (data/embed_cache/**).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from src.index.case_index import index_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 조각 하나에 모으는 벡터 수 (2560차원 fp16 기준 약 20MB). 중단 시 잃는 양의 상한이기도 하다.
SHARD_SIZE = 4096


def cache_dir(case_id: str, profile: str) -> Path:
    """케이스·프로필의 체크포인트 폴더 (색인 루트 옆 embed_cache/, 만들지는 않음)"""
    return index_root().parent / "embed_cache" / case_id / profile


class EmbedCheckpoint:
    """새로 만든 벡터를 조각 파일로 누적 저장하고, 다음 실행에서 재사용 dict로 읽는다

    Args:
        case_id: 케이스 ID
        profile: 색인 프로필 이름
        embed_model: 임베딩 모델 (다른 모델로 만든 조각은 읽지 않는다)
        shard_size: 조각 하나의 벡터 수
    """

    def __init__(self, case_id: str, profile: str, embed_model: str, shard_size: int = SHARD_SIZE) -> None:
        self.root = cache_dir(case_id, profile)
        self.embed_model = embed_model
        self.shard_size = shard_size
        self._keys: list[tuple[str, str]] = []
        self._vecs: list[np.ndarray] = []
        self._next = 0

    def load(self) -> dict[tuple[str, str], np.ndarray]:
        """저장된 조각 → {(part_id, 입력 해시): 벡터}. 깨진 조각·다른 모델 조각은 건너뛴다."""
        out: dict[tuple[str, str], np.ndarray] = {}
        if not self.root.is_dir():
            return out
        shards = sorted(self.root.glob("shard_*.npz"))
        skipped = 0
        for path in shards:
            try:
                with np.load(path, allow_pickle=False) as z:
                    if str(z["model"]) != self.embed_model:
                        skipped += 1
                        continue
                    vecs = z["vecs"]
                    for pid, h, v in zip(z["part_ids"], z["hashes"], vecs, strict=True):
                        out[(str(pid), str(h))] = v
            except Exception as e:
                skipped += 1
                logger.warning(f"임베딩 체크포인트 조각 읽기 실패 (건너뜀): {path.name} — {e}")
        self._next = len(shards)
        if out or skipped:
            logger.info(f"임베딩 체크포인트: 조각 {len(shards)}개에서 벡터 {len(out):,}개 (건너뜀 {skipped})")
        return out

    def add(self, keys: list[tuple[str, str]], vecs: list[np.ndarray]) -> None:
        """벡터 누적. 조각 크기를 채우면 파일로 쓴다."""
        self._keys.extend(keys)
        self._vecs.extend(vecs)
        while len(self._keys) >= self.shard_size:
            self._write(self.shard_size)

    def flush(self) -> None:
        """남은 벡터를 조각 파일로 쓴다 (실행 종료·취소 시)"""
        if self._keys:
            self._write(len(self._keys))

    def clear(self) -> None:
        """체크포인트 삭제 (색인 저장 성공 후)"""
        self._keys.clear()
        self._vecs.clear()
        if self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, n: int) -> None:
        keys, self._keys = self._keys[:n], self._keys[n:]
        vecs, self._vecs = self._vecs[:n], self._vecs[n:]
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"shard_{self._next:06d}.npz"
        tmp = final.with_suffix(".tmp")
        # 쓰는 도중 죽어도 반쪽 조각이 남지 않게 임시 파일 → rename
        with tmp.open("wb") as f:
            np.savez(
                f,
                part_ids=np.array([k[0] for k in keys]),
                hashes=np.array([k[1] for k in keys]),
                vecs=np.stack(vecs).astype(np.float16),
                model=np.array(self.embed_model),
            )
        os.replace(tmp, final)
        self._next += 1
