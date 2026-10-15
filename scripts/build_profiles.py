"""케이스 색인에 실험 프로필 추가 생성 (지시서 §5.5, §6.5)

서비스 인덱싱은 SERVICE_INDEX_PROFILE(기본 hdr-qwen3) 하나만 만든다. 실험용 프로필은 이 스크립트로
같은 units.jsonl(같은 파트·같은 행 순서)에 대해 행렬만 추가한다.

| 프로필       | 입력              | 임베딩 모델                    |
|--------------|-------------------|--------------------------------|
| nohdr-qwen3  | 본문만            | OLLAMA_EMBED_MODEL (Qwen3)     |
| hdr-bgem3    | 헤더 + 본문       | bge-m3 (지시문 없음, R9 전용)  |

같은 파트·같은 입력의 기존 벡터가 그 프로필에 있으면 재사용한다. 임베딩에 영구 실패한 파트가
하나라도 있으면 프로필을 쓰지 않고 실패한다(행렬은 모든 파트를 덮어야 한다).

사용:
    python -m scripts.build_profiles <case_id> --profiles nohdr-qwen3 hdr-bgem3 [--batch 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_service import EmbeddingService  # noqa: E402
from src.index import case_index  # noqa: E402
from src.index.case_index import (  # noqa: E402
    PROFILE_NAMES,
    ProfileVectors,
    embed_input,
    input_hash,
    profile_spec,
)
from src.utils.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class ProfileBuildError(Exception):
    """프로필 행렬을 다 채우지 못함"""


def build_profile(case_id: str, profile: str, batch: int, embedder: EmbeddingService | None = None) -> dict:
    """프로필 하나를 만들어 색인에 쓴다

    Args:
        case_id: 케이스 ID (data/index/{case_id}가 있어야 함)
        profile: 프로필 이름
        batch: embed_texts 1회 호출당 텍스트 수
        embedder: 임베딩 서비스 (테스트 주입용, 기본은 프로필 모델)

    Returns:
        {profile, n_parts, embedded, reused, seconds}

    Raises:
        ProfileBuildError: 영구 실패 파트가 있을 때
    """
    spec = profile_spec(profile)
    idx = case_index.load(case_id)
    embedder = embedder or EmbeddingService(model=spec.embed_model)
    reuse = case_index.reusable_vectors(case_id, profile)
    texts = [embed_input(p, spec.header) for p in idx.parts]
    rows: list[np.ndarray | None] = [
        reuse.get((p.part_id, input_hash(t))) for p, t in zip(idx.parts, texts, strict=True)
    ]
    todo = [i for i, r in enumerate(rows) if r is None]
    t0 = time.monotonic()
    failed: list[int] = []
    for s in range(0, len(todo), batch):
        chunk = todo[s : s + batch]
        vecs = embedder.embed_texts([texts[i] for i in chunk])
        bad = set(embedder._last_failed_indices)
        for j, i in enumerate(chunk):
            if j in bad or not vecs[j]:
                failed.append(i)
            else:
                rows[i] = np.asarray(vecs[j], dtype=np.float16)
        done = min(s + batch, len(todo))
        if done == len(todo) or (s // batch) % 20 == 0:
            logger.info(f"[{profile}] 임베딩 {done:,}/{len(todo):,} ({time.monotonic() - t0:.0f}s)")
    if failed:
        raise ProfileBuildError(
            f"프로필 {profile}: {len(failed)}개 파트 임베딩 실패 (예: {idx.parts[failed[0]].part_id}). "
            f"프로필을 쓰지 않았습니다."
        )
    dim = embedder.get_dimension()
    mat = np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float16)  # type: ignore[arg-type]
    case_index.write_profile(case_id, profile, ProfileVectors(mat, spec.embed_model))
    return {
        "profile": profile,
        "n_parts": idx.n_parts,
        "embedded": len(todo),
        "reused": idx.n_parts - len(todo),
        "seconds": round(time.monotonic() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case_id")
    ap.add_argument("--profiles", nargs="+", required=True, choices=PROFILE_NAMES)
    ap.add_argument("--batch", type=int, default=settings.embed_batch_size * max(1, settings.embed_max_concurrent))
    args = ap.parse_args(argv)

    code = 0
    for profile in args.profiles:
        try:
            info = build_profile(args.case_id, profile, args.batch)
            logger.info(f"프로필 완료: {info}")
        except Exception as e:
            logger.error(f"프로필 {profile} 실패: {e}")
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
