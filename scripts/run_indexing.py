"""인덱싱 파이프라인 CLI — 케이스 단위 인덱싱 실행

사용법:
    python -m scripts.run_indexing <case_id>
"""

import sys

from src.cases.case_store import CaseStore
from src.indexing.pipeline import IndexingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python -m scripts.run_indexing <case_id>")
        sys.exit(1)

    case_id = sys.argv[1]
    store = CaseStore()
    pipeline = IndexingPipeline(case_store=store)

    logger.info(f"인덱싱 시작: {case_id}")
    progress = pipeline.run(case_id)

    logger.info(
        f"인덱싱 완료: {progress.phase.value} — "
        f"파일 {progress.processed_files}/{progress.total_files}, "
        f"청크 {progress.total_chunks}, 에러 {len(progress.errors)}"
    )

    if progress.errors:
        for err in progress.errors:
            logger.warning(f"  에러: {err}")


if __name__ == "__main__":
    main()
