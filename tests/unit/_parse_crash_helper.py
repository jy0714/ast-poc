"""ProcessPool 깨짐 테스트용 워커 — 자식 프로세스가 import할 수 있게 모듈 수준에 둔다"""

import os
from pathlib import Path


def crash_or_parse(args: tuple[str, str, str]):
    """파일명에 'crash'가 있으면 프로세스를 즉시 죽이고, 아니면 원래 워커로 파싱"""
    if "crash" in Path(args[0]).name:
        os._exit(3)
    from src.indexing.pipeline import _process_file_subprocess

    return _process_file_subprocess(args)
