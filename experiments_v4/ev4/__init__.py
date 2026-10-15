"""AuditRAG v4 실험 하네스 (지시서 §6)

서비스의 `src.rag.retrieval.retrieve()`를 직접 호출하는 얇은 스크립트 모음이다.
파서·단위 생성·검색 로직은 `src/`에만 있다 (지시서 §2-9).
"""

from __future__ import annotations

import sys
from pathlib import Path

# experiments_v4/ 에서 `python -m ev4` 로 실행해도 프로젝트 루트의 `src` 를 import 할 수 있게 한다
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
