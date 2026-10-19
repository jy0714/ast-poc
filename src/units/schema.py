"""증거 단위 데이터 클래스

- EvidenceUnit: 판정 단위 (이메일 1통, 채팅 세션 1개, 문서 페이지 1쪽, 첨부 페이지 1쪽 …)
- UnitPart: 검색 단위. 대부분 단위당 파트 1개이고, UNIT_MAX_PART_TOKENS 를 넘는 단위만
  여러 파트로 나뉜다. 모든 파트는 같은 헤더를 갖는다.

qrels(정답)가 unit_id 로 저장되므로 같은 원본은 재색인해도 항상 같은 unit_id 가 나와야 한다
(규칙: src/units/builder.py `make_*_id`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

UNIT_TYPES = ("email", "chat", "document", "attachment")

# 단위 생성 규칙 버전 — 규칙(정제·분할·헤더·ID)이 바뀌면 올린다. 처리 캐시 키에 쓰인다.
UNIT_RULES_VERSION = "units-v3"  # v3: 광고 메일 제외 / v2: 문서·첨부 헤더에 작성자·최종 수정자 (v4.2 U1)


@dataclass
class UnitPart:
    """검색 단위 (단위의 한 조각)

    Attributes:
        part_id: f"{unit_id}#{k}" (k=1..n)
        unit_id: 소속 단위 ID
        unit_type: "email" | "chat" | "document" | "attachment"
        header: 단위 헤더 (모든 파트에 같음)
        text: 정제된 본문 조각 (헤더 제외)
        n_tokens: text 의 토큰 수 (Qwen3 토크나이저 기준)
        meta: 단위 메타데이터 (모든 파트에 같음)
    """

    part_id: str
    unit_id: str
    unit_type: str
    header: str
    text: str
    n_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSONL 직렬화용 dict"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitPart:
        """to_dict() 결과에서 복원"""
        return cls(**data)

    def full_text(self, with_header: bool = True) -> str:
        """임베딩·재순위·BM25 입력 텍스트 (헤더 on: header + 빈 줄 + text)"""
        if with_header and self.header:
            return f"{self.header}\n\n{self.text}"
        return self.text


@dataclass
class EvidenceUnit:
    """증거 단위 (판정 단위)

    Attributes:
        unit_id: 결정적 ID (em:/ch:/doc:/att: 접두어)
        unit_type: "email" | "chat" | "document" | "attachment"
        header: 유형별 헤더 문자열 (src/units/header.py)
        text: 정제된 본문 전체 (헤더 제외)
        meta: 메타데이터 (지시서 §5.1.1 필드)
        parts: 검색 단위 목록 (UNIT_MAX_PART_TOKENS 이하로 분할)
    """

    unit_id: str
    unit_type: str
    header: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    parts: list[UnitPart] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        """단위 본문 전체 토큰 수 (파트 토큰 합)"""
        return sum(p.n_tokens for p in self.parts)
