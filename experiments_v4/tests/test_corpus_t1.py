"""T1 단위 통계 (ev4.corpus) — v4.2 U1·U2·S1~S5. 가명 합성 파트만 쓴다 (P0)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ev4.corpus import (  # noqa: E402
    CorpusStopError,
    code_like_counts,
    excluded_group,
    format_row,
    render_t1,
    stats_from_parts,
)

from src.units.schema import UnitPart  # noqa: E402
from src.units.tokens import TokenCounter  # noqa: E402


class _ExactCounter(TokenCounter):
    """문자 1개 = 토큰 1개이지만 '실제 토크나이저'로 표시되는 테스트용 카운터"""

    def __init__(self) -> None:
        super().__init__(hf_name=None, chars_per_token=1.0)
        self.kind = "hf:test"


def _part(uid: str, k: int, utype: str, n_tokens: int, text: str = "본문", **meta) -> UnitPart:
    meta.setdefault("lang", "ko")
    return UnitPart(part_id=f"{uid}#{k}", unit_id=uid, unit_type=utype, header=f"[{utype}] h",
                    text=text, n_tokens=n_tokens, meta=meta)


def _corpus() -> list[UnitPart]:
    return [
        _part("em:1", 1, "email", 100, "INV-2024-0312 금액 1,200,000원 a@x.com 확인"),
        _part("em:2", 1, "email", 2048, "회신", lang="en"),
        _part("em:3", 1, "email", 2048, "긴 메일", lang="zh", lang_mixed=True),
        _part("em:3", 2, "email", 500, "긴 메일 계속", lang="zh", lang_mixed=True),
        _part("ch:1", 1, "chat", 50, "ok", lang="und"),
        _part("doc:a", 1, "document", 30, file_type="pdf", author="A"),
        _part("doc:b", 1, "document", 40, file_type="docx", author="A", last_modified_by="B"),
        _part("att:c", 1, "attachment", 60, file_type="xlsx", author="", last_modified_by="C",
              lang="cs"),
    ]


BUILD_STATS = {
    "counts": {"forwarded_emails": 1, "orphan_attachments": 2, "attachments_total": 5,
               "attachments_parsed": 3},
    "excluded_types": {".png": 3, ".hwp": 1, ".dwg": 1, "(none)": 2, ".zip": 1},
    "clean_chars": {"ko": [1000, 600]},
    "forwarded_tokens": [300],
    "duplicates_removed": {"email": 4},
}


def test_helpers():
    assert format_row("email", "") == "이메일" and format_row("chat", "") == "채팅"
    assert format_row("attachment", "xls") == "XLSX·XLS" and format_row("document", "html") == "DOCX·TXT·HTML"
    assert format_row("attachment", "eml") == "첨부 메일(EML·MSG)" and format_row("document", "csv") == "기타"
    assert excluded_group(".jpg") == "이미지" and excluded_group(".hwpx") == "HWP"
    assert excluded_group("(none)") == "판별 불가" and excluded_group(".zip") == "기타 형식"
    c = code_like_counts("INV-2024-0312 금액 1,200,000원 a@x.com")
    assert c["code"] >= 1 and c["amount"] == 1 and c["email"] == 1


def test_stats_tables():
    s = stats_from_parts(_corpus(), BUILD_STATS, _ExactCounter(), build_counter_kind="hf:test")
    t6 = s["table6"]
    assert t6["이메일"]["units"] == 3 and t6["이메일"]["split"] == 1
    assert t6["XLSX·XLS"] == {**t6["XLSX·XLS"], "units": 1, "from_attachment": 1}
    assert t6["계"]["units"] == 7 and t6["계"]["from_attachment"] == 1
    assert t6["이메일"]["tokens"]["max"] == 2548
    # U1: 문서·첨부 3개 중 작성자 2, 최종 수정자 2
    assert s["meta_fill"]["계"] == {"n": 3, "author": 2, "modifier": 2}
    # v4.4 §8: 한국어·영어·현지 언어(합산)·계, und 는 집계 메모
    t7 = s["table7"]
    assert t7["und"]["units"] == 1 and t7["현지 언어(합산)"] == {"units": 2, "mixed": 1}
    assert t7["계"]["units"] == 7
    # S2: 본문 2048 이하 이메일 2개(em:1, em:2), 헤더 포함은 em:1만
    assert s["s2"]["body_le_2048"] == 2 and s["s2"]["with_header_le_2048"] == 1
    assert s["s2"]["n_parts"] == 8 and s["s2"]["rerank_over_4096"] == 0
    # S3: 코드형 문자열은 em:1에만
    assert s["s3"]["with_any"] == 1 and s["s3"]["n_email"] == 3
    # S4
    assert s["excluded_grouped"] == {"이미지": 3, "HWP": 1, "DWG": 1, "판별 불가": 2, "기타 형식": 1}
    assert s["forwarded"]["count"] == 1 and s["duplicates_removed"] == {"email": 4}
    md = render_t1(s)
    for needle in ("[표 6]", "[표 7]", "현지 언어(합산)", "판정 불가(und)", "ads_removed", "S2", "S3", "S4",
                   "작성자·최종 수정자"):
        assert needle in md


def test_stops_on_approx_counter():
    approx = TokenCounter(hf_name=None)
    with pytest.raises(CorpusStopError):
        stats_from_parts(_corpus(), BUILD_STATS, approx)
    with pytest.raises(CorpusStopError):  # 색인 때 근사였던 경우도 멈춘다
        stats_from_parts(_corpus(), BUILD_STATS, _ExactCounter(), build_counter_kind="approx:2.0chars/token")
    s = stats_from_parts(_corpus(), BUILD_STATS, approx, require_exact=False)
    assert not s["token_counter_exact"] and "근사" in render_t1(s)
