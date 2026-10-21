"""단위별 언어 감지 테스트 (src/units/lang_detect.py)"""

from __future__ import annotations

from src.units.lang_detect import MIXED_THRESHOLD, UNDETERMINED, detect


def test_korean_email_not_misdetected():
    # 0단계: langdetect 는 이런 짧은 한국어 메일을 vi·et 로 오판정했다
    r = detect(
        "안녕하세요\n생산기술팀 백종산 입니다.\n\nRemind 메일 드립니다.\n\n감사합니다.\n백종산 드림"
    )
    assert r.lang == "ko"
    assert not r.mixed


def test_english_and_chinese():
    assert detect("Please approve the invoice before payment.\nThanks").lang == "en"
    assert detect("请在周五前审批这张发票，谢谢配合。\n收到").lang == "zh"


def test_mixed_korean_english():
    r = detect(
        "안녕하세요 확인 부탁드립니다.\n"
        "Please check the attached contract and confirm the payment schedule by Friday."
    )
    assert r.lang in ("ko", "en")
    assert r.mixed and r.second in ("ko", "en") and r.second != r.lang


def test_short_text_undetermined():
    r = detect("OK")
    assert r.lang == UNDETERMINED and not r.mixed and r.shares == {}
    assert detect("").lang == UNDETERMINED


def test_mixed_threshold_boundary():
    # 두 번째 언어 비중이 20% 미만이면 혼용 아님
    ko = "결재 절차를 건너뛰고 처리한 건에 대해 확인 부탁드립니다 " * 4
    r = detect(f"{ko}\nThanks")
    assert r.lang == "ko"
    assert r.shares.get("en", 0) < MIXED_THRESHOLD
    assert not r.mixed


def test_forwarded_header_lines_and_addresses_ignored():
    # 전달 메일은 인용 헤더가 남는다. 로마자 이름이 줄줄이 나오는 수신자 줄이
    # 베트남어 등으로 오판정되어 주 언어를 뒤집던 문제 (1단계 표본 검증에서 발견)
    names = "; ".join(
        f"김조현(Johyeon Kim{i}) 팀장 <johyeon{i}.kim@example.com<mailto:johyeon{i}.kim@example.com>>"
        for i in range(30)
    )
    text = (
        "매뉴얼 업데이트 현황 공유드립니다.\n금주 중 마무리 부탁드립니다.\n\n"
        f"From: 박화성(Hwasung Park) 수석\nSent: Thursday, August 24, 2023 8:06 AM\nTo: {names}\n"
        f"Cc: {names}\nSubject: DBPS Manual Up-date Guide\n\n업로드 부탁드립니다."
    )
    r = detect(text)
    assert r.lang == "ko"
    assert "vi" not in r.shares


def test_codes_do_not_vote():
    # 2단계 실측: 코드·숫자가 많은 한국어 SAP 표가 독일어로 판정되던 문제
    rows = "\n".join(
        f"EX2023122804{i}   2023.12.28   747962063{i}   G25N7-EB20   0AE1   FGA14   휴먼에러(입력오류)"
        for i in range(10)
    )
    text = f"유관 부서에 아래와 같이 변경 요청 메일이 송부 되었습니다.\n업무에 참고 부탁 드립니다\n{rows}"
    assert detect(text).lang == "ko"
