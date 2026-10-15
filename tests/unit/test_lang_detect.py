"""단위별 언어 감지 테스트 (src/units/lang_detect.py)"""

from __future__ import annotations

from src.units.lang_detect import MIXED_THRESHOLD, UNDETERMINED, detect


def test_korean_email_not_misdetected():
    # 0단계: langdetect 는 이런 짧은 한국어 메일을 vi·et 로 오판정했다
    r = detect("안녕하세요\n생산기술팀 백종산 입니다.\n\nRemind 메일 드립니다.\n\n감사합니다.\n백종산 드림")
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
