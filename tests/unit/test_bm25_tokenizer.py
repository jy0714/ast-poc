"""문자 체계별 BM25 토크나이저 테스트 (ref 테스트 이식 + 코드형 문자열)"""

from __future__ import annotations

from src.text.bm25_tokenizer import TOKENIZER_VERSION, tokenize, tokenize_many


def test_english_sentence_final_word_not_split():
    toks = tokenize("Please approve the invoice before payment. We skipped the approval process.")
    assert "payment" in toks and "process" in toks
    assert "pa" not in toks and "yment." not in toks


def test_czech_spanish_words_kept_and_folded():
    toks = tokenize("Posílám fakturu za březen. Hola, adjunto la factura de marzo.")
    assert "brezen" in toks and "fakturu" in toks and "marzo" in toks
    assert tokenize("Dvořák") == tokenize("Dvorak") == ["dvorak"]


def test_chinese_bigrams():
    toks = tokenize("附件是三月份的发票，请审批。")
    assert "发票" in toks and "审批" in toks
    assert "附件是三月份的发票" not in toks


def test_codes_amounts_emails_kept_whole():
    toks = tokenize("계약번호 INV-2024-0312, 금액 1,200,000원, 담당 hong.gd@abc-corp.com")
    assert "inv-2024-0312" in toks and "inv" in toks and "2024" in toks
    assert "1200000" in toks
    assert "hong.gd@abc-corp.com" in toks


def test_vehicle_serial_code_kept():
    # 0단계: v3.x 토크나이저는 숫자 부분(W_SERIAL)을 버려 'fdb21'만 남았다
    toks = tokenize("경기딜러 D70S-9, FDB21-000963 계기판 에러")
    assert "fdb21-000963" in toks and "000963" in toks
    assert "d70s-9" in toks


def test_korean_morphemes_and_mixed_text():
    toks = tokenize("결재는 일단 건너뛰고 처리했습니다. Please check the contract.")
    assert "결재" in toks and "처리" in toks
    assert "contract" in toks and "con" not in toks
    assert "." not in toks


def test_korean_pos_filter_option():
    toks = tokenize("결재는 일단 건너뛰고 처리했습니다", ko_pos_filter=True)
    assert "결재" in toks and "는" not in toks


def test_query_and_document_share_tokens():
    doc = tokenize("Dobrý den, posílám fakturu za březen.")
    q = tokenize("faktura brezen")
    assert set(q) & set(doc) == {"brezen"}


def test_empty_and_many():
    assert tokenize("") == []
    assert tokenize_many(["a b", ""]) == [["a", "b"], []]
    assert TOKENIZER_VERSION.startswith("script-")


def test_hangul_not_treated_as_cjk_bigrams():
    # 이식 시 NFC 정규화로 CJK 범위가 한글까지 삼켜 '먼에'·'력오' 같은 한글 bigram 이 생기던 회귀
    assert tokenize("휴먼에러(입력오류)") == ["휴먼", "에러", "입력", "오류"]
    assert tokenize("豈") == ["豈"]  # 호환 한자는 여전히 CJK 처리


def test_tokenize_many_parallel_matches_serial():
    from src.text import bm25_tokenizer as bt

    texts = [f"정산 금액 1,200,000원 INV-2024-{i:04d} payment 확인 {i}" for i in range(4500)]
    assert bt.tokenize_many_parallel(texts, workers=2) == bt.tokenize_many(texts)
