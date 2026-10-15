"""참조 구현 테스트. 실행: pytest docs/v4/ref/test_ref.py -q (kiwipiepy 필요)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from email_quote import strip_quoted  # noqa: E402
from qwen3_formats import format_embed_query, format_rerank_pair  # noqa: E402
from script_tokenizer import tokenize  # noqa: E402


# ---------- tokenizer ----------

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


# ---------- email quote ----------

def test_outlook_english_block():
    body = "Please see below.\n\nFrom: Kim <kim@a.com>\nSent: Monday\nTo: Lee\nSubject: RE: test\n\nold text"
    assert strip_quoted(body) == "Please see below."


def test_czech_block_and_separator():
    body = "Souhlasím s návrhem, pošlete fakturu.\n\nOd: Novák Jan\nOdesláno: pondělí\nKomu: Kim\nPředmět: faktura\n\nstarý text"
    assert strip_quoted(body) == "Souhlasím s návrhem, pošlete fakturu."
    body2 = "Prosím o schválení do pátku.\n-----Původní zpráva-----\nstarý text"
    assert strip_quoted(body2) == "Prosím o schválení do pátku."


def test_spanish_de_single_line_not_cut():
    body = "De: acuerdo con lo hablado, enviamos la factura hoy.\nSaludos"
    assert strip_quoted(body) == body.strip()


def test_spanish_and_chinese_blocks():
    es = "Adjunto la factura corregida para su aprobación.\n\nDe: Ana\nEnviado el: lunes\nPara: Kim\nAsunto: factura\n\nviejo"
    assert strip_quoted(es) == "Adjunto la factura corregida para su aprobación."
    zh = "请在周五前审批这张发票，谢谢配合。\n\n发件人: 王\n发送时间: 2024年3月5日\n收件人: Kim\n主题: 发票\n\n旧内容"
    assert strip_quoted(zh) == "请在周五前审批这张发票，谢谢配合。"


def test_wrote_line_and_quote_marks():
    body = "Agreed, go ahead with the payment.\n\nOn Mon, Mar 4, 2024 at 9:00 AM Kim <k@a.com> wrote:\n> old"
    assert strip_quoted(body) == "Agreed, go ahead with the payment."


def test_short_reply_kept_and_empty_returns_original():
    body = "OK\n\nFrom: Kim\nSent: Monday\nTo: Lee\n\nlong old text here"
    assert strip_quoted(body) == "OK"
    only_quote = "> old line 1\n> old line 2"
    assert strip_quoted(only_quote) == only_quote


# ---------- qwen3 formats ----------

def test_qwen3_formats():
    assert format_embed_query("법카 사용", task="T") == "Instruct: T\nQuery:법카 사용"
    assert format_rerank_pair("q", "d", instruction="I") == "<Instruct>: I\n<Query>: q\n<Document>: d"
