"""selftest — GPU·실데이터 없이 가짜 모델로 전체 흐름 점검 (지시서 §6.5, §7 5단계 완료 기준)

임시 폴더에 가짜 케이스 색인(세 프로필)·질의·판정을 만들고
corpus → run(dev·test) → pool → judgments import/agreement/finalize → evaluate → latency → failures → env
를 차례로 실행한 뒤 T1~T7·F1·E1 산출물이 생겼는지 확인한다. 실제 settings·data/ 는 건드리지 않는다.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ev4 import EXPERIMENTS_ROOT
from ev4.io import load_config, read_queries, write_jsonl
from src.utils.config import settings

CASE_ID = "selftest"
DIM = 32

# (unit_id, unit_type, lang, header, text) — 키워드가 질의와 겹치게 만든다
_UNITS: list[tuple[str, str, str, str, str]] = [
    ("em:001", "email", "ko", "[이메일] 제목: 3월 정산 결재 | 보낸이: 김철수", "3월 정산 건 결재를 생략하고 바로 지급했습니다 금액 4,500만원"),
    ("em:002", "email", "ko", "[이메일] 제목: 회식 공지 | 보낸이: 이영희", "금요일 회식 장소는 강남역 근처입니다"),
    ("em:003", "email", "en", "[이메일] 제목: Invoice INV-2026-0042 | 보낸이: John", "Please process invoice INV-2026-0042 without approval this time"),
    ("em:004", "email", "en", "[이메일] 제목: Weekly report | 보낸이: Mary", "Weekly sales report attached, nothing unusual"),
    ("em:005", "email", "zh", "[이메일] 제목: 付款 | 보낸이: 王伟", "请尽快付款 不要经过审批 发票号 INV-2026-0042"),
    ("em:006", "email", "ko", "[이메일] 제목: 거래처 변경 | 보낸이: 박민수", "거래처 계좌를 개인 계좌로 변경해 달라는 요청이 있었습니다 박민수 대리"),
    ("em:007", "email", "ko", "[이메일] 제목: 감사 일정 | 보낸이: 감사팀", "내부 감사 일정은 다음 달 둘째 주입니다"),
    ("ch:001", "chat", "ko", "[채팅] 방: 재무팀 | 참여자: 김철수, 박민수", "김철수: 결재 없이 처리해도 돼? 박민수: 이번만 조용히 넘어가자"),
    ("ch:002", "chat", "ko", "[채팅] 방: 잡담 | 참여자: 이영희, 최지훈", "점심 뭐 먹을까 김치찌개 어때"),
    ("ch:003", "chat", "en", "[채팅] 방: vendor | 참여자: John, 김철수", "John: keep this off the books please 김철수: understood"),
    ("doc:001:p1", "document", "ko", "[문서] 파일: 결재규정.pdf | 1쪽", "모든 지급은 팀장 결재 후 집행한다 예외는 없다"),
    ("doc:002:p1", "document", "en", "[문서] 파일: policy.docx | 1쪽", "Vendor bank account changes require dual approval"),
    ("att:001:p1", "attachment", "ko", "[첨부] 파일: 정산내역.xlsx | 부모: em:001", "정산 내역 4,500만원 지급 결재자 공란"),
    ("att:002:p1", "attachment", "zh", "[첨부] 파일: 发票.pdf | 부모: em:005", "发票 INV-2026-0042 金额 45000000"),
]
# 질의와 일부 단어만 겹치는 채움 단위 — 상위 10 경쟁을 만들어 실패 분석 행이 생기게 한다
_UNITS += [(f"em:f{i:02d}", "email", "ko", f"[이메일] 제목: 업무 연락 {i}", f"일반 업무 연락 {i}번 정황 보고 결재 일정")
           for i in range(1, 26)]
# 긴 단위(파트 2개)
_SPLIT = ("em:008", "email", "ko", "[이메일] 제목: 분기 보고 | 보낸이: 최지훈",
          ["1분기 매출 보고 본문 앞부분 일반 현황", "뒷부분에 리베이트 지급 내역 박민수 대리 승인"])

# (qid, text, type, split, 등급 2 단위, 등급 1 단위)
_QUERIES: list[tuple[str, str, str, str, list[str], list[str]]] = [
    ("q01", "3월 정산 결재 생략", "semantic", "test", ["em:001"], ["att:001:p1", "doc:001:p1"]),
    ("q02", "INV-2026-0042 송장", "exact", "test", ["em:003", "att:002:p1"], ["em:005"]),
    ("q03", "결재 없이 처리 조용히 넘어가자", "semantic", "test", ["ch:001"], ["em:001"]),
    ("q04", "박민수 대리 계좌 변경", "relation", "test", ["em:006"], ["doc:002:p1"]),
    ("q05", "4,500만원 지급", "exact", "test", ["att:001:p1"], ["em:001"]),
    ("q06", "리베이트 지급 박민수 승인", "relation", "test", ["em:008"], []),
    ("q07", "发票 付款 审批", "exact", "test", ["em:005"], ["att:002:p1"]),
    ("q08", "김철수 John 장부 외 처리", "relation", "test", ["ch:003"], ["ch:001"]),
    ("q09", "감사 일정 공지", "semantic", "test", [], ["em:007"]),
    # 결정적 증거(ch:003, 영어)와 단어가 겹치지 않음 → R8 상위 10 밖 → F1 행
    ("q10", "장부 누락 정황", "semantic", "test", ["ch:003"], []),
    ("d01", "회식 장소", "semantic", "dev", ["em:002"], []),
    ("d02", "weekly sales report", "exact", "dev", ["em:004"], []),
    ("d03", "결재규정 예외", "relation", "dev", ["doc:001:p1"], []),
]


@contextmanager
def _fake_settings(index_dir: Path) -> Iterator[None]:
    """가짜 모델·임시 색인 폴더로 settings 를 바꾸고 끝나면 되돌린다"""
    from src.index import case_index
    from src.rag import retrieval
    from src.rag.reranker import release_rerankers

    patch = {"ast_fake_models": True, "index_dir": str(index_dir), "embed_dim": DIM,
             "security_mode": "on", "unit_token_hf_name": None}
    saved = {k: getattr(settings, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(settings, k, v)
        case_index._cache.clear()
        retrieval._embedders.clear()
        release_rerankers()
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)
        case_index._cache.clear()
        retrieval._embedders.clear()
        release_rerankers()


def _build_case() -> None:
    from scripts.build_profiles import build_profile
    from src.embeddings.embedding_service import fake_embed
    from src.index import case_index
    from src.index.case_index import ProfileVectors, embed_input
    from src.units.schema import UnitPart

    parts: list[UnitPart] = []
    for i, (uid, utype, lang, header, text) in enumerate(_UNITS):
        meta = {"lang": lang, "lang_mixed": False, "date": f"2026-03-{i + 1:02d}"}
        parts.append(UnitPart(f"{uid}#1", uid, utype, header, text,
                              len(text) // 2, meta))
    uid, utype, lang, header, texts = _SPLIT
    for k, t in enumerate(texts, 1):
        parts.append(UnitPart(f"{uid}#{k}", uid, utype, header, t, len(t) // 2,
                              {"lang": lang, "lang_mixed": False, "date": "2026-03-20"}))
    vecs = np.array([fake_embed(embed_input(p, True), DIM) for p in parts])
    build_stats = {"counts": {"orphan_attachments": 1, "attachments_total": 3, "forwarded_emails": 1},
                   "clean_chars": {"ko": [1000, 800], "en": [500, 450], "zh": [200, 180]},
                   "duplicates_removed": {"email": 2}}
    case_index.write_index(CASE_ID, parts, {"hdr-qwen3": ProfileVectors(vecs, settings.ollama_embed_model)},
                           build_stats=build_stats)
    build_profile(CASE_ID, "nohdr-qwen3", batch=8)
    build_profile(CASE_ID, "hdr-bgem3", batch=8)


def _write_inputs(root: Path) -> None:
    src_cfg = yaml.safe_load((EXPERIMENTS_ROOT / "config.yaml").read_text(encoding="utf-8"))
    src_cfg.update({"case_id": CASE_ID, "latency_warmup": 1, "latency_repeats": 2})
    (root / "config.yaml").write_text(yaml.safe_dump(src_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_jsonl(root / "data" / "queries.jsonl",
                ({"qid": q, "text": t, "type": ty, "split": sp, "query_lang": "ko",
                  "target_evidence_lang": _TARGET_LANG.get(q, "ko"), "source_question": _SOURCE_Q.get(q, ""),
                  "note": ""}
                 for q, t, ty, sp, _g2, _g1 in _QUERIES))


# 선택 필드 (v4.2 Q2) — 예상 증거 언어, 부록 A 질문 번호
_TARGET_LANG = {"q02": "en", "q07": "zh", "q08": "en", "q10": "en", "d02": "en"}
_SOURCE_Q = {"q01": "1.3", "q02": "1.8", "q05": "1.8", "q07": "2.6", "q10": "1.10"}


def _grade(qid: str, uid: str) -> int:
    for q, _t, _ty, _sp, g2, g1 in _QUERIES:
        if q == qid:
            return 2 if uid in g2 else 1 if uid in g1 else 0
    return 0


def _fill_sheet(path: Path, annotator: str, flip: int = 0) -> int:
    """가짜 판정 입력. flip 개 행은 일부러 등급을 바꿔 불일치를 만든다"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    for i, r in enumerate(rows):
        g = _grade(r["qid"], r["unit_id"])
        if i < flip:
            g = (g + 1) % 3
        r["grade"], r["annotator"] = str(g), annotator
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _adjudicate(path: Path) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    for r in rows:
        r["adjudicated_grade"] = str(_grade(r["qid"], r["unit_id"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


EXPECTED = ["T0_queries.md", "T1_corpus.md", "tables.md", "per_query.csv", "comparisons.csv", "T7_latency.md",
            "F1_failures.csv", "F1_summary.md", "E1_env.md", "env.json", "agreement.json",
            "J1_pool.md", "J3_agreement.md", "J4_qrels.md"]
TABLE_HEADINGS = ["## T2", "## T3", "## 사전 지정 비교", "## T4", "## T5", "## T6"]


def _check_v42(cfg: Any, ev: dict[str, Any]) -> None:
    """v4.2 T4 — T0, T3 유형별 열, T5 Recall@10, E2 차이 CI, F1 category_hint·요약, J2 문서"""
    from ev4.pool import CRITERIA_NAME

    t0 = (cfg.results_dir / "T0_queries.md").read_text(encoding="utf-8")
    assert "## T0 질의 구성" in t0 and "현지어 증거(정답)" in t0 and "source_question" in t0, "T0 형식 이상"
    assert "| 계 | 10 | 3 |" in t0, "T0 계 행(평가용 10, 개발용 3) 이상"
    tables = (cfg.results_dir / "tables.md").read_text(encoding="utf-8")
    assert all(f"{t} 차이" in tables for t in ("exact", "semantic", "relation")), "T3 유형별 열 없음"
    t5 = tables.split("## T5", 1)[1].split("## T6", 1)[0]
    assert "Recall@10" in t5.splitlines()[0], "T5 지표가 Recall@10 이 아님"
    assert "증거 영어" in t5 and "교차 언어 질의" in t5, "T5 행 이상"
    assert "†사례 수준" in tables, "작은 그룹 표시(E8) 없음"
    with open(cfg.results_dir / "comparisons.csv", encoding="utf-8-sig", newline="") as f:
        comps = list(csv.DictReader(f))
    assert len(comps) == 11 and all(c["ci_lo"] and c["ci_hi"] and c["p_check"] for c in comps), "E2 CI 없음"
    assert all(float(c["ci_lo"]) <= float(c["diff"]) <= float(c["ci_hi"]) for c in comps), "E2 CI 가 차이를 안 감쌈"
    with open(cfg.results_dir / "F1_failures.csv", encoding="utf-8-sig", newline="") as f:
        frows = list(csv.DictReader(f))
    assert frows and "category_hint" in frows[0], "F1 category_hint 열 없음"
    assert any(r["category_hint"] for r in frows), "F1 힌트가 하나도 없음"
    f1 = (cfg.results_dir / "F1_summary.md").read_text(encoding="utf-8")
    assert "| 분류 | 단위 수 | 비율" in f1, "F1 요약 형식 이상"
    assert (cfg.judgments_dir / CRITERIA_NAME).exists(), "판정 기준 문서 없음"
    assert ev["n_l2"] == 9, "등급 2 단위 있는 질의 수 이상 (q09 제외)"


def selftest(root: Path | None = None, keep: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    """전체 흐름 점검

    Args:
        root: 실험 루트로 쓸 폴더 (기본: 임시 폴더)
        keep: True 면 끝난 뒤 폴더를 지우지 않는다

    Returns:
        {root, results: [파일], evaluation, failures}

    Raises:
        AssertionError: 산출물이 빠졌을 때
    """
    from ev4 import corpus, evaluate, failures, judgments, latency, pool, querystats, run
    from ev4 import env as env_mod

    tmp = Path(root) if root else Path(tempfile.mkdtemp(prefix="ev4_selftest_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with _fake_settings(tmp / "index"):
            _build_case()
            _write_inputs(tmp)
            cfg = load_config(root=tmp)
            corpus.build_corpus(cfg, log=log)
            querystats.write_t0(cfg, read_queries(cfg.queries_path), log=log)
            dev = read_queries(cfg.queries_path, split="dev")
            test = read_queries(cfg.queries_path, split="test")
            run.run_all(cfg, dev, ["R2", "R4", "R6", "R8", "R9"], "dev", log=log)
            run.run_all(cfg, test, list(cfg.runs), "test", log=log)
            pool.build_pool(cfg, test, log=log)
            n_a = _fill_sheet(cfg.judgments_dir / "pool_A.csv", "A")
            n_b = _fill_sheet(cfg.judgments_dir / "pool_B.csv", "B", flip=2)
            assert n_a > 0 and 0 < n_b < n_a, f"판정 시트 행 수 이상 (A {n_a}, B {n_b})"
            judgments.import_judgments(cfg, [cfg.judgments_dir / "pool_A.csv", cfg.judgments_dir / "pool_B.csv"],
                                       log=log)
            agr = judgments.agreement(cfg, log=log)
            assert agr.get("n_disagree", 0) >= 1, "selftest 는 불일치를 일부러 만든다"
            try:
                judgments.finalize(cfg, log=log)
                raise AssertionError("미중재 불일치가 있는데 finalize 가 통과함")
            except judgments.JudgmentError:
                pass
            _adjudicate(judgments.adjudication_path(cfg))
            judgments.finalize(cfg, log=log)
            ev = evaluate.evaluate_all(cfg, log=log)
            latency.measure(cfg, test, log=log)
            fl = failures.build_failures(cfg, log=log)
            env_mod.write_env(cfg, log=log)
            assert fl["n_rows"] >= 1, "selftest 는 실패 분석 행(q10)을 일부러 만든다"

            missing = [f for f in EXPECTED if not (cfg.results_dir / f).exists()]
            assert not missing, f"산출물 없음: {missing}"
            tables = (cfg.results_dir / "tables.md").read_text(encoding="utf-8")
            absent = [h for h in TABLE_HEADINGS if h not in tables]
            assert not absent, f"tables.md 에 표 없음: {absent}"
            assert len(ev["comparisons"]) == len(cfg.comparison_pairs()), "비교 수 불일치"
            _check_v42(cfg, ev)
            log(f"[selftest] 통과 — T0~T7·J1·J3·J4·F1·E1 생성 ({cfg.results_dir})")
            return {"root": str(tmp), "results": sorted(p.name for p in cfg.results_dir.iterdir()),
                    "evaluation": ev, "failures": fl}
    finally:
        if not keep and root is None:
            shutil.rmtree(tmp, ignore_errors=True)
