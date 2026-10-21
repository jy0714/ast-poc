"""selftest — GPU·실데이터 없이 가짜 모델로 전체 흐름 점검 (지시서 §6.5, §7 5단계 완료 기준)

임시 폴더에 가짜 케이스 색인(두 프로필, v4.4 §1)·질의·판정을 만들고
corpus → queries → smoke → run → pool(A·B·C) → judgments import/agreement/finalize → evaluate → latency(답변 기록)
→ judge gen → 가짜 답변 판정·협의 → judge g1 → failures → env
를 차례로 실행한 뒤 T0~T8·F1·E1 산출물이 생겼는지 확인한다. 실제 settings·data/ 는 건드리지 않는다.
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
from ev4.io import SUPPORT_CLASS_GROUP, load_config, read_queries, write_jsonl
from src.utils.config import settings

CASE_ID = "selftest"
DIM = 32

# (unit_id, unit_type, lang, header, text) — 키워드가 질의와 겹치게 만든다
_UNITS: list[tuple[str, str, str, str, str]] = [
    (
        "em:001",
        "email",
        "ko",
        "[이메일] 제목: 3월 정산 결재 | 보낸이: 김철수",
        "3월 정산 건 결재를 생략하고 바로 지급했습니다 금액 4,500만원",
    ),
    (
        "em:002",
        "email",
        "ko",
        "[이메일] 제목: 회식 공지 | 보낸이: 이영희",
        "금요일 회식 장소는 강남역 근처입니다",
    ),
    (
        "em:003",
        "email",
        "en",
        "[이메일] 제목: Invoice INV-2026-0042 | 보낸이: John",
        "Please process invoice INV-2026-0042 without approval this time",
    ),
    (
        "em:004",
        "email",
        "en",
        "[이메일] 제목: Weekly report | 보낸이: Mary",
        "Weekly sales report attached, nothing unusual",
    ),
    (
        "em:005",
        "email",
        "zh",
        "[이메일] 제목: 付款 | 보낸이: 王伟",
        "请尽快付款 不要经过审批 发票号 INV-2026-0042",
    ),
    (
        "em:006",
        "email",
        "ko",
        "[이메일] 제목: 거래처 변경 | 보낸이: 박민수",
        "거래처 계좌를 개인 계좌로 변경해 달라는 요청이 있었습니다 박민수 대리",
    ),
    (
        "em:007",
        "email",
        "ko",
        "[이메일] 제목: 감사 일정 | 보낸이: 감사팀",
        "내부 감사 일정은 다음 달 둘째 주입니다",
    ),
    (
        "ch:001",
        "chat",
        "ko",
        "[채팅] 방: 재무팀 | 참여자: 김철수, 박민수",
        "김철수: 결재 없이 처리해도 돼? 박민수: 이번만 조용히 넘어가자",
    ),
    (
        "ch:002",
        "chat",
        "ko",
        "[채팅] 방: 잡담 | 참여자: 이영희, 최지훈",
        "점심 뭐 먹을까 김치찌개 어때",
    ),
    (
        "ch:003",
        "chat",
        "en",
        "[채팅] 방: vendor | 참여자: John, 김철수",
        "John: keep this off the books please 김철수: understood",
    ),
    (
        "doc:001:p1",
        "document",
        "ko",
        "[문서] 파일: 결재규정.pdf | 1쪽",
        "모든 지급은 팀장 결재 후 집행한다 예외는 없다",
    ),
    (
        "doc:002:p1",
        "document",
        "en",
        "[문서] 파일: policy.docx | 1쪽",
        "Vendor bank account changes require dual approval",
    ),
    (
        "att:001:p1",
        "attachment",
        "ko",
        "[첨부] 파일: 정산내역.xlsx | 부모: em:001",
        "정산 내역 4,500만원 지급 결재자 공란",
    ),
    (
        "att:002:p1",
        "attachment",
        "zh",
        "[첨부] 파일: 发票.pdf | 부모: em:005",
        "发票 INV-2026-0042 金额 45000000",
    ),
]
# 질의와 일부 단어만 겹치는 채움 단위 — 상위 10 경쟁을 만들어 실패 분석 행이 생기게 한다
_UNITS += [
    (
        f"em:f{i:02d}",
        "email",
        "ko",
        f"[이메일] 제목: 업무 연락 {i}",
        f"일반 업무 연락 {i}번 정황 보고 결재 일정",
    )
    for i in range(1, 26)
]
# 긴 단위(파트 2개)
_SPLIT = (
    "em:008",
    "email",
    "ko",
    "[이메일] 제목: 분기 보고 | 보낸이: 최지훈",
    ["1분기 매출 보고 본문 앞부분 일반 현황", "뒷부분에 리베이트 지급 내역 박민수 대리 승인"],
)

# (qid, text, type, area, support_class, 등급 2 단위, 등급 1 단위) — v4.3 형식, 모두 test
_QUERIES: list[tuple[str, str, str, str, str, list[str], list[str]]] = [
    ("q01", "3월 정산 결재 생략", "semantic", "1", "A1", ["em:001"], ["att:001:p1", "doc:001:p1"]),
    ("q02", "INV-2026-0042 송장", "exact", "1", "A1", ["em:003", "att:002:p1"], ["em:005"]),
    ("q03", "결재 없이 처리 조용히 넘어가자", "semantic", "2", "A2", ["ch:001"], ["em:001"]),
    ("q04", "박민수 대리 계좌 변경", "relation", "2", "A1", ["em:006"], ["doc:002:p1"]),
    ("q05", "4,500만원 지급", "exact", "3", "A2", ["att:001:p1"], ["em:001"]),
    ("q06", "리베이트 지급 박민수 승인", "relation", "4", "A2", ["em:008"], []),
    ("q07", "发票 付款 审批", "none", "4", "C", ["em:005"], ["att:002:p1"]),
    ("q08", "김철수 John 장부 외 처리", "relation", "7", "A1", ["ch:003"], ["ch:001"]),
    ("q09", "감사 일정 공지", "none", "5", "D", [], ["em:007"]),
    # 결정적 증거(ch:003, 영어)와 단어가 겹치지 않음 → R8 상위 10 밖 → F1 행
    ("q10", "장부 누락 정황", "semantic", "7", "A1", ["ch:003"], []),
    ("q11", "회식 장소", "none", "6", "B", ["em:002"], []),
    ("q12", "weekly sales report", "none", "6", "E", ["em:004"], []),
    ("q13", "결재규정 예외", "none", "3", "D", ["doc:001:p1"], []),
]
SMOKE_QIDS = ["q01", "q04", "q10"]


@contextmanager
def _fake_settings(index_dir: Path) -> Iterator[None]:
    """가짜 모델·임시 색인 폴더로 settings 를 바꾸고 끝나면 되돌린다"""
    from src.index import case_index
    from src.rag import retrieval
    from src.rag.reranker import release_rerankers

    patch = {
        "ast_fake_models": True,
        "index_dir": str(index_dir),
        "embed_dim": DIM,
        "security_mode": "on",
        "unit_token_hf_name": None,
    }
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
        parts.append(UnitPart(f"{uid}#1", uid, utype, header, text, len(text) // 2, meta))
    uid, utype, lang, header, texts = _SPLIT
    for k, t in enumerate(texts, 1):
        parts.append(
            UnitPart(
                f"{uid}#{k}",
                uid,
                utype,
                header,
                t,
                len(t) // 2,
                {"lang": lang, "lang_mixed": False, "date": "2026-03-20"},
            )
        )
    vecs = np.array([fake_embed(embed_input(p, True), DIM) for p in parts])
    build_stats = {
        "counts": {"orphan_attachments": 1, "attachments_total": 3, "forwarded_emails": 1},
        "clean_chars": {"ko": [1000, 800], "en": [500, 450], "zh": [200, 180]},
        "duplicates_removed": {"email": 2},
    }
    case_index.write_index(
        CASE_ID,
        parts,
        {"hdr-qwen3": ProfileVectors(vecs, settings.ollama_embed_model)},
        build_stats=build_stats,
    )
    build_profile(CASE_ID, "nohdr-qwen3", batch=8)


def _write_inputs(root: Path) -> None:
    src_cfg = yaml.safe_load((EXPERIMENTS_ROOT / "config.yaml").read_text(encoding="utf-8"))
    src_cfg.update(
        {
            "case_id": CASE_ID,
            "latency_warmup": 1,
            "latency_repeats": 2,
            "smoke_qids": SMOKE_QIDS,
            "c_queries": {"search": 2, "meta": 1, "nonsearch": 1},
        }
    )  # 판정자 C (v4.4 §9)
    (root / "config.yaml").write_text(
        yaml.safe_dump(src_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_jsonl(
        root / "data" / "queries.jsonl",
        (
            {
                "qid": q,
                "text": t,
                "area": area,
                "support_class": cls,
                "support_group": SUPPORT_CLASS_GROUP[cls],
                "type": ty,
                "split": "test",
                "query_lang": "ko",
                "target_evidence_lang": "",
                "source_question": q[1:],
                "note": "",
            }
            for q, t, ty, area, cls, _g2, _g1 in _QUERIES
        ),
    )


def _grade(qid: str, uid: str) -> int:
    for q, _t, _ty, _area, _cls, g2, g1 in _QUERIES:
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


def _fill_gen(cfg: Any, ann: str, flip: int) -> None:
    """가짜 답변 판정 — 인용은 모두 뒷받침(flip 개는 반대로), 유용성 2, 근거 없는 주장 0"""
    for kind in ("citations", "answers"):
        path = cfg.judgments_dir / f"gen_{ann}_{kind}.csv"
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
        for i, r in enumerate(rows):
            if kind == "citations":
                r["supported"] = str(0 if i < flip else 1)
            else:
                r["unsupported_claims"], r["usefulness"] = (
                    "0",
                    str(2 if r["n_citations"] != "0" else 1),
                )
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def _adjudicate_gen(path: Path) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    for r in rows:
        r["final"] = r["value_A"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _check_g1(g1: dict[str, Any]) -> None:
    """v4.4 §4 — G1 지표·일치도 형식"""
    assert g1["n_questions"] == 8 and g1["n_adjudicated"] >= 1, "G1 질문 수·협의 이상"
    ca = g1["citation_accuracy"]
    assert ca["n_citations"] > 0 and ca["pooled"] == 1.0, (
        "G1 인용 정확도 이상 (협의 후 모두 뒷받침)"
    )
    assert set(g1["agreement"]) == {"A-B", "A-C"}, "G1 일치도 쌍 이상"
    assert g1["decisive_citation"]["n_questions"] >= 1, "G1 결정적 증거 인용률 질문 없음"
    assert sum(g1["usefulness"]["dist"].values()) == 8, "G1 유용성 분포 이상"


EXPECTED = [
    "T0_queries.md",
    "T0b_relevant.md",
    "smoke.md",
    "T1_corpus.md",
    "tables.md",
    "per_query.csv",
    "comparisons.csv",
    "T7_latency.md",
    "G1_generation.md",
    "G1.json",
    "F1_failures.csv",
    "F1_summary.md",
    "E1_env.md",
    "env.json",
    "agreement.json",
    "J1_pool.md",
    "J3_agreement.md",
    "J4_qrels.md",
]
TABLE_HEADINGS = ["## T2", "## T2 부가 — ALL45", "## 사전 지정 비교", "## T8", "## 언어별 보조 값"]
REMOVED_HEADINGS = ["## T3", "## T4", "## T5", "## T6"]  # v4.4 §2


def _check_v43(cfg: Any, ev: dict[str, Any]) -> None:
    """v4.3 + v4.4 — T0·T0b, P@10, T2 논문 번호, T8 4열, 언어별 보조 값, E2 차이 CI, F1 S17 한정, J2 문서"""
    from ev4.pool import AGGREGATE_GUIDE, CRITERIA_NAME

    t0 = (cfg.results_dir / "T0_queries.md").read_text(encoding="utf-8")
    assert "## T0 질의 구성" in t0 and "| 영역 | A1 | A2 | C | D | B | E | 계 |" in t0, (
        "T0 형식 이상"
    )
    assert "| 계 | 5 | 3 | 1 | 2 | 1 | 1 | 13 | 2 | 3 | 3 |" in t0, "T0 계 행 이상"
    t0b = (cfg.results_dir / "T0b_relevant.md").read_text(encoding="utf-8")
    assert "| S17 (search) | 8 |" in t0b and "| 전체 (ALL45) | 13 |" in t0b, "T0b 행 이상"
    tables = (cfg.results_dir / "tables.md").read_text(encoding="utf-8")
    assert (
        "| nDCG@10 | P@10 | Recall@10 | Recall@10(등급 2) (n=8) | MRR@10 | Recall@100 |" in tables
    ), "T2 열 순서 이상"
    t8 = tables.split("## T8", 1)[1]
    t8_head = next(ln for ln in t8.splitlines() if ln.startswith("| 집합 |"))
    assert all(f"| {g} |" in t8 for g in ("S17 (search)", "meta", "nonsearch")), "T8 행 이상"
    assert [c.split()[0] for c in t8_head.split("|")[3::2] if c.strip()] == [
        "R1",
        "R2",
        "R3",
        "R5",
    ], "T8 열 이상"
    assert not any(h in tables for h in REMOVED_HEADINGS), "v4.4 에서 뺀 표(T3~T6)가 남음"
    t2 = tables.split("## T2", 1)[1].split("## T2 부가", 1)[0]
    assert [ln.split("|")[1].strip() for ln in t2.splitlines() if ln.startswith("| R")] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ], "T2 행이 논문 번호 R1~R5 가 아님"
    aux = ev["language_aux"]
    assert aux["n_local_queries"] >= 1 and set(aux["hits"]) == {"R1", "R2", "R5"}, (
        "언어별 보조 값 이상"
    )
    with open(cfg.results_dir / "comparisons.csv", encoding="utf-8-sig", newline="") as f:
        comps = list(csv.DictReader(f))
    assert len(comps) == 4 and all(c["ci_lo"] and c["ci_hi"] and c["p_check"] for c in comps), (
        "E2 CI 없음"
    )
    assert [c["comparison_label"] for c in comps] == ["R1→R3", "R2→R3", "R3→R5", "R4→R5"], (
        "사전 비교 4개 이상"
    )
    assert all(float(c["ci_lo"]) <= float(c["diff"]) <= float(c["ci_hi"]) for c in comps), (
        "E2 CI 가 차이를 안 감쌈"
    )
    with open(cfg.results_dir / "F1_failures.csv", encoding="utf-8-sig", newline="") as f:
        frows = list(csv.DictReader(f))
    assert frows and "category_hint" in frows[0], "F1 category_hint 열 없음"
    assert any(r["category_hint"] for r in frows), "F1 힌트가 하나도 없음"
    f1 = (cfg.results_dir / "F1_summary.md").read_text(encoding="utf-8")
    assert "| 분류 | 단위 수 | 비율" in f1, "F1 요약 형식 이상"
    assert (cfg.judgments_dir / CRITERIA_NAME).exists(), "판정 기준 문서 없음"
    assert {r["qid"] for r in frows} <= {q for q, _t, ty, *_r in _QUERIES if ty != "none"}, (
        "F1 에 S17 밖 질의"
    )
    assert AGGREGATE_GUIDE in (cfg.judgments_dir / CRITERIA_NAME).read_text(encoding="utf-8"), (
        "판정 기준 보강 없음"
    )
    assert (ev["n_queries"], ev["n_l2"], ev["n_queries_all"]) == (8, 8, 13), (
        "집합별 평가 질의 수 이상"
    )
    assert all(c["n"] == 8 for c in ev["comparisons"]), "검정이 S17 이 아님"


def selftest(
    root: Path | None = None, keep: bool = False, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """전체 흐름 점검

    Args:
        root: 실험 루트로 쓸 폴더 (기본: 임시 폴더)
        keep: True 면 끝난 뒤 폴더를 지우지 않는다

    Returns:
        {root, results: [파일], evaluation, failures}

    Raises:
        AssertionError: 산출물이 빠졌을 때
    """
    from ev4 import corpus, evaluate, failures, gen, judgments, latency, pool, querystats, run
    from ev4 import env as env_mod
    from ev4 import smoke as smoke_mod

    tmp = Path(root) if root else Path(tempfile.mkdtemp(prefix="ev4_selftest_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with _fake_settings(tmp / "index"):
            _build_case()
            _write_inputs(tmp)
            cfg = load_config(root=tmp)
            corpus.build_corpus(cfg, log=log)
            test = read_queries(cfg.queries_path, split="test")
            querystats.write_t0(cfg, test, log=log)
            sm = smoke_mod.smoke(cfg, test, log=log)
            assert all(v["errors"] == 0 for v in sm["runs"].values()), "동작 점검 오류"
            cfg.check_subset_lock(write=True)
            run.run_all(cfg, test, list(cfg.runs), "test", log=log)
            pool.build_pool(cfg, test, log=log)
            n_a = _fill_sheet(cfg.judgments_dir / "pool_A.csv", "A")
            n_b = _fill_sheet(cfg.judgments_dir / "pool_B.csv", "B", flip=2)
            n_c = _fill_sheet(cfg.judgments_dir / "pool_C.csv", "C", flip=1)
            assert n_a > 0 and 0 < n_b < n_a and 0 < n_c < n_a, (
                f"판정 시트 행 수 이상 (A {n_a}, B {n_b}, C {n_c})"
            )
            judgments.import_judgments(
                cfg, [cfg.judgments_dir / f"pool_{x}.csv" for x in "ABC"], log=log
            )
            agr = judgments.agreement(cfg, log=log)
            assert agr.get("n_disagree", 0) >= 1, "selftest 는 불일치를 일부러 만든다"
            assert set(agr["pairs"]) == {"A-B", "A-C"}, (
                "일치도가 A–B, A–C 따로 나오지 않음 (v4.4 §9)"
            )
            b_q, c_q = set(agr["pairs"]["A-B"]["qids_B"]), set(agr["pairs"]["A-C"]["qids_B"])
            assert c_q and not (b_q & c_q), "판정자 C 질의가 B 와 겹침"
            try:
                judgments.finalize(cfg, log=log)
                raise AssertionError("미중재 불일치가 있는데 finalize 가 통과함")
            except judgments.JudgmentError:
                pass
            _adjudicate(judgments.adjudication_path(cfg))
            judgments.finalize(cfg, log=log)
            ev = evaluate.evaluate_all(cfg, log=log)
            lat = latency.measure(cfg, test, log=log)
            assert lat["answers"]["n"] == len(test) and lat["answers"]["with_citation"] >= 1, (
                "답변 기록 이상"
            )
            gs = gen.build_gen_sheets(cfg, log=log)
            assert gs["n_questions"] == 8 and gs["written"] == ["A", "B", "C"], (
                "답변 판정 시트 이상"
            )
            for ann, flip in (("A", 0), ("B", 1), ("C", 0)):
                _fill_gen(cfg, ann, flip)
            try:
                gen.g1(cfg, log=log)
                raise AssertionError("협의되지 않은 답변 판정 불일치가 있는데 g1 이 통과함")
            except gen.G1Error:
                pass
            _adjudicate_gen(gen.adjudication_path(cfg))
            g1 = gen.g1(cfg, log=log)
            _check_g1(g1)
            fl = failures.build_failures(cfg, log=log)
            env_mod.write_env(cfg, log=log)
            assert fl["n_rows"] >= 1, "selftest 는 실패 분석 행(q10)을 일부러 만든다"

            missing = [f for f in EXPECTED if not (cfg.results_dir / f).exists()]
            assert not missing, f"산출물 없음: {missing}"
            tables = (cfg.results_dir / "tables.md").read_text(encoding="utf-8")
            absent = [h for h in TABLE_HEADINGS if h not in tables]
            assert not absent, f"tables.md 에 표 없음: {absent}"
            assert len(ev["comparisons"]) == len(cfg.comparison_pairs()), "비교 수 불일치"
            _check_v43(cfg, ev)
            log(f"[selftest] 통과 — T0~T8·J1·J3·J4·G1·F1·E1 생성 ({cfg.results_dir})")
            return {
                "root": str(tmp),
                "results": sorted(p.name for p in cfg.results_dir.iterdir()),
                "evaluation": ev,
                "failures": fl,
            }
    finally:
        if not keep and root is None:
            shutil.rmtree(tmp, ignore_errors=True)
