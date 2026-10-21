"""응답 생성 평가 G1 (v4.4 §4, 논문 RQ3·4.6절·[표 14]) — 답변 판정 시트(`judge gen`)·G1 계산(`judge g1`)

- 입력: `latency` 가 남긴 답변 기록(runs 폴더 gen/answers.jsonl, P0). 다시 생성하지 않는다.
  표·판정 시트는 주 분석 집합(S17)만 만든다.
- 판정 시트(판정자마다, 질문 순서는 seed 고정 셔플):
  - `gen_{X}.md`: 질문 문장, 답변 전문(문장 번호 S1…·인용 번호 유지), 컨텍스트 단위 5개(헤더 + 본문,
    text_max_chars 절단, 현지 언어 단위는 로컬 LLM 번역 첨부). 단위 등급은 보여 주지 않는다.
  - `gen_{X}_citations.csv`: (문장 번호, 인용 번호)마다 한 행 — supported 1(뒷받침함)/0(뒷받침하지 않음)
  - `gen_{X}_answers.csv`: 답변마다 한 행 — unsupported_claims(근거 없는 주장 수), no_answer(무근거 응답
    여부, "찾을 수 없다"류 자동 판정값을 미리 채움 → 판정자 확인), usefulness(2 바로 활용 / 1 참고 / 0 무용)
- 판정자 A·B(·C, config `c_queries` 가 있으면)가 모두 S17 전체를 독립 판정한다.
- G1: 판정자 쌍(A–B, A–C) 일치도(인용별 Cohen's kappa, 유용성 QWK n_levels=3, 근거 없는 주장 수 완전 일치율)
  → 불일치는 `gen_adjudication.csv` 의 final 칸에 협의 값을 넣는다 → 최종값으로 G1 지표.
  결정적 증거 인용률·무근거 응답은 qrels(검색 판정 최종 등급)를 쓴다.
- 보고서(results/G1_generation.md, G1.json)에는 집계값만 넣는다. 답변 본문은 원문에 해당한다(P0).
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev4.io import ExperimentConfig, read_jsonl, read_qrels, read_queries, subset_qids
from ev4.stats import cohen_kappa, quadratic_weighted_kappa
from ev4.translate import NO_TRANSLATE_LANGS, Translator

CIT_COLUMNS = ["order", "qid", "sentence_no", "citation_no", "supported", "annotator", "note"]
ANS_COLUMNS = [
    "order",
    "qid",
    "n_citations",
    "no_answer_auto",
    "no_answer",
    "unsupported_claims",
    "usefulness",
    "annotator",
    "note",
]
ITEMS = ("citation", "unsupported_claims", "usefulness", "no_answer")
USEFULNESS = (0, 1, 2)
TRUNC_NOTE = "…(이하 생략)"
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_CITE = re.compile(r"\[출처\s*(\d+)\]")


class G1Error(ValueError):
    """답변 판정 시트 형식 오류·빈 칸·미협의 불일치"""


def answers_path(cfg: ExperimentConfig) -> Path:
    """답변 기록 (latency 가 쓴다)"""
    return cfg.runs_dir / "gen" / "answers.jsonl"


def adjudication_path(cfg: ExperimentConfig) -> Path:
    """답변 판정 협의 시트"""
    return cfg.judgments_dir / "gen_adjudication.csv"


def gen_annotators(cfg: ExperimentConfig) -> list[str]:
    """답변 판정자 — A·B, 판정자 C 가 설정돼 있으면 C 도 (v4.4 §9: C 도 17개 전체)"""
    return ["A", "B", *(["C"] if cfg.c_queries else [])]


def split_sentences(text: str) -> list[str]:
    """답변 → 문장 목록 (마침표·물음표·느낌표 뒤 공백, 줄바꿈 기준). 빈 문장은 뺀다"""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def citation_rows(answer: str) -> list[tuple[int, int]]:
    """(문장 번호, 인용 번호) — 같은 문장 안 같은 인용은 한 번, 나온 순서"""
    out: list[tuple[int, int]] = []
    for i, sent in enumerate(split_sentences(answer), 1):
        for m in _CITE.findall(sent):
            key = (i, int(m))
            if key not in out:
                out.append(key)
    return out


def _numbered(answer: str) -> str:
    return "\n".join(f"(S{i}) {s}" for i, s in enumerate(split_sentences(answer), 1)) or "(빈 답변)"


def _write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_answers(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    path = answers_path(cfg)
    if not path.exists():
        raise G1Error(f"답변 기록이 없습니다: {path} (먼저 latency 실행)")
    return {r["qid"]: r for r in read_jsonl(path)}


def build_gen_sheets(
    cfg: ExperimentConfig,
    translate: bool = True,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """답변 판정 시트 (S17, 판정자마다 md + 인용 CSV + 답변 CSV)

    이미 있는 시트는 판정 내용을 지우지 않도록 건너뛴다 (force=True 면 다시 만든다).

    Returns:
        {n_questions, n_citations, n_no_citation, annotators, written}
    """
    from src.index import case_index
    from src.rag.engine import is_refusal

    answers = _load_answers(cfg)
    queries = {q.qid: q for q in read_queries(cfg.queries_path, split="test")}
    qids = [q for q in subset_qids(list(queries.values()), cfg.primary_subset) if q in answers]
    missing = sorted(set(subset_qids(list(queries.values()), cfg.primary_subset)) - set(answers))
    if missing:
        raise G1Error(
            f"{cfg.primary_subset} 질의 {len(missing)}건의 답변 기록이 없습니다 (예: {missing[:3]})"
        )
    idx = case_index.load(cfg.case_id)
    translator = Translator(cfg.cache_dir / "translations.jsonl") if translate else None

    def _unit_block(n: int, uid: str) -> list[str]:
        parts = idx.unit_parts(uid)
        if not parts:
            return [f"### [출처 {n}] (색인에 없는 단위 {uid})", ""]
        text = idx.unit_text(uid)
        if len(text) > cfg.text_max_chars:
            text = text[: cfg.text_max_chars] + TRUNC_NOTE
        lang = str(parts[0].meta.get("lang", ""))
        out = [
            f"### [출처 {n}] {uid} · {parts[0].unit_type} · 언어 {lang or '-'}",
            "",
            parts[0].header,
            "",
            text,
            "",
        ]
        if translator and lang not in NO_TRANSLATE_LANGS:
            out += ["> 번역(참고용):", "", translator.translate(text), ""]
        return out

    written: list[str] = []
    n_cit = sum(len(citation_rows(answers[q]["answer"])) for q in qids)
    for ann in gen_annotators(cfg):
        md_path = cfg.judgments_dir / f"gen_{ann}.md"
        cit_path = cfg.judgments_dir / f"gen_{ann}_citations.csv"
        ans_path = cfg.judgments_dir / f"gen_{ann}_answers.csv"
        if not force and (cit_path.exists() or ans_path.exists()):
            log(f"[judge gen] 판정자 {ann} 시트가 이미 있어 건너뜀 (다시 만들려면 --force)")
            continue
        order = qids[:]
        random.Random(f"{cfg.seed}:gen:{ann}").shuffle(order)
        md = [
            f"# 답변 판정 시트 — 판정자 {ann}",
            "",
            "- 질문마다: 답변 문장(S 번호)의 인용 [출처 n] 이 그 컨텍스트 단위로 뒷받침되는지 "
            f"`gen_{ann}_citations.csv` 의 supported 에 1/0, 답변별 근거 없는 주장 수·무근거 응답(1/0, 자동값 확인)·"
            f"유용성(2 바로 활용 / 1 참고 / 0 무용)을 `gen_{ann}_answers.csv` 에 적는다.",
            "- 판정은 원문 기준. 번역은 참고용이다. 단위 등급은 보여 주지 않는다.",
            "",
        ]
        cit_rows: list[dict[str, Any]] = []
        ans_rows: list[dict[str, Any]] = []
        for k, qid in enumerate(order, 1):
            a = answers[qid]
            rows = citation_rows(a["answer"])
            auto = int(bool(a.get("refusal")) or is_refusal(a["answer"]))
            md += [
                f"## {k}. {qid}",
                "",
                f"**질문**: {queries[qid].text}",
                "",
                "**답변**",
                "",
                _numbered(a["answer"]),
                "",
                f"**컨텍스트 단위 {len(a['context_unit_ids'])}개**",
                "",
            ]
            for n, uid in enumerate(a["context_unit_ids"], 1):
                md += _unit_block(n, uid)
            cit_rows += [
                {
                    "order": k,
                    "qid": qid,
                    "sentence_no": s,
                    "citation_no": c,
                    "supported": "",
                    "annotator": ann,
                    "note": "" if 1 <= c <= len(a["context_unit_ids"]) else "없는 출처 번호",
                }
                for s, c in rows
            ]
            ans_rows.append(
                {
                    "order": k,
                    "qid": qid,
                    "n_citations": len(rows),
                    "no_answer_auto": auto,
                    "no_answer": auto,
                    "unsupported_claims": "",
                    "usefulness": "",
                    "annotator": ann,
                    "note": "" if rows else "인용 없음",
                }
            )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        _write_csv(cit_path, CIT_COLUMNS, cit_rows)
        _write_csv(ans_path, ANS_COLUMNS, ans_rows)
        written.append(ann)
    if translator:
        translator.save()
    n_no = sum(1 for q in qids if not citation_rows(answers[q]["answer"]))
    log(
        f"[judge gen] 질문 {len(qids)}개 · 인용 {n_cit}개 · 인용 없는 답변 {n_no}개 · "
        f"판정자 {', '.join(written) or '없음(모두 있음)'} → {cfg.judgments_dir}"
    )
    return {
        "n_questions": len(qids),
        "n_citations": n_cit,
        "n_no_citation": n_no,
        "annotators": gen_annotators(cfg),
        "written": written,
    }


# ─────────────────────────── G1 ───────────────────────────


def _int(v: str, allowed: tuple[int, ...] | None, where: str) -> int:
    try:
        x = int(str(v).strip())
    except ValueError:
        raise G1Error(f"{where}: 정수가 아닙니다 ({v!r})") from None
    if (allowed is not None and x not in allowed) or x < 0:
        raise G1Error(f"{where}: 허용 값 {allowed or '0 이상'} (입력 {x})")
    return x


def _read_annotator(cfg: ExperimentConfig, ann: str) -> dict[tuple[str, str, str, str], int]:
    """판정자 한 명의 판정 → {(항목, qid, 문장, 인용): 값}. 빈 칸이 있으면 G1Error"""
    out: dict[tuple[str, str, str, str], int] = {}
    blanks = 0
    for kind, cols in (
        ("citations", ("supported",)),
        ("answers", ("unsupported_claims", "usefulness", "no_answer")),
    ):
        path = cfg.judgments_dir / f"gen_{ann}_{kind}.csv"
        with open(path, encoding="utf-8-sig", newline="") as f:
            for ln, r in enumerate(csv.DictReader(f), 2):
                for col in cols:
                    v = (r.get(col) or "").strip()
                    if not v:
                        blanks += 1
                        continue
                    where = f"{path.name}:{ln} {col}"
                    if col == "supported":
                        out[("citation", r["qid"], r["sentence_no"], r["citation_no"])] = _int(
                            v, (0, 1), where
                        )
                    elif col == "usefulness":
                        out[("usefulness", r["qid"], "", "")] = _int(v, USEFULNESS, where)
                    elif col == "no_answer":
                        out[("no_answer", r["qid"], "", "")] = _int(v, (0, 1), where)
                    else:
                        out[("unsupported_claims", r["qid"], "", "")] = _int(v, None, where)
    if blanks:
        raise G1Error(f"판정자 {ann}: 빈 칸 {blanks}개 — 모두 채운 뒤 다시 실행하세요")
    return out


def _pair_agreement(va: dict, vb: dict) -> dict[str, Any]:
    keys = sorted(set(va) & set(vb))

    def _vals(item: str) -> tuple[list[int], list[int]]:
        ks = [k for k in keys if k[0] == item]
        return [va[k] for k in ks], [vb[k] for k in ks]

    ca, cb = _vals("citation")
    ua, ub = _vals("usefulness")
    xa, xb = _vals("unsupported_claims")
    na, nb = _vals("no_answer")
    return {
        "n_citations": len(ca),
        "citation_kappa": cohen_kappa(ca, cb),
        "citation_agreement": sum(x == y for x, y in zip(ca, cb, strict=True)) / len(ca)
        if ca
        else math.nan,
        "n_answers": len(ua),
        "usefulness_qwk": quadratic_weighted_kappa(ua, ub, 3),
        "unsupported_exact": sum(x == y for x, y in zip(xa, xb, strict=True)) / len(xa)
        if xa
        else math.nan,
        "no_answer_agreement": sum(x == y for x, y in zip(na, nb, strict=True)) / len(na)
        if na
        else math.nan,
    }


def _read_adjudication(path: Path) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {
            (r["item"], r["qid"], r["sentence_no"], r["citation_no"]): (
                (r.get("final") or "").strip(),
                r.get("note") or "",
            )
            for r in csv.DictReader(f)
        }


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else math.nan


def g1(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """판정 시트 → 일치도·협의 시트 → 최종값 → G1 지표 (results/G1_generation.md, G1.json)

    Raises:
        G1Error: 시트 없음·빈 칸·미협의 불일치 (협의 시트는 쓴 뒤에 멈춘다)
    """
    answers = _load_answers(cfg)
    queries = read_queries(cfg.queries_path, split="test")
    qids = [q for q in subset_qids(queries, cfg.primary_subset) if q in answers]
    anns = [a for a in gen_annotators(cfg) if (cfg.judgments_dir / f"gen_{a}_answers.csv").exists()]
    if "A" not in anns or len(anns) < 2:
        raise G1Error(f"답변 판정 시트가 부족합니다 (있음: {anns}) — judge gen 후 A·B 판정")
    vals = {a: _read_annotator(cfg, a) for a in anns}
    agreement = {f"A-{b}": _pair_agreement(vals["A"], vals[b]) for b in anns if b != "A"}

    # 불일치 → 협의 시트 (이미 적은 final 은 보존)
    keys = sorted(set().union(*vals.values()))
    adj_path = adjudication_path(cfg)
    prev = _read_adjudication(adj_path)
    final: dict[tuple[str, str, str, str], int] = {}
    dis: list[tuple[str, str, str, str]] = []
    pending = 0
    for k in keys:
        got = {a: v[k] for a, v in vals.items() if k in v}
        if len(set(got.values())) == 1:
            final[k] = next(iter(got.values()))
            continue
        dis.append(k)
        f, _note = prev.get(k, ("", ""))
        if f:
            final[k] = _int(
                f,
                (0, 1)
                if k[0] in ("citation", "no_answer")
                else USEFULNESS
                if k[0] == "usefulness"
                else None,
                f"{adj_path.name} {k}",
            )
        else:
            pending += 1
    with open(adj_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "item",
                "qid",
                "sentence_no",
                "citation_no",
                *[f"value_{a}" for a in anns],
                "final",
                "note",
            ]
        )
        for k in dis:
            w.writerow([*k, *[vals[a].get(k, "") for a in anns], *prev.get(k, ("", ""))])
    if pending:
        raise G1Error(
            f"협의되지 않은 불일치 {pending}건 — {adj_path} 의 final 을 채운 뒤 다시 실행하세요"
        )

    qrels = read_qrels(cfg.qrels_path) if cfg.qrels_path.exists() else {}
    # 인용 정확도
    cit = {k: v for k, v in final.items() if k[0] == "citation" and k[1] in qids}
    by_q: dict[str, list[int]] = {}
    for k, v in cit.items():
        by_q.setdefault(k[1], []).append(v)
    n_no_cite = sum(1 for q in qids if q not in by_q)
    # 결정적 증거 인용률 (컨텍스트 5개 중 등급 2 단위 가운데 답변이 인용한 비율)
    dec_rates: list[float] = []
    dec_num = dec_den = 0
    for q in qids:
        g2 = [u for u in answers[q]["context_unit_ids"] if qrels.get(q, {}).get(u) == 2]
        if not g2:
            continue
        cited = {answers[q]["citation_map"].get(str(c)) for c in answers[q]["citations"]}
        hit = sum(1 for u in g2 if u in cited)
        dec_rates.append(hit / len(g2))
        dec_num += hit
        dec_den += len(g2)
    # 근거 없는 주장
    claims = [final[("unsupported_claims", q, "", "")] for q in qids]
    # 무근거 응답: 컨텍스트에 등급 ≥1 단위가 없는 질문 중 "찾을 수 없다" 고 답한 비율
    no_ground = [
        q
        for q in qids
        if not any(qrels.get(q, {}).get(u, 0) >= 1 for u in answers[q]["context_unit_ids"])
    ]
    no_ans = [final[("no_answer", q, "", "")] for q in no_ground]
    useful = [final[("usefulness", q, "", "")] for q in qids]
    res = {
        "subset": cfg.primary_subset,
        "n_questions": len(qids),
        "annotators": anns,
        "qrels": bool(qrels),
        "citation_accuracy": {
            "n_citations": len(cit),
            "supported": sum(cit.values()),
            "pooled": sum(cit.values()) / len(cit) if cit else math.nan,
            "per_question_mean": _mean([sum(v) / len(v) for v in by_q.values()]),
            "n_answers_with_citations": len(by_q),
            "n_answers_without_citations": n_no_cite,
        },
        "decisive_citation": {
            "n_questions": len(dec_rates),
            "pooled": dec_num / dec_den if dec_den else math.nan,
            "per_question_mean": _mean(dec_rates),
            "grade2_in_context": dec_den,
            "cited": dec_num,
        },
        "unsupported_claims": {
            "n_answers_with_any": sum(1 for c in claims if c >= 1),
            "n_answers": len(claims),
            "total": sum(claims),
        },
        "no_grounds": {
            "n_questions": len(no_ground),
            "n_said_not_found": sum(no_ans),
            "rate": sum(no_ans) / len(no_ans) if no_ans else math.nan,
        },
        "usefulness": {
            "mean": _mean([float(u) for u in useful]),
            "dist": {str(k): Counter(useful).get(k, 0) for k in USEFULNESS},
        },
        "agreement": agreement,
        "n_adjudicated": len(dis),
    }
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "G1.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.results_dir / "G1_generation.md").write_text(g1_markdown(res), encoding="utf-8")
    log(
        f"[g1] 질문 {len(qids)} · 인용 {len(cit)} · 협의 {len(dis)}건 → {cfg.results_dir / 'G1_generation.md'}"
    )
    return res


def _f(x: float, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def g1_markdown(r: dict[str, Any]) -> str:
    """[표 14] G1 — 집계값만"""
    ca, dc, uc = r["citation_accuracy"], r["decisive_citation"], r["unsupported_claims"]
    ng, us = r["no_grounds"], r["usefulness"]
    n = r["n_questions"]
    lines = [
        f"# G1 응답 생성 평가 ([표 14]) — {r['subset']} 질문 {n}개",
        "",
        f"- 판정자 {', '.join(r['annotators'])} · 불일치 협의 {r['n_adjudicated']}건 (표는 협의 후 최종값)"
        + ("" if r["qrels"] else " · **qrels 없음: 결정적 증거 인용률·무근거 응답 계산 안 됨**"),
        "",
        "| 지표 | 값 | 비고 |",
        "|---|---|---|",
        f"| 인용 정확도 (전체 합산) | {_f(ca['pooled'])} ({ca['supported']}/{ca['n_citations']}) | "
        f"인용 없는 답변 {ca['n_answers_without_citations']}개는 분모에서 뺌 |",
        f"| 인용 정확도 (질문별 평균) | {_f(ca['per_question_mean'])} | n = {ca['n_answers_with_citations']} |",
        f"| 결정적 증거 인용률 (전체 합산) | {_f(dc['pooled'])} ({dc['cited']}/{dc['grade2_in_context']}) | "
        f"컨텍스트에 등급 2 단위가 있는 질문 n = {dc['n_questions']} |",
        f"| 결정적 증거 인용률 (질문별 평균) | {_f(dc['per_question_mean'])} | n = {dc['n_questions']} |",
        f"| 근거 없는 주장이 1개 이상인 답변 | {uc['n_answers_with_any']} / {uc['n_answers']} | "
        f"전체 주장 수 합 {uc['total']} |",
        f"| 무근거 응답 (찾을 수 없다고 답한 비율) | {_f(ng['rate'])} ({ng['n_said_not_found']}/{ng['n_questions']}) | "
        f"컨텍스트에 등급 ≥1 단위가 없는 질문 n = {ng['n_questions']} |",
        f"| 유용성 평균 | {_f(us['mean'], 2)} | 0/1/2 = {us['dist']['0']}/{us['dist']['1']}/{us['dist']['2']} |",
        "",
        "## 판정자 일치도",
        "",
        "| 쌍 | 인용 kappa (n) | 인용 일치율 | 유용성 QWK (n) | 근거 없는 주장 수 완전 일치율 | 무근거 응답 일치율 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for name, a in r["agreement"].items():
        lines.append(
            f"| {name} | {_f(a['citation_kappa'])} ({a['n_citations']}) | {_f(a['citation_agreement'])} | "
            f"{_f(a['usefulness_qwk'])} ({a['n_answers']}) | {_f(a['unsupported_exact'])} | "
            f"{_f(a['no_answer_agreement'])} |"
        )
    return "\n".join(lines) + "\n"
