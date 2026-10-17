"""단위 통계 → results/T1_corpus.md (지시서 v4.1 §6.8 T1, v4.2 §2 U1·U2, §3 S1~S5)

케이스 색인(`data/index/{case}/`)의 파트와 manifest `build_stats` 만으로 만든다. 재파싱하지 않으므로
표는 실험에 실제로 쓰인 색인과 정확히 같다. 같은 계산(`stats_from_parts`)을 `scripts/unit_stats.py`도
쓴다(색인 없이 단위만 만든 초안).

토큰 수는 Qwen3 토크나이저 기준이다. `TokenCounter`가 문자 근사로 떨어졌으면 T1을 만들지 않고
멈춘다(v4.2 S1). 가짜 모델 점검(AST_FAKE_MODELS=1)에서만 근사를 허용하고 표에 표시한다.

출력에는 집계 수치만 넣는다 — 본문·파일명·이름은 넣지 않는다 (v4.2 P0).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from ev4.io import ExperimentConfig
from ev4.stats import percentile
from src.units.schema import UnitPart

UNIT_TYPES = ("email", "chat", "document", "attachment")
MAX_PART_TOKENS = 2048       # 논문 3.3.1절 상한 (settings.unit_max_part_tokens 와 같아야 함)
RERANK_LIMIT = 4096          # 재순위 입력 상한 (settings.rerank_max_length)
QUERY_SLOT_TOKENS = 64       # 재순위 입력 길이 추정에서 질의 자리로 잡는 토큰 수 (S2)

# [표 6] 행: (행 이름, 해당 file_type). 이메일·채팅은 unit_type 으로 나눈다.
FORMAT_ROWS: list[tuple[str, frozenset[str]]] = [
    ("PDF", frozenset({"pdf"})),
    ("PPTX", frozenset({"pptx"})),
    ("DOCX·TXT·HTML", frozenset({"docx", "txt", "html", "htm"})),
    ("XLSX·XLS", frozenset({"xlsx", "xls"})),
    ("첨부 메일(EML·MSG)", frozenset({"eml", "msg"})),
]
ROW_ORDER = ["이메일", "채팅", *(r for r, _ in FORMAT_ROWS), "기타"]

# [표 7] 행
LANG_ROWS = [("한국어", "ko"), ("영어", "en"), ("중국어", "zh")]

# S4 제외 형식 묶음
EXCLUDED_GROUPS: list[tuple[str, frozenset[str]]] = [
    ("이미지", frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp",
                          ".emf", ".wmf", ".ico"})),
    ("HWP", frozenset({".hwp", ".hwpx"})),
    ("DWG", frozenset({".dwg", ".dxf"})),
    ("판별 불가", frozenset({"(none)"})),
    ("빈 첨부", frozenset({"(empty)"})),
]


class CorpusStopError(RuntimeError):
    """T1을 만들 수 없는 조건 (토큰 카운터가 근사 등) — 멈추고 보고한다"""


def _dist(values: list[int]) -> dict[str, float]:
    """n·중앙값·p95·p99·최대"""
    if not values:
        return {"n": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "n": len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def format_row(unit_type: str, file_type: str) -> str:
    """단위 → [표 6] 행 이름"""
    if unit_type == "email":
        return "이메일"
    if unit_type == "chat":
        return "채팅"
    ft = (file_type or "").lower()
    for name, types in FORMAT_ROWS:
        if ft in types:
            return name
    return "기타"


def excluded_group(ext: str) -> str:
    """제외 확장자 → S4 묶음 이름"""
    for name, exts in EXCLUDED_GROUPS:
        if ext in exts:
            return name
    return "기타 형식"


def code_like_counts(text: str) -> dict[str, int]:
    """bm25_tokenizer 가 통째로 보존하는 코드형·금액·이메일 주소 문자열 수 (S3)"""
    from src.text.bm25_tokenizer import _AMOUNT_RE, _CODE_RE, _EMAIL_RE

    return {
        "code": sum(1 for _ in _CODE_RE.finditer(text)),
        "amount": sum(1 for _ in _AMOUNT_RE.finditer(text)),
        "email": sum(1 for _ in _EMAIL_RE.finditer(text)),
    }


def rerank_overhead_tokens(counter: Any) -> int:
    """재순위 입력에서 문서 외 고정 길이: 템플릿 + 지시문 + 질의 자리(QUERY_SLOT_TOKENS)"""
    from src.rag.reranker import RERANK_PREFIX, RERANK_SUFFIX, format_rerank_pair

    fixed = counter.count(RERANK_PREFIX) + counter.count(format_rerank_pair("", "")) \
        + counter.count(RERANK_SUFFIX)
    return fixed + QUERY_SLOT_TOKENS


def _group_units(parts: Iterable[UnitPart]) -> dict[str, list[UnitPart]]:
    """파트 → 단위별 파트 목록 (파트 번호 순)"""
    units: dict[str, list[UnitPart]] = defaultdict(list)
    for p in parts:
        units[p.unit_id].append(p)
    for lst in units.values():
        lst.sort(key=lambda p: int(p.part_id.rsplit("#", 1)[1]))
    return units


def stats_from_parts(
    parts: list[UnitPart],
    build_stats: dict[str, Any],
    counter: Any,
    *,
    require_exact: bool = True,
    build_counter_kind: str = "",
) -> dict[str, Any]:
    """파트와 build_stats 에서 T1 수치를 계산한다 (색인·초안 공용)

    Args:
        parts: 모든 파트 (중복 제거 후)
        build_stats: BuildStats.to_dict() + duplicates_removed (+ token_counter)
        counter: TokenCounter (헤더·재순위 입력 길이 계산용)
        require_exact: True면 근사 카운터에서 CorpusStopError
        build_counter_kind: 색인 때 파트 토큰을 센 카운터 종류 (manifest 기록값)

    Raises:
        CorpusStopError: 토큰 카운터가 근사일 때 (require_exact)
    """
    approx = [k for k in (counter.kind, build_counter_kind) if k and not k.startswith("hf:")]
    if approx and require_exact:
        raise CorpusStopError(
            f"TokenCounter 가 문자 근사입니다({', '.join(approx)}). Qwen3 토크나이저를 로컬 HF 캐시에 "
            "두고 다시 실행하세요 — T1을 만들지 않습니다 (v4.2 S1)."
        )
    bs = build_stats or {}
    counts = bs.get("counts", {}) or {}
    units = _group_units(parts)

    # ── 단위별 기본 값 ──
    rows: dict[str, dict[str, Any]] = {}
    for uid, plist in units.items():
        p0 = plist[0]
        meta = p0.meta or {}
        rows[uid] = {
            "type": p0.unit_type,
            "row": format_row(p0.unit_type, str(meta.get("file_type") or "")),
            "file_type": str(meta.get("file_type") or ""),
            "tokens": sum(p.n_tokens for p in plist),
            "n_parts": len(plist),
            "lang": str(meta.get("lang") or "und"),
            "mixed": bool(meta.get("lang_mixed")),
            "header": p0.header,
            "author": bool(str(meta.get("author") or "").strip()),
            "modifier": bool(str(meta.get("last_modified_by") or "").strip()),
            "forwarded": bool(meta.get("forwarded")),
        }

    # ── S1 [표 6] 형식별 ──
    fmt: dict[str, dict[str, Any]] = {}
    for r in ROW_ORDER + ["계"]:
        fmt[r] = {"units": 0, "from_attachment": 0, "split": 0, "tokens": []}
    for r in rows.values():
        for key in (r["row"], "계"):
            f = fmt[key]
            f["units"] += 1
            f["from_attachment"] += int(r["type"] == "attachment")
            f["split"] += int(r["n_parts"] > 1)
            f["tokens"].append(r["tokens"])
    table6 = {
        name: {"units": f["units"], "from_attachment": f["from_attachment"], "split": f["split"],
               "tokens": _dist(f["tokens"])}
        for name, f in fmt.items() if f["units"] or name == "계"
    }
    all_tokens = [r["tokens"] for r in rows.values()]

    # ── U1 작성자·최종 수정자 채움 비율 (문서·첨부, 형식별) ──
    meta_fill: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "author": 0, "modifier": 0})
    for r in rows.values():
        if r["type"] not in ("document", "attachment"):
            continue
        for key in (r["row"], "계"):
            m = meta_fill[key]
            m["n"] += 1
            m["author"] += int(r["author"])
            m["modifier"] += int(r["modifier"])

    # ── S2 이메일 2,048 이하 비율 · 재순위 입력 길이 ──
    email_ids = [uid for uid, r in rows.items() if r["type"] == "email"]
    email_hdr = counter.count_many([rows[u]["header"] for u in email_ids])
    body_ok = sum(1 for u in email_ids if rows[u]["tokens"] <= MAX_PART_TOKENS)
    # 헤더 포함: 헤더 + 빈 줄(2토큰 내외) + 본문. 빈 줄은 1토큰으로 센다
    with_hdr_ok = sum(
        1 for u, h in zip(email_ids, email_hdr, strict=True)
        if rows[u]["tokens"] + h + 1 <= MAX_PART_TOKENS
    )
    overhead = rerank_overhead_tokens(counter)
    from src.index.case_index import embed_input

    doc_tokens = counter.count_many([embed_input(p, True) for p in parts])
    rerank_len = [overhead + t for t in doc_tokens]
    s2 = {
        "n_email": len(email_ids),
        "body_le_2048": body_ok,
        "with_header_le_2048": with_hdr_ok,
        "rerank_overhead": overhead,
        "query_slot_tokens": QUERY_SLOT_TOKENS,
        "rerank_input": _dist(rerank_len),
        "rerank_over_4096": sum(1 for x in rerank_len if x > RERANK_LIMIT),
        "n_parts": len(rerank_len),
    }

    # ── S3 코드형 문자열 (이메일 단위) ──
    code_totals: list[int] = []
    code_kinds: Counter = Counter()
    for u in email_ids:
        c = code_like_counts("\n".join(p.text for p in units[u]))
        code_kinds.update(c)
        code_totals.append(sum(c.values()))
    s3 = {
        "n_email": len(code_totals),
        "mean": (sum(code_totals) / len(code_totals)) if code_totals else 0.0,
        "median": percentile(code_totals, 50) if code_totals else 0,
        "with_any": sum(1 for x in code_totals if x > 0),
        "by_kind": dict(code_kinds),
    }

    # ── S5 [표 7] 언어 (U2: und 별도 행) ──
    lang_units: Counter = Counter(r["lang"] for r in rows.values())
    lang_mixed: Counter = Counter(r["lang"] for r in rows.values() if r["mixed"])
    named = {code for _, code in LANG_ROWS}
    table7: dict[str, dict[str, int]] = {}
    for label, code in LANG_ROWS:
        table7[label] = {"units": lang_units.get(code, 0), "mixed": lang_mixed.get(code, 0)}
    other = {k: v for k, v in lang_units.items() if k not in named and k != "und"}
    table7["기타"] = {"units": sum(other.values()),
                     "mixed": sum(lang_mixed.get(k, 0) for k in other)}
    table7["und"] = {"units": lang_units.get("und", 0), "mixed": lang_mixed.get("und", 0)}
    table7["계"] = {"units": len(rows), "mixed": sum(lang_mixed.values())}
    lang_by_type: dict[str, Counter] = defaultdict(Counter)
    for r in rows.values():
        lang_by_type[r["type"]][r["lang"]] += 1

    # ── S4 4.1.1절 수치 ──
    excluded = dict(bs.get("excluded_types", {}) or {})
    excluded_grouped: Counter = Counter()
    for ext, n in excluded.items():
        excluded_grouped[excluded_group(ext)] += int(n)
    fwd_tokens = [int(x) for x in bs.get("forwarded_tokens", []) or []]

    return {
        "n_units": len(rows),
        "n_parts": len(parts),
        "token_counter": counter.kind,
        "build_token_counter": build_counter_kind,
        "token_counter_exact": not approx,
        "units_by_type": dict(Counter(r["type"] for r in rows.values())),
        "parts_by_type": dict(Counter(p.unit_type for p in parts)),
        "split_units": dict(Counter(r["type"] for r in rows.values() if r["n_parts"] > 1)),
        "table6": table6,
        "unit_tokens_p95_all": int(percentile(all_tokens, 95)) if all_tokens else 0,
        "part_tokens": _dist([p.n_tokens for p in parts]),
        "meta_fill": {k: dict(v) for k, v in meta_fill.items()},
        "s2": s2,
        "s3": s3,
        "table7": table7,
        "lang_other_detail": dict(sorted(other.items(), key=lambda kv: -kv[1])),
        "lang_by_type": {t: dict(c) for t, c in lang_by_type.items()},
        "duplicates_removed": dict(bs.get("duplicates_removed", {}) or {}),
        "clean_chars": dict(bs.get("clean_chars", {}) or {}),
        "forwarded": {"count": int(counts.get("forwarded_emails", 0)), "tokens": _dist(fwd_tokens)},
        "orphan_attachments": int(counts.get("orphan_attachments", 0)),
        "attachments_total": int(counts.get("attachments_total", 0)),
        "attachments_parsed": int(counts.get("attachments_parsed", 0)),
        "excluded_grouped": dict(excluded_grouped),
        "excluded_types": excluded,
        "errors": dict(bs.get("errors", {}) or {}),
        "embedding": bs.get("embedding", {}),
    }


def corpus_stats(case_id: str) -> dict[str, Any]:
    """케이스 색인에서 T1 수치를 계산한다

    Raises:
        CorpusStopError: 토큰 카운터가 근사일 때 (가짜 모델 점검 제외)
    """
    from src.index import case_index
    from src.units.tokens import get_token_counter
    from src.utils.config import settings

    idx = case_index.load(case_id)
    bs = idx.manifest.get("build_stats", {}) or {}
    s = stats_from_parts(
        list(idx.parts), bs, get_token_counter(),
        require_exact=not settings.ast_fake_models,
        build_counter_kind=str(bs.get("token_counter") or ""),
    )
    s.update({
        "case_id": case_id,
        "created_at": idx.manifest.get("created_at", ""),
        "unit_rules_version": idx.manifest.get("unit_rules_version", ""),
        "tokenizer_version": idx.manifest.get("tokenizer_version", ""),
        "profiles": idx.profiles(),
    })
    return s


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "–"


def render_t1(s: dict[str, Any]) -> str:
    """T1 마크다운 (논문 [표 6]·[표 7] + 근거 수치)"""
    lines = [
        f"# T1 단위 통계 — 케이스 {s.get('case_id', '-')}",
        "",
        f"- 색인 생성 {s.get('created_at', '-')} · 규칙 {s.get('unit_rules_version', '-')} · BM25 토크나이저 "
        f"{s.get('tokenizer_version', '-')} · 프로필 {', '.join(s.get('profiles', [])) or '-'}",
        f"- 토큰 기준 `{s['token_counter']}` (색인 시 `{s['build_token_counter'] or '기록 없음'}`)"
        + ("" if s["token_counter_exact"] else " — **근사: 가짜 모델 점검용, 논문에 쓰지 않음**"),
        f"- 단위 {s['n_units']:,} · 파트 {s['n_parts']:,}",
        "",
        "## [표 6] 자료 유형별 증거 단위",
        "",
        f"토큰은 단위 본문(헤더 제외) 기준. '분할'은 {MAX_PART_TOKENS:,}토큰을 넘어 파트로 나눈 단위. "
        "첨부는 형식별 행 안에서 '첨부에서 만든 단위'로 센다(중복 계산 없음).",
        "",
        "| 자료 유형 | 단위 | 그중 첨부 | 분할 | 토큰 중앙값 | p95 | p99 | 최대 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [*ROW_ORDER, "계"]:
        r = s["table6"].get(name)
        if not r:
            continue
        d = r["tokens"]
        bold = "**" if name == "계" else ""
        lines.append(
            f"| {bold}{name}{bold} | {r['units']:,} | {r['from_attachment']:,} | {r['split']:,} | "
            f"{_n(d['median'])} | {_n(d['p95'])} | {_n(d['p99'])} | {_n(d['max'])} |"
        )
    pt = s["part_tokens"]
    lines += [
        "",
        f"- 파트 토큰: 중앙값 {_n(pt['median'])} · p95 {_n(pt['p95'])} · 최대 {_n(pt['max'])} · "
        f"전체 단위 토큰 p95 {s['unit_tokens_p95_all']:,} (F1 '긴 단위 희석' 기준)",
        "",
        "### 작성자·최종 수정자가 채워진 문서·첨부 단위 (U1)",
        "",
        "| 자료 유형 | 단위 | 작성자 | 최종 수정자 |",
        "|---|---:|---:|---:|",
    ]
    for name in [*ROW_ORDER, "계"]:
        m = s["meta_fill"].get(name)
        if not m:
            continue
        mod = "–" if name == "PDF" else _pct(m["modifier"], m["n"])
        lines.append(f"| {name} | {m['n']:,} | {_pct(m['author'], m['n'])} | {mod} |")

    t7 = s["table7"]
    total = t7["계"]["units"]
    lines += [
        "",
        "## [표 7] 증거 단위 언어 분포 (lingua, 단위 기준)",
        "",
        "| 언어 | 단위 | 비율 | 그중 혼용 |",
        "|---|---:|---:|---:|",
    ]
    for label in [*(lb for lb, _ in LANG_ROWS), "기타", "und", "계"]:
        r = t7[label]
        name = "판정 불가(und)" if label == "und" else label
        bold = "**" if label == "계" else ""
        lines.append(f"| {bold}{name}{bold} | {r['units']:,} | {_pct(r['units'], total)} | {r['mixed']:,} |")
    if s["lang_other_detail"]:
        lines += ["", "기타 언어 세부: " + ", ".join(f"{k} {v:,}" for k, v in s["lang_other_detail"].items())]

    s2 = s["s2"]
    ri = s2["rerank_input"]
    lines += [
        "",
        f"## 근거 수치: {MAX_PART_TOKENS:,}토큰 상한 (S2, 논문 3.3.1절)",
        "",
        f"- 이메일 단위 {s2['n_email']:,}개 중 본문 {MAX_PART_TOKENS:,}토큰 이하: "
        f"헤더 제외 {s2['body_le_2048']:,} ({_pct(s2['body_le_2048'], s2['n_email'])}) · "
        f"헤더 포함 {s2['with_header_le_2048']:,} ({_pct(s2['with_header_le_2048'], s2['n_email'])})",
        f"- 재순위 입력 길이(템플릿·지시문 + 질의 자리 {s2['query_slot_tokens']} + 헤더 + 파트, 파트 "
        f"{s2['n_parts']:,}개): p95 {_n(ri['p95'])} · p99 {_n(ri['p99'])} · 최대 {_n(ri['max'])} · "
        f"{RERANK_LIMIT:,} 초과 {s2['rerank_over_4096']:,} ({_pct(s2['rerank_over_4096'], s2['n_parts'])})",
    ]
    s3 = s["s3"]
    kinds = s3["by_kind"]
    lines += [
        "",
        "## 근거 수치: 코드형 문자열 (S3, 논문 3.4절)",
        "",
        f"- 이메일 단위 {s3['n_email']:,}개: 평균 {s3['mean']:.2f}개 · 중앙값 {_n(s3['median'])} · "
        f"1개 이상 포함 {s3['with_any']:,} ({_pct(s3['with_any'], s3['n_email'])})",
        f"- 종류별 합계: 코드 {kinds.get('code', 0):,} · 금액 {kinds.get('amount', 0):,} · "
        f"이메일 주소 {kinds.get('email', 0):,} (bm25_tokenizer 통째 보존 규칙 기준)",
        "",
        "## 4.1.1절 수치 (S4)",
        "",
        f"- 중복 제거로 지운 단위: {', '.join(f'{k} {v:,}' for k, v in s['duplicates_removed'].items()) or '0'}",
    ]
    fw = s["forwarded"]
    lines += [
        f"- 전달 메일(인용 유지) {fw['count']:,}건 · 토큰 중앙값 {_n(fw['tokens']['median'])} · "
        f"p95 {_n(fw['tokens']['p95'])} · 최대 {_n(fw['tokens']['max'])}",
        f"- 첨부 {s['attachments_total']:,}건 → 파싱 {s['attachments_parsed']:,}건 · "
        f"부모를 찾지 못한 첨부 {s['orphan_attachments']:,}건",
        "",
        "| 언어 | 정제 전 문자 | 정제 후 문자 | 감소율 |",
        "|---|---:|---:|---:|",
    ]
    for lang, (b, a) in sorted(s["clean_chars"].items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {lang} | {b:,} | {a:,} | {(1 - a / b) if b else 0:.1%} |")
    lines += ["", "| 제외 형식 | 건수 |", "|---|---:|"]
    for name in [*(g for g, _ in EXCLUDED_GROUPS), "기타 형식"]:
        if name in s["excluded_grouped"]:
            lines.append(f"| {name} | {s['excluded_grouped'][name]:,} |")
    if s["excluded_types"]:
        lines += ["", "제외 확장자 세부: " + ", ".join(
            f"{k} {v:,}" for k, v in sorted(s["excluded_types"].items(), key=lambda kv: -kv[1]))]
    if s["errors"]:
        lines += ["", "| 오류 유형 | 건수 |", "|---|---:|"]
        for k, n in sorted(s["errors"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {n:,} |")
    return "\n".join(lines) + "\n"


def build_corpus(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """T1 생성 → results/T1_corpus.md, results/corpus.json

    Raises:
        CorpusStopError: 토큰 카운터가 근사일 때 (파일을 쓰지 않는다)
    """
    s = corpus_stats(cfg.case_id)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "corpus.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.results_dir / "T1_corpus.md").write_text(render_t1(s), encoding="utf-8")
    log(f"[corpus] 단위 {s['n_units']:,} · 파트 {s['n_parts']:,} → {cfg.results_dir / 'T1_corpus.md'}")
    return s
