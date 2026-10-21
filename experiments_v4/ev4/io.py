"""설정·질의·qrels·실행(TREC) 입출력 (지시서 §6.3, §6.4)

경로는 모두 실험 루트(기본 `experiments_v4/`) 기준이다. selftest 는 임시 폴더를 루트로 쓴다.

- 질의: `data/queries.jsonl` — 감사인 표준 질문 45개(논문 부록 A, v4.3 §1)
  {qid, text, area, support_class, support_group, type, split, query_lang, target_evidence_lang,
  source_question, note}. `type` 은 search 집합만 exact·semantic·relation, 나머지는 none
- 최종 qrels: `data/qrels.txt` — `qid 0 unit_id grade`
- 실행: `runs/{RUN}.trec` (test), `runs/dev/{RUN}.trec` (dev) — `qid Q0 unit_id rank score RUN`
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ev4 import EXPERIMENTS_ROOT

QUERY_TYPES = ("exact", "semantic", "relation")
# 논문 부록 [표 A-1] 이름·정의 (v4.2 Q1) — README·판정 기준 문서·표 머리에 그대로 쓴다
QUERY_TYPE_NAMES = {"exact": "정확 일치", "semantic": "정황", "relation": "관계"}
QUERY_TYPE_DEFS = {
    "exact": ("인명, 업체명, 계약번호, 금액, 특정 문구처럼 본문이나 파일명에 같은 문자열로 나타나는 대상을 찾는 질의",
              "문자열 일치(BM25)"),
    "semantic": ("본문에 직접 드러나지 않는 행위, 회피 표현, 사내 은어를 서술하여 찾는 질의", "의미 유사도(밀집 검색)"),
    "relation": ("특정 인물이나 업체 사이 소통, 또는 날짜 선후를 조건으로 하는 질의", "메타데이터 헤더와 의미 유사도"),
}
TYPE_NONE = "none"  # search 집합이 아닌 질의의 type (v4.3 §1)
# 부록 A 지원 분류 → 분석 집합 (v4.3 §1, §3)
SUPPORT_GROUPS = ("search", "meta", "nonsearch")
# v4.4 §3: A → A1, A′ → A2
SUPPORT_CLASS_GROUP = {"A1": "search", "A2": "search", "C": "meta", "D": "nonsearch", "B": "nonsearch",
                       "E": "nonsearch"}
SUPPORT_CLASSES = tuple(SUPPORT_CLASS_GROUP)
AREAS = tuple(str(i) for i in range(1, 8))
EXPECTED_QUERIES = 45
SPLITS = ("test",)  # 개발용(dev) 질의는 없다 (v4.3 §2)
GRADES = (0, 1, 2)
DEFAULT_B_QUERIES = {"search": 4, "meta": 1, "nonsearch": 6}  # 판정자 B 질의 수 (v4.3 §5)
DEFAULT_SMOKE_QIDS = ["q1.1", "q2.6", "q7.2"]  # 실데이터 동작 점검 (v4.3 §2)

# RetrievalConfig 에서 실행별로 바꿀 수 있는 필드 — k값은 고정(지시서 §6.4: config 에서 덮어쓰지 않음)
_RUN_FIELDS = {"profile", "use_bm25", "use_dense", "rerank", "reranker"}


class ConfigError(ValueError):
    """config.yaml·queries.jsonl 형식 오류"""


@dataclass
class ExperimentConfig:
    """config.yaml 내용 + 경로"""

    root: Path
    case_id: str
    depth: int
    pool_depth: int
    seed: int
    runs: dict[str, dict[str, Any]]
    pool_runs: list[str]
    comparisons: dict[str, list[tuple[str, str]]]
    latency_runs: list[str]
    primary_subset: str = "search"
    b_queries: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_B_QUERIES))
    c_queries: dict[str, int] = field(default_factory=dict)  # 판정자 C 질의 수 (v4.4 §9, 비면 C 없음)
    smoke_qids: list[str] = field(default_factory=lambda: list(DEFAULT_SMOKE_QIDS))
    text_max_chars: int = 6000
    latency_warmup: int = 3
    latency_repeats: int = 5
    labels: dict[str, str] = field(default_factory=dict)  # 하네스 실행 이름 → 논문 번호 (v4.4 §2)
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self, run: str) -> str:
        """표·보고서에 쓰는 실행 이름 (labels 에 없으면 하네스 이름 그대로)"""
        return self.labels.get(run, run)

    def runs_by_label(self, runs: list[str] | None = None) -> list[str]:
        """하네스 실행 이름을 논문 번호 순으로 정렬"""
        names = list(self.runs) if runs is None else list(runs)

        def _key(r: str) -> tuple[int, str]:
            digits = "".join(c for c in self.label(r) if c.isdigit())
            return (int(digits) if digits else 0, self.label(r))

        return sorted(names, key=_key)

    @property
    def data_dir(self) -> Path:
        """판정·질의 등 입력 데이터 (gitignore)"""
        return self.root / "data"

    @property
    def runs_dir(self) -> Path:
        """실행 결과 (gitignore)"""
        return self.root / "runs"

    @property
    def results_dir(self) -> Path:
        """표·CSV 산출물 (gitignore)"""
        return self.root / "results"

    @property
    def cache_dir(self) -> Path:
        """번역 캐시 등 (gitignore)"""
        return self.root / "cache"

    @property
    def queries_path(self) -> Path:
        return self.data_dir / "queries.jsonl"

    @property
    def qrels_path(self) -> Path:
        return self.data_dir / "qrels.txt"

    @property
    def judgments_dir(self) -> Path:
        return self.data_dir / "judgments"

    @property
    def subset_lock_path(self) -> Path:
        """첫 test 실행 때 기록한 주 분석 집합 (v4.3 §3: 실행 전에 고정하고 바꾸지 않는다)"""
        return self.runs_dir / "primary_subset.json"

    def check_subset_lock(self, write: bool = False) -> None:
        """config 의 primary_subset 이 첫 실행 때 기록한 값과 같은지 확인한다

        Args:
            write: 기록이 없으면 지금 값을 기록한다 (run 에서 True)

        Raises:
            ConfigError: 기록된 값과 다를 때
        """
        path = self.subset_lock_path
        if path.exists():
            locked = json.loads(path.read_text(encoding="utf-8")).get("primary_subset")
            if locked != self.primary_subset:
                raise ConfigError(f"primary_subset 이 실행 뒤에 바뀌었습니다 (기록 {locked!r} → config "
                                  f"{self.primary_subset!r}). 실행 전에 고정한 값으로 되돌리세요 ({path})")
        elif write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"primary_subset": self.primary_subset}) + "\n", encoding="utf-8")

    def run_path(self, run: str, split: str = "test") -> Path:
        """실행 파일 경로 (dev 는 runs/dev/ 아래 — test 실행을 덮어쓰지 않게)"""
        base = self.runs_dir if split == "test" else self.runs_dir / split
        return base / f"{run}.trec"

    def timing_path(self, run: str, split: str = "test") -> Path:
        return self.run_path(run, split).with_suffix(".timing.jsonl")

    def comparison_pairs(self) -> list[tuple[str, str, str]]:
        """(계열 이름, A, B) — Holm 보정은 전체를 한 계열로 한다"""
        return [(fam, a, b) for fam, pairs in self.comparisons.items() for a, b in pairs]

    def retrieval_config(self, run: str) -> Any:
        """실행 이름 → RetrievalConfig (depth 만 config 값, k값은 기본값)"""
        from src.rag.retrieval import RetrievalConfig

        return RetrievalConfig(depth=self.depth, **self.runs[run])


def load_config(path: Path | None = None, root: Path | None = None) -> ExperimentConfig:
    """config.yaml 읽기·검증

    Args:
        path: config 경로 (기본 {root}/config.yaml)
        root: 실험 루트 (기본 experiments_v4/)

    Raises:
        ConfigError: 필수 키 누락, 알 수 없는 실행 이름·필드
    """
    root = Path(root or EXPERIMENTS_ROOT)
    path = Path(path or root / "config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("case_id", "runs"):
        if key not in raw:
            raise ConfigError(f"{path}: '{key}' 가 없습니다")
    runs: dict[str, dict[str, Any]] = {}
    for name, spec in raw["runs"].items():
        bad = set(spec) - _RUN_FIELDS
        if bad:
            raise ConfigError(f"{path}: 실행 {name} 에 바꿀 수 없는 필드 {sorted(bad)} (k값은 고정)")
        runs[str(name)] = dict(spec)
    comparisons = {
        fam: [(str(a), str(b)) for a, b in pairs] for fam, pairs in (raw.get("comparisons") or {}).items()
    }
    cfg = ExperimentConfig(
        root=root,
        case_id=str(raw["case_id"]),
        depth=int(raw.get("depth", 100)),
        pool_depth=int(raw.get("pool_depth", 10)),
        seed=int(raw.get("seed", 20261015)),
        runs=runs,
        pool_runs=[str(r) for r in raw.get("pool_runs", list(runs))],
        comparisons=comparisons,
        latency_runs=[str(r) for r in raw.get("latency_runs", [])],
        primary_subset=str(raw.get("primary_subset", "search")),
        b_queries={str(k): int(v) for k, v in (raw.get("b_queries") or DEFAULT_B_QUERIES).items()},
        c_queries={str(k): int(v) for k, v in (raw.get("c_queries") or {}).items()},
        smoke_qids=[str(q) for q in raw.get("smoke_qids", DEFAULT_SMOKE_QIDS)],
        text_max_chars=int(raw.get("text_max_chars", 6000)),
        latency_warmup=int(raw.get("latency_warmup", 3)),
        latency_repeats=int(raw.get("latency_repeats", 5)),
        labels={str(k): str(v) for k, v in (raw.get("labels") or {}).items()},
        extra={k: v for k, v in raw.items() if k not in {
            "case_id", "depth", "pool_depth", "seed", "runs", "pool_runs", "comparisons",
            "latency_runs", "primary_subset", "b_queries", "c_queries", "smoke_qids", "text_max_chars", "latency_warmup",
            "latency_repeats", "labels"}},
    )
    bad_labels = set(cfg.labels) - set(runs)
    if bad_labels:
        raise ConfigError(f"{path}: labels 에 정의되지 않은 실행 {sorted(bad_labels)}")
    if len(set(cfg.labels.values())) != len(cfg.labels):
        raise ConfigError(f"{path}: labels 의 논문 번호가 겹칩니다")
    if cfg.primary_subset not in SUPPORT_GROUPS:
        raise ConfigError(f"{path}: primary_subset 은 {SUPPORT_GROUPS} 중 하나 (입력 {cfg.primary_subset!r})")
    for key, quota in (("b_queries", cfg.b_queries), ("c_queries", cfg.c_queries)):
        bad_q = set(quota) - set(SUPPORT_GROUPS)
        if bad_q:
            raise ConfigError(f"{path}: {key} 의 집합 {sorted(bad_q)} 는 {SUPPORT_GROUPS} 가 아닙니다")
    used = set(cfg.pool_runs) | set(cfg.latency_runs) | {r for _f, a, b in cfg.comparison_pairs() for r in (a, b)}
    unknown = used - set(runs)
    if unknown:
        raise ConfigError(f"{path}: 정의되지 않은 실행 {sorted(unknown)}")
    return cfg


# ─────────────────────────── JSONL ───────────────────────────


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 읽기 (빈 줄 무시)"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """JSONL 쓰기 (폴더 생성)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────── 질의 ───────────────────────────


@dataclass(frozen=True)
class Query:
    """평가 질의 한 건"""

    qid: str
    text: str
    type: str
    split: str
    query_lang: str = "ko"
    note: str = ""
    target_evidence_lang: str = ""  # v4.3: 빈칸 (증거 언어는 판정 후 등급 2 단위로 정한다)
    source_question: str = ""       # 부록 A 감사인 질문 번호, 예 "1.3"
    area: str = ""                  # 부록 A 영역 1~7
    support_class: str = ""         # 부록 A 분류 A1·A2·C·D·B·E
    support_group: str = ""         # 분석 집합 search·meta·nonsearch


def read_queries(path: Path, split: str | None = None) -> list[Query]:
    """queries.jsonl 읽기·검증 (v4.3 §1 — 질의 수 45개는 query_warnings 에서 본다)

    Args:
        path: queries.jsonl
        split: "test" | None(전체)

    Raises:
        ConfigError: 중복 qid, 빈 질의, 정해진 6개 밖의 support_class, 분류와 맞지 않는 support_group,
            search 인데 type 이 exact·semantic·relation 이 아니거나 search 가 아닌데 none 이 아님, split 오류
    """
    out: list[Query] = []
    seen: set[str] = set()
    for i, r in enumerate(read_jsonl(path), 1):
        q = Query(
            qid=str(r["qid"]).strip(),
            text=str(r["text"]).strip(),
            type=str(r.get("type", "")).strip(),
            split=str(r.get("split", "test")).strip(),
            query_lang=str(r.get("query_lang", "ko")).strip() or "ko",
            note=str(r.get("note", "")),
            target_evidence_lang=str(r.get("target_evidence_lang") or "").strip(),
            source_question=str(r.get("source_question") or "").strip(),
            area=str(r.get("area") or "").strip(),
            support_class=str(r.get("support_class") or "").strip(),
            support_group=str(r.get("support_group") or "").strip(),
        )
        if not q.qid or not q.text:
            raise ConfigError(f"{path}:{i}: qid·text 가 비어 있습니다")
        if q.qid in seen:
            raise ConfigError(f"{path}:{i}: 중복 qid {q.qid}")
        if q.support_class not in SUPPORT_CLASSES:
            raise ConfigError(f"{path}:{i}: support_class 는 {SUPPORT_CLASSES} 중 하나 (입력 {q.support_class!r})")
        if q.support_group != SUPPORT_CLASS_GROUP[q.support_class]:
            raise ConfigError(f"{path}:{i}: support_class {q.support_class} 의 support_group 은 "
                              f"{SUPPORT_CLASS_GROUP[q.support_class]} (입력 {q.support_group!r})")
        allowed = QUERY_TYPES if q.support_group == "search" else (TYPE_NONE,)
        if q.type not in allowed:
            raise ConfigError(f"{path}:{i}: {q.support_group} 집합의 type 은 {allowed} 중 하나 (입력 {q.type!r})")
        if q.split not in SPLITS:
            raise ConfigError(f"{path}:{i}: split 은 {SPLITS} 중 하나 (입력 {q.split!r})")
        seen.add(q.qid)
        out.append(q)
    return [q for q in out if split is None or q.split == split]


def query_warnings(queries: list[Query]) -> list[str]:
    """질의 구성 경고 목록 (v4.3 §1, 질의 문장은 넣지 않는다)

    필드 검증은 read_queries 가 오류로 막고, 여기서는 질의 수가 45개가 아닌 것만 경고한다.
    """
    if len(queries) != EXPECTED_QUERIES:
        return [f"질의 {len(queries)}개 — 표준 질문 {EXPECTED_QUERIES}개가 아님"]
    return []


def subset_qids(queries: list[Query], group: str | None) -> list[str]:
    """분석 집합의 qid (group 이 None 이면 전체 = ALL45)"""
    return [q.qid for q in queries if group is None or q.support_group == group]


# ─────────────────────────── qrels ───────────────────────────

Qrels = dict[str, dict[str, int]]
Run = dict[str, list[str]]  # qid → unit_id 순위 목록 (1위부터)


def read_qrels(path: Path) -> Qrels:
    """TREC qrels (`qid 0 unit_id grade`) → {qid: {unit_id: grade}}"""
    out: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            cols = line.split()
            if not cols:
                continue
            if len(cols) != 4:
                raise ConfigError(f"{path}:{ln}: 열 4개가 필요합니다 (qid 0 unit_id grade)")
            grade = int(cols[3])
            if grade not in GRADES:
                raise ConfigError(f"{path}:{ln}: 등급은 0~2 (입력 {grade})")
            out.setdefault(cols[0], {})[cols[2]] = grade
    return out


def write_qrels(path: Path, qrels: Qrels) -> None:
    """{qid: {unit_id: grade}} → TREC qrels (qid·unit_id 순 정렬)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(qrels):
            for uid in sorted(qrels[qid]):
                f.write(f"{qid} 0 {uid} {qrels[qid][uid]}\n")


# ─────────────────────────── 실행(TREC) ───────────────────────────


def write_run(path: Path, run_name: str, run: Run, depth: int) -> None:
    """순위 목록 → TREC 실행 파일

    단계마다 점수 척도가 달라서(P(yes)·RRF·BM25·코사인) 점수는 순위 기반(depth+1-rank)으로 쓴다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(run):
            for rank, uid in enumerate(run[qid][:depth], 1):
                f.write(f"{qid} Q0 {uid} {rank} {depth + 1 - rank} {run_name}\n")


def read_run(path: Path) -> Run:
    """TREC 실행 파일 → {qid: [unit_id, ...]} (rank 순)"""
    rows: dict[str, list[tuple[int, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 4:
                rows.setdefault(cols[0], []).append((int(cols[3]), cols[2]))
    return {qid: [u for _r, u in sorted(v)] for qid, v in rows.items()}
