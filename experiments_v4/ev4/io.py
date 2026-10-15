"""설정·질의·qrels·실행(TREC) 입출력 (지시서 §6.3, §6.4)

경로는 모두 실험 루트(기본 `experiments_v4/`) 기준이다. selftest 는 임시 폴더를 루트로 쓴다.

- 질의: `data/queries.jsonl` — {qid, text, type, split, query_lang, note}
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
SPLITS = ("test", "dev")
GRADES = (0, 1, 2)

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
    b_ratio: float = 0.25
    text_max_chars: int = 6000
    latency_warmup: int = 3
    latency_repeats: int = 5
    extra: dict[str, Any] = field(default_factory=dict)

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
        b_ratio=float(raw.get("b_ratio", 0.25)),
        text_max_chars=int(raw.get("text_max_chars", 6000)),
        latency_warmup=int(raw.get("latency_warmup", 3)),
        latency_repeats=int(raw.get("latency_repeats", 5)),
        extra={k: v for k, v in raw.items() if k not in {
            "case_id", "depth", "pool_depth", "seed", "runs", "pool_runs", "comparisons",
            "latency_runs", "b_ratio", "text_max_chars", "latency_warmup", "latency_repeats"}},
    )
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


def read_queries(path: Path, split: str | None = None) -> list[Query]:
    """queries.jsonl 읽기·검증

    Args:
        path: queries.jsonl
        split: "test" | "dev" | None(전체)

    Raises:
        ConfigError: 중복 qid, 빈 질의, 알 수 없는 type·split
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
        )
        if not q.qid or not q.text:
            raise ConfigError(f"{path}:{i}: qid·text 가 비어 있습니다")
        if q.qid in seen:
            raise ConfigError(f"{path}:{i}: 중복 qid {q.qid}")
        if q.type not in QUERY_TYPES:
            raise ConfigError(f"{path}:{i}: type 은 {QUERY_TYPES} 중 하나 (입력 {q.type!r})")
        if q.split not in SPLITS:
            raise ConfigError(f"{path}:{i}: split 은 {SPLITS} 중 하나 (입력 {q.split!r})")
        seen.add(q.qid)
        out.append(q)
    return [q for q in out if split is None or q.split == split]


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
