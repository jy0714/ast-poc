"""CLI — `cd experiments_v4 && python -m ev4 <명령>` (지시서 §6.5)

    corpus                                   단위 통계 → results/T1_corpus.md
    queries                                  질의 구성 T0(판정 전) + 입력 검증 경고 → results/T0_queries.md
    run --split dev|test --runs R2 R4 | all  retrieve() 실행 → runs/
    pool [--split test] [--no-translate]     판정 시트 → data/judgments/pool_{A,B}.csv
    judgments import <csv...>                판정 가져오기
    judgments agreement                      일치도 + 중재 시트
    judgments finalize                       최종 qrels → data/qrels.txt
    evaluate                                 T2~T6 → results/tables.md
    latency                                  T7 → results/T7_latency.md
    failures                                 F1 → results/F1_failures.csv
    env                                      E1 → results/E1_env.md
    selftest [--keep DIR]                    가짜 모델 전체 흐름 점검 (GPU·실데이터 없음)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ev4", description="AuditRAG v4 실험 하네스")
    ap.add_argument("--root", type=Path, default=None, help="실험 루트 (기본 experiments_v4/)")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml 경로 (기본 {root}/config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus")
    sub.add_parser("queries")
    p = sub.add_parser("run")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--runs", nargs="+", default=["all"])
    p = sub.add_parser("pool")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--no-translate", action="store_true", help="현지어 단위 번역 생략")
    p = sub.add_parser("judgments")
    js = p.add_subparsers(dest="jcmd", required=True)
    pi = js.add_parser("import")
    pi.add_argument("csv", nargs="+", type=Path)
    js.add_parser("agreement")
    js.add_parser("finalize")
    sub.add_parser("evaluate")
    sub.add_parser("latency")
    sub.add_parser("failures")
    sub.add_parser("env")
    p = sub.add_parser("selftest")
    p.add_argument("--keep", type=Path, default=None, help="이 폴더에 산출물을 남긴다")
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점"""
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.disable(logging.INFO)  # src 로거의 INFO 콘솔 출력은 끈다 (경고 이상만)

    if args.cmd == "selftest":
        from ev4.selftest import selftest

        selftest(root=args.keep, keep=args.keep is not None)
        return 0

    from ev4.io import load_config, query_warnings, read_queries

    cfg = load_config(args.config, args.root)
    if args.cmd in ("run", "pool"):
        for w in query_warnings(read_queries(cfg.queries_path)):  # 경고만 내고 실행은 계속 (v4.2 Q2)
            print(f"[경고] {w}")
    if args.cmd == "corpus":
        from ev4.corpus import build_corpus

        build_corpus(cfg)
    elif args.cmd == "queries":
        from ev4.querystats import write_t0

        write_t0(cfg, read_queries(cfg.queries_path))
    elif args.cmd == "run":
        from ev4.run import run_all

        names = list(cfg.runs) if args.runs == ["all"] else args.runs
        unknown = [r for r in names if r not in cfg.runs]
        if unknown:
            raise SystemExit(f"config 에 없는 실행: {unknown}")
        run_all(cfg, read_queries(cfg.queries_path, split=args.split), names, args.split)
    elif args.cmd == "pool":
        from ev4.pool import build_pool

        build_pool(cfg, read_queries(cfg.queries_path, split=args.split), args.split,
                   translate=not args.no_translate)
    elif args.cmd == "judgments":
        from ev4 import judgments

        if args.jcmd == "import":
            judgments.import_judgments(cfg, args.csv)
        elif args.jcmd == "agreement":
            judgments.agreement(cfg)
        else:
            judgments.finalize(cfg)
    elif args.cmd == "evaluate":
        from ev4.evaluate import evaluate_all

        evaluate_all(cfg)
    elif args.cmd == "latency":
        from ev4.latency import measure

        measure(cfg, read_queries(cfg.queries_path, split="test"))
    elif args.cmd == "failures":
        from ev4.failures import build_failures

        build_failures(cfg)
    elif args.cmd == "env":
        from ev4.env import write_env

        write_env(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
