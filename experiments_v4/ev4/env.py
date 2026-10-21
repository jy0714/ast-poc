"""환경 기록 → results/env.json, results/E1_env.md (지시서 §6.8 E1, v4.2 X1)

하드웨어(CPU·코어·RAM·GPU·드라이버·CUDA·OS), 라이브러리·Ollama·OCR 버전, BM25 매개변수(k1·b 실제 값),
Ollama 태그·양자화·파일 크기, HF 모델 리비전(로컬 캐시), 지시문 원문(EMBED_QUERY_TASK·RERANK_INSTRUCTION),
검색·단위 설정값, 토크나이저 버전, git 커밋. GPU·Ollama가 없거나 가짜 모델 모드면 빈 값(–)으로 남긴다.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from ev4 import PROJECT_ROOT, ollama
from ev4.io import ExperimentConfig
from src.utils.config import settings

LIBRARIES = (
    "torch",
    "transformers",
    "FlagEmbedding",
    "sentence-transformers",
    "numpy",
    "ranx",
    "rank-bm25",
    "kiwipiepy",
    "lingua-language-detector",
    "langchain-ollama",
    "httpx",
    "PyMuPDF",
    "pytesseract",
    "paddleocr",
    "paddlepaddle",
    "fastapi",
)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:
        return ""


# ─────────────────────────── 하드웨어 ───────────────────────────


def cpu_name() -> str:
    """CPU 모델 이름 (Windows 레지스트리 → /proc/cpuinfo → platform.processor 순)"""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ram_gib() -> float | None:
    """물리 메모리 총량 (GiB)"""
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        return None


def hardware_info() -> dict[str, Any]:
    """CPU 모델·물리/논리 코어·RAM"""
    physical: int | None = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        pass
    ram = ram_gib()
    return {
        "cpu": cpu_name(),
        "cores_physical": physical,
        "cores_logical": os.cpu_count(),
        "ram_gib": round(ram, 1) if ram else None,
    }


def gpu_info() -> dict[str, Any]:
    """nvidia-smi 와 torch 로 본 GPU·드라이버·CUDA"""
    out: dict[str, Any] = {}
    smi = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    if smi:
        name, driver, mem = (x.strip() for x in smi.splitlines()[0].split(","))
        out.update({"name": name, "driver": driver, "memory_total": mem})
        m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", _run(["nvidia-smi"]))
        if m:
            out["driver_cuda"] = m.group(1)  # 드라이버가 지원하는 CUDA 버전
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda  # torch 빌드 CUDA 버전
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["bf16_supported"] = torch.cuda.is_bf16_supported()
            out["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:
        out["torch_error"] = str(e)[:200]
    return out


# ─────────────────────────── 소프트웨어 ───────────────────────────


def lib_versions() -> dict[str, str]:
    """설치된 라이브러리 버전 (없으면 "")"""
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = ""
    return out


def ocr_info() -> dict[str, Any]:
    """설정 OCR 엔진, 실제 선택된 엔진, Tesseract 바이너리 버전·설치 언어 데이터"""
    out: dict[str, Any] = {
        "configured": settings.ocr_engine,
        "languages": settings.ocr_languages,
        "selected": "",
        "tesseract_version": "",
        "tesseract_langs": [],
    }
    try:
        from src.parsers.ocr import get_ocr_engine

        out["selected"] = get_ocr_engine().name
    except Exception as e:
        out["selected_error"] = str(e)[:200]
    try:
        import pytesseract

        out["tesseract_version"] = str(pytesseract.get_tesseract_version())
        out["tesseract_langs"] = sorted(pytesseract.get_languages(config=""))
    except Exception:
        pass
    return out


def bm25_params() -> dict[str, Any]:
    """색인에 쓰는 BM25Okapi 의 실제 k1·b·epsilon (라이브러리 기본값을 객체에서 읽는다)"""
    try:
        from rank_bm25 import BM25Okapi

        bm = BM25Okapi([["x"]])
        return {"class": "rank_bm25.BM25Okapi", "k1": bm.k1, "b": bm.b, "epsilon": bm.epsilon}
    except Exception as e:
        return {"error": str(e)[:200]}


def hf_revision(repo: str) -> str:
    """HF 로컬 캐시의 refs/main 커밋 (없으면 스냅샷 폴더 이름, 그것도 없으면 "")"""
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    base = hub / f"models--{repo.replace('/', '--')}"
    try:
        return (base / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        snaps = sorted(p.name for p in (base / "snapshots").iterdir() if p.is_dir())
        return ",".join(snaps)
    except OSError:
        return ""


def ollama_models(models: list[str]) -> dict[str, dict[str, Any]]:
    """`ollama show` 요약 + `/api/tags` 의 모델 파일 크기(bytes)"""
    sizes: dict[str, int] = {}
    if not settings.ast_fake_models:
        try:
            for m in ollama._get("/api/tags").get("models", []):
                sizes[m.get("name", "")] = int(m.get("size") or 0)
        except Exception:
            pass
    out = {}
    for m in models:
        info = ollama.show(m)
        size = sizes.get(m) or sizes.get(f"{m}:latest")
        if size:
            info["file_size_bytes"] = size
        out[m] = info
    return out


def collect_env(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    """환경 정보 dict

    Args:
        cfg: 실험 설정 (seed·depth·실행 설정 기록용, 없으면 생략)
    """
    from src.embeddings.embedding_service import format_embed_query
    from src.index.case_index import BGE_M3_MODEL
    from src.rag.reranker import RERANK_PREFIX, RERANK_SUFFIX, format_rerank_pair
    from src.rag.retrieval import RetrievalConfig
    from src.text.bm25_tokenizer import TOKENIZER_VERSION
    from src.units.schema import UNIT_RULES_VERSION
    from src.units.tokens import get_token_counter

    hf_models = [
        settings.rerank_model,
        settings.rerank_bge_model,
        settings.unit_token_hf_name or "",
    ]
    rc = RetrievalConfig()
    token_kind = get_token_counter().kind
    return {
        "hardware": hardware_info(),
        "gpu": gpu_info(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": lib_versions(),
        "ocr": ocr_info(),
        "bm25": bm25_params(),
        "ollama": {
            "version": ollama.version(),
            "models": ollama_models(
                [settings.ollama_embed_model, BGE_M3_MODEL, settings.ollama_llm_model]
            ),
        },
        "hf_revisions": {m: hf_revision(m) for m in hf_models if m},
        "tokenizers": {
            "bm25": TOKENIZER_VERSION,
            "unit_rules": UNIT_RULES_VERSION,
            "token_counter": token_kind,
            "token_counter_is_real": token_kind.startswith("hf:"),
        },
        "prompts": {
            "EMBED_QUERY_TASK": settings.embed_query_task,
            "embed_query_format": format_embed_query("{query}"),
            "RERANK_INSTRUCTION": settings.rerank_instruction,
            "rerank_input_format": RERANK_PREFIX
            + format_rerank_pair("{query}", "{document}")
            + RERANK_SUFFIX,
        },
        "git_commit": _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(
            _run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src"])
        ),
        "settings": {
            "embed_model": settings.ollama_embed_model,
            "llm_model": settings.ollama_llm_model,
            "rerank_model": settings.rerank_model,
            "rerank_bge_model": settings.rerank_bge_model,
            "rerank_dtype": settings.rerank_dtype,
            "rerank_batch_size": settings.rerank_batch_size,
            "unit_max_part_tokens": settings.unit_max_part_tokens,
            "rerank_max_length": settings.rerank_max_length,
            "rrf_k": rc.rrf_k,
            "k_bm25": rc.k_bm25,
            "k_dense": rc.k_dense,
            "rerank_candidates": rc.rerank_candidates,
            "depth": cfg.depth if cfg else rc.depth,
            "chat_session_gap_minutes": settings.chat_session_gap_minutes,
            "embed_num_ctx": settings.embed_num_ctx,
            "llm_num_ctx": settings.llm_num_ctx,
            "seed": cfg.seed if cfg else None,
            "primary_subset": cfg.primary_subset if cfg else None,
        },
        "fake_models": settings.ast_fake_models,
    }


def _gib(n: Any) -> str:
    return f"{int(n) / 2**30:.2f} GiB" if n else "–"


def render_e1(e: dict[str, Any]) -> str:
    """E1 마크다운"""
    g, h, s, ocr = e["gpu"], e["hardware"], e["settings"], e["ocr"]
    bm = e["bm25"]
    cores = f"{h.get('cores_physical') or '–'}코어 / {h.get('cores_logical') or '–'}스레드"
    lines = [
        "# E1 실험 환경",
        ""
        if not e["fake_models"]
        else "\n> **가짜 모델 모드** (AST_FAKE_MODELS=1) — 모델 정보는 실측이 아니다\n",
        "## 하드웨어·OS",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| CPU | {h.get('cpu') or '–'} ({cores}) |",
        f"| RAM | {h['ram_gib'] if h.get('ram_gib') else '–'} GiB |",
        f"| GPU | {g.get('name', '–')} ({g.get('memory_total', '–')}, compute {g.get('compute_capability', '–')}, "
        f"BF16 {'지원' if g.get('bf16_supported') else '–'}) |",
        f"| 드라이버 / CUDA(드라이버) / CUDA(torch) | {g.get('driver', '–')} / {g.get('driver_cuda', '–')} / "
        f"{g.get('torch_cuda') or '–'} |",
        f"| OS | {e['os']} |",
        f"| Python | {e['python']} |",
        f"| git | {e['git_commit']}{' (src 수정 있음)' if e['git_dirty'] else ''} |",
        f"| Ollama | {e['ollama']['version'] or '–'} |",
        f"| OCR | 설정 {ocr['configured']} → 선택 {ocr.get('selected') or '–'} · Tesseract "
        f"{ocr.get('tesseract_version') or '–'} · 언어 데이터 {', '.join(ocr.get('tesseract_langs') or []) or '–'} "
        f"(요청 {ocr['languages']}) |",
        f"| 토크나이저 | BM25 {e['tokenizers']['bm25']} · 단위 규칙 {e['tokenizers']['unit_rules']} · "
        f"토큰 수 {e['tokenizers']['token_counter']}"
        f"{'' if e['tokenizers']['token_counter_is_real'] else ' (**근사**)'} |",
        f"| BM25 | {bm.get('class', '–')} k1={bm.get('k1', '–')}, b={bm.get('b', '–')}, "
        f"epsilon={bm.get('epsilon', '–')} |",
        "",
        "## 모델",
        "",
        "| 모델 | 양자화 | 매개변수 | 파일 크기 | digest / revision |",
        "|---|---|---|---|---|",
    ]
    for m, info in e["ollama"]["models"].items():
        lines.append(
            f"| {m} (Ollama) | {info.get('quantization') or '–'} | {info.get('parameter_size') or '–'} | "
            f"{_gib(info.get('file_size_bytes'))} | {str(info.get('digest', ''))[:12] or '–'} |"
        )
    for m, rev in e["hf_revisions"].items():
        lines.append(f"| {m} (HF) | – | – | – | {rev[:12] or '–'} |")
    lines += [
        "",
        "## 설정값",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| UNIT_MAX_PART_TOKENS | {s['unit_max_part_tokens']} |",
        f"| 재순위 max_length | {s['rerank_max_length']} |",
        f"| 재순위 dtype / 배치 | {s['rerank_dtype']} / {s['rerank_batch_size']} |",
        f"| rrf_k | {s['rrf_k']} |",
        f"| k (BM25 / 밀집) | {s['k_bm25']} / {s['k_dense']} |",
        f"| 재순위 후보 수 | {s['rerank_candidates']} |",
        f"| 평가 깊이 | {s['depth']} |",
        f"| CHAT_SESSION_GAP_MINUTES | {s['chat_session_gap_minutes']} |",
        f"| num_ctx (임베딩 / LLM) | {s['embed_num_ctx']} / {s['llm_num_ctx']} |",
        f"| seed | {s['seed'] if s['seed'] is not None else '–'} |",
        f"| 주 분석 집합 (primary_subset) | {s.get('primary_subset') or '–'} |",
        "",
        "## 지시문 원문",
        "",
        "EMBED_QUERY_TASK (질의에만 붙임, 문서에는 붙이지 않음):",
        "",
        "```",
        e["prompts"]["EMBED_QUERY_TASK"],
        "```",
        "",
        "질의 임베딩 입력 형식:",
        "",
        "```",
        e["prompts"]["embed_query_format"],
        "```",
        "",
        "RERANK_INSTRUCTION:",
        "",
        "```",
        e["prompts"]["RERANK_INSTRUCTION"],
        "```",
        "",
        "재순위 입력 형식 (Qwen3-Reranker):",
        "",
        "```",
        e["prompts"]["rerank_input_format"],
        "```",
        "",
        "## 라이브러리",
        "",
        "| 라이브러리 | 버전 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v or '–'} |" for k, v in e["libraries"].items()]
    return "\n".join(lines) + "\n"


def write_env(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """환경 기록 저장"""
    e = collect_env(cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "env.json").write_text(
        json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.results_dir / "E1_env.md").write_text(render_e1(e), encoding="utf-8")
    log(f"[env] → {cfg.results_dir / 'E1_env.md'}")
    return e
