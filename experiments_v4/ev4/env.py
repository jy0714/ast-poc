"""환경 기록 → results/env.json, results/E1_env.md (지시서 §6.8 E1)

GPU·드라이버·CUDA·OS·Python·라이브러리 버전, Ollama 버전·모델 태그(digest·양자화),
HF 모델 리비전(로컬 캐시 refs/main), 토크나이저 버전, git 커밋.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from ev4 import PROJECT_ROOT, ollama
from ev4.io import ExperimentConfig
from src.utils.config import settings

LIBRARIES = ("torch", "transformers", "FlagEmbedding", "sentence-transformers", "numpy", "ranx", "kiwipiepy",
             "lingua-language-detector", "langchain-ollama", "httpx", "PyMuPDF", "fastapi")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return ""


def gpu_info() -> dict[str, Any]:
    """nvidia-smi 와 torch 로 본 GPU·드라이버·CUDA"""
    out: dict[str, Any] = {}
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if smi:
        name, driver, mem = (x.strip() for x in smi.splitlines()[0].split(","))
        out.update({"name": name, "driver": driver, "memory_total": mem})
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as e:
        out["torch_error"] = str(e)[:200]
    return out


def lib_versions() -> dict[str, str]:
    """설치된 라이브러리 버전 (없으면 "")"""
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = ""
    return out


def hf_revision(repo: str) -> str:
    """HF 로컬 캐시의 refs/main 커밋 (없으면 "")"""
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def collect_env() -> dict[str, Any]:
    """환경 정보 dict"""
    from src.index.case_index import BGE_M3_MODEL
    from src.text.bm25_tokenizer import TOKENIZER_VERSION
    from src.units.schema import UNIT_RULES_VERSION
    from src.units.tokens import get_token_counter

    hf_models = [settings.rerank_model, settings.rerank_bge_model, settings.unit_token_hf_name or ""]
    return {
        "gpu": gpu_info(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": lib_versions(),
        "ollama": {
            "version": ollama.version(),
            "models": {m: ollama.show(m) for m in (settings.ollama_embed_model, BGE_M3_MODEL,
                                                   settings.ollama_llm_model)},
        },
        "hf_revisions": {m: hf_revision(m) for m in hf_models if m},
        "tokenizers": {"bm25": TOKENIZER_VERSION, "unit_rules": UNIT_RULES_VERSION,
                       "token_counter": get_token_counter().kind},
        "git_commit": _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src"])),
        "settings": {
            "embed_model": settings.ollama_embed_model, "llm_model": settings.ollama_llm_model,
            "rerank_model": settings.rerank_model, "unit_max_part_tokens": settings.unit_max_part_tokens,
        },
        "fake_models": settings.ast_fake_models,
    }


def render_e1(e: dict[str, Any]) -> str:
    """E1 마크다운"""
    g = e["gpu"]
    lines = [
        "# E1 실험 환경",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| GPU | {g.get('name', '–')} ({g.get('memory_total', '–')}) |",
        f"| 드라이버 / CUDA(torch) | {g.get('driver', '–')} / {g.get('torch_cuda') or '–'} |",
        f"| OS | {e['os']} |",
        f"| Python | {e['python']} |",
        f"| git | {e['git_commit']}{' (src 수정 있음)' if e['git_dirty'] else ''} |",
        f"| Ollama | {e['ollama']['version'] or '–'} |",
        f"| 토크나이저 | BM25 {e['tokenizers']['bm25']} · 단위 규칙 {e['tokenizers']['unit_rules']} · "
        f"토큰 수 {e['tokenizers']['token_counter']} |",
        "",
        "## 모델",
        "",
        "| 모델 | 양자화 | 크기 | digest / revision |",
        "|---|---|---|---|",
    ]
    for m, info in e["ollama"]["models"].items():
        lines.append(f"| {m} (Ollama) | {info.get('quantization', '–')} | {info.get('parameter_size', '–')} | "
                     f"{str(info.get('digest', ''))[:12] or '–'} |")
    for m, rev in e["hf_revisions"].items():
        lines.append(f"| {m} (HF) | – | – | {rev[:12] or '–'} |")
    lines += ["", "## 라이브러리", "", "| 라이브러리 | 버전 |", "|---|---|"]
    lines += [f"| {k} | {v or '–'} |" for k, v in e["libraries"].items()]
    return "\n".join(lines) + "\n"


def write_env(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """환경 기록 저장"""
    e = collect_env()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "env.json").write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.results_dir / "E1_env.md").write_text(render_e1(e), encoding="utf-8")
    log(f"[env] → {cfg.results_dir / 'E1_env.md'}")
    return e
