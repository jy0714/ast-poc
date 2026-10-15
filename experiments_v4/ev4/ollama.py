"""Ollama 상태 조회 (적재 모델·태그·버전) — 실행 조건 확인과 환경 기록용

가짜 모델 모드나 Ollama 가 없을 때는 빈 값을 돌려준다 (실험을 막지 않는다).
"""

from __future__ import annotations

from typing import Any

import httpx

from src.utils.config import settings

_TIMEOUT = 10.0


def _get(path: str) -> Any:
    resp = httpx.get(f"{settings.ollama_base_url}{path}", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def version() -> str:
    """Ollama 서버 버전 (없으면 "")"""
    if settings.ast_fake_models:
        return ""
    try:
        return str(_get("/api/version").get("version", ""))
    except Exception:
        return ""


def loaded_models() -> list[dict[str, Any]]:
    """`ollama ps` — 현재 적재된 모델 [{name, size, size_vram, details}]"""
    if settings.ast_fake_models:
        return []
    try:
        return list(_get("/api/ps").get("models", []))
    except Exception:
        return []


def show(model: str) -> dict[str, Any]:
    """`ollama show` 요약 — {model, digest, quantization, parameter_size, family}"""
    if settings.ast_fake_models:
        return {"model": model, "fake": True}
    out: dict[str, Any] = {"model": model}
    try:
        tags = {m["name"]: m for m in _get("/api/tags").get("models", [])}
        tag = tags.get(model) or tags.get(f"{model}:latest") or {}
        out["digest"] = tag.get("digest", "")
        resp = httpx.post(f"{settings.ollama_base_url}/api/show", json={"model": model}, timeout=_TIMEOUT)
        resp.raise_for_status()
        details = resp.json().get("details", {})
        out.update({
            "quantization": details.get("quantization_level", ""),
            "parameter_size": details.get("parameter_size", ""),
            "family": details.get("family", ""),
        })
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def is_loaded(model: str) -> bool:
    """모델이 적재돼 있는지 (태그 ':latest' 생략 허용)"""
    names = {m.get("name", "") for m in loaded_models()} | {m.get("model", "") for m in loaded_models()}
    return model in names or f"{model}:latest" in names
