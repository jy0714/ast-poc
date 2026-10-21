"""E1 환경 기록(v4.2 X1)과 T7 표 형식(v4.2 L1~L4) 테스트 — 합성 값만 쓴다"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ev4 import env, latency  # noqa: E402


def _q(p50: float, p95: float) -> dict[str, float]:
    return {"p50": p50, "p95": p95, "n": 10}


def _summary() -> dict:
    stages_fused = {k: _q(1.0, 2.0) for k in ("bm25", "embed_query", "dense", "rrf", "total")}
    return {
        "runs": {
            "R6": stages_fused,
            "R8": {**stages_fused, "rerank": _q(800.0, 950.0)},
            "R9": {**stages_fused, "rerank": _q(300.0, 400.0)},
        },
        "rerank_truncation": {"R8": 0.05, "R9": 0.0},
        "generation": {k: _q(10.0, 20.0) for k in ("ttft_ms", "total_ms", "tokens_per_s")},
        "generation_run": "R8",
        "vram_max_mib": 24100,
        "torch_peak_mib": 9500.0,
        "service_memory": {
            "vram_max_mib": 23800,
            "vram_now_mib": 23700,
            "torch_peak_mib": 8400.0,
            "loaded_models": [
                {
                    "name": "gemma4:12b",
                    "size": 9 * 2**30,
                    "size_vram": 9 * 2**30,
                    "context_length": 16384,
                }
            ],
        },
        "loaded_models": [{"name": "gemma4:12b"}],
        "models": {
            "gemma4:12b": {
                "quantization": "Q4_K_M",
                "parameter_size": "11.9B",
                "file_size_bytes": 8 * 2**30,
                "digest": "abc",
            }
        },
        "llm_num_ctx": 16384,
        "rerank_max_length": 4096,
        "n_queries": 10,
        "warmup": 3,
        "repeats": 5,
        "fake_models": True,
    }


def test_t7_stage_rows_fixed_and_r6_rerank_dash() -> None:
    md = latency.render_t7(_summary())
    labels = [label for _k, label in latency.STAGE_LABELS]
    assert labels == ["질의 임베딩", "밀집 검색", "BM25 검색", "RRF 결합", "재순위", "검색 전체"]
    assert "| 단계 | R6 | R8 | R9 |" in md
    rerank_row = next(line for line in md.splitlines() if line.startswith("| 재순위 |"))
    assert rerank_row == "| 재순위 | – | 800.0 / 950.0 | 300.0 / 400.0 |"
    # 표 행 순서가 STAGE_LABELS 순서와 같다
    positions = [md.index(f"| {label} |") for label in labels]
    assert positions == sorted(positions)


def test_t7_truncation_and_memory_sections() -> None:
    md = latency.render_t7(_summary())
    assert "4,096토큰 초과" in md
    assert "| R8 | 5.0% |" in md and "| R6 | – |" in md
    assert "서비스 세 모델 적재 시점까지 | 23,800 |" in md
    assert "서비스 재순위기만 | 8,400 |" in md
    assert "| gemma4:12b | 9,216 | 9,216 | 16384 |" in md
    assert "| gemma4:12b | Q4_K_M | 11.9B | 8,192 | abc |" in md
    assert "LLM_NUM_CTX" in md and "16384" in md


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils.config import settings

    monkeypatch.setattr(settings, "ast_fake_models", True)


def test_e1_has_x1_fields(fake_models: None) -> None:
    from src.utils.config import settings

    e = env.collect_env()
    assert e["bm25"]["class"] == "rank_bm25.BM25Okapi"
    assert e["bm25"]["k1"] > 0 and 0 <= e["bm25"]["b"] <= 1
    assert e["prompts"]["EMBED_QUERY_TASK"] == settings.embed_query_task
    assert e["prompts"]["RERANK_INSTRUCTION"] == settings.rerank_instruction
    assert settings.rerank_instruction in e["prompts"]["rerank_input_format"]
    for key in (
        "unit_max_part_tokens",
        "rerank_max_length",
        "rrf_k",
        "k_bm25",
        "k_dense",
        "rerank_candidates",
        "chat_session_gap_minutes",
        "seed",
    ):
        assert key in e["settings"]
    assert e["hardware"]["cores_logical"]
    md = env.render_e1(e)
    for needle in (
        "## 지시문 원문",
        "UNIT_MAX_PART_TOKENS",
        "CHAT_SESSION_GAP_MINUTES",
        "k1=",
        "가짜 모델 모드",
        "단위 규칙",
        "rank-bm25",
    ):
        assert needle in md


def test_e1_degrades_without_gpu(fake_models: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_run", lambda cmd: "")
    e = env.collect_env()
    assert "name" not in e["gpu"]
    md = env.render_e1(e)
    assert "| GPU | – (–" in md
