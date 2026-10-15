"""통계: 질의 재표집 bootstrap CI, Holm 보정, 2차 가중 카파, 백분위 (v3.3 하네스 이식)

검색은 결정적이라 같은 질의를 반복해도 품질 지표의 분산은 0 이다. 불확실성은 질의 표본에서
오므로 질의 재표집 bootstrap 이 올바른 CI 다. 유의성 검정은 ranx fisher(무작위화 검정)를 쓴다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _clean(values: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(values), dtype=float)
    return v[~np.isnan(v)]


def bootstrap_ci(
    values: Iterable[float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """평균과 질의 재표집 bootstrap (1-alpha) 구간 → (mean, lo, hi). 값이 없으면 NaN"""
    v = _clean(values)
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni 보정 (한 계열). NaN 은 계열에서 빼고 그대로 둔다"""
    items = [(k, p) for k, p in pvals.items() if not (p is None or math.isnan(p))]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {k: math.nan for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def quadratic_weighted_kappa(a: list[int], b: list[int], n_levels: int = 3) -> float:
    """판정자 간 일치도 (등급 0/1/2, 순서형 → 2차 가중 카파)"""
    if not a:
        return math.nan
    obs = np.zeros((n_levels, n_levels))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(n_levels)] for i in range(n_levels)], dtype=float)
    w /= (n_levels - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    """백분위 (q: 0~100). 값이 없으면 NaN"""
    v = _clean(values)
    return float(np.percentile(v, q)) if v.size else math.nan
