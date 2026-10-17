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


def paired_bootstrap_ci(
    a: Iterable[float], b: Iterable[float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """짝지은 bootstrap — 같은 질의 재표집에서 (b − a) 평균 차이 → (diff, lo, hi) (v4.2 E2)

    a·b 는 같은 질의 순서로 정렬된 질의별 값이다. 둘 중 하나가 NaN 인 질의는 뺀다.
    """
    x, y = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"짝지은 값의 길이가 다릅니다 ({x.size} vs {y.size})")
    keep = ~(np.isnan(x) | np.isnan(y))
    d = (y - x)[keep]
    if d.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))


def paired_randomization_p(
    a: Iterable[float], b: Iterable[float], n: int = 10000, seed: int = 0
) -> float:
    """양측 짝지은 무작위화 검정 p (질의마다 두 실행 값을 무작위로 맞바꿈 = 차이 부호 뒤집기)

    ranx fisher 와 같은 귀무가설·통계량(|평균 차이|)이다. 단일 스레드 numpy 난수라
    seed 가 같으면 결과가 같다. ranx 결과의 교차 확인용으로 함께 보고한다 (v4.2 E3).
    """
    x, y = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    d = (y - x)[keep]
    if d.size == 0:
        return math.nan
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    # 부동소수 오차로 관측값과 같은 순열을 놓치지 않게 작은 허용 오차를 둔다
    return float((perm >= obs - 1e-12).mean())
