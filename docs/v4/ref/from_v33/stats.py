"""통계: 질의 재표집 bootstrap CI, 대응 t-검정, 대응 무작위화(sign-flip) 검정, Holm 보정.

검색은 결정적(deterministic)이라 같은 질의를 5번 돌려도 품질 지표의 표준편차는 0 이다.
따라서 품질 지표의 불확실성은 '질의 표본' 에서 오며, 질의 재표집 bootstrap 이 올바른 CI 다.
IR 에서는 무작위화 검정이 t-검정보다 권장된다 (Smucker et al., CIKM 2007). 둘 다 보고한다.
"""

from __future__ import annotations

import math

import numpy as np


def _clean(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[~np.isnan(v)]


def bootstrap_ci(values, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    v = _clean(values)
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def _paired(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_tests(a, b, n_perm: int = 10000, n_boot: int = 1000, seed: int = 0) -> dict:
    """a - b 에 대한 대응 검정. 반환: n, mean_diff, diff CI, p_ttest, p_perm."""
    a, b = _paired(a, b)
    out = {"n": int(a.size), "mean_diff": math.nan, "diff_lo": math.nan, "diff_hi": math.nan,
           "p_ttest": math.nan, "p_perm": math.nan}
    if a.size < 2:
        return out
    d = a - b
    out["mean_diff"], out["diff_lo"], out["diff_hi"] = bootstrap_ci(d, n=n_boot, seed=seed)
    if np.allclose(d, 0):
        out["p_ttest"] = out["p_perm"] = 1.0
        return out
    if np.allclose(d, d[0]):
        out["p_ttest"] = 0.0  # 차이가 상수(분산 0)면 t 는 무한대
    else:
        try:
            import warnings

            from scipy import stats as st

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out["p_ttest"] = float(st.ttest_rel(a, b).pvalue)
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    out["p_perm"] = float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni 보정. NaN 은 그대로 둔다."""
    items = [(k, p) for k, p in pvals.items() if not (p is None or math.isnan(p))]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {k: math.nan for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def quadratic_weighted_kappa(a: list[int], b: list[int], n_levels: int = 4) -> float:
    """판정자 간 일치도 (등급 0~3, 순서형 → 2차 가중 카파)."""
    if not a:
        return math.nan
    O = np.zeros((n_levels, n_levels))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(n_levels)] for i in range(n_levels)], dtype=float)
    W /= (n_levels - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else math.nan


def percentile(values, q: float) -> float:
    v = _clean(values)
    return float(np.percentile(v, q)) if v.size else math.nan
