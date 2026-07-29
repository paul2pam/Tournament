"""Sequential contest resolution on trust-weighted votes.

Beta-posterior rule: with weighted challenger wins w and weighted total n,
posterior over the challenger's true win rate is Beta(1 + w, 1 + n - w).
Resolve 'challenger_won' when P(p > 0.5) > threshold, 'incumbent_held' when
P(p > 0.5) < 1 - threshold, or at the comparison cap (tie -> incumbent holds).

Obvious mismatches resolve in 3-4 comparisons; close ones run to the cap (spec §5).
Shared by the worker (authoritative) and the server's inline check on POST /vote.
"""
import math

from common.config import CONTEST_MAX_COMPARISONS, CONTEST_POSTERIOR_THRESHOLD

MIN_COMPARISONS = 3


def _beta_cdf_at_half(a: float, b: float, n: int = 4096) -> float:
    """P(X < 0.5) for X ~ Beta(a, b) via Simpson integration of the pdf over [0, 0.5].

    a, b >= 1 always (prior Beta(1,1) + non-negative weighted counts), so the pdf is
    finite on the interval; counts stay small (<= ~21), so the pdf is smooth and
    Simpson at n=4096 is accurate far past what contest resolution needs.
    """
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    def pdf(x: float) -> float:
        if x == 0.0:
            return math.exp(lbeta) if a == 1.0 else 0.0
        return math.exp(lbeta + (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x))

    h = 0.5 / n
    s = pdf(0.0) + pdf(0.5)
    s += 4.0 * sum(pdf((2 * i + 1) * h) for i in range(n // 2))
    s += 2.0 * sum(pdf(2 * i * h) for i in range(1, n // 2))
    return min(1.0, max(0.0, s * h / 3.0))


def p_challenger_better(weighted_wins: float, weighted_total: float) -> float:
    """Posterior P(challenger win rate > 0.5)."""
    a = 1.0 + weighted_wins
    b = 1.0 + (weighted_total - weighted_wins)
    return 1.0 - _beta_cdf_at_half(a, b)


def resolve(
    weighted_wins: float,
    weighted_total: float,
    n_comparisons: int,
) -> str | None:
    """Returns 'challenger_won' | 'incumbent_held' | None (still open)."""
    if n_comparisons < MIN_COMPARISONS:
        return None
    p = p_challenger_better(weighted_wins, weighted_total)
    if p > CONTEST_POSTERIOR_THRESHOLD:
        return "challenger_won"
    if p < 1.0 - CONTEST_POSTERIOR_THRESHOLD:
        return "incumbent_held"
    if n_comparisons >= CONTEST_MAX_COMPARISONS:
        # cap reached without a verdict: incumbency wins ties
        return "challenger_won" if p > CONTEST_POSTERIOR_THRESHOLD else "incumbent_held"
    return None
