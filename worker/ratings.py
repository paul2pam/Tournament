"""Policy ratings aggregated per policy across ALL its clips (spec §5).

rating_mean = weighted win rate; rating_lb = Wilson lower bound at 95%.
Selection anywhere in the system uses rating_lb — a mediocre policy with one
lucky 3-second window must not beat a consistent one.
"""
import math

Z95 = 1.959963984540054


def wilson_lb(wins: float, total: float, z: float = Z95) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denom)


def policy_rating(weighted_wins: float, weighted_total: float) -> tuple[float, float]:
    """Returns (rating_mean, rating_lb)."""
    if weighted_total <= 0:
        return 0.0, 0.0
    return weighted_wins / weighted_total, wilson_lb(weighted_wins, weighted_total)
