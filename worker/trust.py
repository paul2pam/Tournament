"""Per-session trust weights. Down-weight, never ban (spec §5, §9).

Signals:
  - attention-check pass rate (known-good vs. ragdoll pairs)
  - dt_ms distribution (sub-300ms votes are bot-speed; spec floor)
  - vote entropy (all-left or all-right clicking)

Weight in [0.1, 1.0]. Recomputed over each session's full history every worker
cycle and cached in session_weights for the server's inline contest check.
The weighting logic is deliberately not published (spec §9).
"""
import math

FAST_MS = 300
MIN_WEIGHT = 0.1


def session_weight(
    n_checks: int,
    n_checks_passed: int,
    dt_ms_list: list[int],
    n_left: int,
    n_right: int,
) -> float:
    w = 1.0

    # Attention checks: harsh — failing known-good-vs-ragdoll is a strong signal.
    if n_checks > 0:
        pass_rate = n_checks_passed / n_checks
        w *= max(0.1, pass_rate**2)

    # Fast votes: fraction under the 300ms floor.
    if dt_ms_list:
        fast_frac = sum(1 for t in dt_ms_list if t < FAST_MS) / len(dt_ms_list)
        w *= 1.0 - 0.9 * fast_frac

    # Positional entropy: all-one-side clicking gets discounted once there's
    # enough history to matter (>= 8 votes).
    n = n_left + n_right
    if n >= 8:
        p = n_left / n
        entropy = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        w *= 0.25 + 0.75 * entropy

    return max(MIN_WEIGHT, min(1.0, w))
