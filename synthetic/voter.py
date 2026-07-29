"""Synthetic voter (Phase 1): hits the real HTTP API and prefers higher mean torso height.

Simulates a mixed population so trust scoring gets exercised:
  - honest sessions: prefer higher mean torso z, ~10% noise, human-ish dt_ms
  - bot sessions:    always click left, sub-300ms dt_ms

Usage (library):  run_votes(base_url, n_votes, rng, bot_frac=0.2)
"""
import numpy as np
import httpx


def mean_torso_h(trajectory: dict) -> float:
    return float(np.mean([p[2] for p in trajectory["root_pos"]]))


def run_votes(
    base_url: str,
    n_votes: int,
    rng: np.random.Generator,
    bot_frac: float = 0.2,
    noise: float = 0.10,
    n_sessions: int = 8,
) -> dict:
    """Casts up to n_votes votes. Returns counters incl. resolutions seen."""
    sessions = [
        {"id": f"synth-{i}", "bot": rng.random() < bot_frac} for i in range(n_sessions)
    ]
    stats = {"votes": 0, "resolutions": [], "checks_passed": 0, "checks_failed": 0}
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        while stats["votes"] < n_votes:
            pairs = client.get("/pairs", params={"n": 5}).json()["pairs"]
            if not pairs:
                break
            for pair in pairs:
                if stats["votes"] >= n_votes:
                    break
                s = sessions[int(rng.integers(len(sessions)))]
                a, b = pair["clips"]
                if s["bot"]:
                    winner = a["id"]                      # always left
                    dt = int(rng.integers(50, 250))       # bot-speed
                else:
                    ha, hb = mean_torso_h(a["trajectory"]), mean_torso_h(b["trajectory"])
                    winner = a["id"] if ha >= hb else b["id"]
                    if rng.random() < noise:
                        winner = b["id"] if winner == a["id"] else a["id"]
                    dt = int(rng.integers(700, 2500))
                r = client.post(
                    "/vote",
                    json={
                        "pair_token": pair["pair_token"],
                        "winner_clip": winner,
                        "dt_ms": dt,
                        "session_id": s["id"],
                    },
                )
                if r.status_code != 200:
                    continue
                fb = r.json()
                stats["votes"] += 1
                if fb.get("resolution"):
                    stats["resolutions"].append(fb["resolution"]["outcome"])
    return stats
