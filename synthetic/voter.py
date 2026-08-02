"""Synthetic voter (Phase 1): hits the real HTTP API with a scripted preference.

Metrics:
  torso_h  — prefers higher mean torso z (spec's example; near-ceiling once the
             seed policy stands at full height, so contests stall at ties)
  velocity — prefers higher mean horizontal root speed (wide-open headroom from
             a stationary stander; the official Phase 1 axis)

Simulates a mixed population so trust scoring gets exercised:
  - honest sessions: prefer higher metric, ~10% noise, human-ish dt_ms
  - bot sessions:    always click left, sub-300ms dt_ms

Usage (library):  run_votes(base_url, n_votes, rng, bot_frac=0.2, metric="velocity")
"""
import numpy as np
import httpx


def mean_torso_h(trajectory: dict) -> float:
    return float(np.mean([p[2] for p in trajectory["root_pos"]]))


def mean_velocity(trajectory: dict) -> float:
    xy = np.asarray(trajectory["root_pos"], dtype=float)[:, :2]
    speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) * trajectory["fps"]
    return float(speeds.mean())


def is_dead(trajectory: dict) -> bool:
    """Prone AND still in the final second — a settled corpse. A worming crawler
    keeps its joints moving and is NOT dead, however low its torso."""
    joints = np.asarray(trajectory["joints"], dtype=float)
    final_speed = float(np.abs(np.diff(joints[-30:], axis=0)).max() * trajectory["fps"])
    return mean_torso_h(trajectory) < 0.35 and final_speed < 0.3


METRICS = {"torso_h": mean_torso_h, "velocity": mean_velocity}


def run_votes(
    base_url: str,
    n_votes: int,
    rng: np.random.Generator,
    bot_frac: float = 0.2,
    noise: float = 0.10,
    n_sessions: int = 8,
    metric: str = "velocity",
) -> dict:
    """Casts up to n_votes votes. Returns counters incl. resolutions seen."""
    score = METRICS[metric]
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
                    # Coherent total order: a settled corpse loses to anything alive
                    # (this is what passes attention checks, as a human would); among
                    # the living, the metric decides. Earlier variants — posture-first
                    # (scored fast crawling as fallen) and near-tie posture breaks
                    # (the ragdoll's fall itself out-runs a stander) — were incoherent
                    # with the metric and are deliberately gone.
                    ha = -1.0 if is_dead(a["trajectory"]) else score(a["trajectory"])
                    hb = -1.0 if is_dead(b["trajectory"]) else score(b["trajectory"])
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
