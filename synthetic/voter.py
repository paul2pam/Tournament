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
                    # Coherent total order: the metric decides; posture breaks only
                    # near-ties. (An earlier posture-first rule vetoed the metric —
                    # it scored fast CRAWLING as "fallen" and promoted a motionless
                    # stander over the lineage's fastest mover.) The tie-break is
                    # what passes attention checks: stander vs ragdoll is a 0-vs-0
                    # velocity tie, and upright wins it, as a human would vote.
                    ha, hb = score(a["trajectory"]), score(b["trajectory"])
                    if abs(ha - hb) < 0.05:
                        fa = mean_torso_h(a["trajectory"]) < 0.5
                        fb = mean_torso_h(b["trajectory"]) < 0.5
                        if fa != fb:
                            winner = b["id"] if fa else a["id"]
                        else:
                            winner = a["id"] if ha >= hb else b["id"]
                    else:
                        winner = a["id"] if ha > hb else b["id"]
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
