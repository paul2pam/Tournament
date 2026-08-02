"""Phase 1 checkpoint (spec §8, non-negotiable): the full closed loop driven by a
synthetic voter must converge before humans touch it.

Resets the DB, bootstraps from the standing checkpoint, then alternates synthetic
vote batches (10% noise, 20% bot sessions) with worker cycles until 10 generations
have passed. Default preference metric is velocity — the seed stander is
stationary, so "prefers moving" has real headroom; height is at its ceiling once
Phase 0 passes (the run that used it stalled at gen 5/10 on pure tie-breaking).
Asserts:
  1. The incumbent's preferred metric is non-decreasing across promotions
     (small tolerance — vote noise can promote a marginal challenger).
  2. Every sequentially-resolved contest consumed 3-20 comparisons, counting
     only votes cast before resolution (stragglers on already-issued pairs
     land after) and excluding contests mooted at generation turnover.

Requires: Postgres up, server running on --base-url, NO external worker loop
(this harness owns worker cadence).

Run:  python -m synthetic.run_phase1 [--target-gens 10]
Exits nonzero on checkpoint failure. Writes synthetic/phase1_report.json.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import asyncpg
import httpx
import numpy as np

from common.config import DATABASE_URL
from synthetic.voter import run_votes

METRIC_COLUMN = {"torso_h": "desc_torso_h", "velocity": "desc_velocity"}
METRIC_TOLERANCE = {"torso_h": 0.01, "velocity": 0.02}


def run_worker(args_list, env_overrides):
    env = {**os.environ, **env_overrides}
    r = subprocess.run(
        [sys.executable, "-m", "worker.loop", *args_list],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise RuntimeError(f"worker.loop {args_list} failed")
    return r.stdout


async def reset_db(conn):
    await conn.execute(
        "TRUNCATE votes, pair_queue, contests, clips, policies, session_weights, "
        "worker_state RESTART IDENTITY CASCADE"
    )


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--target-gens", type=int, default=10)
    p.add_argument("--stand-ckpt", default="checkpoints/stand.pt")
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--finetune-steps", type=int, default=16384)
    p.add_argument("--votes-per-batch", type=int, default=30)
    p.add_argument("--max-batches", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--metric", choices=list(METRIC_COLUMN), default="velocity")
    # Spec's Phase 1 gate is a CLEAN synthetic signal (noise/bots belong to
    # robustness runs, not the checkpoint): defaults are clean.
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--bot-frac", type=float, default=0.0)
    args = p.parse_args()
    metric_col = METRIC_COLUMN[args.metric]
    tolerance = METRIC_TOLERANCE[args.metric]

    env_overrides = {
        "POOL_SIZE": str(args.pool_size),
        "FINETUNE_STEPS": str(args.finetune_steps),
        "OMP_NUM_THREADS": "4",
    }
    rng = np.random.default_rng(args.seed)
    conn = await asyncpg.connect(DATABASE_URL)

    # sanity: server must be up before we wipe anything
    with httpx.Client(timeout=10) as c:
        c.get(f"{args.base_url}/state").raise_for_status()

    print("== resetting DB + bootstrapping from", args.stand_ckpt, flush=True)
    await reset_db(conn)
    subprocess.run(["rm", "-rf", "blobs/clips", "checkpoints/policies"], check=True)
    run_worker(["--bootstrap", "--stand-ckpt", args.stand_ckpt, "--seed", str(args.seed)], env_overrides)

    metrics = []  # (generation, incumbent policy id, metric value) at each promotion
    t0 = time.time()
    batches = 0

    async def incumbent_row():
        return await conn.fetchrow(
            f"SELECT id, generation, {metric_col} AS m FROM policies WHERE status='incumbent'"
        )

    inc = await incumbent_row()
    metrics.append((inc["generation"], inc["id"], inc["m"]))
    print(f"gen {inc['generation']}: incumbent p{inc['id']} {args.metric}={inc['m']:.4f}", flush=True)

    while metrics[-1][0] < args.target_gens and batches < args.max_batches:
        stats = run_votes(
            args.base_url, args.votes_per_batch, rng,
            bot_frac=args.bot_frac, noise=args.noise, metric=args.metric,
        )
        batches += 1
        run_worker(["--cycle", "--seed", str(args.seed + batches)], env_overrides)
        inc = await incumbent_row()
        if inc["generation"] != metrics[-1][0]:
            metrics.append((inc["generation"], inc["id"], inc["m"]))
            print(
                f"gen {inc['generation']}: incumbent p{inc['id']} "
                f"{args.metric}={inc['m']:.4f}  "
                f"(batch {batches}, {stats['votes']} votes cast, {time.time()-t0:.0f}s elapsed)",
                flush=True,
            )

    # -- checkpoint assertions ------------------------------------------------
    failures = []

    if metrics[-1][0] < args.target_gens:
        failures.append(
            f"only reached gen {metrics[-1][0]} of {args.target_gens} "
            f"after {batches} vote batches"
        )

    for (g0, p0, m0), (g1, p1, m1) in zip(metrics, metrics[1:]):
        if m1 < m0 - tolerance:
            failures.append(
                f"{args.metric} regressed: gen {g0} p{p0} {m0:.4f} -> gen {g1} p{p1} {m1:.4f}"
            )

    # Only sequential-test resolutions; only votes cast before the verdict —
    # in-flight pairs can legitimately deliver votes after a contest closes.
    contest_counts = await conn.fetch(
        """SELECT c.id, c.status, count(v.id) AS n
           FROM contests c LEFT JOIN votes v
             ON v.contest_id = c.id AND v.pair_type='contest'
                AND v.created_at <= c.resolved_at
           WHERE c.status IN ('challenger_won', 'incumbent_held')
           GROUP BY c.id, c.status"""
    )
    out_of_range = [
        (r["id"], r["n"]) for r in contest_counts if not (3 <= r["n"] <= 20)
    ]
    if out_of_range:
        failures.append(f"contests resolved outside 3-20 comparisons: {out_of_range}")

    report = {
        "metric": args.metric,
        "target_gens": args.target_gens,
        "reached_gen": metrics[-1][0],
        "vote_batches": batches,
        "elapsed_s": round(time.time() - t0),
        "incumbent_metrics": [
            {"generation": g, "policy": p, args.metric: m} for g, p, m in metrics
        ],
        "resolved_contests": len(contest_counts),
        "resolution_counts": sorted(r["n"] for r in contest_counts),
        "failures": failures,
    }
    with open("synthetic/phase1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    await conn.close()
    print(json.dumps(report, indent=2))
    if failures:
        print("PHASE 1 CHECKPOINT: FAIL")
        sys.exit(1)
    print("PHASE 1 CHECKPOINT: PASS")


if __name__ == "__main__":
    asyncio.run(amain())
