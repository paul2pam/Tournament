"""Phase 1 checkpoint (spec §8, non-negotiable): the full closed loop driven by a
synthetic voter must converge before humans touch it.

Resets the DB, bootstraps from the standing checkpoint, then alternates synthetic
vote batches (torso-height preference, 10% noise, 20% bot sessions) with worker
cycles until 10 generations have passed. Asserts:
  1. desc_torso_h of successive incumbents is non-decreasing (tolerance 1cm —
     vote noise can promote a marginal challenger; a real regression fails).
  2. Every resolved contest consumed 3-20 comparisons (counted from votes, not
     the cached column).

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

HEIGHT_TOLERANCE = 0.01


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
    args = p.parse_args()

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

    heights = []  # (generation, incumbent policy id, desc_torso_h) at each promotion
    t0 = time.time()
    batches = 0

    async def incumbent_row():
        return await conn.fetchrow(
            "SELECT id, generation, desc_torso_h FROM policies WHERE status='incumbent'"
        )

    inc = await incumbent_row()
    heights.append((inc["generation"], inc["id"], inc["desc_torso_h"]))
    print(f"gen {inc['generation']}: incumbent p{inc['id']} torso_h={inc['desc_torso_h']:.4f}", flush=True)

    while heights[-1][0] < args.target_gens and batches < args.max_batches:
        stats = run_votes(
            args.base_url, args.votes_per_batch, rng, bot_frac=0.2, noise=0.10
        )
        batches += 1
        run_worker(["--cycle", "--seed", str(args.seed + batches)], env_overrides)
        inc = await incumbent_row()
        if inc["generation"] != heights[-1][0]:
            heights.append((inc["generation"], inc["id"], inc["desc_torso_h"]))
            print(
                f"gen {inc['generation']}: incumbent p{inc['id']} "
                f"torso_h={inc['desc_torso_h']:.4f}  "
                f"(batch {batches}, {stats['votes']} votes cast, {time.time()-t0:.0f}s elapsed)",
                flush=True,
            )

    # -- checkpoint assertions ------------------------------------------------
    failures = []

    if heights[-1][0] < args.target_gens:
        failures.append(
            f"only reached gen {heights[-1][0]} of {args.target_gens} "
            f"after {batches} vote batches"
        )

    for (g0, p0, h0), (g1, p1, h1) in zip(heights, heights[1:]):
        if h1 < h0 - HEIGHT_TOLERANCE:
            failures.append(
                f"torso height regressed: gen {g0} p{p0} {h0:.4f} -> gen {g1} p{p1} {h1:.4f}"
            )

    contest_counts = await conn.fetch(
        """SELECT c.id, c.status, count(v.id) AS n
           FROM contests c LEFT JOIN votes v
             ON v.contest_id = c.id AND v.pair_type='contest'
           WHERE c.status <> 'open'
           GROUP BY c.id, c.status"""
    )
    out_of_range = [
        (r["id"], r["n"]) for r in contest_counts if not (3 <= r["n"] <= 20)
    ]
    if out_of_range:
        failures.append(f"contests resolved outside 3-20 comparisons: {out_of_range}")

    report = {
        "target_gens": args.target_gens,
        "reached_gen": heights[-1][0],
        "vote_batches": batches,
        "elapsed_s": round(time.time() - t0),
        "incumbent_heights": [
            {"generation": g, "policy": p, "torso_h": h} for g, p, h in heights
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
