"""Batch worker cycle (spec §5).

One cycle: pull vote deltas -> rescore session trust -> resolve contests ->
promote/reject -> (re)generate challenger pool -> roll out + filter -> refill queue.

Run:  python -m worker.loop --cycle          # one cycle (Phase 1 harness calls this)
      python -m worker.loop --bootstrap      # seed gen 0 from checkpoints/stand.pt
In production this is triggered by vote volume, floored at one hour — the trigger
lives in ops (cron on CSL), not here.
"""
import argparse
import asyncio
import secrets
import shutil
from pathlib import Path

import asyncpg
import numpy as np

from common import blobs
from common.config import (
    ANCESTOR_PROB,
    CKPT_DIR,
    DATABASE_URL,
    LEAK_RATE,
    POOL_SIZE,
    QUEUE_SHARE_ANCESTOR,
    QUEUE_SHARE_CHECK,
    QUEUE_TARGET,
)
from sim.challengers import challenger_scales, make_challenger
from sim.degeneracy import check_clip
from sim.descriptors import policy_descriptors
from sim.policy import load_policy, save_policy
from sim.rollout import rollout_policy
from worker.contests import resolve
from worker.ratings import policy_rating
from worker.trust import session_weight

POLICY_CKPT_DIR = Path(CKPT_DIR) / "policies"


class ZeroPolicy:
    """Ragdoll generator for attention checks: zero torque, humanoid collapses."""

    def act_deterministic(self, obs):
        return np.zeros(21)


# ---------------------------------------------------------------- state helpers

async def get_state(conn, key: str, default: str) -> str:
    row = await conn.fetchrow("SELECT value FROM worker_state WHERE key=$1", key)
    return row["value"] if row else default


async def set_state(conn, key: str, value: str) -> None:
    await conn.execute(
        """INSERT INTO worker_state (key, value) VALUES ($1, $2)
           ON CONFLICT (key) DO UPDATE SET value=$2""",
        key,
        value,
    )


# ---------------------------------------------------------------- clip creation

async def add_policy_clips(
    conn, policy_id: int, net, rng, skip_filter: bool = False, still: bool = False
) -> int:
    """Roll out, filter (with 15% leak), store descriptors + clip rows. Returns kept count.
    skip_filter also biases window sampling toward upright frames (bootstrap seed);
    still=True instead keeps the least dynamic windows (obvious-ragdoll checks)."""
    clips, episodes = rollout_policy(
        net,
        policy_id,
        base_seed=policy_id * 1000,
        prefer_alive=skip_filter and not still,
        prefer_still=still,
    )
    dv, dh = policy_descriptors(episodes)
    await conn.execute(
        "UPDATE policies SET desc_velocity=$2, desc_torso_h=$3 WHERE id=$1",
        policy_id, dv, dh,
    )
    kept = 0
    for c in clips:
        reason = None if skip_filter else check_clip(c["frames"])
        leaked = False
        if reason is not None:
            if rng.random() < LEAK_RATE:
                leaked = True   # leak the reject through, flagged, so the filter is auditable
            else:
                blobs.delete(c["blob_key"])
                continue
        await conn.execute(
            """INSERT INTO clips (policy_id, seed, window_start_s, blob_key, leaked)
               VALUES ($1, $2, $3, $4, $5)""",
            policy_id, c["seed"], c["window_start_s"], c["blob_key"], leaked,
        )
        kept += 1
    return kept


async def insert_policy(conn, generation: int, parent_id, status: str) -> int:
    return await conn.fetchval(
        """INSERT INTO policies (generation, parent_id, ckpt_path, status)
           VALUES ($1, $2, '', $3) RETURNING id""",
        generation, parent_id, status,
    )


async def create_policy(conn, net, generation: int, parent_id, status: str, rng) -> int:
    pid = await insert_policy(conn, generation, parent_id, status)
    POLICY_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = str(POLICY_CKPT_DIR / f"p{pid}.pt")
    save_policy(net, ckpt)
    await conn.execute("UPDATE policies SET ckpt_path=$2 WHERE id=$1", pid, ckpt)
    kept = await add_policy_clips(conn, pid, net, rng)
    if kept == 0 and status == "challenger":
        # fully degenerate challenger: nothing showable, auto-reject
        await conn.execute("UPDATE policies SET status='rejected' WHERE id=$1", pid)
    return pid


# ---------------------------------------------------------------- trust scoring

async def rescore_sessions(conn) -> dict[str, float]:
    rows = await conn.fetch(
        """SELECT session_id,
                  count(*) FILTER (WHERE pair_type='attention_check') AS n_checks,
                  count(*) FILTER (WHERE pair_type='attention_check' AND passed_check) AS n_passed,
                  array_agg(dt_ms) AS dts,
                  count(*) FILTER (WHERE winner_clip = shown_left) AS n_left,
                  count(*) FILTER (WHERE winner_clip <> shown_left) AS n_right
           FROM votes GROUP BY session_id"""
    )
    weights = {}
    for r in rows:
        w = session_weight(r["n_checks"], r["n_passed"], list(r["dts"]), r["n_left"], r["n_right"])
        weights[r["session_id"]] = w
        await conn.execute(
            """INSERT INTO session_weights (session_id, weight, updated_at)
               VALUES ($1, $2, now())
               ON CONFLICT (session_id) DO UPDATE SET weight=$2, updated_at=now()""",
            r["session_id"], w,
        )
    return weights


# ---------------------------------------------------------------- contests

async def contest_tally(conn, contest_id: int, challenger_policy: int, weights: dict):
    rows = await conn.fetch(
        """SELECT v.session_id, c.policy_id AS winner_policy
           FROM votes v JOIN clips c ON c.id = v.winner_clip
           WHERE v.contest_id=$1 AND v.pair_type='contest'""",
        contest_id,
    )
    w_total = sum(weights.get(r["session_id"], 1.0) for r in rows)
    w_wins = sum(
        weights.get(r["session_id"], 1.0) for r in rows if r["winner_policy"] == challenger_policy
    )
    return w_wins, w_total, len(rows)


async def resolve_contests(conn, weights: dict) -> bool:
    """Two steps. (1) Re-tally still-open contests with fresh trust weights and mark any
    that cross the threshold — the server also does this inline per vote; the worker is
    the recomputed, authoritative pass. (2) APPLY every marked-but-unapplied resolution:
    promotion, retirement, rejection. Returns True if the generation turned over."""
    for ct in await conn.fetch("SELECT * FROM contests WHERE status='open'"):
        w_wins, w_total, n = await contest_tally(conn, ct["id"], ct["challenger_policy"], weights)
        await conn.execute(
            "UPDATE contests SET n_comparisons=$2, n_challenger_wins=$3 WHERE id=$1",
            ct["id"], n, int(round(w_wins)),
        )
        outcome = resolve(w_wins, w_total, n)
        if outcome:
            await conn.execute(
                "UPDATE contests SET status=$2, resolved_at=now() WHERE id=$1 AND status='open'",
                ct["id"], outcome,
            )

    # Apply resolutions whose challenger hasn't been promoted/rejected yet (server marks
    # eagerly; this is where statuses actually change). First win takes the crown;
    # simultaneous wins are demoted — single lineage, one incumbent.
    pending = await conn.fetch(
        """SELECT ct.* FROM contests ct
           JOIN policies ch ON ch.id = ct.challenger_policy
           WHERE ct.status <> 'open' AND ch.status = 'challenger'
           ORDER BY ct.resolved_at"""
    )
    promoted = False
    for ct in pending:
        incumbent_now = await conn.fetchval(
            "SELECT status FROM policies WHERE id=$1", ct["incumbent_policy"]
        )
        if ct["status"] == "challenger_won" and not promoted and incumbent_now == "incumbent":
            await conn.execute(
                "UPDATE policies SET status='retired' WHERE id=$1", ct["incumbent_policy"]
            )
            await conn.execute(
                # generation = incumbent's + 1 for fresh mutants AND returning
                # ancestors alike (a re-crowned ancestor becomes a NEW generation)
                """UPDATE policies SET status='incumbent', promoted_at=now(),
                       generation=$2 WHERE id=$1""",
                ct["challenger_policy"], ct["generation"] + 1,
            )
            await set_state(conn, "generation", str(ct["generation"] + 1))
            promoted = True
        else:
            if ct["status"] == "challenger_won":
                await conn.execute(
                    "UPDATE contests SET status='incumbent_held' WHERE id=$1", ct["id"]
                )
            await _demote_challenger(conn, ct["challenger_policy"])

    if promoted:
        # Generation turned over: remaining open contests are against a retired
        # incumbent. 'mooted', not 'incumbent_held' — these were never resolved by
        # the sequential test and must not count as resolutions.
        for ct in await conn.fetch("SELECT * FROM contests WHERE status='open'"):
            await conn.execute(
                "UPDATE contests SET status='mooted', resolved_at=now() WHERE id=$1",
                ct["id"],
            )
            await _demote_challenger(conn, ct["challenger_policy"])
    return promoted


async def _demote_challenger(conn, policy_id: int) -> None:
    """Losing fresh mutants are rejected (blobs pruned). A losing ANCESTOR
    challenger (replay buffer entrant, recognizable by promoted_at) returns to
    'retired' — it keeps its timeline entry and its clips."""
    was_incumbent = await conn.fetchval(
        "SELECT promoted_at IS NOT NULL FROM policies WHERE id=$1", policy_id
    )
    await conn.execute(
        "UPDATE policies SET status=$2 WHERE id=$1",
        policy_id, "retired" if was_incumbent else "rejected",
    )


async def prune_rejected_blobs(conn) -> None:
    """Spec §9: prune trajectory blobs for rejected policies once their contest is over.
    DB rows stay (vote history); only blob files go. Queue rows referencing them are removed."""
    rows = await conn.fetch(
        """SELECT c.id, c.blob_key FROM clips c
           JOIN policies p ON p.id = c.policy_id
           WHERE p.status='rejected' AND c.blob_key <> ''"""
    )
    for r in rows:
        await conn.execute(
            "DELETE FROM pair_queue WHERE NOT consumed AND (clip_a=$1 OR clip_b=$1)", r["id"]
        )
        blobs.delete(r["blob_key"])
        await conn.execute("UPDATE clips SET blob_key='' WHERE id=$1", r["id"])


# ---------------------------------------------------------------- ratings

async def update_ratings(conn, weights: dict) -> None:
    rows = await conn.fetch(
        """SELECT v.session_id, v.winner_clip, ca.policy_id AS pa, cb.policy_id AS pb,
                  cw.policy_id AS pw
           FROM votes v
           JOIN clips ca ON ca.id = v.clip_a
           JOIN clips cb ON cb.id = v.clip_b
           JOIN clips cw ON cw.id = v.winner_clip
           WHERE v.pair_type <> 'attention_check'"""
    )
    tal: dict[int, list[float]] = {}
    for r in rows:
        w = weights.get(r["session_id"], 1.0)
        for pid in (r["pa"], r["pb"]):
            t = tal.setdefault(pid, [0.0, 0.0, 0, 0])   # w_wins, w_total, n_wins, n_losses
            t[1] += w
            won = r["pw"] == pid
            t[0] += w * won
            t[2] += won
            t[3] += not won
    for pid, (w_wins, w_total, n_wins, n_losses) in tal.items():
        mean, lb = policy_rating(w_wins, w_total)
        await conn.execute(
            """UPDATE policies SET rating_mean=$2, rating_lb=$3, n_wins=$4, n_losses=$5
               WHERE id=$1""",
            pid, mean, lb, n_wins, n_losses,
        )


# ---------------------------------------------------------------- pool + queue

async def spawn_pool(conn, rng) -> None:
    """If no open contests exist, breed a fresh challenger pool from the incumbent."""
    n_open = await conn.fetchval("SELECT count(*) FROM contests WHERE status='open'")
    if n_open > 0:
        return
    inc = await conn.fetchrow("SELECT * FROM policies WHERE status='incumbent'")
    incumbent_net = load_policy(inc["ckpt_path"])
    print(f"spawning pool of {POOL_SIZE} from incumbent p{inc['id']} (gen {inc['generation']})")
    for i, scale in enumerate(challenger_scales()):
        seed = int(rng.integers(1, 2**31))
        child = make_challenger(incumbent_net, scale, seed)
        pid = await create_policy(
            conn, child, inc["generation"] + 1, inc["id"], "challenger", rng
        )
        status = await conn.fetchval("SELECT status FROM policies WHERE id=$1", pid)
        if status == "challenger":
            await conn.execute(
                """INSERT INTO contests (generation, incumbent_policy, challenger_policy, status)
                   VALUES ($1, $2, $3, 'open')""",
                inc["generation"], inc["id"], pid,
            )
        print(f"  challenger p{pid} scale={scale} status={status}")

    # Replay buffer: some pools also field a PAST INCUMBENT as a challenger.
    # Zero compute (checkpoint and clips already exist); acts as drift rollback
    # (an ancestor can reclaim the crown), diversity injection, and theater.
    if rng.random() < ANCESTOR_PROB:
        anc = await conn.fetchrow(
            """SELECT p.id, p.generation FROM policies p
               WHERE p.status='retired' AND p.promoted_at IS NOT NULL AND p.id <> $1
                 AND EXISTS (SELECT 1 FROM clips c
                             WHERE c.policy_id=p.id AND c.blob_key <> '')
               ORDER BY coalesce(p.rating_lb, 0) DESC, random() LIMIT 1""",
            inc["id"],
        )
        if anc is not None:
            await conn.execute(
                "UPDATE policies SET status='challenger' WHERE id=$1", anc["id"]
            )
            await conn.execute(
                """INSERT INTO contests (generation, incumbent_policy, challenger_policy, status)
                   VALUES ($1, $2, $3, 'open')""",
                inc["generation"], inc["id"], anc["id"],
            )
            print(f"  ancestor challenger p{anc['id']} (was gen {anc['generation']}) re-enters")


async def _clip_ids(conn, policy_id: int) -> list[int]:
    return [
        r["id"]
        for r in await conn.fetch(
            "SELECT id FROM clips WHERE policy_id=$1 AND blob_key <> ''", policy_id
        )
    ]


async def refill_queue(conn, rng) -> int:
    n_unconsumed = await conn.fetchval("SELECT count(*) FROM pair_queue WHERE NOT consumed")
    deficit = QUEUE_TARGET - n_unconsumed
    if deficit <= 0:
        return 0

    inc = await conn.fetchrow("SELECT id FROM policies WHERE status='incumbent'")
    inc_clips = await _clip_ids(conn, inc["id"])
    open_contests = await conn.fetch(
        "SELECT id, incumbent_policy, challenger_policy FROM contests WHERE status='open'"
    )
    contest_clips = [
        (ct["id"], await _clip_ids(conn, ct["incumbent_policy"]),
         await _clip_ids(conn, ct["challenger_policy"]))
        for ct in open_contests
    ]
    contest_clips = [(cid, a, b) for cid, a, b in contest_clips if a and b]

    ancestors = await conn.fetch(
        """SELECT id FROM policies
           WHERE status='retired' AND promoted_at IS NOT NULL AND id <> $1""",
        inc["id"],
    )
    ancestor_clips = []
    for a in ancestors:
        ancestor_clips.extend(await _clip_ids(conn, a["id"]))

    ragdoll = await conn.fetchrow("SELECT id FROM policies WHERE status='ragdoll'")
    ragdoll_clips = await _clip_ids(conn, ragdoll["id"]) if ragdoll else []

    added = 0
    for _ in range(deficit):
        r = rng.random()
        if r < QUEUE_SHARE_CHECK and ragdoll_clips and inc_clips:
            pair_type, contest_id = "attention_check", None
            a = int(rng.choice(inc_clips))
            b = int(rng.choice(ragdoll_clips))
        elif r < QUEUE_SHARE_CHECK + QUEUE_SHARE_ANCESTOR and ancestor_clips and inc_clips:
            pair_type, contest_id = "ancestor", None
            a = int(rng.choice(inc_clips))
            b = int(rng.choice(ancestor_clips))
        elif contest_clips:
            i = int(rng.integers(len(contest_clips)))
            contest_id, ia, ib = contest_clips[i]
            pair_type = "contest"
            # Stratified pairing: cycle a per-contest shuffled deck of all clip
            # matchups instead of sampling with replacement — otherwise a 4-vote
            # sweep can ride one lucky window (spec §5's "mediocre policy with one
            # lucky 3-second window" failure, observed as velocity regressions in
            # Phase 1 run 5).
            deck = [(a_, b_) for a_ in ia for b_ in ib]
            np.random.default_rng(contest_id).shuffle(deck)
            n_prior = await conn.fetchval(
                "SELECT count(*) FROM pair_queue WHERE contest_id=$1", contest_id
            )
            a, b = deck[n_prior % len(deck)]
        elif ancestor_clips and inc_clips:
            # no open contests (mid-breeding): serve ancestor pairs rather than
            # starving the queue — voters always have something to judge
            pair_type, contest_id = "ancestor", None
            a = int(rng.choice(inc_clips))
            b = int(rng.choice(ancestor_clips))
        else:
            break
        if rng.random() < 0.5:   # randomize left/right; clip_a renders left
            a, b = b, a
        await conn.execute(
            """INSERT INTO pair_queue (pair_token, clip_a, clip_b, contest_id, pair_type)
               VALUES ($1, $2, $3, $4, $5)""",
            secrets.token_urlsafe(16), a, b, contest_id, pair_type,
        )
        added += 1
    return added


# ---------------------------------------------------------------- bootstrap

async def bootstrap(conn, rng, stand_ckpt: str = "checkpoints/stand.pt") -> None:
    existing = await conn.fetchval("SELECT count(*) FROM policies")
    if existing:
        print("already bootstrapped")
        return
    print(f"bootstrapping gen 0 from {stand_ckpt}")
    net = load_policy(stand_ckpt)
    inc_id = await insert_policy(conn, 0, None, "incumbent")
    POLICY_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = str(POLICY_CKPT_DIR / f"p{inc_id}.pt")
    shutil.copy(stand_ckpt, ckpt)
    await conn.execute(
        "UPDATE policies SET ckpt_path=$2, promoted_at=now() WHERE id=$1", inc_id, ckpt
    )
    # skip_filter: the seed incumbent must always be showable, however wobbly —
    # "seed for coverage, not competence" (spec §7). The filter gates challengers.
    kept = await add_policy_clips(conn, inc_id, net, rng, skip_filter=True)
    print(f"  incumbent p{inc_id}: {kept} clips")

    rag_id = await insert_policy(conn, 0, None, "ragdoll")
    await conn.execute("UPDATE policies SET ckpt_path='zero' WHERE id=$1", rag_id)
    kept = await add_policy_clips(conn, rag_id, ZeroPolicy(), rng, skip_filter=True, still=True)
    print(f"  ragdoll p{rag_id}: {kept} clips")

    await set_state(conn, "generation", "0")
    await spawn_pool(conn, rng)
    await refill_queue(conn, rng)


# ---------------------------------------------------------------- cycle

async def cycle(conn, rng) -> None:
    last_seen = int(await get_state(conn, "last_vote_id", "0"))
    new_last = await conn.fetchval("SELECT coalesce(max(id), 0) FROM votes WHERE id > $1", last_seen)
    weights = await rescore_sessions(conn)
    promoted = await resolve_contests(conn, weights)
    await update_ratings(conn, weights)
    await prune_rejected_blobs(conn)
    await spawn_pool(conn, rng)
    added = await refill_queue(conn, rng)
    if new_last:
        await set_state(conn, "last_vote_id", str(new_last))
    gen = await get_state(conn, "generation", "0")
    print(f"cycle done: gen={gen} promoted={promoted} queue+={added}")


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--cycle", action="store_true")
    p.add_argument("--stand-ckpt", default="checkpoints/stand.pt")
    # default None -> fresh entropy each cycle; pass a seed only for reproducible harness runs
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if args.bootstrap:
            await bootstrap(conn, rng, args.stand_ckpt)
        if args.cycle:
            await cycle(conn, rng)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(amain())
