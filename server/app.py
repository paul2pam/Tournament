"""Web tier: four routes (spec §4) + static frontend.

POST /vote runs the same sequential rule as the worker inline (weights cached by the
last worker cycle) so the response can carry the resolution moment immediately and no
further votes are wasted on a decided contest. The worker remains authoritative for
promotion, retirement, and breeding the next pool.

Run:  .venv/bin/uvicorn server.app:app --port 8080
"""
import hashlib
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common import blobs
from common.config import CONTEST_MAX_COMPARISONS, IP_HASH_SALT, REPO_ROOT
from server import db
from worker.contests import resolve

app = FastAPI(title="doabackflip.ai")
app.add_middleware(GZipMiddleware, minimum_size=1024)   # trajectories compress ~10x


@app.on_event("shutdown")
async def _shutdown():
    await db.close()


def _traj(blob_key: str):
    return blobs.get_json(blob_key)


# ------------------------------------------------------------------ GET /pairs

@app.get("/pairs")
async def get_pairs(n: int = 5):
    n = max(1, min(n, 10))
    p = await db.pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """WITH picked AS (
                     SELECT pq.id FROM pair_queue pq
                     LEFT JOIN contests ct ON ct.id = pq.contest_id
                     WHERE NOT pq.consumed AND (pq.contest_id IS NULL OR ct.status = 'open')
                     LIMIT $1
                     FOR UPDATE OF pq SKIP LOCKED
                   )
                   UPDATE pair_queue pq SET consumed = TRUE
                   FROM picked WHERE pq.id = picked.id
                   RETURNING pq.pair_token, pq.clip_a, pq.clip_b""",
                n,
            )
            pairs = []
            for r in rows:
                ka, kb = await conn.fetchval(
                    "SELECT blob_key FROM clips WHERE id=$1", r["clip_a"]
                ), await conn.fetchval("SELECT blob_key FROM clips WHERE id=$1", r["clip_b"])
                if not ka or not kb:
                    continue   # blob pruned between queueing and serving; skip
                try:
                    trajs = [_traj(ka), _traj(kb)]
                except FileNotFoundError:
                    continue   # pruned between the SELECT above and the disk read
                await conn.execute(
                    "UPDATE clips SET n_views = n_views + 1 WHERE id = ANY($1::bigint[])",
                    [r["clip_a"], r["clip_b"]],
                )
                pairs.append(
                    {
                        "pair_token": r["pair_token"],
                        "clips": [
                            {"id": r["clip_a"], "trajectory": trajs[0]},
                            {"id": r["clip_b"], "trajectory": trajs[1]},
                        ],
                    }
                )
    return {"pairs": pairs}


# ------------------------------------------------------------------ POST /vote

class VoteIn(BaseModel):
    pair_token: str
    winner_clip: int
    dt_ms: int
    session_id: str


@app.post("/vote")
async def post_vote(vote: VoteIn, request: Request):
    p = await db.pool()
    ip = request.client.host if request.client else "unknown"
    ip_hash = hashlib.sha256((IP_HASH_SALT + ip).encode()).hexdigest()[:32]

    async with p.acquire() as conn:
        pq = await conn.fetchrow(
            "SELECT * FROM pair_queue WHERE pair_token=$1", vote.pair_token
        )
        if pq is None:
            raise HTTPException(404, "unknown pair_token")
        if vote.winner_clip not in (pq["clip_a"], pq["clip_b"]):
            raise HTTPException(400, "winner_clip not in pair")

        passed_check = None
        if pq["pair_type"] == "attention_check":
            winner_status = await conn.fetchval(
                """SELECT p.status FROM clips c JOIN policies p ON p.id=c.policy_id
                   WHERE c.id=$1""",
                vote.winner_clip,
            )
            passed_check = winner_status != "ragdoll"

        try:
            await conn.execute(
                """INSERT INTO votes (pair_token, clip_a, clip_b, winner_clip, contest_id,
                                      pair_type, session_id, ip_hash, shown_left, dt_ms,
                                      passed_check)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                vote.pair_token, pq["clip_a"], pq["clip_b"], vote.winner_clip,
                pq["contest_id"], pq["pair_type"], vote.session_id, ip_hash,
                pq["clip_a"],   # clip_a is always rendered on the left
                vote.dt_ms, passed_check,
            )
        except Exception:
            raise HTTPException(409, "already voted on this pair")

        # agreement: other votes on the same unordered clip pair
        agree = await conn.fetchrow(
            """SELECT count(*) AS n,
                      count(*) FILTER (WHERE winner_clip=$3) AS same
               FROM votes
               WHERE ((clip_a=$1 AND clip_b=$2) OR (clip_a=$2 AND clip_b=$1))
                 AND pair_token <> $4""",
            pq["clip_a"], pq["clip_b"], vote.winner_clip, vote.pair_token,
        )
        agreement_pct = round(100 * agree["same"] / agree["n"]) if agree["n"] else None

        contest_progress = None
        resolution = None
        if pq["contest_id"] is not None:
            ct = await conn.fetchrow("SELECT * FROM contests WHERE id=$1", pq["contest_id"])
            tally = await conn.fetch(
                """SELECT v.session_id, c.policy_id AS wp
                   FROM votes v JOIN clips c ON c.id=v.winner_clip
                   WHERE v.contest_id=$1 AND v.pair_type='contest'""",
                pq["contest_id"],
            )
            wrows = await conn.fetch("SELECT session_id, weight FROM session_weights")
            weights = {r["session_id"]: r["weight"] for r in wrows}
            w_total = sum(weights.get(r["session_id"], 1.0) for r in tally)
            w_wins = sum(
                weights.get(r["session_id"], 1.0)
                for r in tally
                if r["wp"] == ct["challenger_policy"]
            )
            n = len(tally)
            contest_progress = {"n": n, "target": CONTEST_MAX_COMPARISONS}

            if ct["status"] == "open":
                outcome = resolve(w_wins, w_total, n)
                if outcome:
                    await conn.execute(
                        """UPDATE contests SET status=$2, resolved_at=now(),
                               n_comparisons=$3, n_challenger_wins=$4
                           WHERE id=$1 AND status='open'""",
                        ct["id"], outcome, n, int(round(w_wins)),
                    )
                    # old/new relative to this pair: which shown clip belongs to whom
                    a_policy = await conn.fetchval(
                        "SELECT policy_id FROM clips WHERE id=$1", pq["clip_a"]
                    )
                    inc_clip = (
                        pq["clip_a"] if a_policy == ct["incumbent_policy"] else pq["clip_b"]
                    )
                    chal_clip = pq["clip_b"] if inc_clip == pq["clip_a"] else pq["clip_a"]
                    resolution = {
                        "outcome": outcome,
                        "old_clip": inc_clip,
                        "new_clip": chal_clip if outcome == "challenger_won" else None,
                    }

    return {
        "agreement_pct": agreement_pct,
        "contest_progress": contest_progress,
        "resolution": resolution,
    }


# ------------------------------------------------------------------ GET /state

@app.get("/state")
async def get_state():
    p = await db.pool()
    async with p.acquire() as conn:
        gen = await conn.fetchval("SELECT value FROM worker_state WHERE key='generation'")
        total_votes = await conn.fetchval("SELECT count(*) FROM votes")
        inc = await conn.fetchrow("SELECT id, generation FROM policies WHERE status='incumbent'")
        clip = None
        if inc:
            c = await conn.fetchrow(
                """SELECT id, blob_key FROM clips
                   WHERE policy_id=$1 AND blob_key <> '' AND NOT leaked
                   ORDER BY id LIMIT 1""",
                inc["id"],
            )
            if c:
                clip = {"id": c["id"], "trajectory": _traj(c["blob_key"])}
    return {
        "generation": int(gen) if gen else 0,
        "total_votes": total_votes,
        "incumbent": {"policy_id": inc["id"] if inc else None, "clip": clip},
    }


# ---------------------------------------------------------------- GET /timeline

@app.get("/timeline")
async def get_timeline():
    p = await db.pool()
    async with p.acquire() as conn:
        rows = await conn.fetch(
            # status <> 'rejected' rather than IN (incumbent, retired): an ancestor
            # mid-comeback is briefly status='challenger' and must stay listed
            """SELECT id, generation, promoted_at, status FROM policies
               WHERE promoted_at IS NOT NULL AND status <> 'rejected'
               ORDER BY promoted_at"""
        )
        out = []
        for r in rows:
            c = await conn.fetchrow(
                """SELECT id, blob_key FROM clips
                   WHERE policy_id=$1 AND blob_key <> '' AND NOT leaked
                   ORDER BY id LIMIT 1""",
                r["id"],
            )
            out.append(
                {
                    "policy_id": r["id"],
                    "generation": r["generation"],
                    "promoted_at": r["promoted_at"].isoformat(),
                    "current": r["status"] == "incumbent",
                    "clip": {"id": c["id"], "trajectory": _traj(c["blob_key"])} if c else None,
                }
            )
    return {"timeline": out, "as_of": time.time()}


# static frontend last so API routes win
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")
