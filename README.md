# doabackflip.ai — crowd-evolved humanoid

A humanoid in a physics simulation. Visitors see two 3-second clips side by side and
click whichever is **closer to doing a backflip**. Votes resolve contests between the
current champion policy (the *incumbent*) and mutated *challengers*; winners breed the
next generation. The creature visibly evolves while people watch.

No reward model, no gradient RLHF: votes are used purely for **selection** among
perturbation-mutants. The crowd is the fitness function, legibly — every promotion
traces to specific contests and specific votes.

## Architecture

```
batch worker ──push──> Postgres + blobs ──> web tier (FastAPI) ──> browser (three.js)
 (GPU/CPU box)                                                        │
      └──────────────── vote deltas pulled by id ────────────────────┘
```

- **Clips ship as trajectories** (joint angles + root transform, ~7KB gzipped), rendered
  client-side via a forward-kinematics port verified against MuJoCo to 1e-16
  (`sim/verify_fk.py`). No video, no egress problem.
- **Physics runs server-side only** — rigid-body sim is chaotic across machines, and two
  voters must see byte-identical clips.
- **Votes are scarce**: sequential Beta-posterior contests (3–20 comparisons), per-policy
  Wilson-lower-bound ratings, per-session trust weights (attention checks, dt floor,
  click entropy), and a precomputed pair queue with stratified clip matchups.

## Repo layout

| Path | What |
|---|---|
| `sim/` | MuJoCo env (240Hz physics, 30Hz control, EMA action filter), PPO trainer, challenger breeding (perturb + neutral-reward repair fine-tune), rollout → clip pipeline, degeneracy filter, FK reference |
| `worker/` | Batch cycle: trust rescoring, contest resolution, promotion, pool breeding (incl. ancestor "replay buffer" challengers), queue refill, blob pruning |
| `server/` | Four routes: `GET /pairs`, `POST /vote` (inline sequential check → instant resolution moments), `GET /state`, `GET /timeline` + static frontend |
| `web/` | No-build vanilla JS + three.js: one canvas / two viewports, clip-is-the-button voting, timeline page |
| `synthetic/` | Phase 1 harness: scripted voter (velocity preference, deadness rule, optional noise/bots) driving the full HTTP loop |
| `db/` | Postgres schema |
| `ops/` | Dev docker-compose (Postgres on :5433), CSL cluster train/pull scripts |

## Quickstart (dev)

```bash
python3.11 -m venv .venv
.venv/bin/pip install mujoco==3.2.7 "numpy<2" torch fastapi 'uvicorn[standard]' asyncpg httpx
docker compose -f ops/docker-compose.yml up -d          # Postgres :5433, schema auto-loads

# Phase 0: a standing seed policy (hours; or use ops/csl_train.sh on a cluster)
.venv/bin/python -m sim.train_stand --subproc --num-envs 48 --total-steps 40000000 \
    --num-steps 256 --ent-coef 0.005
.venv/bin/python -m sim.eval_stand                      # gate: 10s upright x 20 seeds

# Seed the lineage and serve
.venv/bin/python -m worker.loop --bootstrap
.venv/bin/uvicorn server.app:app --port 8080            # http://localhost:8080
while true; do .venv/bin/python -m worker.loop --cycle; sleep 20; done   # dev worker
```

## Verification gates

- **Phase 0** (`sim/eval_stand.py`): seed policy survives 10s upright on 20 seeds. ✅
- **Phase 1** (`synthetic/run_phase1.py`): the full pipeline — server, worker, queue,
  trust, contests — must drive 10 generations of monotone improvement against a clean
  scripted preference, with every contest resolving in 3–20 votes. Non-negotiable
  before human voters: if the loop can't converge on a clean signal, it can't converge
  on a noisy one. Runs entirely over the real HTTP API.

Phase 1 earned its keep: it caught six real defects before any human voted, including
a fine-tune reward that dragged every mutant back toward standing and a trust-scoring
death spiral triggered by attention-check ragdolls out-running the stander.

## Training notes

- Big PPO runs want big batches: the humanoid plateaus hard below ~50k samples/batch
  (20M steps of 6k-batch PPO fell at 1.3s forever; 49k-batch + entropy bonus stood up
  by 15M).
- `ops/csl_train.sh <host>` rsyncs the repo to a cluster sandbox, resumes from the
  shipped checkpoint, and runs detached. `ops/csl_pull.sh` fetches results. See script
  comments for shared-account etiquette and driver/CUDA pinning.

## Design doc

The full build spec (data model, API shapes, queue composition, launch phases, known
weak points) lives in the project planning notes; `db/schema.sql` and the module
docstrings mirror its decisions section-by-section (spec § references throughout).
