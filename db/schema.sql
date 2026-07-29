-- Crowd-Evolved Humanoid schema (spec §3)

CREATE TABLE IF NOT EXISTS policies (
  id              BIGSERIAL PRIMARY KEY,
  generation      INT NOT NULL,
  parent_id       BIGINT REFERENCES policies(id),
  ckpt_path       TEXT NOT NULL,
  status          TEXT NOT NULL,        -- 'incumbent'|'challenger'|'retired'|'rejected'|'seed'
  rating_mean     REAL,
  rating_lb       REAL,                 -- lower confidence bound; selection uses THIS
  n_wins          INT DEFAULT 0,
  n_losses        INT DEFAULT 0,
  desc_velocity   REAL,                 -- stored, unused in v1
  desc_torso_h    REAL,                 -- stored, unused in v1
  created_at      TIMESTAMPTZ DEFAULT now(),
  promoted_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS clips (
  id              BIGSERIAL PRIMARY KEY,
  policy_id       BIGINT NOT NULL REFERENCES policies(id),
  seed            INT NOT NULL,
  window_start_s  REAL NOT NULL,        -- sampled from a longer episode
  blob_key        TEXT NOT NULL,
  leaked          BOOLEAN DEFAULT FALSE, -- degeneracy-filter reject leaked through for audit
  n_views         INT DEFAULT 0,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS contests (
  id                  BIGSERIAL PRIMARY KEY,
  generation          INT NOT NULL,
  incumbent_policy    BIGINT NOT NULL REFERENCES policies(id),
  challenger_policy   BIGINT NOT NULL REFERENCES policies(id),
  n_comparisons       INT DEFAULT 0,
  n_challenger_wins   INT DEFAULT 0,
  status              TEXT NOT NULL,    -- 'open'|'challenger_won'|'incumbent_held'
  resolved_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS votes (
  id                BIGSERIAL PRIMARY KEY,
  pair_token        TEXT UNIQUE NOT NULL,
  clip_a            BIGINT NOT NULL REFERENCES clips(id),
  clip_b            BIGINT NOT NULL REFERENCES clips(id),
  winner_clip       BIGINT NOT NULL REFERENCES clips(id),
  contest_id        BIGINT REFERENCES contests(id),   -- null for ancestor/anchor pairs
  pair_type         TEXT NOT NULL,      -- 'contest'|'ancestor'|'attention_check'
  session_id        TEXT NOT NULL,
  ip_hash           TEXT NOT NULL,
  shown_left        BIGINT NOT NULL,    -- which clip rendered on the left
  dt_ms             INT NOT NULL,
  passed_check      BOOLEAN,            -- null unless pair_type='attention_check'
  created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pair_queue (
  id            BIGSERIAL PRIMARY KEY,
  pair_token    TEXT UNIQUE NOT NULL,
  clip_a        BIGINT NOT NULL REFERENCES clips(id),
  clip_b        BIGINT NOT NULL REFERENCES clips(id),
  contest_id    BIGINT REFERENCES contests(id),
  pair_type     TEXT NOT NULL,
  consumed      BOOLEAN DEFAULT FALSE
);

-- Session trust weights, recomputed each worker cycle; read by the server for
-- the inline sequential check on POST /vote.
CREATE TABLE IF NOT EXISTS session_weights (
  session_id    TEXT PRIMARY KEY,
  weight        REAL NOT NULL DEFAULT 1.0,   -- in [0.1, 1.0]; down-weight, never ban
  updated_at    TIMESTAMPTZ DEFAULT now()
);

-- Worker bookkeeping (last seen vote id, current generation, etc.)
CREATE TABLE IF NOT EXISTS worker_state (
  key           TEXT PRIMARY KEY,
  value         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS votes_created_at_idx ON votes (created_at);
CREATE INDEX IF NOT EXISTS votes_contest_id_idx ON votes (contest_id);
CREATE INDEX IF NOT EXISTS pair_queue_unconsumed_idx ON pair_queue (consumed) WHERE consumed = FALSE;
