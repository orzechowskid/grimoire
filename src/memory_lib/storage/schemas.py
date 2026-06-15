# SPDX-License-Identifier: MIT

"""SQLite table schemas for the memory library."""

SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    created_at              INTEGER,
    importance              TEXT,
    age_signal              TEXT,
    tags                    TEXT,    -- JSON array
    brief                   TEXT,    -- max 500 chars (configurable)
    conflict                INTEGER DEFAULT 0,
    session_type            TEXT,    -- context / content / research
    use_count               INTEGER DEFAULT 0,
    deep_use_count          INTEGER DEFAULT 0,
    last_use_ts             INTEGER,
    implicit_score          REAL    DEFAULT 0.5,
    resolution              REAL    DEFAULT 1.0,
    intensity               REAL    DEFAULT 0.0,

    -- v1.3/v1.4 columns
    urgency                 TEXT    DEFAULT 'none',
    deadline_ts             INTEGER,
    urgency_active          INTEGER DEFAULT 0,
    urgency_expired         INTEGER DEFAULT 0,
    bare_entity             INTEGER DEFAULT 0,
    embedding_model_version TEXT    DEFAULT 'multilingual-e5-small',
    embedding               BLOB    -- float16 768d binary
);
"""

SCHEMA_ANCHORS = """
CREATE TABLE IF NOT EXISTS anchors (
    anchor_id        TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    anchor_type      TEXT NOT NULL,
    brief            TEXT NOT NULL,
    key_facts        TEXT NOT NULL DEFAULT '[]',
    flags            TEXT NOT NULL DEFAULT '{}',
    decay_level      INTEGER NOT NULL DEFAULT 0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at INTEGER NOT NULL DEFAULT 0,
    t_rel            TEXT    NOT NULL DEFAULT '{"after":[],"before":[],"caused_by":[],"during":[]}',
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    embedding        BLOB    -- float16 normalized, for Guardian/Surfacing semantic match
);
"""

SCHEMA_EXPERIENCE = """
CREATE TABLE IF NOT EXISTS experience_metrics (
    tag                   TEXT PRIMARY KEY,
    session_count         INTEGER NOT NULL DEFAULT 0,
    score_sum             REAL    NOT NULL DEFAULT 0.0,
    conflict_count        INTEGER NOT NULL DEFAULT 0,
    last_updated          INTEGER NOT NULL,
    emotion_positive      INTEGER NOT NULL DEFAULT 0,
    emotion_negative      INTEGER NOT NULL DEFAULT 0,
    emotion_intensity_sum REAL    NOT NULL DEFAULT 0.0
);
"""

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_importance ON sessions(importance);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_age ON sessions(age_signal);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_urgency ON sessions(urgency_active, deadline_ts) WHERE urgency_active = 1;",
    "CREATE INDEX IF NOT EXISTS idx_sessions_principle ON sessions(importance) WHERE importance = 'principle';",
    "CREATE INDEX IF NOT EXISTS idx_anchors_type ON anchors(anchor_type);",
    "CREATE INDEX IF NOT EXISTS idx_anchors_decay ON anchors(decay_level);",
    "CREATE INDEX IF NOT EXISTS idx_anchors_accessed ON anchors(last_accessed_at);",
    "CREATE INDEX IF NOT EXISTS idx_anchors_session ON anchors(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_anchors_flags_outcome ON anchors(json_extract(flags, '$.outcome'));",
]

INDICES_EXPERIENCE = [
    "CREATE INDEX IF NOT EXISTS idx_exp_score ON experience_metrics(score_sum);",
    "CREATE INDEX IF NOT EXISTS idx_exp_count ON experience_metrics(session_count);",
]

ALL_SCHEMAS = (
    [SCHEMA_SESSIONS, SCHEMA_ANCHORS, SCHEMA_EXPERIENCE]
    + INDICES
    + INDICES_EXPERIENCE
)
