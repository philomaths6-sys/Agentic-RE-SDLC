-- db/schema.sql
-- Synchronized Statutory Financial Engineering & SDLC Governance Schema

-- 1. Users & Authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL DEFAULT '',
    role VARCHAR(100) NOT NULL DEFAULT 'Principal Architect',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 2. Chat Sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);

-- 3. Chat Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);

-- 4. Session State Records (Requirements & SDLC Specification State)
CREATE TABLE IF NOT EXISTS session_state_records (
    session_id VARCHAR(64) PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
    analysis_json JSONB,
    sdlc_json JSONB,
    requirements_approved BOOLEAN DEFAULT FALSE,
    sdlc_approved BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Append-Only Immutable Audit Log (Statutory Accountability)
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    actor_email VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    session_id VARCHAR(64),
    payload JSONB,
    requirement_version VARCHAR(50) DEFAULT '1.0',
    model_version VARCHAR(100) DEFAULT 'Qwen-2.5-3B-LoRA',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);

-- 6. Stage-Gate Governance Decisions (Identity, Role & Decision Sign-off)
CREATE TABLE IF NOT EXISTS gate_decisions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    gate_name VARCHAR(50) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    reviewer_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reviewer_email VARCHAR(255) NOT NULL,
    reviewer_role VARCHAR(100) NOT NULL,
    rationale TEXT,
    model_name VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gate_decisions_session_id ON gate_decisions(session_id);
