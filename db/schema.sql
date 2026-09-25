-- db/schema.sql

-- Sessions table
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Requirements table
CREATE TABLE requirements (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    category TEXT[],
    source VARCHAR(255),
    priority VARCHAR(20),
    dependencies TEXT[],
    acceptance_criteria TEXT,
    applicable_regulations TEXT[],
    confidence_score REAL,
    approval_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- SDLC Recommendations table
CREATE TABLE sdlc_recommendations (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    model_name VARCHAR(255),
    confidence_pct REAL,
    justification TEXT,
    workflow_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Approvals table
CREATE TABLE approvals (
    id SERIAL PRIMARY KEY,
    item_type VARCHAR(50),
    item_id INTEGER,
    action VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    actor VARCHAR(255)
);

-- Audit log table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    actor VARCHAR(255),
    action TEXT,
    payload JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
