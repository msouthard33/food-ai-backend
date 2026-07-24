-- Food AI — Test harness Postgres bootstrap
-- Runs on first container init via /docker-entrypoint-initdb.d/.
-- Also idempotent for local (Homebrew) Postgres — safe to re-run via `psql -f`.
--
-- Wave 2 / Pillar 1 (1a-inference) needs pgvector for semantic food search.
-- The existing schema already uses Postgres ARRAY + JSONB + UUID, which is why
-- the SQLite test harness was torn out in favor of real Postgres.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

-- Sanity check: fail loudly if pgvector is missing so CI does not silently
-- fall back to a broken harness.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE EXCEPTION 'pgvector extension failed to install — aborting test DB bootstrap';
  END IF;
END
$$;
