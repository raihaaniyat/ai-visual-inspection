-- ==============================================================================
-- Inspectra — Supabase PostgreSQL Database Schema
-- Run this script in your Supabase project SQL Editor to initialize tables.
-- ==============================================================================

-- 1. Profiles Table
CREATE TABLE IF NOT EXISTS public.profiles (
    profile_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    reference_image_id TEXT DEFAULT NULL,
    reference_corners_json TEXT DEFAULT NULL,
    checklist_json TEXT NOT NULL,
    regions_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Inspections Table
CREATE TABLE IF NOT EXISTS public.inspections (
    inspection_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES public.profiles(profile_id) ON DELETE CASCADE,
    profile_version INTEGER DEFAULT 1,
    verdict TEXT NOT NULL,
    verdict_reason TEXT NOT NULL,
    inspection_issue_summary TEXT DEFAULT NULL,
    reference_identity_ok INTEGER DEFAULT 1,
    image_quality_ok INTEGER DEFAULT 1,
    geometric_alignment_ok INTEGER DEFAULT 1,
    counts_summary_json TEXT NOT NULL DEFAULT '{}',
    findings_json TEXT NOT NULL DEFAULT '[]',
    rectified_ref_b64 TEXT DEFAULT NULL,
    rectified_test_b64 TEXT DEFAULT NULL,
    annotated_b64 TEXT DEFAULT NULL,
    opencv_diff_b64 TEXT DEFAULT NULL,
    engine TEXT NOT NULL DEFAULT 'Inspection Engine',
    model_identifier TEXT NOT NULL DEFAULT 'Inspection Engine',
    inference_latency_ms REAL NOT NULL DEFAULT 0.0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ground_truth_verdict TEXT DEFAULT NULL,
    ground_truth_notes TEXT DEFAULT NULL,
    inspection_state TEXT DEFAULT 'COMPLETED',
    failure_category TEXT DEFAULT NULL,
    request_id TEXT DEFAULT NULL
);

-- 3. Stored Images Table (Durable BLOB storage across serverless containers with SHA-256 deduplication)
CREATE TABLE IF NOT EXISTS public.stored_images (
    image_id TEXT PRIMARY KEY,
    image_bytes BYTEA NOT NULL,
    format TEXT DEFAULT 'PNG',
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotent deduplication migration & index (ensures identical images are never stored twice)
ALTER TABLE public.stored_images ADD COLUMN IF NOT EXISTS content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_stored_images_content_hash ON public.stored_images (content_hash);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.inspections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.stored_images ENABLE ROW LEVEL SECURITY;

-- 5. Public RLS Policies (Allows read & write for API service)
CREATE POLICY "Allow all operations on profiles" ON public.profiles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations on inspections" ON public.inspections FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations on stored_images" ON public.stored_images FOR ALL USING (true) WITH CHECK (true);

