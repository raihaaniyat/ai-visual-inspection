# Local Development & Deployment Guide

This guide walks through local setup, testing, and deployment for the Visual Quality Inspection Platform.

## 1. Local Prerequisites

- **Python 3.10+**
- **Git**
- **Vision API Key** (e.g. from Google AI Studio)

## 2. Quick Local Setup

1. **Clone and Navigate:**
   ```powershell
   git clone <repo-url>
   cd AI-Visual-Inspector
   ```

2. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create `.env` based on `.env.example`:
   ```powershell
   Copy-Item .env.example .env
   ```
   Add your primary and fallback API keys:
   ```ini
   GEMINI_API_KEY=your_actual_primary_key_here
   GEMINI_API_KEY_FALLBACK=your_actual_fallback_key_here
   GEMINI_MODEL=gemini-3.5-flash
   ```

5. **Start Inspection Server:**
   ```powershell
   .\run_inspector.ps1
   # Or manually:
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. Open [http://localhost:8000](http://localhost:8000) in your browser.

## 3. Running Automated Tests

Run the full pytest suite:

```powershell
pytest -v
```

All 40 automated tests cover:
- Dual API key automatic failover on 429 quota exhaustion
- Planar homography perspective rectification and coordinate spaces
- Visual style overlay distinction (dashed expected vs solid detected)
- Reference difference heatmaps with reliability gating
- Operational statistics calculation and error segregation
- Ground-truth-only accuracy benchmarking
- Vercel serverless request guards (4.0 MB payload limit, sliding-window rate limiting)
- Durable database BLOB image persistence across disk wipes

## 4. Vercel Serverless Deployment

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Initialize Supabase Database Tables:
   - Open Supabase SQL Editor: `https://supabase.com/dashboard/project/<your-project-id>/sql`
   - Paste and execute `supabase_schema.sql` to initialize `profiles`, `inspections`, and `stored_images`.

3. Deploy:
   ```bash
   vercel
   ```

4. Set Environment Variables in Vercel Project Settings:
   - `GEMINI_API_KEY`: Primary API key
   - `GEMINI_API_KEY_FALLBACK`: Automatic fallback key when primary hits quota limit
   - `GEMINI_MODEL`: `gemini-3.5-flash`
   - `DATABASE_URL`: Direct PostgreSQL connection string (from Supabase Database Settings: `postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres`)
   - `SUPABASE_URL`: `https://[ref].supabase.co`
   - `SUPABASE_KEY`: Supabase publishable/anon key
   - `INSPECTION_API_KEY`: (Optional) Secret key required in `X-Inspection-Key` header for public access control

5. Production deployment:
   ```bash
   vercel --prod
   ```
