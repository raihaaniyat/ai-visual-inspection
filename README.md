# Visual Quality Inspection Platform

A configurable, automated visual quality inspection platform for manufactured assemblies, electronics, and mechanical workpieces. Inspects production assemblies against approved reference baselines and operator-defined acceptance criteria using planar perspective homography, server-side multimodal vision inference, and OpenCV difference analysis.

---

## 1. Project Purpose & Scope

Visual quality control in manufacturing requires reliable verification that all components, fasteners, connectors, and sub-assemblies are correctly installed and seated without relying on brittle, fixed-template algorithms or expensive on-premise GPU clusters.

This platform provides:
- Arbitrary workpiece and component support (not hardcoded to specific parts, screw counts, or grids).
- Geometric planar perspective rectification aligning tilted camera angles to a normalized golden reference plane.
- Deterministic decision logic ensuring model hallucinations or raw confidence scores never directly decide product quality.
- Production operational statistics tracking throughput, pass/reject/review rates, failure distributions, and ground-truth-verified accuracy.
- Serverless readiness for public deployment on Vercel with strict payload limits, durable database persistence, and rate-limiting safeguards.

---

## 2. Key Features

- **Arbitrary Inspection Profiles**: Create custom profiles for any manufactured product with customizable acceptance checklists (presence, count, alignment, condition).
- **Golden Reference Baselines**: Associate each profile with an operator-approved reference photograph and workpiece corners. Replacing a reference automatically increments the profile version and invalidates prior cached results.
- **Planar Perspective Homography**: Warps angled or perspective-distorted production photographs into rectified workpiece coordinate space using OpenCV homography transforms.
- **Strict Coordinate Space Segregation**:
  - `original_reference_image`: Coordinates in raw reference photograph.
  - `original_test_image`: Coordinates in raw camera inspection photograph.
  - `rectified_workpiece`: Normalized `[0, 1]` coordinates in rectified workpiece space.
- **Clear Visual Distinction**:
  - *Dashed neutral box*: Operator-defined expected inspection region.
  - *Solid colored box*: Actually observed component detection.
  - *Green / Red / Amber*: Verified checklist item status (PASS / REJECT / REVIEW), never box existence alone.
  - Missing components display dashed expected regions without inventing solid bounding boxes.
- **Reference Difference Heatmap**: Illumination-normalized difference heatmap rendered with a continuous Turbo spectrum inside a 4% inward workpiece margin, protected by disparity reliability gating.
- **Deterministic Python Decision Gate**: Verdicts are computed strictly in Python code. Processing failures (timeouts, invalid credentials, quota exhaustion) are strictly isolated from product defects.
- **Meaningful Operational Statistics**:
  - Completed inspections and total attempt denominators.
  - PASS, REJECT, and REVIEW counts and percentages among completed runs.
  - Processing errors tracked outside product rejection rates.
  - Average inference and total processing latencies.
  - Pure SVG volume & verdict trends timeline.
  - Frequently failed checklist criteria ranked by frequency.
  - Ground-truth benchmarking (Accuracy %, False Accept Rate %, False Reject Rate %) computed strictly on independently labelled samples.
- **Vercel Serverless Ready**:
  - Single-entry ASGI handler (`api/index.py`) and `vercel.json` rewrites.
  - Lightweight CPU-only dependencies (no PyTorch, Anomalib, or Ollama in production path).
  - Durable image persistence via database BLOB storage across ephemeral container restarts.
  - Client-side image compression and server-side 4.0 MB payload validation to respect Vercel's 4.5 MB request limit.
  - Server-side sliding-window rate limiting and optional `X-Inspection-Key` authentication.

---

## 3. Architecture & Provider Usage

```text
┌────────────────────────────────────────────────────────┐
│               Operator Web Dashboard                   │
│      Overview  •  Inspection  •  Profiles  •  History  │
└───────────────────────────┬────────────────────────────┘
                            │ (HTTP / JSON / Images <= 4MB)
                            ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Application (api/index.py)          │
│   Rate Limiter • Auth Guard • Payload Validation       │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ OpenCV Geometry & Diff   │  │ Server-Side Vision Engine │
│ • Planar Homography      │  │ • Structured JSON finding │
│ • LAB Light Normalization│  │ • Component localization  │
│ • Boundary Masking       │  │ • Confidence & semantics  │
└─────────────┬────────────┘  └───────────┬──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│           Deterministic Python Decision Gate           │
│   Pass / Reject / Review Logic & Error Isolation       │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│          Durable Storage Layer (storage.py)            │
│  PostgreSQL (DATABASE_URL) or SQLite (Local/Tmp)       │
│  Profiles • Inspections • Images (BLOB) • Ground Truth │
└────────────────────────────────────────────────────────┘
```

### Real Model & Provider Details
- Inference is performed server-side via the `google-genai` / `google-generativeai` SDK configured with `GEMINI_MODEL` (default: `gemini-3.5-flash`).
- No credentials or API keys are ever bundled into client-side code, frontend templates, or browser bundles.
- If no external vision credentials are configured or provider fails, the system safely marks inspections as `REVIEW` with an explicit failure category (`Authentication/configuration error`, `Provider unavailable`, or `Quota or rate limit`).

---

## 4. Verdict Definitions

| Verdict | Definition | Action Required |
| :--- | :--- | :--- |
| **PASS** | Every mandatory checklist criterion is satisfied and component counts match expectations. | Workpiece passes inspection line. |
| **REJECT** | One or more components are confirmed missing, extra, misplaced, or defective. | Workpiece diverted for repair or scrap. |
| **REVIEW** | Optical ambiguity (glare, severe occlusion), geometric misalignment, or unexpected finding. | Operator visual confirmation required. |

> [!NOTE]
> Processing errors (network timeouts, quota limits, preparation failures) result in an `ERROR` state with a `REVIEW` verdict, but are explicitly segregated from product quality statistics.

---

## 5. Local Prerequisites & Installation

### Prerequisites
- Python 3.10 or higher.
- A Vision API key (e.g. from Google AI Studio).
- Git.

### Step-by-Step Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AI-Visual-Inspector
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and provide your API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```ini
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-3.5-flash
   ```

5. **Run the local inspection server:**
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the application:**
   - Web Dashboard: [http://localhost:8000](http://localhost:8000)
   - Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Environment Variables Reference

| Variable Name | Required | Default | Purpose |
| :--- | :---: | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** | `""` | API key for vision model inference (server-side only). |
| `GEMINI_MODEL` | No | `gemini-3.5-flash` | Multimodal vision model identifier. |
| `DATABASE_URL` | No | `""` | PostgreSQL connection string for durable cloud persistence. If omitted, falls back to SQLite. |
| `INSPECTION_API_KEY` | No | `""` | Optional secret key. When set, requests must pass `X-Inspection-Key`. |
| `RATE_LIMIT_PER_MINUTE` | No | `30` | Max requests per minute per IP to prevent quota exhaustion. |
| `MAX_UPLOAD_SIZE_BYTES` | No | `4194304` | Max image payload (4.0 MB) to keep within Vercel's 4.5 MB limit. |
| `INSPECTOR_STORAGE_DIR` | No | `./backend/data/assembly_storage` | Local directory for SQLite DB and fallback image cache. |
| `OPENCV_DIFF_THRESHOLD` | No | `40` | Intensity threshold for supporting difference analysis. |
| `OPENCV_MIN_DEFECT_AREA`| No | `150` | Minimum pixel area to consider a candidate difference contour. |

---

## 7. Inspection Workflow

1. **Select or Create Profile**: Choose an existing profile or navigate to `Profiles & References` to create a new profile with an approved golden reference photograph.
2. **Workpiece Pinning**: When uploading an inspection image, align the 4 corner pins to match the workpiece boundary.
3. **Execute Inspection**: Click `Run Inspection`. The image is automatically validated, compressed if oversized, perspective-rectified, and evaluated against the profile's checklist.
4. **Review Findings**:
   - Inspect the Annotated Overlays (solid detected component boxes vs dashed expected regions).
   - Review the continuous Turbo difference heatmap inside the workpiece boundary.
   - Examine checklist items with individual observed states and coordinates.
5. **Verify Ground Truth**: Operators can record verified ground truth (`PASS`, `REJECT`, `REVIEW`) to build independent accuracy benchmarks.

---

## 8. Storage Configuration

- **Serverless Cloud Persistence (Recommended on Vercel)**:
  Configure `DATABASE_URL` with a serverless PostgreSQL database (e.g., Neon, Supabase). All profiles, checklist criteria, inspection logs, and image BLOBs are stored durably across container lifecycles.
- **Local Fallback**:
  When `DATABASE_URL` is omitted, SQLite is automatically initialized at `INSPECTOR_STORAGE_DIR/inspector.db` (or in `/tmp` when running in a read-only serverless environment).

---

## 9. Vercel Deployment

The application is structured for direct deployment on Vercel using the official Python Serverless Functions runtime:

- **Root Directory**: Repository root (`/`).
- **Entrypoint**: `api/index.py` (exposes FastAPI `app`).
- **Routing**: `vercel.json` rewrites all traffic `/(.*)` to `/api/index.py`.
- **Static Frontend**: Embedded directly via FastAPI at `/`, eliminating multi-origin CORS friction.

### Deploying to Vercel via CLI:
```bash
# Install Vercel CLI if needed
npm install -g vercel

# Link and deploy
vercel

# Deploy to production with environment variables
vercel --prod
```

### Environment Variables to set in Vercel Project Settings:
- `GEMINI_API_KEY`: Your Vision API key.
- `GEMINI_MODEL`: `gemini-3.5-flash`
- `DATABASE_URL`: (Optional but recommended) PostgreSQL connection string.
- `INSPECTION_API_KEY`: (Optional) Secret key to restrict public inference.

---

## 10. Automated Testing

Run the automated test suite with pytest:

```bash
pytest -v
```

### Verified Test Suites (38 Passing Tests):
- `tests/test_assembly_inspection.py`: Profile versioning, reference invalidation, arbitrary component checklists, planar perspective homography, Python decision gate, and physical photo verifications.
- `tests/test_geometry_mapping_and_overlays.py`: Coordinate space data contracts, scale invariance, round-trip mapping accuracy, visual style segregation (dashed vs solid), continuous difference heatmap legend, and mapping diagnostics.
- `tests/test_inspection_failures_and_mapping.py`: Error categorization, provider unavailability handling, credential errors, and workpiece corner validation.
- `tests/test_operational_statistics.py`: Statistics calculation, error isolation from product defects, ground-truth-only accuracy metrics, durable image BLOB persistence across disk wipes, upload size limits (413), and rate limiting (429).
- `tests/test_schemas_and_guardrails.py`: Schema validation, guardrails, and contradiction handling.

---

## 11. Limitations & Physical Setup Guidelines

- **Board-Plane Parallax**: Planar homography assumes all target components lie on a single 2D plane. Tall protruding components (such as screw heads) photographed at steep camera angles will exhibit parallax shift. Use a consistent camera mount perpendicular to the workpiece.
- **Serverless Upload Limits**: Vercel limits serverless function request payloads to 4.5 MB. The frontend automatically downsizes oversized images client-side before transmission.
- **Lighting Consistency**: While illumination normalization in LAB color space compensates for moderate exposure drift, extreme glare or harsh shadows can obscure component edges and trigger `REVIEW`.

---

## 12. Attribution & License

- **License**: MIT License.
- **Attribution**: OpenCV difference analysis and baseline template concepts adapted from `Automated-Optical-Inspection` (MIT License).
