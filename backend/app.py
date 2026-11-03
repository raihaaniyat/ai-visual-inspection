"""
FastAPI application for AI-Visual-Inspector.

Endpoints:
  ASSEMBLY QUALITY INSPECTION (Gemini Vision + SQLite):
    POST /profiles                            - Create new inspection profile with reference image
    GET  /profiles                            - List all saved profiles
    GET  /profiles/{profile_id}               - Get specific profile configuration
    DELETE /profiles/{profile_id}            - Delete profile
    GET  /profiles/{profile_id}/reference-image - Download approved reference image
    POST /profiles/{profile_id}/inspect       - Run inspection on uploaded/captured image
    GET  /inspections                         - List inspection history
    GET  /inspections/{inspection_id}         - Get single inspection details
    POST /inspections/{inspection_id}/ground-truth - Record operator ground truth for accuracy testing
    GET  /reports/accuracy                    - Accuracy & confusion matrix from ground truth
    GET  /reports/export                      - Export inspections as JSON or CSV
    POST /demo/generate-samples               - Generate synthetic test images for physical demo

  LEGACY LOCAL ANOMALY DETECTION (Optional, MVTecAD):
    POST /inspect                             - Anomaly detection via PatchCore / EfficientAd
    GET  /metrics                             - Anomaly benchmarks

Run from project root:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

logger = logging.getLogger("backend.app")

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

from backend.config import settings
from backend.default_profiles import (
    PERMANENT_PROFILE_IDS,
    _load_library_image,
    create_synthetic_demo_board_image,
    seed_general_inspection_profile,
    seed_permanent_reference_library,
    seed_physical_experiment_profile,
)
from backend.geometry import compute_mapping_diagnostics, rectify_workpiece
from backend.gemini_inspector import run_workpiece_inspection, validate_workpiece_corners
from backend.schemas import (
    AccuracyReportResponse,
    AssemblyInspectionResponse,
    CategoryMetrics,
    CheckItem,
    ExpectedRegion,
    GroundTruthUpdateRequest,
    InspectionResult,
    MappingDiagnosticResponse,
    MetricsResponse,
    OperationalStatisticsResponse,
    ProfileResponse,
    WorkpieceCorners,
)
from backend.storage import (
    clear_all_inspections,
    create_or_save_profile,
    delete_all_references,
    delete_profile,
    get_accuracy_report,
    get_operational_statistics,
    get_profile,
    init_db,
    list_inspections,
    list_profiles,
    load_image_file,
    remove_profile_reference,
    save_inspection,
    update_ground_truth,
    update_profile_reference,
)

METRICS_PATH = Path("training/results/metrics.json")
ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = (
    ROOT_DIR / "public" / "index.html"
    if (ROOT_DIR / "public" / "index.html").exists()
    else ROOT_DIR / "frontend" / "index.html"
)
ASSETS_DIR = (
    ROOT_DIR / "public" / "assets"
    if (ROOT_DIR / "public" / "assets").exists()
    else ROOT_DIR / "frontend" / "assets"
)

# In-memory sliding window rate limiter
_rate_limit_records: dict[str, list[float]] = defaultdict(list)


def check_rate_limit_and_auth(request: Request) -> None:
    """Protects serverless endpoints against unauthorized abuse and rate limit exhaustion."""
    if settings.inspection_api_key:
        client_key = request.headers.get("X-Inspection-Key")
        if not client_key or client_key != settings.inspection_api_key:
            raise HTTPException(401, "Invalid or missing X-Inspection-Key authentication header.")

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    timestamps = [t for t in _rate_limit_records[client_ip] if t > window_start]
    if len(timestamps) >= settings.rate_limit_per_minute:
        raise HTTPException(429, f"Rate limit exceeded (max {settings.rate_limit_per_minute} requests/min). Please try again shortly.")
    timestamps.append(now)
    _rate_limit_records[client_ip] = timestamps


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        seed_permanent_reference_library()
        seed_physical_experiment_profile()
        seed_general_inspection_profile()
    except Exception as exc:
        logger.info("Profile initialization notice: %s", exc)
    yield


app = FastAPI(
    title="Visual Quality Inspection Platform",
    description="Configurable Visual Quality Control & Component Verification Dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VisionInspect AI</h1><p>API docs at <a href='/docs'>/docs</a></p>")


@app.get("/assets/{filename}")
def get_asset(filename: str) -> Response:
    """Serves static assets with long-lived immutable caching."""
    path = ASSETS_DIR / filename
    if not path.exists():
        path = ROOT_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Asset '{filename}' not found.")
    media_type = (
        "image/png"
        if filename.endswith(".png")
        else "image/svg+xml"
        if filename.endswith(".svg")
        else "application/octet-stream"
    )
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/inspectra-dark-mode.png")
def get_dark_logo() -> Response:
    return get_asset("inspectra-dark-mode.png")


@app.get("/inspectra-light-mode.png")
def get_light_logo() -> Response:
    return get_asset("inspectra-light-mode.png")


@app.get("/health")
def health() -> dict:
    gemini_key_configured = bool(settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")

    ollama_reachable = False
    ollama_error = None
    if settings.vlm_enabled:
        try:
            import ollama
            ollama.Client(host=settings.ollama_host).list()
            ollama_reachable = True
        except Exception as exc:
            ollama_reachable = False
            ollama_error = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok",
        "engine": "Inspection Engine",
        "engine_ready": gemini_key_configured,
        "storage_dir": str(settings.storage_dir_path),
        "legacy_local": {
            "ad_backend": settings.ad_backend,
            "vlm_enabled": settings.vlm_enabled,
            "ollama_host": settings.ollama_host,
            "ollama_reachable": ollama_reachable,
            "ollama_error": ollama_error,
        },
    }


# ==============================================================================
# Inspection Engine Health Verification & Diagnostics Endpoints
# ==============================================================================

class EngineHealthRequest(BaseModel):
    model: str | None = None


@app.post("/api/engine/health-check")
@app.post("/api/gemini/test")
def check_engine_health(_req: EngineHealthRequest | None = None) -> dict:
    """Backend verification for the visual inspection engine.
    
    Credentials and provider-specific error payloads are kept strictly on the backend.
    Diagnostic details are recorded in backend logs with secrets redacted.
    """
    key = settings.gemini_api_key
    model = settings.gemini_model

    if not key or key == "your_gemini_api_key_here":
        logger.warning("Visual inspection engine health check failed: backend API key is unconfigured.")
        return {
            "success": False,
            "status": "unavailable",
            "message": "Inspection unavailable. Please retry.",
            "latency_ms": 0.0,
        }

    start = time.perf_counter()
    try:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            client.models.generate_content(
                model=model,
                contents="Ping test for quality inspection engine: respond with PONG",
            )
        except ImportError:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=key)
            gmodel = legacy_genai.GenerativeModel(model)
            gmodel.generate_content("Ping test for quality inspection engine: respond with PONG")

        latency = (time.perf_counter() - start) * 1000.0
        logger.info("Visual inspection engine ping successful (model: %s, latency: %.1fms)", model, latency)
        return {
            "success": True,
            "status": "ready",
            "message": "Inspection Engine is ready.",
            "latency_ms": round(latency, 1),
        }
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000.0
        # Redact secrets and log to backend diagnostic logger only
        logger.error("Visual inspection engine health check error: %s (model: %s)", exc, model)
        return {
            "success": False,
            "status": "unavailable",
            "message": "Inspection unavailable. Please retry.",
            "latency_ms": round(latency, 1),
        }


@app.get("/api/system/diagnostics")
def system_diagnostics() -> dict:
    """Returns system diagnostic status, storage health, and database metrics without exposing provider keys."""
    import platform
    import shutil

    k = settings.gemini_api_key
    is_set = bool(k and k != "your_gemini_api_key_here")

    disk_c = shutil.disk_usage("C:") if Path("C:").exists() else None
    disk_d = shutil.disk_usage("D:") if Path("D:").exists() else None

    c_free = round(disk_c.free / (1024**3), 2) if disk_c else None
    d_free = round(disk_d.free / (1024**3), 2) if disk_d else None

    profiles = list_profiles()
    acc = get_accuracy_report()

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "engine_type": "Inspection Engine",
        "engine_status": "ready" if is_set else "unavailable",
        "storage_dir": str(settings.storage_dir_path),
        "disk_c_free_gb": c_free,
        "disk_d_free_gb": d_free,
        "profiles_count": len(profiles),
        "inspections_count": acc.total_inspections,
        "ground_truth_count": acc.ground_truth_count,
        "database_online": True,
    }


# ==============================================================================
# Assembly Profile Endpoints
# ==============================================================================

@app.get("/profiles", response_model=list[ProfileResponse])
def get_profiles() -> list[ProfileResponse]:
    return list_profiles()


@app.post("/profiles", response_model=ProfileResponse)
async def create_profile(
    name: str = Form(...),
    description: str = Form(""),
    checklist_json: str = Form("[]"),
    regions_json: str = Form("[]"),
    corners_json: str | None = Form(None),
    reference_file: UploadFile | None = File(None),
    reference_image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
) -> ProfileResponse:
    try:
        checklist_data = json.loads(checklist_json)
        checklist = [CheckItem(**item) for item in checklist_data]
    except Exception as exc:
        raise HTTPException(400, f"Invalid checklist JSON: {exc}")

    try:
        regions_data = json.loads(regions_json)
        regions = [ExpectedRegion(**r) for r in regions_data]
    except Exception as exc:
        raise HTTPException(400, f"Invalid regions JSON: {exc}")

    ref_corners = None
    if corners_json:
        try:
            ref_corners = WorkpieceCorners(**json.loads(corners_json))
        except Exception as exc:
            raise HTTPException(400, f"Invalid workpiece corners JSON: {exc}")

    # Robust upload field resolution: accepts 'reference_image', 'reference_file', or 'file'
    ref_upload = reference_image if reference_image is not None else (reference_file if reference_file is not None else file)

    ref_image = None
    if ref_upload is not None:
        contents = await ref_upload.read()
        if contents:
            if len(contents) > settings.max_upload_size_bytes:
                max_mb = settings.max_upload_size_bytes // (1024 * 1024)
                raise HTTPException(
                    413,
                    f"Reference image ({len(contents) // 1024} KB) exceeds the maximum upload limit of {max_mb} MB. Please compress or resize before upload."
                )
            try:
                ref_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
            except Exception as exc:
                raise HTTPException(400, f"Invalid reference image file: {exc}")

    return create_or_save_profile(
        name=name,
        description=description,
        reference_image=ref_image,
        reference_corners=ref_corners,
        checklist=checklist,
        regions=regions,
    )


@app.post("/profiles/{profile_id}/reference", response_model=ProfileResponse)
async def upload_profile_reference(
    profile_id: str,
    reference_file: UploadFile | None = File(None),
    reference_image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    corners_json: str | None = Form(None),
) -> ProfileResponse:
    """
    Requirement 1: Replacing a reference must invalidate its old regions
    and any cached inspection results. Associate each reference, checklist
    and set of regions with a profile version.
    """
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")

    ref_upload = reference_image if reference_image is not None else (reference_file if reference_file is not None else file)
    if ref_upload is None:
        raise HTTPException(400, "Reference image file is required.")

    contents = await ref_upload.read()
    if not contents:
        raise HTTPException(400, "Reference image file is empty.")

    if len(contents) > settings.max_upload_size_bytes:
        max_mb = settings.max_upload_size_bytes // (1024 * 1024)
        raise HTTPException(
            413,
            f"Reference image ({len(contents) // 1024} KB) exceeds the maximum upload limit of {max_mb} MB."
        )

    try:
        ref_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid reference image file: {exc}")

    corners = None
    if corners_json:
        try:
            corners = WorkpieceCorners(**json.loads(corners_json))
        except Exception as exc:
            raise HTTPException(400, f"Invalid workpiece corners JSON: {exc}")

    return update_profile_reference(
        profile_id=profile_id,
        reference_image=ref_image,
        reference_corners=corners,
    )


@app.delete("/profiles/{profile_id}/reference", response_model=ProfileResponse)
def delete_profile_reference(profile_id: str) -> ProfileResponse:
    """
    Deletes the approved reference image and associated corners from a profile,
    clearing regions and resetting the reference status to unconfigured.
    """
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")
    return remove_profile_reference(profile_id)


@app.post("/profiles/{profile_id}/config", response_model=ProfileResponse)
def update_profile_config(
    profile_id: str,
    checklist_json: str = Form(...),
    regions_json: str = Form(...),
) -> ProfileResponse:
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")

    try:
        checklist = [CheckItem(**item) for item in json.loads(checklist_json)]
    except Exception as exc:
        raise HTTPException(400, f"Invalid checklist JSON: {exc}")

    try:
        regions = [ExpectedRegion(**r) for r in json.loads(regions_json)]
    except Exception as exc:
        raise HTTPException(400, f"Invalid regions JSON: {exc}")

    return create_or_save_profile(
        name=profile.name,
        description=profile.description,
        checklist=checklist,
        regions=regions,
        profile_id=profile_id,
    )


@app.post("/geometry/rectify-preview")
async def rectify_preview(
    file: UploadFile = File(...),
    corners_json: str | None = Form(None),
) -> dict:
    """
    Requirement 3: Rectify planar workpiece to common board coordinate system
    and return preview. Validates that corners enclose the intended assembly.
    """
    contents = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image file: {exc}")

    corners = None
    if corners_json:
        try:
            corners = WorkpieceCorners(**json.loads(corners_json))
        except Exception as exc:
            raise HTTPException(400, f"Invalid corners JSON: {exc}")

    corners_valid, corner_warning = validate_workpiece_corners(corners)

    try:
        rectified_img, _, _ = rectify_workpiece(img, corners)
        buf = io.BytesIO()
        rectified_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "rectified_image_png_base64": b64,
            "width": rectified_img.width,
            "height": rectified_img.height,
            "corners_valid": corners_valid,
            "warning": corner_warning,
            "notice": "This mapping assumes a planar board; it does not handle arbitrary 3D machinery.",
        }
    except Exception as exc:
        raise HTTPException(400, f"Perspective transformation failed: {exc}")


@app.post("/geometry/mapping-diagnostic", response_model=MappingDiagnosticResponse)
async def geometry_mapping_diagnostic(
    file: UploadFile | None = File(None),
    corners_json: str | None = Form(None),
    profile_id: str | None = Form(None),
) -> MappingDiagnosticResponse:
    """
    Requirement 2: Verifies geometric coordinate mapping without AI.
    Tests:
    - 4 workpiece corners round-trip mapping.
    - Center and edge point mappings.
    - Scale invariance and display sizes.
    - Euclidean pixel error and normalized error.
    - Returns side-by-side diagnostic visualization overlay and board-plane notice.
    """
    img = None
    corners = None

    if file is not None and file.filename:
        contents = await file.read()
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
        except Exception as exc:
            raise HTTPException(400, f"Invalid image file: {exc}")

    if corners_json:
        try:
            corners = WorkpieceCorners(**json.loads(corners_json))
        except Exception as exc:
            raise HTTPException(400, f"Invalid corners JSON: {exc}")

    if img is None and profile_id:
        profile = get_profile(profile_id)
        if profile and profile.reference_image_id:
            img = load_image_file(profile.reference_image_id)
            if corners is None:
                corners = profile.reference_corners

    if img is None:
        ref_photo_path = Path("data/physical_samples/90645232-ae17-491d-a5f4-b9408128cead.jpg")
        if ref_photo_path.exists():
            img = ImageOps.exif_transpose(Image.open(ref_photo_path)).convert("RGB")
            if corners is None:
                corners = WorkpieceCorners(
                    top_left=[0.24, 0.17],
                    top_right=[0.74, 0.16],
                    bottom_right=[0.75, 0.81],
                    bottom_left=[0.21, 0.81],
                )
        else:
            img = Image.new("RGB", (800, 600), color=(180, 195, 210))

    try:
        return compute_mapping_diagnostics(image=img, corners=corners)
    except Exception as exc:
        raise HTTPException(500, f"Geometry mapping diagnostic failed: {exc}")


@app.get("/profiles/{profile_id}/mapping-diagnostic", response_model=MappingDiagnosticResponse)
def get_profile_mapping_diagnostic(profile_id: str) -> MappingDiagnosticResponse:
    """Runs geometry mapping diagnostic on a saved profile's reference image and corners."""
    profile = get_profile(profile_id)
    if not profile or not profile.reference_image_id:
        raise HTTPException(404, f"Profile '{profile_id}' has no reference image.")

    ref_img = load_image_file(profile.reference_image_id)
    if not ref_img:
        raise HTTPException(404, "Reference image file missing from storage.")

    return compute_mapping_diagnostics(image=ref_img, corners=profile.reference_corners)


@app.get("/profiles/{profile_id}", response_model=ProfileResponse)
def get_single_profile(profile_id: str) -> ProfileResponse:
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")
    return profile


@app.delete("/profiles/references/all", response_model=dict)
@app.post("/profiles/references/delete-all", response_model=dict)
@app.delete("/profiles/reference/all", response_model=dict)
def delete_all_profile_references() -> dict:
    """
    Deletes all approved reference images and regions across all profiles,
    and purges all stored inspection audit history.
    """
    return delete_all_references()


@app.delete("/profiles/{profile_id}")
def remove_profile(profile_id: str) -> dict:
    ok = delete_profile(profile_id)
    if not ok:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")
    return {"status": "deleted", "profile_id": profile_id}


@app.get("/profiles/{profile_id}/reference-image")
def get_profile_reference_image(profile_id: str):
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")

    img = None
    if profile.reference_image_id:
        img = load_image_file(profile.reference_image_id)

    # Automatic durable fallback for permanent library profiles
    if not img and profile.profile_id in PERMANENT_PROFILE_IDS:
        img_name_map = {
            "prof_speaker": "speaker.png",
            "prof_transistor": "transistor.png",
            "prof_screws_plywood": "screws_plywood.jpg",
            "prof_cable_assembly": "cable_assembly.png",
            "prof_zipper": "zipper.png",
            "prof_physical_6screw": "screws_plywood.jpg",
            "prof_general_inspection": "screws_plywood.jpg",
        }
        filename = img_name_map.get(profile.profile_id)
        if filename:
            img = _load_library_image(filename)

    if not img:
        raise HTTPException(404, "No approved reference image has been uploaded for this profile yet.")

    disp_img = img.copy()
    if max(disp_img.size) > 1600:
        disp_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    disp_img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    img_bytes = buf.getvalue()
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Content-Length": str(len(img_bytes)),
        },
    )


# ==============================================================================
# Assembly Inspection Endpoints
# ==============================================================================

@app.post("/profiles/{profile_id}/inspect", response_model=AssemblyInspectionResponse)
async def run_profile_inspection(
    profile_id: str,
    request: Request,
    file: UploadFile = File(...),
    test_corners_json: str | None = Form(None),
    ground_truth_verdict: str | None = Form(None),
    ground_truth_notes: str | None = Form(None),
) -> AssemblyInspectionResponse:
    # Rate limit and access control verification
    check_rate_limit_and_auth(request)

    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")

    ref_img = load_image_file(profile.reference_image_id) if profile.reference_image_id else None
    if not ref_img and profile.profile_id in PERMANENT_PROFILE_IDS:
        img_name_map = {
            "prof_speaker": "speaker.png",
            "prof_transistor": "transistor.png",
            "prof_screws_plywood": "screws_plywood.jpg",
            "prof_cable_assembly": "cable_assembly.png",
            "prof_zipper": "zipper.png",
            "prof_physical_6screw": "screws_plywood.jpg",
            "prof_general_inspection": "screws_plywood.jpg",
        }
        filename = img_name_map.get(profile.profile_id)
        if filename:
            ref_img = _load_library_image(filename)

    if not ref_img:
        raise HTTPException(400, "Add an approved reference first.")
    if not ref_img:
        raise HTTPException(400, "Add an approved reference first.")

    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            413,
            f"Image file size ({len(contents) // 1024} KB) exceeds the serverless limit of "
            f"{settings.max_upload_size_bytes // 1024 // 1024} MB. Please compress or resize before upload.",
        )

    try:
        test_img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid inspection image file: {exc}")

    test_corners = None
    if test_corners_json:
        try:
            test_corners = WorkpieceCorners(**json.loads(test_corners_json))
        except Exception as exc:
            raise HTTPException(400, f"Invalid test workpiece corners JSON: {exc}")

    # Run Planar Workpiece Inspection
    result = run_workpiece_inspection(
        reference_image=ref_img,
        test_image=test_img,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        checklist=profile.checklist,
        regions=profile.regions,
        reference_corners=profile.reference_corners,
        test_corners=test_corners,
    )

    # Persist inspection to database
    saved_res = save_inspection(
        profile_id=result.profile_id,
        profile_version=result.profile_version,
        verdict=result.verdict,
        verdict_reason=result.verdict_reason,
        findings=result.findings,
        counts_summary=result.counts_summary,
        engine=result.engine,
        model_identifier=result.model_identifier,
        inference_latency_ms=result.inference_latency_ms,
        total_latency_ms=result.total_latency_ms,
        inspection_state=result.inspection_state,
        failure_category=result.failure_category,
        request_id=result.request_id,
        inspection_issue_summary=result.inspection_issue_summary,
        reference_identity_ok=result.reference_identity_ok,
        image_quality_ok=result.image_quality_ok,
        geometric_alignment_ok=result.geometric_alignment_ok,
        rectified_ref_png_base64=result.rectified_ref_png_base64,
        rectified_test_png_base64=result.rectified_test_png_base64,
        annotated_image_png_base64=result.annotated_image_png_base64,
        opencv_diff_png_base64=result.opencv_diff_png_base64,
        ground_truth_verdict=ground_truth_verdict,
        ground_truth_notes=ground_truth_notes,
    )

    return saved_res


@app.post("/profiles/{profile_id}/propose-inventory", response_model=list[ExpectedRegion])
def propose_profile_inventory(profile_id: str, request: Request) -> list[ExpectedRegion]:
    """
    Requirement 4: Proposes an AI-generated candidate component inventory from the approved
    reference image for operator review and approval.
    """
    check_rate_limit_and_auth(request)
    profile = get_profile(profile_id)
    if not profile or not profile.reference_image_id:
        raise HTTPException(400, "Add an approved reference first.")

    ref_img = load_image_file(profile.reference_image_id)
    if not ref_img:
        raise HTTPException(400, "Reference image missing from storage.")

    try:
        from backend.gemini_inspector import propose_reference_inventory
        return propose_reference_inventory(
            reference_image=ref_img,
            reference_corners=profile.reference_corners,
        )
    except Exception as exc:
        logger.error("Failed to propose reference inventory for %s: %s", profile_id, exc)
        raise HTTPException(500, f"Inventory proposal failed: {exc}")


@app.put("/profiles/{profile_id}/approve-inventory", response_model=ProfileResponse)
def approve_profile_inventory(
    profile_id: str,
    regions_json: str = Form(...),
) -> ProfileResponse:
    """
    Requirement 4: Saves operator-reviewed and approved components to the profile,
    replacing old regions and incrementing the profile version.
    """
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, f"Profile '{profile_id}' not found.")

    try:
        approved_regions = [ExpectedRegion(**r) for r in json.loads(regions_json)]
    except Exception as exc:
        raise HTTPException(400, f"Invalid regions JSON: {exc}")

    new_checklist = list(profile.checklist)
    existing_region_ids = {item.region_id for item in new_checklist if item.region_id}
    for r in approved_regions:
        if r.region_id not in existing_region_ids:
            new_checklist.append(
                CheckItem(
                    check_id=f"chk_{r.region_id}",
                    check_type="presence",
                    region_id=r.region_id,
                    description=f"{r.label} present and seated",
                    expected_requirement=f"Verify {r.label} is installed in its designated position.",
                    required=True,
                )
            )

    ref_img = load_image_file(profile.reference_image_id) if profile.reference_image_id else None
    return create_or_save_profile(
        name=profile.name,
        description=profile.description,
        reference_image=ref_img,
        reference_corners=profile.reference_corners,
        checklist=new_checklist,
        regions=approved_regions,
        profile_id=profile_id,
    )


@app.post("/api/test/simulate-inspection-error", response_model=AssemblyInspectionResponse)
def simulate_inspection_error(
    category: str = Query("Provider unavailable"),
    profile_id: str = Query("prof_general_inspection"),
) -> AssemblyInspectionResponse:
    """
    Test endpoint for Requirement 6 to verify reporting for each failure category:
    - Authentication/configuration error
    - Quota or rate limit
    - Provider unavailable
    - Request timeout
    - Invalid inspection response
    - Image preparation failure
    """
    from backend.gemini_inspector import build_error_inspection_response
    profile = get_profile(profile_id)
    checklist = profile.checklist if profile else []
    return build_error_inspection_response(
        profile_id=profile_id,
        profile_version=profile.version if profile else 1,
        checklist=checklist,
        request_id="req_test_sim",
        stage="provider_call" if "Provider" in category or "Quota" in category else "response_parsing",
        failure_category=category,
        latency_ms=120.0,
    )


@app.get("/inspections", response_model=list[AssemblyInspectionResponse])
def get_inspections(
    profile_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> list[AssemblyInspectionResponse]:
    return list_inspections(profile_id=profile_id, limit=limit)


@app.delete("/inspections", response_model=dict)
def delete_all_inspections() -> dict:
    """Purges all stored inspection audit records from local and remote database storage."""
    deleted_count = clear_all_inspections()
    return {"status": "cleared", "deleted_count": deleted_count}


@app.post("/inspections/{inspection_id}/ground-truth", response_model=dict)
def set_ground_truth(
    inspection_id: str,
    req: GroundTruthUpdateRequest,
) -> dict:
    ok = update_ground_truth(inspection_id, req.expected_verdict, req.notes)
    if not ok:
        raise HTTPException(404, f"Inspection '{inspection_id}' not found.")
    return {"status": "updated", "inspection_id": inspection_id, "ground_truth": req.expected_verdict}


@app.get("/reports/statistics", response_model=OperationalStatisticsResponse)
def get_statistics(days: int | None = Query(None, ge=1, le=365)) -> OperationalStatisticsResponse:
    """
    Computes comprehensive operational inspection statistics from real stored records:
    - Total attempts & completed inspections
    - PASS / REJECT / REVIEW counts and rates (% of completed)
    - Processing error counts (isolated from product quality rejections)
    - Average inference and total latency
    - Volume trends grouped by date
    - Most frequently failed checks with rates
    - Ground truth metrics computed strictly on independently labelled records
    """
    return get_operational_statistics(days=days)


@app.get("/reports/accuracy", response_model=AccuracyReportResponse)
def get_accuracy() -> AccuracyReportResponse:
    return get_accuracy_report()


@app.get("/reports/export")
def export_reports(format: Literal["json", "csv"] = "json"):
    inspections = list_inspections(limit=500)

    if format == "json":
        data = [insp.model_dump() for insp in inspections]
        # Exclude huge base64 strings in export for lightweight reporting
        for item in data:
            item["rectified_ref_png_base64"] = None
            item["rectified_test_png_base64"] = None
            item["annotated_image_png_base64"] = None
            item["opencv_diff_png_base64"] = None
            # Neutralize provider naming while accurately identifying local models only if they actually ran
            if not str(item.get("engine", "")).startswith("patchcore"):
                item["engine"] = "Inspection Engine"
            if not str(item.get("model_identifier", "")).startswith("patchcore"):
                item["model_identifier"] = "Inspection Engine"
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="inspection_report.json"'},
        )

    # Format CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "inspection_id", "profile_id", "profile_version", "created_at", "verdict", "verdict_reason",
        "geometric_alignment_ok", "inspection_issue_summary", "ground_truth_verdict", "ground_truth_notes",
        "total_checks", "satisfied", "failed", "uncertain", "not_evaluated",
        "engine", "inference_latency_ms", "total_latency_ms"
    ])

    for insp in inspections:
        counts = insp.counts_summary
        engine_label = insp.model_identifier
        if not str(engine_label).startswith("patchcore"):
            engine_label = "Inspection Engine"
        writer.writerow([
            insp.inspection_id,
            insp.profile_id,
            insp.profile_version,
            insp.created_at,
            insp.verdict,
            insp.verdict_reason,
            insp.geometric_alignment_ok,
            insp.inspection_issue_summary or "",
            insp.ground_truth_verdict or "",
            insp.ground_truth_notes or "",
            counts.get("total_checks", 0),
            counts.get("satisfied", 0),
            counts.get("failed", 0),
            counts.get("uncertain", 0),
            counts.get("not_evaluated", 0),
            engine_label,
            insp.inference_latency_ms,
            insp.total_latency_ms,
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="inspection_report.csv"'},
    )


@app.post("/demo/generate-samples")
def generate_demo_samples() -> dict:
    """Generates synthetic demonstration images for demo mode."""
    img_dir = settings.storage_dir_path / "demo_samples"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Complete board (PASS)
    pass_img = create_synthetic_demo_board_image()
    pass_path = img_dir / "sample_complete_pass.png"
    pass_img.save(pass_path)

    # 2. Missing screw at lower-centre (REJECT)
    screws_5 = ["upper-centre", "upper-left", "upper-right", "lower-left", "lower-right"]
    missing_img = create_synthetic_demo_board_image(screws_present=screws_5)
    missing_path = img_dir / "sample_missing_screw_reject.png"
    missing_img.save(missing_path)

    # 3. Misplaced screw outside lower-centre (REJECT)
    misplaced_screws = {"lower-centre": (0.50, 0.50)}
    misplaced_img = create_synthetic_demo_board_image(misplaced_screws=misplaced_screws)
    misplaced_path = img_dir / "sample_misplaced_screw_reject.png"
    misplaced_img.save(misplaced_path)

    # 4. Obscured / Blurred screw at upper-left (REVIEW)
    blur_img = create_synthetic_demo_board_image(blur_region="upper-left")
    blur_path = img_dir / "sample_obscured_review.png"
    blur_img.save(blur_path)

    return {
        "status": "generated",
        "demo_mode_notice": "These are synthetic demonstration images watermarked for Demo Mode only.",
        "samples": {
            "complete_pass": str(pass_path),
            "missing_screw_reject": str(missing_path),
            "misplaced_screw_reject": str(misplaced_path),
            "obscured_review": str(blur_path),
        },
    }


# ==============================================================================
# Legacy Anomaly Detection Endpoints (Lazy-loaded, optional)
# ==============================================================================

@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    if not METRICS_PATH.exists():
        entries = [
            CategoryMetrics(category=c, model_backend=settings.ad_backend, trained=False)
            for c in settings.categories
        ]
        return MetricsResponse(categories=entries)

    raw = json.loads(METRICS_PATH.read_text())
    entries = [CategoryMetrics(**row) for row in raw]
    return MetricsResponse(categories=entries)


@app.post("/inspect", response_model=InspectionResult)
async def inspect(
    file: UploadFile = File(...),
    category: str = "bottle",
    backend: str | None = None,
) -> InspectionResult:
    """Legacy local anomaly detection (requires PyTorch & trained Anomalib checkpoints)."""
    backend = backend or settings.ad_backend
    if category not in settings.categories:
        raise HTTPException(400, f"Unknown category '{category}'. Known: {settings.categories}")

    contents = await file.read()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc

    try:
        from backend.detector import get_detector, make_heatmap_overlay, ModelNotTrainedError
        detector = get_detector(category, backend)
    except Exception as exc:
        raise HTTPException(503, f"Local anomaly detector unavailable: {exc}") from exc

    detection = detector.predict(image)

    overlay_b64 = None
    description = cause = severity = None
    explanation_source = "none"
    explanation_latency_ms = None

    if detection.heatmap is not None:
        overlay_image = make_heatmap_overlay(image, detection.heatmap)
        buf = io.BytesIO()
        overlay_image.save(buf, format="PNG")
        overlay_b64 = buf.getvalue().hex()
    else:
        overlay_image = image

    if detection.is_defective and settings.vlm_enabled:
        from backend.vlm_explainer import explain_defect
        explanation = explain_defect(overlay_image, category)
        description = explanation.defect_description
        cause = explanation.likely_cause
        severity = explanation.severity
        explanation_source = explanation.source
        explanation_latency_ms = explanation.latency_ms

    return InspectionResult(
        category=category,
        model_backend=backend,
        is_defective=detection.is_defective,
        anomaly_score=detection.anomaly_score,
        heatmap_png_base64=overlay_b64,
        defect_description=description,
        likely_cause=cause,
        severity=severity,
        explanation_source=explanation_source,
        inference_latency_ms=detection.latency_ms,
        explanation_latency_ms=explanation_latency_ms,
    )
