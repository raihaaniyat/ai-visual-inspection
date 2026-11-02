"""
Inspection Engine & Vision Diagnostics.

Executes visual inspection comparing assembly workpieces against an approved
golden reference image and defined acceptance criteria.

Guarantees:
- Stage-by-stage error tracing with request IDs, HTTP status codes, and secret redaction.
- Structured failure categorization into neutral categories:
    * Authentication/configuration error
    * Quota or rate limit
    * Provider unavailable
    * Request timeout
    * Invalid inspection response
    * Image preparation failure
- On failure: inspection_state=ERROR, verdict=REVIEW, checks marked NOT_EVALUATED.
- UNCERTAIN is reserved strictly for successful inspections with ambiguous visual evidence.
- Full photograph presentation with correctly transformed polygon overlays.
- Separation of broad acceptance criteria from individual component-level findings.
- Automatic AI proposal of reference inventory for operator review and approval.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import uuid
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.config import settings
from backend.decision import compute_inspection_verdict
from backend.geometry import (
    RECTIFIED_HEIGHT,
    RECTIFIED_WIDTH,
    ensure_correct_orientation,
    rectify_workpiece,
    transform_box_to_original_polygon,
)
from backend.opencv_diff import (
    compute_rectified_visual_difference,
    extract_difference_regions,
)
from backend.schemas import (
    AssemblyInspectionResponse,
    CheckItem,
    CheckObservation,
    ComponentFinding,
    ExpectedRegion,
    GeminiInspectionPayload,
    NormalizedBox,
    WorkpieceCorners,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Error Categorization & Sanitization
# ==============================================================================

FAILURE_CATEGORIES = [
    "Authentication/configuration error",
    "Quota or rate limit",
    "Provider unavailable",
    "Request timeout",
    "Invalid inspection response",
    "Image preparation failure",
]


def get_configured_api_keys() -> list[str]:
    """Returns list of unique configured API keys (primary followed by fallback key)."""
    keys = []
    for k in (settings.gemini_api_key, settings.gemini_api_key_fallback):
        k_clean = (k or "").strip()
        if k_clean and k_clean not in (
            "your_gemini_api_key_here",
            "your_primary_api_key_here",
            "your_backup_api_key_here",
            "your_api_key_here",
        ):
            if k_clean not in keys:
                keys.append(k_clean)
    return keys


def sanitize_log_message(msg: str) -> str:
    """Redacts API keys and sensitive tokens from diagnostic log strings."""
    # Redact query params key=... or api_key=...
    msg = re.sub(r'(api_key|key|token)=["\']?[A-Za-z0-9_\-\.]+["\']?', r'\1=[REDACTED]', msg, flags=re.IGNORECASE)
    # Redact Google API key format AIzaSy...
    msg = re.sub(r'AIzaSy[A-Za-z0-9_-]{33}', '[REDACTED_API_KEY]', msg)
    return msg


def classify_inspection_error(exc: Exception | None, stage: str) -> tuple[str, int | None]:
    """
    Deterministically maps an exception and pipeline stage to one of the 6 neutral categories:
    - Authentication/configuration error
    - Quota or rate limit
    - Provider unavailable
    - Request timeout
    - Invalid inspection response
    - Image preparation failure
    """
    if stage == "image_preparation":
        return "Image preparation failure", None

    if exc is None:
        if stage == "configuration":
            return "Authentication/configuration error", 401
        return "Provider unavailable", 503

    msg = str(exc)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, str) and code.isdigit():
        code = int(code)

    # 1. Authentication / Configuration
    if code in (401, 403) or any(k in msg for k in ["API_KEY_INVALID", "invalid api key", "PERMISSION_DENIED", "unauthenticated", "Unauthorized"]):
        return "Authentication/configuration error", 401

    # 2. Quota / Rate Limit
    if code == 429 or any(k in msg for k in ["RESOURCE_EXHAUSTED", "quota", "rate limit", "Too Many Requests"]):
        return "Quota or rate limit", 429

    # 3. Timeout
    if isinstance(exc, (TimeoutError,)) or any(k in msg.lower() for k in ["deadlineexceeded", "timed out", "timeout"]):
        return "Request timeout", 408

    # 4. Parsing / Schema Validation
    if stage in ("response_parsing", "schema_validation") or any(k in msg for k in ["JSONDecodeError", "ValidationError", "Expecting value"]):
        return "Invalid inspection response", None

    # 5. Provider Unavailable / Service demand
    if code in (500, 502, 503, 504) or any(k in msg for k in ["UNAVAILABLE", "high demand", "ServiceUnavailable", "BadGateway", "Connection refused"]):
        return "Provider unavailable", code if isinstance(code, int) else 503

    # Fallback by stage
    if stage == "provider_call":
        return "Provider unavailable", code if isinstance(code, int) else 503
    elif stage in ("response_parsing", "schema_validation"):
        return "Invalid inspection response", None

    return "Provider unavailable", code if isinstance(code, int) else 500


def build_error_inspection_response(
    profile_id: str,
    profile_version: int,
    checklist: list[CheckItem],
    request_id: str,
    stage: str,
    failure_category: str,
    latency_ms: float,
    orig_test_image: Image.Image | None = None,
) -> AssemblyInspectionResponse:
    """
    Builds a clean, non-hallucinating failed-inspection response:
    - inspection_state: ERROR
    - verdict: REVIEW
    - required checks: NOT_EVALUATED
    - overlays and difference maps: cleared
    """
    counts = {
        "total_checks": len(checklist),
        "satisfied": 0,
        "failed": 0,
        "uncertain": 0,
        "not_evaluated": len(checklist),
    }

    findings = [
        ComponentFinding(
            check_id=item.check_id,
            component_id=item.region_id,
            label=item.description,
            check_type=item.check_type,
            expected_requirement=item.expected_requirement,
            observed_state_or_count="Not evaluated",
            status="not_evaluated",
            evidence=f"Inspection failed during {stage} ({failure_category}). Check was not evaluated by inspection engine.",
            rectified_box=None,
            original_polygon=None,
        )
        for item in checklist
    ]

    clean_ann_b64 = None
    if orig_test_image:
        try:
            disp_orig = orig_test_image.copy()
            if max(disp_orig.size) > 1280:
                disp_orig.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            disp_orig.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
            clean_ann_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            clean_ann_b64 = None

    return AssemblyInspectionResponse(
        inspection_id=f"insp_err_{uuid.uuid4().hex[:8]}",
        profile_id=profile_id,
        profile_version=profile_version,
        inspection_state="ERROR",
        failure_category=failure_category,  # type: ignore
        request_id=request_id,
        verdict="REVIEW",
        verdict_reason=f"Inspection unavailable during {stage}: {failure_category}. Manual review required.",
        inspection_issue_summary=f"{failure_category}: Inspection unavailable. Please retry.",
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
        findings=findings,
        counts_summary=counts,
        rectified_ref_png_base64=None,
        rectified_test_png_base64=None,
        annotated_image_png_base64=clean_ann_b64,
        opencv_diff_png_base64=None,
        engine="Inspection Engine",
        model_identifier="Inspection Engine",
        inference_latency_ms=0.0,
        total_latency_ms=round(latency_ms, 1),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ==============================================================================
# Workpiece Selection Geometry Validation
# ==============================================================================

def validate_workpiece_corners(corners: WorkpieceCorners | None) -> tuple[bool, str | None]:
    """
    Checks that the 4 corner handles encompass a plausible complete workpiece
    rather than a narrow or tiny central strip.
    """
    if not corners:
        return True, None

    pts = [corners.top_left, corners.top_right, corners.bottom_right, corners.bottom_left]
    # Shoelace formula for quadrilateral area in normalized [0.0, 1.0] coordinates
    area = 0.5 * abs(
        (pts[0][0]*pts[1][1] - pts[1][0]*pts[0][1]) +
        (pts[1][0]*pts[2][1] - pts[2][0]*pts[1][1]) +
        (pts[2][0]*pts[3][1] - pts[3][0]*pts[2][1]) +
        (pts[3][0]*pts[0][1] - pts[0][0]*pts[3][1])
    )

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    width_span = max(xs) - min(xs)
    height_span = max(ys) - min(ys)

    if area < 0.10:
        return False, f"Selected workpiece covers only {area*100:.1f}% of image. Corners may enclose a narrow strip rather than the complete board."
    if width_span > 0 and height_span > 0:
        aspect = width_span / height_span
        if aspect < 0.22 or aspect > 4.5:
            return False, f"Selected workpiece has an extreme aspect ratio ({aspect:.2f}). Please ensure corners include the full intended assembly."
    return True, None


# ==============================================================================
# Visual Annotations & Drawing
# ==============================================================================

def _crop_region(image: Image.Image, box: NormalizedBox) -> Image.Image:
    """Crops a region from an image based on normalized coordinates [ymin, xmin, ymax, xmax]."""
    w, h = image.size
    left = int(max(0.0, box.xmin) * w)
    top = int(max(0.0, box.ymin) * h)
    right = int(min(1.0, box.xmax) * w)
    bottom = int(min(1.0, box.ymax) * h)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _draw_dashed_polygon(
    img_bgr: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    """Draws a dashed polygon onto an OpenCV image."""
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist <= 0:
            continue
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        curr = 0.0
        while curr < dist:
            start_pt = (int(p1[0] + dx * curr), int(p1[1] + dy * curr))
            end_len = min(dist, curr + dash_len)
            end_pt = (int(p1[0] + dx * end_len), int(p1[1] + dy * end_len))
            cv2.line(img_bgr, start_pt, end_pt, color, thickness, cv2.LINE_AA)
            curr += dash_len + gap_len


def _draw_polygon_annotations_on_original(
    orig_image: Image.Image,
    findings: list[ComponentFinding],
) -> Image.Image:
    """
    Draws oriented transformed quadrilaterals onto the original inspection photograph:
    - Dashed neutral box: operator-defined expected inspection region.
    - Solid box: actual observed component detection on the workpiece.
    - Green/red/amber: verified check status, not box existence.
    - Labels positioned outside component boxes with clean readable backdrop pills.
    """
    img_bgr = cv2.cvtColor(np.array(orig_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    color_map = {
        "satisfied": (46, 204, 113),      # Green
        "failed": (46, 46, 231),          # Red
        "uncertain": (0, 165, 255),        # Amber
        "not_evaluated": (160, 160, 160),  # Gray
    }
    dashed_neutral_color = (210, 210, 215)  # Neutral silver-white for expected regions

    # Pass 1: Draw operator-defined expected inspection regions as DASHED neutral boxes
    for finding in findings:
        poly = finding.expected_polygon_original
        if not poly or len(poly) != 4:
            continue

        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
        # Dashed neutral box for expected region
        _draw_dashed_polygon(img_bgr, pts, color=dashed_neutral_color, thickness=2, dash_len=7, gap_len=5)

        # If component is missing, draw label tag outside the expected region indicating missing status
        if not finding.has_detected_box and finding.status == "failed":
            min_x = int(np.min(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            max_y = int(np.max(pts[:, 1]))
            tag = f"{finding.label}: MISSING"
            tag_y = max(22, min_y - 8) if min_y > 28 else (max_y + 18)

            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(img_bgr, (min_x, tag_y - th - 4), (min_x + tw + 6, tag_y + 4), (20, 24, 33), -1)
            cv2.rectangle(img_bgr, (min_x, tag_y - th - 4), (min_x + tw + 6, tag_y + 4), (46, 46, 231), 1)
            cv2.putText(img_bgr, tag, (min_x + 3, tag_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Pass 2: Draw actual observed component detections as SOLID boxes
    for finding in findings:
        if not finding.has_detected_box:
            continue

        poly = finding.detected_polygon_original or finding.original_polygon
        if not poly or len(poly) != 4:
            continue

        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
        color = color_map.get(finding.status, (160, 160, 160))

        # Solid box for actual observed component
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

        min_x = int(np.min(pts[:, 0]))
        min_y = int(np.min(pts[:, 1]))
        max_y = int(np.max(pts[:, 1]))
        tag = f"{finding.label}: {finding.status.upper()}"
        tag_y = max(22, min_y - 8) if min_y > 28 else (max_y + 18)

        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(img_bgr, (min_x, tag_y - th - 4), (min_x + tw + 6, tag_y + 4), (20, 24, 33), -1)
        cv2.rectangle(img_bgr, (min_x, tag_y - th - 4), (min_x + tw + 6, tag_y + 4), color, 1)
        cv2.putText(img_bgr, tag, (min_x + 3, tag_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _clean_json_response(text: str) -> str:
    clean = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        return match.group(1).strip()
    return clean


# ==============================================================================
# Core Inspection Pipeline
# ==============================================================================

_SYSTEM_PROMPT = """You are an automated industrial quality-control visual inspection engine.
Your task is to thoroughly inspect a test workpiece against an approved golden reference standard to identify what components are present, what is missing, and what is misplaced.

CRITICAL INSPECTION RULES:
1. UNIVERSAL WORKPIECE SUPPORT:
   - You inspect ANY physical product, manufactured assembly, circuit board, mechanical apparatus, fastener array, enclosure, or kit.
   - You are NOT restricted to screws or to 6 positions! Components can be fasteners, electronic chips, resistors, connectors, brackets, pins, washers, gears, or any visual assembly parts.
2. SYSTEMATIC REFERENCE COMPARISON:
   - Image 1 is the APPROVED GOLDEN REFERENCE (shows all required components in their correct positions).
   - Image 2 is the TEST WORKPIECE UNDER INSPECTION (workpiece to evaluate for presence, absence, and correctness).
   - Compare Image 1 and Image 2 systematically across the entire workpiece.
3. IDENTIFY ALL COMPONENTS & DEFECTS:
   - Identify every distinct required component, part, and fastener shown in Image 1.
   - For EACH component:
     a) Determine if it is PRESENT and properly installed in Image 2 (status: "satisfied").
     b) Or if it is MISSING in Image 2 (status: "failed", e.g. empty hole, vacant mounting slot, vacant pad where a component exists in Image 1).
     c) Or if it is MISPLACED / EXTRA / WRONG in Image 2 (status: "failed").
     d) Or if it is UNCERTAIN (status: "uncertain", e.g. obscured by glare, blur, or obstruction).
   - If configured checklist items or regions are given below, evaluate those items with their check_id.
   - In addition (and whenever inspecting any image), identify and evaluate all other components present in Image 1.
4. PRECISE BOUNDING BOXES:
   - For EVERY component observation (present or missing), provide its normalized bounding box `box: {"ymin": float, "xmin": float, "ymax": float, "xmax": float}` (values 0.0 to 1.0) in Image 2 coordinates (or the vacant location where the component should be if missing).
5. WORKPIECE FOCUS:
   - Inspect only the primary workpiece. Ignore loose clutter or tabletop surface far outside the workpiece.
6. CONCISE VISUAL EVIDENCE:
   - State the direct visual evidence clearly (e.g. "PRESENT: Installed and seated properly" or "MISSING: Empty countersunk hole detected; component in reference is absent")."""


def run_workpiece_inspection(
    reference_image: Image.Image,
    test_image: Image.Image,
    profile_id: str,
    profile_version: int,
    checklist: list[CheckItem],
    regions: list[ExpectedRegion],
    reference_corners: WorkpieceCorners | None = None,
    test_corners: WorkpieceCorners | None = None,
    opencv_diff_overlay_b64: str | None = None,
) -> AssemblyInspectionResponse:
    """
    Executes visual inspection comparing test workpiece against approved reference.
    Enforces strict error tracing, secret redaction, and failed-inspection reporting.
    """
    start_total = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    # Correct EXIF orientations immediately
    reference_image = ensure_correct_orientation(reference_image)
    test_image = ensure_correct_orientation(test_image)

    api_keys = get_configured_api_keys()
    model_name = settings.gemini_model

    # Check for unconfigured / missing API keys
    if not api_keys:
        logger.error("[%s] Provider configuration error: No valid API key is configured.", request_id)
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage="configuration",
            failure_category="Authentication/configuration error",
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )

    # Validate workpiece boundary selection
    corners_valid, corner_warning = validate_workpiece_corners(test_corners)
    if not corners_valid and corner_warning:
        logger.warning("[%s] Workpiece selection warning: %s", request_id, corner_warning)

    # Stage 1: Image Preparation & Perspective Rectification
    current_stage = "image_preparation"
    try:
        rect_ref, M_ref, M_ref_inv = rectify_workpiece(reference_image, reference_corners)
        rect_test, M_test, M_test_inv = rectify_workpiece(test_image, test_corners)
        alignment_ok = True
        alignment_summary = corner_warning
    except Exception as exc:
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        logger.error("[%s] Failure at stage '%s' (status: None): %s", request_id, current_stage, sanitize_log_message(str(exc)))
        cat, status_c = classify_inspection_error(exc, current_stage)
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage=current_stage,
            failure_category=cat,
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )

    # Compute supporting difference map and candidate difference regions
    diff_b64 = None
    diff_bytes = None
    diff_reliable = True
    diff_notice = None
    diff_boxes: list[NormalizedBox] = []
    try:
        if alignment_ok:
            diff_res = compute_rectified_visual_difference(rect_ref, rect_test)
            diff_b64 = diff_res.overlay_b64
            diff_reliable = getattr(diff_res, "reliable", True)
            diff_notice = getattr(diff_res, "notice", None)
            if diff_b64:
                diff_bytes = base64.b64decode(diff_b64)
            diff_boxes = extract_difference_regions(rect_ref, rect_test, max_hotspots=6)
    except Exception as diff_exc:
        logger.warning("[%s] Supporting OpenCV difference calculation skipped: %s", request_id, diff_exc)

    # Build prompt instructions
    criteria_lines = []
    for item in checklist:
        reg_info = f" [Region: {item.region_id}]" if item.region_id else ""
        cnt_info = f" [Expected Count: {item.target_count}]" if item.target_count is not None else ""
        criteria_lines.append(f"- ID: {item.check_id} | Type: {item.check_type}{reg_info}{cnt_info} | Requirement: {item.expected_requirement}")

    defined_regions_text = ""
    if regions:
        region_lines = [f"- {r.region_id}: {r.label} ({r.component_type}) at box [ymin={r.box.ymin:.2f}, xmin={r.box.xmin:.2f}, ymax={r.box.ymax:.2f}, xmax={r.box.xmax:.2f}]" for r in regions]
        defined_regions_text = "\nDEFINED COMPONENT POSITIONS (Rectified Space):\n" + "\n".join(region_lines)
    else:
        defined_regions_text = "\nNO PRE-DEFINED REGIONS CONFIGURED: Dynamically discover ALL components on the reference and systematically verify each on the test workpiece."

    prompt_content = f"""{_SYSTEM_PROMPT}

ACCEPTANCE CHECKLIST:
{chr(10).join(criteria_lines)}
{defined_regions_text}

INSPECTION INSTRUCTIONS:
1. Examine Image 1 (Approved Reference) and Image 2 (Test Workpiece Under Inspection).
2. For each acceptance checklist item above, record an observation with its check_id and status.
3. IN ADDITION, discover and catalogue ALL individual components, parts, and fasteners on Image 1.
4. For every discovered component, output an observation with:
   - "check_id": a descriptive identifier (e.g. comp_top_left, comp_fastener_1, comp_chip_u1, etc.)
   - "label": component name and location
   - "status": "satisfied" (present & correct), "failed" (MISSING or MISPLACED), or "uncertain" (obscured)
   - "observed_state_or_count": clear summary (e.g. "PRESENT", "MISSING: Empty mounting location", "MISPLACED")
   - "box": normalized bounding box with keys ymin, xmin, ymax, xmax (values 0.0 to 1.0) in Image 2 (rectified workpiece) coordinates for an observed component.
     IMPORTANT RULES FOR "box":
     * If a component is MISSING (empty hole/pad), set "box": null. Describe the vacant location in "visible_evidence" and "location". Do NOT invent a false component detection box for absent items!
     * If this is an overall count or assembly check, set "box": null. Never provide a box that covers the entire workpiece [0, 0, 1, 1] as a component detection!
5. If difference hotspot crops are attached, examine them closely for missing screws, vacant pads, or unpopulated sockets.

Return ONLY a valid JSON object matching this schema:
{{
  "reference_identity_ok": true,
  "image_quality_ok": true,
  "geometric_alignment_ok": true,
  "alignment_issue_summary": null,
  "overall_visible_count": <integer or null>,
  "observations": [
    {{
      "check_id": "<check_id>",
      "label": "<Component Label>",
      "expected_requirement": "<requirement>",
      "observed_state_or_count": "<PRESENT or MISSING>",
      "status": "satisfied|failed|uncertain|not_evaluated",
      "visible_evidence": "<concise visual evidence>",
      "location": "<location description>",
      "box": null
    }}
  ]
}}"""

    # Stage 2: Call Vision Inspection Model (with automatic fallback on quota limits)
    current_stage = "provider_call"
    start_inf = time.perf_counter()
    raw_text = ""

    from google import genai
    from google.genai import types

    ref_io = io.BytesIO()
    rect_ref.convert("RGB").save(ref_io, format="JPEG", quality=90)
    ref_bytes = ref_io.getvalue()

    test_io = io.BytesIO()
    rect_test.convert("RGB").save(test_io, format="JPEG", quality=90)
    test_bytes = test_io.getvalue()

    contents_list = [
        types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=test_bytes, mime_type="image/jpeg"),
    ]

    if diff_bytes:
        contents_list.append(types.Part.from_bytes(data=diff_bytes, mime_type="image/jpeg"))

    contents_list.append(prompt_content)

    # Attach high-resolution crops
    crops_to_attach = regions if regions else [ExpectedRegion(region_id=f"diff_{i}", label=f"Difference Spot #{i+1}", box=b) for i, b in enumerate(diff_boxes)]
    for r in crops_to_attach[:6]:
        crop_img = _crop_region(rect_test, r.box)
        c_io = io.BytesIO()
        crop_img.save(c_io, format="JPEG", quality=85)
        contents_list.append(types.Part.from_bytes(data=c_io.getvalue(), mime_type="image/jpeg"))

    last_exc: Exception | None = None
    for key_idx, current_api_key in enumerate(api_keys):
        try:
            client = genai.Client(api_key=current_api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=contents_list,
            )
            raw_text = response.text or ""
            inf_lat = (time.perf_counter() - start_inf) * 1000.0
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            is_quota_or_rate = (
                "resource_exhausted" in err_str
                or "429" in err_str
                or "quota" in err_str
                or "rate limit" in err_str
            )
            if is_quota_or_rate and key_idx < len(api_keys) - 1:
                logger.warning(
                    "[%s] API key #%d hit quota/rate limit. Automatically falling back to backup API key...",
                    request_id,
                    key_idx + 1,
                )
                continue
            break

    if last_exc is not None:
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        cat, status_c = classify_inspection_error(last_exc, current_stage)
        logger.error("[%s] Failure at stage '%s' (status: %s): %s", request_id, current_stage, status_c, sanitize_log_message(str(last_exc)))
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage=current_stage,
            failure_category=cat,
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )

    # Stage 3: Response Parsing
    current_stage = "response_parsing"
    try:
        clean_json = _clean_json_response(raw_text)
        data = json.loads(clean_json)
    except Exception as exc:
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        cat, status_c = classify_inspection_error(exc, current_stage)
        logger.error("[%s] Failure at stage '%s' (status: None): %s. Raw preview: %.200s", request_id, current_stage, exc, raw_text)
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage=current_stage,
            failure_category=cat,
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )

    # Stage 4: Schema Validation
    current_stage = "schema_validation"
    try:
        parsed_payload = GeminiInspectionPayload(**data)
    except Exception as exc:
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        cat, status_c = classify_inspection_error(exc, current_stage)
        logger.error("[%s] Failure at stage '%s' (status: None): %s", request_id, current_stage, exc)
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage=current_stage,
            failure_category=cat,
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )

    # Stage 5: Verdict Calculation & Findings Assembly
    current_stage = "verdict_calculation"
    try:
        identity_ok = parsed_payload.reference_identity_ok
        quality_ok = parsed_payload.image_quality_ok
        geom_ok = alignment_ok and parsed_payload.geometric_alignment_ok
        alignment_note = alignment_summary or parsed_payload.alignment_issue_summary

        verdict, reason, issue_summary, counts, findings = compute_inspection_verdict(
            checklist=checklist,
            regions=regions,
            observations=parsed_payload.observations,
            reference_identity_ok=identity_ok,
            image_quality_ok=quality_ok,
            geometric_alignment_ok=geom_ok,
            alignment_issue_summary=alignment_note,
        )

        # Transform Rectified Boxes to Polygons on Original Photograph
        test_w, test_h = test_image.size
        for f in findings:
            # 1. Transform expected inspection region if present
            if f.expected_region_rectified and geom_ok:
                f.expected_polygon_original = transform_box_to_original_polygon(
                    box=f.expected_region_rectified,
                    M_inv=M_test_inv,
                    rect_w=RECTIFIED_WIDTH,
                    rect_h=RECTIFIED_HEIGHT,
                    orig_w=test_w,
                    orig_h=test_h,
                )
            # 2. Transform detected component box if present
            if f.detected_box_rectified and geom_ok:
                f.detected_polygon_original = transform_box_to_original_polygon(
                    box=f.detected_box_rectified,
                    M_inv=M_test_inv,
                    rect_w=RECTIFIED_WIDTH,
                    rect_h=RECTIFIED_HEIGHT,
                    orig_w=test_w,
                    orig_h=test_h,
                )
            # Backward compatibility alias
            if f.rectified_box and geom_ok and not f.original_polygon:
                f.original_polygon = transform_box_to_original_polygon(
                    box=f.rectified_box,
                    M_inv=M_test_inv,
                    rect_w=RECTIFIED_WIDTH,
                    rect_h=RECTIFIED_HEIGHT,
                    orig_w=test_w,
                    orig_h=test_h,
                )
            if not f.original_polygon:
                f.original_polygon = f.detected_polygon_original or f.expected_polygon_original

        # Render Visual Annotations on Original Photograph
        has_polygons = any((f.detected_polygon_original or f.expected_polygon_original or f.original_polygon) for f in findings)
        if has_polygons:
            annotated_img = _draw_polygon_annotations_on_original(test_image, findings)
        else:
            # Preserve original photograph as-is without cropping or distortion
            annotated_img = test_image

        # Constrain display overlays to crisp dimensions & JPEG compression
        # to ensure the full inspection response stays well within Vercel's 4.5MB serverless limit
        disp_ann = annotated_img.copy()
        if max(disp_ann.size) > 1280:
            disp_ann.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buf_ann = io.BytesIO()
        disp_ann.convert("RGB").save(buf_ann, format="JPEG", quality=85, optimize=True)
        annotated_b64 = base64.b64encode(buf_ann.getvalue()).decode("utf-8")

        # Encode rectified images separately
        disp_rref = rect_ref.copy()
        if max(disp_rref.size) > 800:
            disp_rref.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buf_rref = io.BytesIO()
        disp_rref.convert("RGB").save(buf_rref, format="JPEG", quality=85, optimize=True)
        rect_ref_b64 = base64.b64encode(buf_rref.getvalue()).decode("utf-8")

        disp_rtest = rect_test.copy()
        if max(disp_rtest.size) > 800:
            disp_rtest.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buf_rtest = io.BytesIO()
        disp_rtest.convert("RGB").save(buf_rtest, format="JPEG", quality=85, optimize=True)
        rect_test_b64 = base64.b64encode(buf_rtest.getvalue()).decode("utf-8")

        if not geom_ok:
            diff_b64 = None

        tot_lat = (time.perf_counter() - start_total) * 1000.0

        return AssemblyInspectionResponse(
            inspection_id=f"insp_{uuid.uuid4().hex[:10]}",
            profile_id=profile_id,
            profile_version=profile_version,
            inspection_state="COMPLETED",
            failure_category=None,
            request_id=request_id,
            verdict=verdict,
            verdict_reason=reason,
            inspection_issue_summary=issue_summary,
            reference_identity_ok=identity_ok,
            image_quality_ok=quality_ok,
            geometric_alignment_ok=geom_ok,
            findings=findings,
            counts_summary=counts,
            rectified_ref_png_base64=rect_ref_b64,
            rectified_test_png_base64=rect_test_b64,
            annotated_image_png_base64=annotated_b64,
            opencv_diff_png_base64=diff_b64,
            difference_map_reliable=diff_reliable,
            difference_map_notice=diff_notice,
            engine="Inspection Engine",
            model_identifier="Inspection Engine",
            inference_latency_ms=round(inf_lat, 1),
            total_latency_ms=round(tot_lat, 1),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    except Exception as exc:
        tot_lat = (time.perf_counter() - start_total) * 1000.0
        logger.error("[%s] Failure at stage '%s': %s", request_id, current_stage, exc)
        return build_error_inspection_response(
            profile_id=profile_id,
            profile_version=profile_version,
            checklist=checklist,
            request_id=request_id,
            stage=current_stage,
            failure_category="Image preparation failure",
            latency_ms=tot_lat,
            orig_test_image=test_image,
        )


# ==============================================================================
# AI-Proposed Reference Inventory Proposal
# ==============================================================================

def propose_reference_inventory(
    reference_image: Image.Image,
    reference_corners: WorkpieceCorners | None = None,
) -> list[ExpectedRegion]:
    """
    Scans an approved reference standard and automatically detects all installed components.
    Returns candidate regions for operator review and approval.
    """
    api_keys = get_configured_api_keys()
    model_name = settings.gemini_model
    if not api_keys:
        raise ValueError("Inspection Engine is unconfigured.")

    reference_image = ensure_correct_orientation(reference_image)
    rect_ref, _, _ = rectify_workpiece(reference_image, reference_corners)

    prompt = """Analyze this approved reference standard assembly workpiece.
Identify and catalogue every distinct component, fastener, connector, chip, pin, or visual part present on the board.
For each component, provide:
- region_id: e.g. "comp_screw_1", "comp_chip_u1", "comp_connector_j1"
- label: descriptive name and location, e.g. "Upper-Left Fastener" or "USB Connector J1"
- component_type: e.g. "screw", "connector", "ic", "capacitor", "bracket"
- box: normalized coordinates {"ymin": float, "xmin": float, "ymax": float, "xmax": float} in range 0.0 to 1.0

Return ONLY a valid JSON object matching this schema:
{
  "proposed_components": [
    {
      "region_id": "comp_1",
      "label": "Upper-Left Screw",
      "component_type": "screw",
      "box": {"ymin": 0.15, "xmin": 0.20, "ymax": 0.35, "xmax": 0.40}
    }
  ]
}"""

    from google import genai
    from google.genai import types

    ref_io = io.BytesIO()
    rect_ref.convert("RGB").save(ref_io, format="JPEG", quality=90)

    contents = [
        types.Part.from_bytes(data=ref_io.getvalue(), mime_type="image/jpeg"),
        prompt,
    ]

    last_exc = None
    data = {}
    for key_idx, current_api_key in enumerate(api_keys):
        try:
            client = genai.Client(api_key=current_api_key)
            response = client.models.generate_content(model=model_name, contents=contents)
            clean_json = _clean_json_response(response.text or "")
            data = json.loads(clean_json)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if ("resource_exhausted" in err_str or "429" in err_str or "quota" in err_str) and key_idx < len(api_keys) - 1:
                logger.warning("Propose inventory key #%d quota exhausted, trying fallback key...", key_idx + 1)
                continue
            break

    if last_exc is not None:
        raise last_exc

    proposed = []
    for item in data.get("proposed_components", []):
        try:
            proposed.append(
                ExpectedRegion(
                    region_id=item.get("region_id", f"comp_{uuid.uuid4().hex[:6]}"),
                    label=item.get("label", "Component"),
                    component_type=item.get("component_type", "part"),
                    box=NormalizedBox(**item.get("box", {})),
                )
            )
        except Exception:
            continue
    return proposed
