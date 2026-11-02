"""
Data schemas for VisionInspect AI.

Supports:
1. General Assembly Inspection Profiles (Arbitrary Components, Quantities, Positions, Check Types)
2. Workpiece Selection & Perspective Rectification Coordinates
3. Concise Structured Observations with Clean Status Hierarchy:
   - satisfied
   - failed
   - uncertain
   - not_evaluated
4. Persistent Inspection Records and Benchmark Ground-Truth
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Workpiece Geometry & Rectification Schemas
# ==============================================================================

class NormalizedBox(BaseModel):
    """Normalized coordinates [0.0, 1.0] in rectified workpiece coordinate space."""
    ymin: float = Field(default=0.0)
    xmin: float = Field(default=0.0)
    ymax: float = Field(default=1.0)
    xmax: float = Field(default=1.0)

    @model_validator(mode="before")
    @classmethod
    def clamp_coordinates(cls, data: Any) -> Any:
        if isinstance(data, dict):
            try:
                ymin = max(0.0, min(1.0, float(data.get("ymin", 0.0))))
                xmin = max(0.0, min(1.0, float(data.get("xmin", 0.0))))
                ymax = max(0.0, min(1.0, float(data.get("ymax", 1.0))))
                xmax = max(0.0, min(1.0, float(data.get("xmax", 1.0))))
                # Ensure ymin <= ymax and xmin <= xmax
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                return {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
            except Exception:
                return data
        return data

    def is_workpiece_scale(self, threshold: float = 0.85) -> bool:
        """Determines if the bounding box spans the entire workpiece rather than a specific component."""
        w = max(0.0, self.xmax - self.xmin)
        h = max(0.0, self.ymax - self.ymin)
        return (w >= threshold and h >= threshold) or (w * h >= 0.75)


class WorkpieceCorners(BaseModel):
    """
    4 corresponding board corners in normalized image coordinates [x, y] (0.0 to 1.0).
    Ordered: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    top_left: list[float] = Field(default_factory=lambda: [0.1, 0.1])
    top_right: list[float] = Field(default_factory=lambda: [0.9, 0.1])
    bottom_right: list[float] = Field(default_factory=lambda: [0.9, 0.9])
    bottom_left: list[float] = Field(default_factory=lambda: [0.1, 0.9])


# ==============================================================================
# General Inspection Profiles & Checklists
# ==============================================================================

CheckType = Literal["presence", "count", "arrangement", "visible_quality"]


class CheckItem(BaseModel):
    """Acceptance criterion item on an editable checklist."""
    check_id: str = Field(..., validation_alias=AliasChoices("check_id", "item_id"), description="Unique check identifier, e.g. chk_count or chk_lower_centre")
    check_type: CheckType = Field(default="presence")
    description: str = Field(..., description="Human-readable requirement")
    expected_requirement: str = Field(default="", description="Strict requirement for the model, e.g. 'Must have 1 screw seated'")
    region_id: str | None = Field(default=None, description="Optional region ID if this is a positional check")
    target_count: int | None = Field(default=None, description="Expected quantity if count check")
    required: bool = True


class ExpectedRegion(BaseModel):
    """A labelled component position defined in rectified reference coordinates."""
    region_id: str = Field(..., description="Unique position identifier, e.g. lower-centre")
    label: str = Field(..., description="Display label, e.g. Lower-Centre Screw")
    component_type: str = Field(default="screw", validation_alias=AliasChoices("component_type", "expected_component"), description="Arbitrary component type, e.g. screw, washer, pin, chip")
    box: NormalizedBox


class ProfileResponse(BaseModel):
    profile_id: str
    name: str
    description: str = ""
    version: int = 1
    reference_image_id: str | None = None
    reference_corners: WorkpieceCorners | None = None
    checklist: list[CheckItem] = Field(default_factory=list)
    regions: list[ExpectedRegion] = Field(default_factory=list)
    created_at: str
    updated_at: str


# ==============================================================================
# Structured Observations & Findings
# ==============================================================================

CheckStatus = Literal["satisfied", "failed", "uncertain", "not_evaluated"]
InspectionVerdict = Literal["PASS", "REJECT", "REVIEW"]


class CheckObservation(BaseModel):
    """Observation returned from visual inspection for a check item or discovered component."""
    check_id: str
    label: str | None = None
    expected_requirement: str = "Component must be present and correctly installed."
    observed_state_or_count: str
    status: CheckStatus
    visible_evidence: str
    location: str | None = None
    box: NormalizedBox | None = None

    @field_validator("observed_state_or_count", mode="before")
    @classmethod
    def stringify_observed_state(cls, v: Any) -> str:
        if v is None:
            return "None"
        return str(v)


class GeminiInspectionPayload(BaseModel):
    """Clean structured output schema requested from Gemini."""
    reference_identity_ok: bool = True
    image_quality_ok: bool = True
    geometric_alignment_ok: bool = True
    alignment_issue_summary: str | None = None
    overall_visible_count: int | None = None
    observations: list[CheckObservation]


CoordinateSpace = Literal[
    "original_reference_image",
    "original_inspection_image",
    "rectified_workpiece",
    "model_input_image",
    "displayed_frontend_image",
]


class ComponentFinding(BaseModel):
    """
    Final finding per checklist item with explicit coordinate spaces, separating
    operator-defined expected inspection regions from actual component detections.
    """
    check_id: str
    component_id: str | None = None
    label: str
    check_type: str
    expected_requirement: str
    observed_state_or_count: str
    status: CheckStatus
    evidence: str

    # Semantic location if precise bounding box is not available or not applicable
    semantic_location: str | None = None

    # 1. Expected Inspection Region (operator-defined in profile or reference standard)
    has_expected_region: bool = False
    expected_region_rectified: NormalizedBox | None = None  # in 'rectified_workpiece' space
    expected_polygon_original: list[list[float]] | None = None  # in 'original_inspection_image' space

    # 2. Detected Component Box (actual physical observation localized on workpiece)
    has_detected_box: bool = False
    detected_box_rectified: NormalizedBox | None = None  # in 'rectified_workpiece' space
    detected_polygon_original: list[list[float]] | None = None  # in 'original_inspection_image' space

    # Explicit mapping of each box/polygon to its coordinate space
    coordinate_spaces: dict[str, str] = Field(
        default_factory=lambda: {
            "expected_region_rectified": "rectified_workpiece",
            "expected_polygon_original": "original_inspection_image",
            "detected_box_rectified": "rectified_workpiece",
            "detected_polygon_original": "original_inspection_image",
        }
    )

    # Primary coordinate space represented by the original polygon
    coordinate_space: CoordinateSpace = "original_inspection_image"

    # Backward compatibility aliases
    rectified_box: NormalizedBox | None = None
    original_polygon: list[list[float]] | None = None


class AssemblyInspectionResponse(BaseModel):
    """Complete inspection response returned by API."""
    inspection_id: str
    profile_id: str
    profile_version: int
    inspection_state: Literal["COMPLETED", "ERROR"] = "COMPLETED"
    failure_category: Literal[
        "Authentication/configuration error",
        "Quota or rate limit",
        "Provider unavailable",
        "Request timeout",
        "Invalid inspection response",
        "Image preparation failure",
    ] | None = None
    request_id: str | None = None
    verdict: InspectionVerdict
    verdict_reason: str
    inspection_issue_summary: str | None = None
    reference_identity_ok: bool
    image_quality_ok: bool
    geometric_alignment_ok: bool
    findings: list[ComponentFinding]
    counts_summary: dict[str, int]
    rectified_ref_png_base64: str | None = None
    rectified_test_png_base64: str | None = None
    annotated_image_png_base64: str | None = None
    opencv_diff_png_base64: str | None = None
    difference_map_reliable: bool = True
    difference_map_notice: str | None = None
    engine: str = "Inspection Engine"
    model_identifier: str = "Inspection Engine"
    inference_latency_ms: float
    total_latency_ms: float
    created_at: str
    ground_truth_verdict: str | None = None
    ground_truth_notes: str | None = None


# ==============================================================================
# Geometry Mapping Diagnostics (AI-Independent)
# ==============================================================================

class MappingDiagnosticPoint(BaseModel):
    name: str
    orig_point: list[float]  # [norm_x, norm_y] in original photograph
    rect_point: list[float]  # [norm_x, norm_y] in rectified workpiece space
    roundtrip_point: list[float]  # [norm_x, norm_y] mapped back to original
    pixel_error: float  # Euclidean distance in original camera pixels
    norm_error: float  # Euclidean distance in normalized [0, 1] coordinates


class MappingDiagnosticResponse(BaseModel):
    status: str = "ok"
    image_dimensions: list[int]  # [width, height]
    rectified_dimensions: list[int]  # [width, height]
    num_points_tested: int
    max_pixel_error: float
    mean_pixel_error: float
    max_norm_error: float
    board_plane_notice: str
    points: list[MappingDiagnosticPoint]
    diagnostic_visualization_png_base64: str


# ==============================================================================
# Benchmarking & Ground Truth & Operational Statistics
# ==============================================================================

class GroundTruthUpdateRequest(BaseModel):
    expected_verdict: InspectionVerdict
    notes: str = ""


class AccuracyReportResponse(BaseModel):
    total_inspections: int
    ground_truth_count: int
    pass_count: int
    reject_count: int
    review_count: int
    accuracy: float | None = None
    confusion_matrix: dict[str, dict[str, int]]


class TrendPoint(BaseModel):
    date: str
    total_attempts: int
    completed: int
    pass_count: int
    reject_count: int
    review_count: int
    error_count: int


class FailedCheckStat(BaseModel):
    check_id: str
    description: str
    failure_count: int
    failure_rate_percent: float


class RecentInspectionSummary(BaseModel):
    inspection_id: str
    profile_id: str
    profile_version: int
    created_at: str
    verdict: InspectionVerdict
    inspection_state: str
    total_latency_ms: float
    ground_truth_verdict: str | None = None


class GroundTruthMetrics(BaseModel):
    total_labelled: int
    total_unlabelled: int
    accuracy_percent: float | None = None
    false_accept_count: int
    false_accept_rate_percent: float | None = None
    false_reject_count: int
    false_reject_rate_percent: float | None = None
    review_coverage_percent: float | None = None
    error_coverage_percent: float | None = None
    confusion_matrix: dict[str, dict[str, int]]


class OperationalStatisticsResponse(BaseModel):
    selected_range_days: int | None = None
    total_attempts: int
    completed_inspections: int
    processing_errors_count: int
    pass_count: int
    reject_count: int
    review_count: int
    pass_rate_percent: float | None = None
    reject_rate_percent: float | None = None
    review_rate_percent: float | None = None
    avg_inference_latency_ms: float | None = None
    avg_total_latency_ms: float | None = None
    volume_trends: list[TrendPoint]
    frequently_failed_checks: list[FailedCheckStat]
    recent_inspections: list[RecentInspectionSummary]
    ground_truth: GroundTruthMetrics


# ==============================================================================
# Legacy Anomaly Detection Compatibility Schemas
# ==============================================================================

class InspectionResult(BaseModel):
    category: str
    model_backend: str
    is_defective: bool
    anomaly_score: float
    heatmap_png_base64: str | None = None
    defect_description: str | None = None
    likely_cause: str | None = None
    severity: str | None = None
    explanation_source: str = "none"
    inference_latency_ms: float
    explanation_latency_ms: float | None = None


class CategoryMetrics(BaseModel):
    category: str
    model_backend: str
    image_auroc: float | None = None
    pixel_auroc: float | None = None
    eval_time_s: float | None = None
    trained: bool = True


class MetricsResponse(BaseModel):
    categories: list[CategoryMetrics]
