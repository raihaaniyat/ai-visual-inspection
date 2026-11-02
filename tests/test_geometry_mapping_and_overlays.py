"""
Tests for Coordinate Spaces, AI-Independent Geometry Mapping,
Separation of Expected Regions vs Detected Component Boxes,
and Continuous Reference Difference Maps.
"""
from __future__ import annotations

import base64
import numpy as np
import pytest
from PIL import Image

from backend.default_profiles import get_physical_6_screw_checklist, get_physical_6_screw_regions
from backend.decision import compute_inspection_verdict
from backend.geometry import (
    RECTIFIED_HEIGHT,
    RECTIFIED_WIDTH,
    compute_mapping_diagnostics,
    rectify_workpiece,
    transform_box_to_original_polygon,
    transform_orig_point_to_rect,
    transform_rect_point_to_orig,
)
from backend.opencv_diff import compute_rectified_visual_difference, extract_difference_regions
from backend.schemas import (
    CheckItem,
    CheckObservation,
    ComponentFinding,
    ExpectedRegion,
    NormalizedBox,
    WorkpieceCorners,
)


def test_round_trip_geometry_mapping_corners_and_edges():
    """
    Requirement 2: Verify round-trip mapping accuracy without AI:
    - 4 workpiece corners
    - Center point
    - Points near edges
    - Measures Euclidean pixel error and normalized error
    """
    # Create test image with non-square dimensions
    orig_w, orig_h = 900, 1600
    img = Image.new("RGB", (orig_w, orig_h), color=(140, 120, 100))

    corners = WorkpieceCorners(
        top_left=[0.24, 0.17],
        top_right=[0.74, 0.16],
        bottom_right=[0.75, 0.81],
        bottom_left=[0.21, 0.81],
    )

    diag = compute_mapping_diagnostics(img, corners)

    assert diag.status == "ok"
    assert diag.num_points_tested >= 12
    # Mathematical round-trip error for homography must be well below 0.05px
    assert diag.max_pixel_error < 0.05
    assert diag.mean_pixel_error < 0.01
    assert diag.max_norm_error < 1e-4

    # Verify board-plane disclaimer notice is explicitly included
    assert "board-plane homography" in diag.board_plane_notice.lower()
    assert "perspective parallax" in diag.board_plane_notice.lower()

    # Visualization base64 string must be populated
    assert len(diag.diagnostic_visualization_png_base64) > 1000

    # Verify specific point names
    point_names = [p.name for p in diag.points]
    assert "Top-Left Corner" in point_names
    assert "Top-Right Corner" in point_names
    assert "Bottom-Right Corner" in point_names
    assert "Bottom-Left Corner" in point_names
    assert "Workpiece Centre" in point_names
    assert "Top Edge Midpoint" in point_names


def test_scale_invariance_across_different_resolutions():
    """
    Requirement 2: Test mapping consistency across different image dimensions
    and frontend display sizes.
    """
    corners = WorkpieceCorners(
        top_left=[0.20, 0.15],
        top_right=[0.80, 0.15],
        bottom_right=[0.78, 0.82],
        bottom_left=[0.22, 0.82],
    )

    for w, h in [(1200, 1600), (800, 600), (500, 500)]:
        img = Image.new("RGB", (w, h), color=(100, 100, 100))
        diag = compute_mapping_diagnostics(img, corners, display_dimensions=[(w, h), (400, 300)])
        assert diag.max_pixel_error < 0.05
        assert diag.mean_pixel_error < 0.01


def test_separation_of_expected_regions_from_detections():
    """
    Requirement 3:
    - Dashed neutral box for operator-defined expected inspection region.
    - Solid box for actual observed component detection.
    - Missing components must NOT have a detected box.
    - Whole workpiece box must NEVER be used as a component detection.
    """
    checklist = [
        CheckItem(check_id="chk_count", check_type="count", target_count=6, description="Total 6 installed screws"),
        CheckItem(check_id="chk_top", check_type="presence", region_id="r_top", description="Top screw"),
        CheckItem(check_id="chk_bottom", check_type="presence", region_id="r_bottom", description="Bottom screw"),
    ]
    regions = [
        ExpectedRegion(region_id="r_top", label="Top Screw", box=NormalizedBox(ymin=0.15, xmin=0.45, ymax=0.25, xmax=0.55)),
        ExpectedRegion(region_id="r_bottom", label="Bottom Screw", box=NormalizedBox(ymin=0.75, xmin=0.45, ymax=0.85, xmax=0.55)),
    ]

    observations = [
        # Count check returned whole-workpiece box (must be discarded!)
        CheckObservation(
            check_id="chk_count",
            observed_state_or_count="5 installed screws",
            status="failed",
            visible_evidence="5 screws installed; 1 screw is missing.",
            box=NormalizedBox(ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0),
        ),
        # Top screw is satisfied and detected
        CheckObservation(
            check_id="chk_top",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
            box=NormalizedBox(ymin=0.16, xmin=0.46, ymax=0.24, xmax=0.54),
        ),
        # Bottom screw is MISSING (empty hole)
        CheckObservation(
            check_id="chk_bottom",
            observed_state_or_count="Empty hole",
            status="failed",
            visible_evidence="Empty mounting hole detected; screw is absent.",
            box=None,
        ),
    ]

    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=checklist,
        regions=regions,
        observations=observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
    )

    count_f = next(f for f in findings if f.check_id == "chk_count")
    top_f = next(f for f in findings if f.check_id == "chk_top")
    bottom_f = next(f for f in findings if f.check_id == "chk_bottom")

    # 1. Whole workpiece box was rejected as a component box
    assert not count_f.has_detected_box
    assert count_f.detected_box_rectified is None

    # 2. Top screw (satisfied): has both expected region and detected box
    assert top_f.has_expected_region
    assert top_f.expected_region_rectified is not None
    assert top_f.has_detected_box
    assert top_f.detected_box_rectified is not None
    assert top_f.detected_box_rectified.ymin == 0.16

    # 3. Bottom screw (missing): has expected region, but NEVER a detected box!
    assert bottom_f.has_expected_region
    assert bottom_f.expected_region_rectified is not None
    assert not bottom_f.has_detected_box
    assert bottom_f.detected_box_rectified is None
    assert "r_bottom" in (bottom_f.component_id or "")


def test_reference_difference_map_continuous_overlay_and_legend():
    """
    Requirement 4:
    - Continuous heatmap with understandable legend.
    - Named 'Reference difference map'.
    - Only computed inside valid workpiece area.
    - Slat edges not converted into 20+ false defect boxes.
    """
    ref_img = Image.new("RGB", (700, 600), color=(180, 160, 140))
    test_img = ref_img.copy()

    diff_res = compute_rectified_visual_difference(ref_img, test_img)
    assert not diff_res.has_diff
    assert diff_res.count == 0
    assert diff_res.reliable
    assert "Reference difference map" in diff_res.notes

    # Decode overlay image and verify dimensions match
    overlay_bytes = base64.b64decode(diff_res.overlay_b64)
    assert len(overlay_bytes) > 5000

    # Ensure extract_difference_regions returns empty list for identical inputs
    boxes = extract_difference_regions(ref_img, test_img)
    assert len(boxes) == 0


def test_unreliable_reference_difference_when_disparity_extreme():
    """
    Requirement 4: If lighting or disparity is extreme, mark comparison unreliable.
    """
    # Extremely bright vs extremely dark
    ref_img = Image.new("RGB", (700, 600), color=(250, 250, 250))
    test_img = Image.new("RGB", (700, 600), color=(10, 10, 10))

    diff_res = compute_rectified_visual_difference(ref_img, test_img)
    assert not diff_res.reliable
    assert "unreliable" in diff_res.notice.lower()


def test_coordinate_space_data_contract():
    """
    Requirement 1: Verify explicit coordinate spaces representation in data contract.
    """
    f = ComponentFinding(
        check_id="chk_screw_1",
        label="Screw 1",
        check_type="presence",
        expected_requirement="Seated screw",
        observed_state_or_count="Present",
        status="satisfied",
        evidence="Observed",
        has_expected_region=True,
        expected_region_rectified=NormalizedBox(ymin=0.1, xmin=0.1, ymax=0.2, xmax=0.2),
        has_detected_box=True,
        detected_box_rectified=NormalizedBox(ymin=0.11, xmin=0.11, ymax=0.21, xmax=0.21),
    )

    assert f.coordinate_spaces["expected_region_rectified"] == "rectified_workpiece"
    assert f.coordinate_spaces["expected_polygon_original"] == "original_inspection_image"
    assert f.coordinate_spaces["detected_box_rectified"] == "rectified_workpiece"
    assert f.coordinate_spaces["detected_polygon_original"] == "original_inspection_image"


def test_geometry_mapping_diagnostic_api_endpoint():
    """
    Requirement 2: Verify geometry mapping endpoint returns 200 with complete diagnostics.
    """
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    resp = client.post("/geometry/mapping-diagnostic")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["num_points_tested"] >= 12
    assert data["max_pixel_error"] < 0.05
    assert "board-plane homography" in data["board_plane_notice"].lower()
    assert len(data["diagnostic_visualization_png_base64"]) > 500

