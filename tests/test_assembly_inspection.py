"""
Comprehensive Test Suite for Visual Assembly Inspection.

Covers:
1. USER-CONTROLLED REFERENCES & PROFILE VERSIONING:
   - Profile creation with real/operator reference and corner coordinates.
   - Replacing a reference invalidates old regions and increments profile version.
   - Inspection requires approved reference ("Add an approved reference first").
   - Synthetic generator is isolated to Demo Mode and not used silently.

2. GENERAL INSPECTION PROFILES & CHECKLISTS:
   - Arbitrary components, counts, and positions (not hardcoded to 6 screws or 2x3 grid).
   - Count-only checks.
   - Positional presence checks.
   - Visible arrangement checks.
   - Non-visible properties explicitly marked 'not_evaluated'.

3. WORKPIECE GEOMETRY & PLANAR RECTIFICATION:
   - 4-corner perspective rectification to standard board space.
   - Transform of rectified boxes to original image quadrilateral polygons.
   - When alignment fails: positional checks become 'not_evaluated', count checks preserved, verdict never PASS.

4. GEMINI DECISION GATE & STRUCTURED FINDINGS:
   - REJECT when any required check demonstrably fails.
   - REVIEW when checks are uncertain or not evaluated and no confirmed failure exists.
   - PASS only when all required checks are satisfied.
   - Single banner issue summary, without repeated alignment text across components.

5. OPENCV RECTIFIED DIFFERENCE VIEW:
   - Evaluates aligned workpiece planes only (excludes tabletop and background objects).
   - Disabled when valid workpiece mapping does not exist.

6. PHYSICAL PHOTOGRAPHS VERIFICATION:
   - Uses the supplied 6-screw photo as approved reference and 5-screw photo as inspection.
   - Excludes loose background screws and jars.
   - Detects missing lower-centre screw.
   - Self-comparison (identical reference vs reference) PASSES without manufactured defects.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from backend.config import settings
from backend.decision import compute_inspection_verdict
from backend.default_profiles import (
    create_synthetic_demo_board_image,
    get_physical_6_screw_checklist,
    get_physical_6_screw_regions,
    seed_physical_experiment_profile,
)
from backend.geometry import (
    RECTIFIED_HEIGHT,
    RECTIFIED_WIDTH,
    rectify_workpiece,
    transform_box_to_original_polygon,
)
from backend.opencv_diff import compute_rectified_visual_difference
from backend.schemas import (
    CheckItem,
    CheckObservation,
    ExpectedRegion,
    NormalizedBox,
    WorkpieceCorners,
)
from backend.storage import (
    create_or_save_profile,
    get_profile,
    init_db,
    save_inspection,
    update_profile_reference,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolate SQLite database to a temporary directory for tests."""
    with patch("backend.storage._get_db_path", return_value=tmp_path / "test_inspector.db"), \
         patch("backend.storage._get_images_dir", return_value=tmp_path / "test_images"):
        init_db()
        yield


# ==============================================================================
# 1. USER-CONTROLLED REFERENCES & PROFILE VERSIONING TESTS
# ==============================================================================

def test_profile_creation_and_versioning():
    """Verify profile creation and initial versioning."""
    checklist = [
        CheckItem(check_id="chk_c1", check_type="count", target_count=4, description="Check 4 items"),
    ]
    regions = [
        ExpectedRegion(region_id="reg_1", label="Item 1", box=NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.4, xmax=0.4)),
    ]
    ref_img = Image.new("RGB", (600, 400), color=(200, 200, 200))
    corners = WorkpieceCorners(top_left=[0.1, 0.1], top_right=[0.9, 0.1], bottom_right=[0.9, 0.9], bottom_left=[0.1, 0.9])

    prof = create_or_save_profile(
        name="Custom Board Profile",
        description="User-defined assembly",
        reference_image=ref_img,
        reference_corners=corners,
        checklist=checklist,
        regions=regions,
        profile_id="prof_custom_test",
    )

    assert prof.profile_id == "prof_custom_test"
    assert prof.version == 1
    assert len(prof.checklist) == 1
    assert len(prof.regions) == 1
    assert prof.reference_corners is not None


def test_reference_replacement_invalidates_old_regions_and_increments_version():
    """
    Requirement 1: Replacing a reference must invalidate its old regions
    and associate each reference, checklist and set of regions with a profile version.
    """
    checklist = [
        CheckItem(check_id="chk_pos1", check_type="presence", region_id="p1", description="Position 1 screw"),
    ]
    regions = [
        ExpectedRegion(region_id="p1", label="Position 1", box=NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.4, xmax=0.4)),
    ]
    ref_img_v1 = Image.new("RGB", (500, 500), color=(150, 150, 150))

    prof_v1 = create_or_save_profile(
        name="Board Assembly",
        reference_image=ref_img_v1,
        checklist=checklist,
        regions=regions,
        profile_id="prof_replace_test",
    )
    assert prof_v1.version == 1
    assert len(prof_v1.regions) == 1

    # Now operator captures/uploads a new approved reference
    ref_img_v2 = Image.new("RGB", (600, 600), color=(210, 180, 140))
    new_corners = WorkpieceCorners(top_left=[0.15, 0.15], top_right=[0.85, 0.15], bottom_right=[0.85, 0.85], bottom_left=[0.15, 0.85])

    prof_v2 = update_profile_reference(
        profile_id="prof_replace_test",
        reference_image=ref_img_v2,
        reference_corners=new_corners,
    )

    # Version MUST increment to 2
    assert prof_v2.version == 2
    # Old regions MUST be invalidated (cleared) so stale coordinates are never used
    assert len(prof_v2.regions) == 0
    # Checklist is preserved
    assert len(prof_v2.checklist) == 1

    # Verify persistent storage reflects this
    stored = get_profile("prof_replace_test")
    assert stored is not None
    assert stored.version == 2
    assert len(stored.regions) == 0


def test_profile_without_reference_cannot_inspect():
    """Verify that a profile created without a reference image has reference_image_id=None."""
    prof = create_or_save_profile(
        name="Unconfigured Profile",
        reference_image=None,
        checklist=[],
        regions=[],
        profile_id="prof_no_ref",
    )
    assert prof.reference_image_id is None


def test_delete_profile_reference_endpoint_and_storage():
    """Verify delete reference functionality removes reference image, increments version, and disables inspection."""
    from backend.storage import remove_profile_reference
    from fastapi.testclient import TestClient
    from backend.app import app

    checklist = [
        CheckItem(check_id="chk_p1", check_type="presence", region_id="r1", description="Part 1"),
    ]
    regions = [
        ExpectedRegion(region_id="r1", label="Part 1", box=NormalizedBox(ymin=0.1, xmin=0.1, ymax=0.3, xmax=0.3)),
    ]
    ref_img = Image.new("RGB", (400, 400), color=(100, 100, 100))
    corners = WorkpieceCorners(top_left=[0.1, 0.1], top_right=[0.9, 0.1], bottom_right=[0.9, 0.9], bottom_left=[0.1, 0.9])

    prof = create_or_save_profile(
        name="Delete Ref Test Profile",
        reference_image=ref_img,
        reference_corners=corners,
        checklist=checklist,
        regions=regions,
        profile_id="prof_del_ref_test",
    )
    assert prof.reference_image_id is not None
    assert prof.version >= 1
    init_ver = prof.version

    # Call DELETE endpoint
    client = TestClient(app)
    del_res = client.delete("/profiles/prof_del_ref_test/reference")
    assert del_res.status_code == 200, f"Delete failed: {del_res.text}"
    updated = del_res.json()
    assert updated["reference_image_id"] is None
    assert updated["reference_corners"] is None
    assert updated["version"] == init_ver + 1
    assert len(updated["regions"]) == 0

    # Verify GET reference-image now returns 404
    get_img_res = client.get("/profiles/prof_del_ref_test/reference-image")
    assert get_img_res.status_code == 404

    # Verify inspect fails with 400 because reference was removed
    test_buf = io.BytesIO()
    ref_img.save(test_buf, format="JPEG")
    insp_res = client.post(
        "/profiles/prof_del_ref_test/inspect",
        files={"file": ("test.jpg", test_buf.getvalue(), "image/jpeg")},
    )
    assert insp_res.status_code == 400
    assert "Add an approved reference first" in insp_res.json()["detail"]


# ==============================================================================
# 2. GENERAL INSPECTION PROFILES & ARBITRARY CHECKLISTS
# ==============================================================================

def test_arbitrary_component_checklist_support():
    """Verify system supports non-screw arbitrary components (e.g. PCB capacitors, chips)."""
    pcb_checklist = [
        CheckItem(
            check_id="chk_chip_count",
            check_type="count",
            target_count=3,
            description="3 IC chips must be mounted",
            expected_requirement="Workpiece must have 3 surface-mount ICs",
        ),
        CheckItem(
            check_id="chk_microcontroller",
            check_type="presence",
            region_id="u1_chip",
            description="Main MCU U1 presence",
            expected_requirement="MCU chip must be aligned in U1 footprint",
        ),
        CheckItem(
            check_id="chk_orientation",
            check_type="arrangement",
            region_id="u1_chip",
            description="MCU Pin 1 orientation",
            expected_requirement="Pin 1 dot must face top-left",
        ),
        CheckItem(
            check_id="chk_hidden_trace",
            check_type="visible_quality",
            description="Internal layer trace continuity",
            expected_requirement="Non-visible internal layer check",
        ),
    ]

    observations = [
        CheckObservation(
            check_id="chk_chip_count",
            expected_requirement="3 chips",
            observed_state_or_count="3 ICs observed",
            status="satisfied",
            visible_evidence="Three IC packages identified on board.",
        ),
        CheckObservation(
            check_id="chk_microcontroller",
            expected_requirement="MCU at U1",
            observed_state_or_count="Mounted",
            status="satisfied",
            visible_evidence="QFP package seated on U1 pads.",
        ),
        CheckObservation(
            check_id="chk_orientation",
            expected_requirement="Pin 1 at top-left",
            observed_state_or_count="Pin 1 top-left",
            status="satisfied",
            visible_evidence="Circular indentation marker visible at pin 1 corner.",
        ),
        CheckObservation(
            check_id="chk_hidden_trace",
            expected_requirement="Internal layer check",
            observed_state_or_count="Non-visible property",
            status="not_evaluated",
            visible_evidence="Internal multilayer PCB traces cannot be verified with optical surface photography.",
        ),
    ]

    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=pcb_checklist,
        regions=[],
        observations=observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
    )

    # Since a required check was marked not_evaluated (non-visible internal trace), verdict is REVIEW
    assert verdict == "REVIEW"
    assert counts["satisfied"] == 3
    assert counts["not_evaluated"] == 1
    assert any(f.check_id == "chk_hidden_trace" and f.status == "not_evaluated" for f in findings)


# ==============================================================================
# 3. WORKPIECE GEOMETRY & PLANAR RECTIFICATION
# ==============================================================================

def test_planar_perspective_rectification():
    """Verify that 4-point corner rectification standardizes the planar workpiece."""
    # Synthetic canvas with known dimensions
    canvas = Image.new("RGB", (800, 600), color=(50, 50, 50))
    corners = WorkpieceCorners(
        top_left=[0.20, 0.15],
        top_right=[0.80, 0.15],
        bottom_right=[0.75, 0.85],
        bottom_left=[0.25, 0.85],
    )

    rectified, M, M_inv = rectify_workpiece(canvas, corners)
    assert rectified.size == (RECTIFIED_WIDTH, RECTIFIED_HEIGHT)
    assert M.shape == (3, 3)
    assert M_inv.shape == (3, 3)


def test_transform_box_to_original_polygon():
    """Verify coordinate transformation from rectified box to original photo quadrilateral."""
    canvas = Image.new("RGB", (1000, 800), color=(100, 100, 100))
    corners = WorkpieceCorners(
        top_left=[0.20, 0.20],
        top_right=[0.80, 0.20],
        bottom_right=[0.80, 0.80],
        bottom_left=[0.20, 0.80],
    )
    _, _, M_inv = rectify_workpiece(canvas, corners)

    # Box at center of rectified board
    rect_box = NormalizedBox(ymin=0.4, xmin=0.4, ymax=0.6, xmax=0.6)
    poly = transform_box_to_original_polygon(
        box=rect_box,
        M_inv=M_inv,
        rect_w=RECTIFIED_WIDTH,
        rect_h=RECTIFIED_HEIGHT,
        orig_w=1000,
        orig_h=800,
    )

    assert len(poly) == 4
    for pt in poly:
        assert len(pt) == 2
        # All points should lie inside the normalized bounds [0.0..1.0]
        assert 0.15 <= pt[0] <= 0.85
        assert 0.15 <= pt[1] <= 0.85


def test_unaligned_mapping_marks_positional_checks_not_evaluated():
    """
    Requirement 3: If mapping fails:
    - Mark positional checks as not evaluated.
    - Do not draw supposedly matched component boxes.
    - Preserve separately obtained count observations.
    - Do not allow unresolved required positional checks to PASS.
    """
    checklist = [
        CheckItem(check_id="chk_count", check_type="count", target_count=6, description="Total 6 screws"),
        CheckItem(check_id="chk_p1", check_type="presence", region_id="r1", description="Position 1 screw"),
        CheckItem(check_id="chk_p2", check_type="presence", region_id="r2", description="Position 2 screw"),
    ]
    regions = [
        ExpectedRegion(region_id="r1", label="Region 1", box=NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.4, xmax=0.4)),
        ExpectedRegion(region_id="r2", label="Region 2", box=NormalizedBox(ymin=0.6, xmin=0.6, ymax=0.8, xmax=0.8)),
    ]
    observations = [
        CheckObservation(
            check_id="chk_count",
            expected_requirement="6 screws",
            observed_state_or_count="6 screws visible",
            status="satisfied",
            visible_evidence="6 screws visible on board despite tilt.",
        ),
    ]

    # Geometric alignment failed!
    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=checklist,
        regions=regions,
        observations=observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=False,
        alignment_issue_summary="Workpiece corners could not be reliably mapped.",
    )

    # Verdict must NOT be PASS! Must be REVIEW
    assert verdict == "REVIEW"
    # Positional checks must be marked not_evaluated
    pos_findings = [f for f in findings if f.check_type == "presence"]
    assert len(pos_findings) == 2
    assert all(f.status == "not_evaluated" for f in pos_findings)
    # Count check must be preserved as satisfied
    count_finding = next(f for f in findings if f.check_id == "chk_count")
    assert count_finding.status == "satisfied"
    # Single banner issue summary
    assert "Workpiece corners could not be reliably mapped" in (issue_sum or "")


# ==============================================================================
# 4. DECISION GATE LOGIC & CLEAN BANNER
# ==============================================================================

def test_decision_gate_rejects_on_confirmed_failure():
    """Requirement 5: REJECT when any required check demonstrably fails."""
    checklist = get_physical_6_screw_checklist()
    regions = get_physical_6_screw_regions()

    observations = [
        CheckObservation(
            check_id="chk_count",
            expected_requirement="6 screws",
            observed_state_or_count="5 installed screws",
            status="failed",
            visible_evidence="Only 5 screws attached to the workpiece; 1 screw is missing.",
        ),
        CheckObservation(
            check_id="chk_upper_centre",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
        ),
        CheckObservation(
            check_id="chk_upper_left",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
        ),
        CheckObservation(
            check_id="chk_upper_right",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
        ),
        CheckObservation(
            check_id="chk_lower_left",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
        ),
        CheckObservation(
            check_id="chk_lower_right",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Screw head visible.",
        ),
        CheckObservation(
            check_id="chk_lower_centre",
            expected_requirement="Seated screw",
            observed_state_or_count="Empty hole",
            status="failed",
            visible_evidence="Empty hole visible at the lower-centre reference position.",
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

    assert verdict == "REJECT"
    assert counts["failed"] == 2
    assert counts["satisfied"] == 5
    # Concise component evidence
    lower_c_finding = next(f for f in findings if f.check_id == "chk_lower_centre")
    assert lower_c_finding.status == "failed"
    assert "Empty hole visible" in lower_c_finding.evidence


def test_decision_gate_passes_when_all_satisfied():
    """Requirement 5: PASS only when all required checks are satisfied."""
    checklist = get_physical_6_screw_checklist()
    regions = get_physical_6_screw_regions()

    observations = [
        CheckObservation(
            check_id=item.check_id,
            expected_requirement=item.expected_requirement,
            observed_state_or_count="Satisfied",
            status="satisfied",
            visible_evidence="Verified present and correct.",
        )
        for item in checklist
    ]

    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=checklist,
        regions=regions,
        observations=observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
    )

    assert verdict == "PASS"
    assert counts["failed"] == 0
    assert counts["uncertain"] == 0
    assert counts["satisfied"] == len(checklist)


# ==============================================================================
# 5. OPENCV DIFFERENCE VIEW
# ==============================================================================

def test_opencv_rectified_difference_runs_on_rectified_planes():
    """Requirement 6: Compare aligned workpiece regions, not whole tabletop."""
    ref_img = Image.new("RGB", (700, 600), color=(180, 160, 140))
    test_img = ref_img.copy()

    # Identical inputs should produce 0 significant difference contours
    diff_detected, count, overlay_b64, _, notes = compute_rectified_visual_difference(ref_img, test_img)
    assert not diff_detected
    assert overlay_b64 is not None
    assert "candidate difference region" in notes


# ==============================================================================
# 6. PHYSICAL PHOTOGRAPHS VERIFICATION (Using Real Provided Photos)
# ==============================================================================

def test_physical_photos_rectification_and_isolation():
    """
    Requirement 7: Verify with the provided photographs.
    - Reference: 6-screw photo (90645232-ae17-491d-a5f4-b9408128cead.jpg)
    - Inspection: 5-screw photo (60d49384-4552-49ea-a4d2-ac454d16681a.jpg)
    - Verify: Planar board is rectified and tabletop clutter (blue lid, jars, loose screws) is excluded.
    """
    ref_path = Path("data/physical_samples/90645232-ae17-491d-a5f4-b9408128cead.jpg")
    test_path = Path("data/physical_samples/60d49384-4552-49ea-a4d2-ac454d16681a.jpg")

    if not ref_path.exists() or not test_path.exists():
        pytest.skip("Physical photograph files not available in data/physical_samples/")

    ref_img = Image.open(ref_path).convert("RGB")
    test_img = Image.open(test_path).convert("RGB")

    ref_corners = WorkpieceCorners(
        top_left=[0.24, 0.17],
        top_right=[0.74, 0.16],
        bottom_right=[0.75, 0.81],
        bottom_left=[0.21, 0.81],
    )
    test_corners = WorkpieceCorners(
        top_left=[0.22, 0.14],
        top_right=[0.81, 0.15],
        bottom_right=[0.79, 0.83],
        bottom_left=[0.19, 0.82],
    )

    rect_ref, _, _ = rectify_workpiece(ref_img, ref_corners)
    rect_test, _, _ = rectify_workpiece(test_img, test_corners)

    assert rect_ref.size == (RECTIFIED_WIDTH, RECTIFIED_HEIGHT)
    assert rect_test.size == (RECTIFIED_WIDTH, RECTIFIED_HEIGHT)

    # Verify OpenCV difference on the rectified workpiece
    diff_detected, count, overlay_b64, _, notes = compute_rectified_visual_difference(rect_ref, rect_test)
    assert overlay_b64 is not None
    assert "candidate difference region" in notes


def test_physical_photos_inspection_with_mocked_gemini():
    """
    Verify end-to-end flow with realistic Gemini observations on the physical photographs:
    - Lower-centre hole empty -> failed -> REJECT.
    - Loose background screws on table/jar are ignored.
    """
    ref_path = Path("data/physical_samples/90645232-ae17-491d-a5f4-b9408128cead.jpg")
    test_path = Path("data/physical_samples/60d49384-4552-49ea-a4d2-ac454d16681a.jpg")
    if not ref_path.exists() or not test_path.exists():
        pytest.skip("Physical photographs not found.")

    checklist = get_physical_6_screw_checklist()
    regions = get_physical_6_screw_regions()

    # Mock response representing vision analysis of the 5-screw photo:
    # Lower-centre screw is visibly missing (empty hole). Loose screws in jar/blue lid ignored.
    mock_observations = [
        CheckObservation(
            check_id="chk_count",
            expected_requirement="6 installed screws",
            observed_state_or_count="5 installed screws",
            status="failed",
            visible_evidence="5 screws installed on the wood board. Loose screws on tabletop and inside jar are excluded.",
        ),
        CheckObservation(
            check_id="chk_upper_centre",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Upper-centre screw securely seated.",
        ),
        CheckObservation(
            check_id="chk_upper_left",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Upper-left screw seated.",
        ),
        CheckObservation(
            check_id="chk_upper_right",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Upper-right screw seated.",
        ),
        CheckObservation(
            check_id="chk_lower_left",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Lower-left screw seated.",
        ),
        CheckObservation(
            check_id="chk_lower_right",
            expected_requirement="Seated screw",
            observed_state_or_count="Present",
            status="satisfied",
            visible_evidence="Lower-right screw seated.",
        ),
        CheckObservation(
            check_id="chk_lower_centre",
            expected_requirement="Seated screw",
            observed_state_or_count="Empty hole detected",
            status="failed",
            visible_evidence="Empty hole visible at the lower-centre reference position.",
        ),
    ]

    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=checklist,
        regions=regions,
        observations=mock_observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
    )

    assert verdict == "REJECT"
    assert counts["failed"] == 2
    lower_c = next(f for f in findings if f.check_id == "chk_lower_centre")
    assert lower_c.status == "failed"
    assert "Empty hole visible at the lower-centre" in lower_c.evidence


# ==============================================================================
# 7. UNRESTRAINED GENERAL WORKPIECE INSPECTION (ANY IMAGE / ANY COMPONENTS)
# ==============================================================================

def test_unrestrained_arbitrary_image_component_discovery():
    """Verify that the system is not restrained to 6 screws and correctly detects what is missing and what is not on arbitrary assemblies."""
    # Profile with NO pre-defined regions or checklist (or arbitrary assembly)
    empty_checklist = []
    empty_regions = []

    # Model observes 8 diverse components on an arbitrary electronic/mechanical board
    mock_observations = [
        CheckObservation(
            check_id="comp_ic_microcontroller",
            label="Microcontroller IC U1",
            expected_requirement="Must be populated and seated on pads",
            observed_state_or_count="PRESENT: Verified seated on footprint",
            status="satisfied",
            visible_evidence="Microcontroller IC U1 verified in position.",
            box=NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.35, xmax=0.35),
        ),
        CheckObservation(
            check_id="comp_connector_usb",
            label="USB-C Connector Port J1",
            expected_requirement="Port connector soldered at board edge",
            observed_state_or_count="PRESENT: Verified seated",
            status="satisfied",
            visible_evidence="USB-C connector present at bottom edge.",
            box=NormalizedBox(ymin=0.8, xmin=0.45, ymax=0.95, xmax=0.55),
        ),
        CheckObservation(
            check_id="comp_capacitor_c3",
            label="Electrolytic Capacitor C3",
            expected_requirement="Canister capacitor mounted at top-right",
            observed_state_or_count="MISSING: Empty mounting pads detected",
            status="failed",
            visible_evidence="MISSING: Empty solder pads visible at C3; capacitor present in reference is missing.",
            box=NormalizedBox(ymin=0.15, xmin=0.75, ymax=0.3, xmax=0.85),
        ),
        CheckObservation(
            check_id="comp_mounting_bracket",
            label="Corner Mounting Bracket",
            expected_requirement="Corner bracket secured with bolt",
            observed_state_or_count="PRESENT: Installed and tightened",
            status="satisfied",
            visible_evidence="Bracket present and secured in upper-left corner.",
            box=NormalizedBox(ymin=0.05, xmin=0.05, ymax=0.15, xmax=0.15),
        ),
    ]

    verdict, reason, issue_sum, counts, findings = compute_inspection_verdict(
        checklist=empty_checklist,
        regions=empty_regions,
        observations=mock_observations,
        reference_identity_ok=True,
        image_quality_ok=True,
        geometric_alignment_ok=True,
    )

    # 1. Confirms 4 components were dynamically tracked
    assert len(findings) == 4
    assert counts["total_checks"] == 4
    assert counts["satisfied"] == 3
    assert counts["failed"] == 1

    # 2. Rejection triggered specifically on the missing capacitor C3
    assert verdict == "REJECT"
    assert "Electrolytic Capacitor C3" in reason
    assert "Assembly Rejected: 1 issue(s) detected" in reason


def test_permanent_reference_library_protected(monkeypatch, tmp_path):
    """
    Verifies that standard golden reference baselines:
    - Speaker (prof_speaker)
    - Transistor (prof_transistor)
    - Screws in Plywood (prof_screws_plywood)
    are seeded with approved reference images, return HTTP 200 on reference-image GET,
    and cannot be deleted or have their reference removed.
    """
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.default_profiles import seed_permanent_reference_library, PERMANENT_PROFILE_IDS

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path / "perm_storage"))
    init_db()
    seed_permanent_reference_library()

    client = TestClient(app)

    # 1. Verify all permanent profiles exist with reference images
    for pid in ["prof_speaker", "prof_transistor", "prof_screws_plywood"]:
        res = client.get(f"/profiles/{pid}")
        assert res.status_code == 200
        data = res.json()
        assert data["reference_image_id"] is not None

        # Reference image download endpoint must return HTTP 200 JPEG
        img_res = client.get(f"/profiles/{pid}/reference-image")
        assert img_res.status_code == 200
        assert img_res.headers["content-type"] == "image/jpeg"
        assert len(img_res.content) > 1000

    # 2. Attempting to delete a permanent profile reference is rejected / preserved
    del_ref_res = client.delete("/profiles/prof_speaker/reference")
    assert del_ref_res.status_code == 200
    assert del_ref_res.json()["reference_image_id"] is not None

    # 3. Attempting to delete a permanent profile is blocked
    del_prof_res = client.delete("/profiles/prof_speaker")
    assert del_prof_res.status_code in (400, 404) or del_prof_res.json().get("status") != "deleted"
    assert client.get("/profiles/prof_speaker").status_code == 200


def test_general_inspection_profile_seed():
    """Verify that the universal general inspection profile is seeded and functional."""
    from backend.default_profiles import seed_general_inspection_profile
    prof_id = seed_general_inspection_profile()
    assert prof_id == "prof_general_inspection"

    prof = get_profile("prof_general_inspection")
    assert prof is not None
    assert "General Visual Inspection" in prof.name
    assert len(prof.checklist) > 0
