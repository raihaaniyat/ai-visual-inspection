"""
Tests for Inspection Engine Failure Categorization, Error Reporting,
Workpiece Corner Geometry, and Presentation Guarantees.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.schemas import CheckItem, WorkpieceCorners, NormalizedBox
from backend.gemini_inspector import (
    classify_inspection_error,
    validate_workpiece_corners,
    build_error_inspection_response,
    run_workpiece_inspection,
)


def test_classify_inspection_errors():
    """Verify exact categorization of exceptions into 6 neutral categories."""
    # 1. Authentication / Configuration error
    exc_401 = Exception("401 API_KEY_INVALID: User api key is invalid")
    cat_401, code_401 = classify_inspection_error(exc_401, "provider_call")
    assert cat_401 == "Authentication/configuration error"
    assert code_401 == 401

    # Missing API key configuration
    cat_cfg, _ = classify_inspection_error(None, "configuration")
    assert cat_cfg == "Authentication/configuration error"

    # 2. Quota or rate limit
    exc_429 = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded or quota exhausted")
    cat_429, code_429 = classify_inspection_error(exc_429, "provider_call")
    assert cat_429 == "Quota or rate limit"
    assert code_429 == 429

    # 3. Provider unavailable
    exc_503 = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")
    cat_503, code_503 = classify_inspection_error(exc_503, "provider_call")
    assert cat_503 == "Provider unavailable"
    assert code_503 == 503

    # 4. Request timeout
    exc_timeout = TimeoutError("DeadlineExceeded: The socket timed out after 30000ms")
    cat_to, code_to = classify_inspection_error(exc_timeout, "provider_call")
    assert cat_to == "Request timeout"
    assert code_to == 408

    # 5. Invalid inspection response
    exc_json = json.JSONDecodeError("Expecting value", "raw_output", 0)
    cat_json, _ = classify_inspection_error(exc_json, "response_parsing")
    assert cat_json == "Invalid inspection response"

    # 6. Image preparation failure
    exc_img = ValueError("Perspective transform failed: points collinear")
    cat_prep, _ = classify_inspection_error(exc_img, "image_preparation")
    assert cat_prep == "Image preparation failure"


def test_failed_inspection_reporting_requirements():
    """
    Requirement 2:
    - Set inspection state to ERROR.
    - Set the assembly verdict to REVIEW.
    - Mark required checks NOT_EVALUATED.
    - Do not claim the model inspected them.
    - Clear findings and overlays from previous runs.
    - Reserve UNCERTAIN for a successful inspection where visual evidence is ambiguous.
    """
    checklist = [
        CheckItem(
            check_id="chk_count",
            check_type="count",
            description="Required parts present",
            expected_requirement="All parts installed",
        ),
        CheckItem(
            check_id="chk_seat",
            check_type="presence",
            description="Mounting fastener seated",
            expected_requirement="Fastener seated",
        ),
    ]

    res = build_error_inspection_response(
        profile_id="prof_test",
        profile_version=1,
        checklist=checklist,
        request_id="req_test_123",
        stage="provider_call",
        failure_category="Provider unavailable",
        latency_ms=45.0,
    )

    assert res.inspection_state == "ERROR"
    assert res.failure_category == "Provider unavailable"
    assert res.verdict == "REVIEW"
    assert res.counts_summary["total_checks"] == 2
    assert res.counts_summary["not_evaluated"] == 2
    assert res.counts_summary["satisfied"] == 0
    assert res.counts_summary["failed"] == 0
    assert res.counts_summary["uncertain"] == 0  # UNCERTAIN must NOT be set on error!

    # Check each individual finding
    for f in res.findings:
        assert f.status == "not_evaluated"
        assert f.observed_state_or_count == "Not evaluated"
        assert f.rectified_box is None
        assert f.original_polygon is None
        assert "not evaluated" in f.evidence.lower()

    # Overlays must be cleared
    assert res.opencv_diff_png_base64 is None


def test_workpiece_corners_validation():
    """
    Requirement 3:
    Detect when corner handles enclose only a narrow central strip
    rather than the complete assembly board.
    """
    # 1. Narrow central strip (as seen in the failure screenshot: x from 0.33 to 0.70, width 0.37)
    narrow_strip_corners = WorkpieceCorners(
        top_left=[0.45, 0.20],
        top_right=[0.55, 0.20],
        bottom_right=[0.55, 0.80],
        bottom_left=[0.45, 0.80],
    )
    is_valid, warning = validate_workpiece_corners(narrow_strip_corners)
    assert not is_valid
    assert warning is not None
    assert "strip" in warning.lower() or "too small" in warning.lower() or "aspect ratio" in warning.lower()

    # 2. Complete board corners (full board, e.g. 0.1 to 0.9)
    valid_board_corners = WorkpieceCorners(
        top_left=[0.15, 0.15],
        top_right=[0.85, 0.15],
        bottom_right=[0.85, 0.85],
        bottom_left=[0.15, 0.85],
    )
    is_valid_board, warning_board = validate_workpiece_corners(valid_board_corners)
    assert is_valid_board
    assert warning_board is None


def test_inspection_provider_unavailability_handling():
    """
    Tests that a 503 UNAVAILABLE or high-demand error during inspection returns
    state=ERROR, verdict=REVIEW, category='Provider unavailable', checks=not_evaluated.
    """
    ref_img = Image.new("RGB", (300, 300), color="white")
    test_img = Image.new("RGB", (300, 300), color="white")
    checklist = [
        CheckItem(
            check_id="chk_part_a",
            check_type="presence",
            description="Part A presence",
            expected_requirement="Part A installed",
        )
    ]

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
        )
        mock_client_cls.return_value = mock_instance

        res = run_workpiece_inspection(
            reference_image=ref_img,
            test_image=test_img,
            profile_id="prof_general_inspection",
            profile_version=1,
            checklist=checklist,
            regions=[],
        )

        assert res.inspection_state == "ERROR"
        assert res.verdict == "REVIEW"
        assert res.failure_category == "Provider unavailable"
        assert res.counts_summary["not_evaluated"] == 1
        assert res.counts_summary["uncertain"] == 0
        assert res.findings[0].status == "not_evaluated"


def test_inspection_invalid_credentials_handling():
    """
    Tests that an invalid API key error returns
    state=ERROR, verdict=REVIEW, category='Authentication/configuration error', checks=not_evaluated.
    """
    ref_img = Image.new("RGB", (300, 300), color="white")
    test_img = Image.new("RGB", (300, 300), color="white")
    checklist = [
        CheckItem(
            check_id="chk_part_a",
            check_type="presence",
            description="Part A presence",
            expected_requirement="Part A installed",
        )
    ]

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception(
            "400 API_KEY_INVALID: API key not valid. Please pass a valid API key."
        )
        mock_client_cls.return_value = mock_instance

        res = run_workpiece_inspection(
            reference_image=ref_img,
            test_image=test_img,
            profile_id="prof_general_inspection",
            profile_version=1,
            checklist=checklist,
            regions=[],
        )

        assert res.inspection_state == "ERROR"
        assert res.verdict == "REVIEW"
        assert res.failure_category == "Authentication/configuration error"
        assert res.counts_summary["not_evaluated"] == 1
        assert res.findings[0].status == "not_evaluated"


def test_inspection_malformed_json_response_handling():
    """
    Tests that a malformed / non-JSON model response returns
    state=ERROR, verdict=REVIEW, category='Invalid inspection response', checks=not_evaluated.
    """
    ref_img = Image.new("RGB", (300, 300), color="white")
    test_img = Image.new("RGB", (300, 300), color="white")
    checklist = [
        CheckItem(
            check_id="chk_part_a",
            check_type="presence",
            description="Part A presence",
            expected_requirement="Part A installed",
        )
    ]

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "I am unable to inspect this image because of lighting issues."  # Malformed non-JSON
        mock_instance.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = mock_instance

        res = run_workpiece_inspection(
            reference_image=ref_img,
            test_image=test_img,
            profile_id="prof_general_inspection",
            profile_version=1,
            checklist=checklist,
            regions=[],
        )

        assert res.inspection_state == "ERROR"
        assert res.verdict == "REVIEW"
        assert res.failure_category == "Invalid inspection response"
        assert res.counts_summary["not_evaluated"] == 1
        assert res.findings[0].status == "not_evaluated"
