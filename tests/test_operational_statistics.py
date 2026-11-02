"""
Unit & Integration Tests for:
1. Operational Statistics Endpoint (/reports/statistics)
2. Durable Image BLOB Storage (Serverless container persistence)
3. Rate Limiting & Upload Size Controls (Public Vercel deployment guards)
4. Ground-Truth Benchmarking Separation (Accuracy strictly on labelled data)
"""
from __future__ import annotations

import io
import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app, _rate_limit_records
from backend.config import settings
from backend.schemas import (
    CheckItem,
    ComponentFinding,
    NormalizedBox,
    WorkpieceCorners,
)
from backend.storage import (
    init_db,
    save_image_file,
    load_image_file,
    save_inspection,
    update_ground_truth,
    get_operational_statistics,
    _get_images_dir,
)


@pytest.fixture(autouse=True)
def reset_rate_limits():
    _rate_limit_records.clear()
    yield
    _rate_limit_records.clear()


def test_durable_image_blob_persistence_across_disk_wipe():
    """
    Verifies that images saved to durable database storage can still be loaded
    even if local disk cache is wiped (simulating serverless container restart on Vercel).
    """
    init_db()
    # Create test image
    img = Image.new("RGB", (120, 120), color=(50, 150, 200))
    file_id = save_image_file(img, prefix="test_durable")

    # Verify disk cache exists
    cached_path = _get_images_dir() / file_id
    assert cached_path.exists()

    # Simulate serverless container restart / local disk wipe
    cached_path.unlink()
    assert not cached_path.exists()

    # Load image: should fall back to database BLOB
    loaded_img = load_image_file(file_id)
    assert loaded_img is not None
    assert loaded_img.size == (120, 120)
    # Check pixel value
    r, g, b = loaded_img.getpixel((60, 60))
    assert (r, g, b) == (50, 150, 200)


def test_image_deduplication_prevents_storage_bloat(monkeypatch, tmp_path):
    """
    Verifies that saving the identical image multiple times does NOT insert duplicate rows
    or re-upload binary BLOBs, preventing Supabase and local DB storage from getting choked out.
    """
    import sqlite3
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path / "dedup_storage"))
    init_db()

    # Create identical test image
    img1 = Image.new("RGB", (100, 100), color=(88, 120, 240))
    img1_identical = Image.new("RGB", (100, 100), color=(88, 120, 240))
    img2_different = Image.new("RGB", (100, 100), color=(255, 0, 0))

    # Save identical image multiple times (different calls, even different prefixes)
    id1 = save_image_file(img1, prefix="ref")
    id2 = save_image_file(img1_identical, prefix="ref")
    id3 = save_image_file(img1, prefix="test")

    # All identical calls must reuse the same image ID
    assert id1 == id2
    assert id1 == id3

    # Check database row count: must only have 1 row for this image
    db_path = settings.storage_dir_path / "inspector.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stored_images")
        count_after_identical = cursor.fetchone()[0]
        assert count_after_identical == 1

        cursor.execute("SELECT content_hash FROM stored_images WHERE image_id = ?", (id1,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert len(row[0]) == 64  # valid SHA-256 hex string

    # Now save a different image
    id_diff = save_image_file(img2_different, prefix="ref")
    assert id_diff != id1

    # Row count should now be exactly 2
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stored_images")
        count_after_different = cursor.fetchone()[0]
        assert count_after_different == 2

    # Verify both images can be loaded cleanly
    loaded1 = load_image_file(id1)
    loaded_diff = load_image_file(id_diff)
    assert loaded1 is not None and loaded1.getpixel((10, 10)) == (88, 120, 240)
    assert loaded_diff is not None and loaded_diff.getpixel((10, 10)) == (255, 0, 0)


def test_operational_statistics_empty_state(monkeypatch, tmp_path):
    """
    Verifies that when no inspections exist, empty state is returned cleanly
    without division by zero or errors.
    """
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path / "empty_storage"))
    init_db()

    stats = get_operational_statistics()
    assert stats.total_attempts == 0
    assert stats.completed_inspections == 0
    assert stats.processing_errors_count == 0
    assert stats.pass_count == 0
    assert stats.reject_count == 0
    assert stats.review_count == 0
    assert stats.pass_rate_percent is None
    assert stats.reject_rate_percent is None
    assert stats.review_rate_percent is None
    assert stats.avg_inference_latency_ms is None
    assert stats.volume_trends == []
    assert stats.frequently_failed_checks == []
    assert stats.recent_inspections == []
    assert stats.ground_truth.total_labelled == 0
    assert stats.ground_truth.total_unlabelled == 0
    assert stats.ground_truth.accuracy_percent is None


def test_operational_statistics_computation_and_ground_truth_isolation(monkeypatch, tmp_path):
    """
    Verifies:
    1. Processing errors are kept outside product rejections.
    2. Pass, Reject, Review rates calculated only among completed inspections.
    3. Failure frequency ranked correctly.
    4. Accuracy, FAR, FRR calculated strictly on independently labelled ground truth.
    """
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path / "stats_storage"))
    init_db()

    # 1. Save a PASS inspection (unlabelled)
    f_pass = [
        ComponentFinding(
            check_id="chk_c1",
            region_id="c1",
            label="Component 1",
            description="Component 1",
            check_type="presence",
            expected_requirement="Verify component is present",
            observed_state_or_count="Present and aligned",
            evidence="Visual confirmation",
            status="satisfied",
            confidence=0.98,
        )
    ]
    insp_pass = save_inspection(
        profile_id="prof_test",
        profile_version=1,
        verdict="PASS",
        verdict_reason="All components present",
        findings=f_pass,
        counts_summary={"total": 1, "satisfied": 1, "failed": 0, "uncertain": 0},
        engine="Inspection Engine",
        model_identifier="Inspection Engine",
        inference_latency_ms=100.0,
        total_latency_ms=150.0,
        inspection_state="COMPLETED",
    )

    # 2. Save a REJECT inspection (labelled ground truth: REJECT)
    f_rej = [
        ComponentFinding(
            check_id="chk_c1",
            region_id="c1",
            label="Component 1",
            description="Component 1",
            check_type="presence",
            expected_requirement="Verify component is present",
            observed_state_or_count="Missing",
            evidence="Empty hole",
            status="failed",
            confidence=0.95,
        ),
        ComponentFinding(
            check_id="chk_c2",
            region_id="c2",
            label="Component 2",
            description="Component 2",
            check_type="presence",
            expected_requirement="Verify component is present",
            observed_state_or_count="Missing",
            evidence="Empty hole",
            status="failed",
            confidence=0.90,
        ),
    ]
    insp_rej = save_inspection(
        profile_id="prof_test",
        profile_version=1,
        verdict="REJECT",
        verdict_reason="Missing components",
        findings=f_rej,
        counts_summary={"total": 2, "satisfied": 0, "failed": 2, "uncertain": 0},
        engine="Inspection Engine",
        model_identifier="Inspection Engine",
        inference_latency_ms=200.0,
        total_latency_ms=250.0,
        inspection_state="COMPLETED",
        ground_truth_verdict="REJECT",
    )

    # 3. Save a REVIEW inspection (labelled ground truth: PASS -> this is not a false accept or reject, but uncertain)
    f_rev = [
        ComponentFinding(
            check_id="chk_c1",
            region_id="c1",
            label="Component 1",
            description="Component 1",
            check_type="presence",
            expected_requirement="Verify component is present",
            observed_state_or_count="Obscured",
            evidence="Shadow or partial occlusion",
            status="uncertain",
            confidence=0.50,
        )
    ]
    insp_rev = save_inspection(
        profile_id="prof_test",
        profile_version=1,
        verdict="REVIEW",
        verdict_reason="Ambiguous edge",
        findings=f_rev,
        counts_summary={"total": 1, "satisfied": 0, "failed": 0, "uncertain": 1},
        engine="Inspection Engine",
        model_identifier="Inspection Engine",
        inference_latency_ms=150.0,
        total_latency_ms=200.0,
        inspection_state="COMPLETED",
        ground_truth_verdict="PASS",
    )

    # 4. Save a PROCESSING ERROR (e.g. Quota/Timeout) - must NOT count as a product defect
    insp_err = save_inspection(
        profile_id="prof_test",
        profile_version=1,
        verdict="REVIEW",
        verdict_reason="Provider timeout during inference",
        findings=[],
        counts_summary={},
        engine="Inspection Engine",
        model_identifier="Inspection Engine",
        inference_latency_ms=0.0,
        total_latency_ms=500.0,
        inspection_state="ERROR",
        failure_category="Request timeout",
    )

    stats = get_operational_statistics()
    assert stats.total_attempts == 4
    assert stats.completed_inspections == 3
    assert stats.processing_errors_count == 1  # Separated from product rejections

    # Completed inspection rates
    assert stats.pass_count == 1
    assert stats.reject_count == 1
    assert stats.review_count == 1
    assert stats.pass_rate_percent == 33.3
    assert stats.reject_rate_percent == 33.3
    assert stats.review_rate_percent == 33.3

    # Latencies
    assert stats.avg_inference_latency_ms == 150.0  # (100 + 200 + 150) / 3
    assert stats.avg_total_latency_ms == 200.0  # (150 + 250 + 200) / 3

    # Frequently failed checks
    assert len(stats.frequently_failed_checks) == 2
    top_failed = stats.frequently_failed_checks[0]
    assert top_failed.failure_count == 1
    assert top_failed.failure_rate_percent == 33.3

    # Ground Truth Benchmarking
    # Only 2 inspections were labelled (insp_rej: REJECT, insp_rev: PASS)
    # insp_pass and insp_err were unlabelled
    assert stats.ground_truth.total_labelled == 2
    assert stats.ground_truth.total_unlabelled == 2
    # Correct = insp_rej (verdict REJECT == gt REJECT). insp_rev had verdict REVIEW != gt PASS.
    # Accuracy = 1 / 2 = 50.0%
    assert stats.ground_truth.accuracy_percent == 50.0
    # False Accept: gt is REJECT, verdict is PASS -> 0
    assert stats.ground_truth.false_accept_count == 0
    # False Reject: gt is PASS, verdict is REJECT -> 0 (verdict was REVIEW, not REJECT)
    assert stats.ground_truth.false_reject_count == 0
    assert stats.ground_truth.review_coverage_percent == 50.0


def test_api_reports_statistics_endpoint():
    client = TestClient(app)
    response = client.get("/reports/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "total_attempts" in data
    assert "completed_inspections" in data
    assert "processing_errors_count" in data
    assert "pass_rate_percent" in data
    assert "ground_truth" in data
    assert "confusion_matrix" in data["ground_truth"]


def test_upload_payload_size_limit_rejection(monkeypatch):
    """Verifies that an image upload exceeding max size is rejected with HTTP 413."""
    client = TestClient(app)
    monkeypatch.setattr(settings, "max_upload_size_bytes", 1024)  # 1 KB limit for test

    # Create dummy bytes > 1024 bytes
    oversized_data = b"x" * 2048
    response = client.post(
        "/profiles/prof_general_inspection/inspect",
        files={"file": ("large.jpg", oversized_data, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


def test_rate_limiting_enforcement(monkeypatch):
    """Verifies that exceeding rate_limit_per_minute raises HTTP 429."""
    import backend.gemini_inspector
    monkeypatch.setattr(backend.gemini_inspector, "propose_reference_inventory", lambda **kwargs: [])
    client = TestClient(app)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)

    # 1st call
    client.post("/profiles/prof_general_inspection/propose-inventory")
    # 2nd call
    client.post("/profiles/prof_general_inspection/propose-inventory")
    # 3rd call should trigger rate limit
    response = client.post("/profiles/prof_general_inspection/propose-inventory")
    assert response.status_code == 429
    assert "rate limit exceeded" in response.json()["detail"].lower()


def test_inspectra_brand_logos_and_theme_assets():
    """Verifies that Inspectra dual-theme logos and dashboard are served cleanly."""
    client = TestClient(app)

    # 1. Root dashboard contains Inspectra logo elements and theme hooks
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "inspectra-light-mode.png" in res_root.text
    assert "inspectra-dark-mode.png" in res_root.text
    assert "data-theme" in res_root.text
    assert "toggleTheme" in res_root.text

    # 2. Both dark and light mode logo endpoints return valid PNGs with immutable caching
    res_dark = client.get("/assets/inspectra-dark-mode.png")
    assert res_dark.status_code == 200
    assert res_dark.headers["content-type"] == "image/png"
    assert "max-age" in res_dark.headers.get("cache-control", "")
    assert len(res_dark.content) > 10000

    res_light = client.get("/assets/inspectra-light-mode.png")
    assert res_light.status_code == 200
    assert res_light.headers["content-type"] == "image/png"
    assert "max-age" in res_light.headers.get("cache-control", "")
    assert len(res_light.content) > 10000


def test_api_key_automatic_fallback_on_quota_limit(monkeypatch):
    """
    Verifies that when the primary API key encounters a 429 / RESOURCE_EXHAUSTED quota error,
    the inspector automatically fails over to the backup API key and completes inspection.
    """
    from backend.gemini_inspector import get_configured_api_keys, run_workpiece_inspection
    from PIL import Image

    monkeypatch.setattr(settings, "gemini_api_key", "primary_key_123")
    monkeypatch.setattr(settings, "gemini_api_key_fallback", "backup_key_456")

    keys = get_configured_api_keys()
    assert keys == ["primary_key_123", "backup_key_456"]

    called_keys = []

    class FakeClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.models = self

        def generate_content(self, model: str, contents: list):
            called_keys.append(self.api_key)
            if self.api_key == "primary_key_123":
                # Simulate 429 quota exhaustion on primary key
                raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'GenerateContent'")
            
            # Backup key succeeds
            class FakeResp:
                text = json.dumps({
                    "reference_identity_ok": True,
                    "image_quality_ok": True,
                    "geometric_alignment_ok": True,
                    "overall_visible_count": 1,
                    "observations": [
                        {
                            "check_id": "chk_c1",
                            "region_id": "c1",
                            "observed_state_or_count": "Present",
                            "visible_evidence": "Clearly visible in designated position",
                            "status": "satisfied",
                            "confidence": 0.99
                        }
                    ]
                })
            return FakeResp()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", FakeClient)

    img = Image.new("RGB", (100, 100), (200, 200, 200))
    res = run_workpiece_inspection(
        reference_image=img,
        test_image=img,
        profile_id="prof_fallback_test",
        profile_version=1,
        checklist=[],
        regions=[],
    )

    # Primary key was tried first, failed on 429, backup key was called next and succeeded
    assert called_keys == ["primary_key_123", "backup_key_456"]
    assert res.verdict == "PASS"
    assert "All 1 inspection criteria satisfied" in res.verdict_reason
