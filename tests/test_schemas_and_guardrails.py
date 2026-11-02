"""
Unit tests that don't need a GPU, anomalib, or Ollama installed - they
cover the parts of the logic that are pure Python: the guardrail in
vlm_explainer._try_parse_json / _NO_DEFECT_PATTERNS, and the schema
validation. Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.vlm_explainer import _NO_DEFECT_PATTERNS, _try_parse_json  # noqa: E402
from backend.schemas import InspectionResult  # noqa: E402


def test_parse_valid_json():
    text = '{"defect_type": "scratch", "likely_cause": "handling", "severity": "medium", "note": "x"}'
    parsed = _try_parse_json(text)
    assert parsed is not None
    assert parsed["defect_type"] == "scratch"


def test_parse_json_wrapped_in_fence():
    text = "```json\n{\"defect_type\": \"crack\", \"severity\": \"high\", \"note\": \"x\"}\n```"
    parsed = _try_parse_json(text)
    assert parsed is not None
    assert parsed["severity"] == "high"


def test_parse_garbage_returns_none():
    assert _try_parse_json("I cannot help with that.") is None


def test_no_defect_pattern_catches_contradiction():
    assert _NO_DEFECT_PATTERNS.search("The part looks fine, no defect visible.")
    assert not _NO_DEFECT_PATTERNS.search("A clear scratch is visible on the surface.")


def test_inspection_result_requires_score_and_verdict():
    result = InspectionResult(
        category="bottle",
        model_backend="patchcore",
        is_defective=True,
        anomaly_score=0.87,
        inference_latency_ms=12.3,
    )
    assert result.is_defective is True
    assert result.explanation_source == "none"  # default before VLM runs
