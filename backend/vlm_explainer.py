"""
Natural-language explanation layer, powered by a local vision-language
model (Qwen2.5-VL) served through Ollama - same local-inference pattern
as AI-Portfolio-Agent's agent.py.

Design rule (enforced in code, not just in the prompt): this module is
DESCRIPTIVE ONLY. It is only ever called after the deterministic
AnomalyDetector has already decided a part is defective, and its output
is never used to flip that decision. If the VLM's own text claims
something like "no defect" or otherwise contradicts the fact that it was
called to explain a defect, that contradiction is logged and the
structured fields are discarded rather than trusted - the same "a
model's self-report is a claim, not a fact" principle used in
AI-Dev-Crew's independent test verification.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass

from PIL import Image

from backend.config import settings

_NO_DEFECT_PATTERNS = re.compile(
    r"\b(no defect|no anomaly|looks (fine|good|normal)|appears normal|passes inspection)\b",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE = """You are assisting a factory quality-control system. A separate, \
already-final measurement system has determined that this "{category}" part IS \
DEFECTIVE, and highlighted the suspect region in red/yellow on the image below \
(a heatmap overlay).

Your only job is to describe what a human inspector would see - you are not being \
asked to judge whether the part passes or fails, that has already been decided.

Look only at the highlighted region and respond with STRICT JSON, no extra text, \
in this exact shape:
{{
  "defect_type": "<short phrase, e.g. 'surface scratch', 'crack', 'discoloration'>",
  "likely_cause": "<one plausible cause, e.g. 'handling damage during transport'>",
  "severity": "<low|medium|high>",
  "note": "<one short sentence, plain language>"
}}
"""


@dataclass
class ExplanationResult:
    defect_description: str | None
    likely_cause: str | None
    severity: str | None
    source: str  # "vlm" | "vlm_unparsed" | "vlm_discarded_inconsistent" | "disabled" | "error"
    latency_ms: float


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def explain_defect(image_with_overlay: Image.Image, category: str) -> ExplanationResult:
    """Only call this when the detector has already flagged the part as defective."""
    start = time.perf_counter()

    if not settings.vlm_enabled:
        return ExplanationResult(None, None, None, "disabled", 0.0)

    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        prompt = _PROMPT_TEMPLATE.format(category=category)
        img_b64 = _image_to_base64_png(image_with_overlay)

        response = client.chat(
            model=settings.vlm_model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        raw_text = response["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as a graceful "no explanation"
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ExplanationResult(None, None, None, f"error: {exc}", latency_ms)

    latency_ms = (time.perf_counter() - start) * 1000.0

    # Guardrail: the detector already established this part IS defective.
    # If the model's raw text disagrees, don't trust any of its structured
    # fields either - something went wrong (hallucination, wrong region).
    if _NO_DEFECT_PATTERNS.search(raw_text):
        return ExplanationResult(None, None, None, "vlm_discarded_inconsistent", latency_ms)

    parsed = _try_parse_json(raw_text)
    if parsed is None:
        # Model didn't follow the JSON format - fall back to using the raw
        # text as a plain description rather than throwing it away.
        return ExplanationResult(raw_text[:500], None, None, "vlm_unparsed", latency_ms)

    severity = str(parsed.get("severity", "")).lower()
    if severity not in {"low", "medium", "high"}:
        severity = None

    description = parsed.get("defect_type") or parsed.get("note")
    return ExplanationResult(
        defect_description=description,
        likely_cause=parsed.get("likely_cause"),
        severity=severity,
        source="vlm",
        latency_ms=latency_ms,
    )


def _try_parse_json(text: str) -> dict | None:
    # Models sometimes wrap JSON in ```json fences despite instructions.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
