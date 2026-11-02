"""
Deterministic decision engine for assembly quality inspection.

Evaluates structured checklist observations and computes the final verdict strictly in Python:
- REJECT: When any required check demonstrably fails ('failed').
- REVIEW: When no required check failed, but one or more required checks are 'uncertain' or 'not_evaluated'.
- PASS: Only when ALL required checks are satisfied ('satisfied').

Displays inspection-level issues (alignment, reference identity, blur) ONCE in a clear summary banner,
without repeating the same message across every component.
"""
from __future__ import annotations

from typing import Tuple
from backend.schemas import (
    CheckItem,
    CheckObservation,
    ComponentFinding,
    ExpectedRegion,
    InspectionVerdict,
)


def compute_inspection_verdict(
    checklist: list[CheckItem],
    regions: list[ExpectedRegion],
    observations: list[CheckObservation],
    reference_identity_ok: bool = True,
    image_quality_ok: bool = True,
    geometric_alignment_ok: bool = True,
    alignment_issue_summary: str | None = None,
) -> Tuple[InspectionVerdict, str, str | None, dict[str, int], list[ComponentFinding]]:
    """
    Computes final verdict and compiles concise component findings.

    Returns:
        (verdict, verdict_reason, inspection_issue_summary, counts_summary, findings)
    """
    regions_by_id = {r.region_id: r for r in regions}
    obs_by_id = {obs.check_id: obs for obs in observations}
    handled_obs_ids = set()

    findings: list[ComponentFinding] = []
    counts_summary = {
        "total_checks": 0,
        "satisfied": 0,
        "failed": 0,
        "uncertain": 0,
        "not_evaluated": 0,
    }

    # Collect high-level inspection issues to show ONCE in the banner
    inspection_issues = []
    if not reference_identity_ok:
        inspection_issues.append("Workpiece identity mismatch: The inspection image does not appear to be the same part type as the approved reference.")
    if not image_quality_ok:
        inspection_issues.append("Image quality issue: Severe blur, glare, or camera occlusion detected on the workpiece.")
    if not geometric_alignment_ok:
        issue_msg = alignment_issue_summary or "Workpiece geometric alignment failure: Planar board corners could not be reliably transformed."
        inspection_issues.append(f"Workpiece alignment issue: {issue_msg}")

    combined_issue_summary = "; ".join(inspection_issues) if inspection_issues else None

    # 1. Evaluate predefined checklist items (if any are configured in the profile)
    for item in checklist:
        reg = regions_by_id.get(item.region_id) if item.region_id else None
        obs = obs_by_id.get(item.check_id)
        if obs:
            handled_obs_ids.add(item.check_id)

        # If geometric alignment failed and this is a positional check, mark as not_evaluated
        if not geometric_alignment_ok and item.check_type == "presence":
            status = "not_evaluated"
            evidence = "Positional check skipped: Planar workpiece alignment not established."
            observed_state = "Not evaluated"
        elif obs is None:
            # Model response omitted this check
            status = "not_evaluated" if not geometric_alignment_ok else "uncertain"
            evidence = "Check item was omitted from inspection observations."
            observed_state = "Omitted"
        else:
            status = obs.status
            if status not in ("satisfied", "failed", "uncertain", "not_evaluated"):
                status = "uncertain"
            evidence = obs.visible_evidence
            observed_state = obs.observed_state_or_count

        counts_summary[status] += 1

        # 1. Expected Inspection Region (operator-defined in profile)
        has_expected = bool(reg and geometric_alignment_ok)
        expected_box = reg.box if has_expected else None

        # 2. Detected Component Box (actual physical component observation)
        # Never use the whole workpiece box as a component detection
        obs_box = obs.box if (obs and obs.box and geometric_alignment_ok) else None
        is_whole_board = obs_box.is_workpiece_scale(0.80) if obs_box else False
        if is_whole_board:
            obs_box = None

        has_detected = False
        detected_box = None

        if status == "satisfied":
            if obs_box:
                has_detected = True
                detected_box = obs_box
            elif has_expected:
                # Component verified satisfied in expected region
                has_detected = True
                detected_box = expected_box
        elif status == "failed":
            # For missing components, an empty hole is NOT a detected component!
            # Do not label transformed reference regions as detected component boxes.
            has_detected = False
            detected_box = None

        semantic_loc = obs.location if (obs and obs.location) else (reg.label if reg else item.description)

        # Backward compatibility alias
        compat_box = detected_box or expected_box

        findings.append(
            ComponentFinding(
                check_id=item.check_id,
                component_id=item.region_id,
                label=reg.label if reg else item.description,
                check_type=item.check_type,
                expected_requirement=item.expected_requirement or item.description,
                observed_state_or_count=observed_state,
                status=status,
                evidence=evidence,
                semantic_location=semantic_loc,
                has_expected_region=has_expected,
                expected_region_rectified=expected_box,
                expected_polygon_original=None,
                has_detected_box=has_detected,
                detected_box_rectified=detected_box,
                detected_polygon_original=None,
                rectified_box=compat_box,
                original_polygon=None,
            )
        )

    # 2. Process dynamically discovered components across the workpiece
    # (Enables inspection for ANY image and ANY components without fixed 6-position restraint)
    for obs in observations:
        if obs.check_id in handled_obs_ids:
            continue

        status = obs.status
        if status not in ("satisfied", "failed", "uncertain", "not_evaluated"):
            status = "uncertain"

        counts_summary[status] += 1
        label = obs.label or (obs.location and f"{obs.location.title()} Component") or obs.check_id.replace("_", " ").title()

        obs_box = obs.box if geometric_alignment_ok else None
        is_whole_board = obs_box.is_workpiece_scale(0.80) if obs_box else False
        if is_whole_board:
            obs_box = None

        has_expected = False
        expected_box = None
        has_detected = False
        detected_box = None

        is_missing = any(k in obs.observed_state_or_count.lower() or k in obs.visible_evidence.lower() for k in ["missing", "absent", "empty", "vacant", "unpopulated"])

        if status == "satisfied" and obs_box:
            has_detected = True
            detected_box = obs_box
        elif status == "failed" and is_missing and obs_box:
            # Solder pad or hole was expected to have component
            has_expected = True
            expected_box = obs_box
            has_detected = False
            detected_box = None
        elif status == "failed" and obs_box:
            has_detected = True
            detected_box = obs_box

        compat_box = detected_box or expected_box or obs_box

        findings.append(
            ComponentFinding(
                check_id=obs.check_id,
                component_id=obs.location or obs.check_id,
                label=label,
                check_type="presence" if "count" not in obs.check_id.lower() else "count",
                expected_requirement=obs.expected_requirement or "Must be present and installed as shown in reference",
                observed_state_or_count=obs.observed_state_or_count,
                status=status,
                evidence=obs.visible_evidence,
                semantic_location=obs.location or label,
                has_expected_region=has_expected,
                expected_region_rectified=expected_box,
                expected_polygon_original=None,
                has_detected_box=has_detected,
                detected_box_rectified=detected_box,
                detected_polygon_original=None,
                rectified_box=compat_box,
                original_polygon=None,
            )
        )

    counts_summary["total_checks"] = len(findings)

    # Deterministic Verdict Gate:
    # 1. Any confirmed failure -> REJECT
    if counts_summary["failed"] > 0:
        failed_items = [f.label for f in findings if f.status == "failed"]
        reason = f"Assembly Rejected: {counts_summary['failed']} issue(s) detected ({', '.join(failed_items)})."
        return "REJECT", reason, combined_issue_summary, counts_summary, findings

    # 2. If there are unresolved checks (uncertain or not_evaluated) -> REVIEW
    unresolved_count = counts_summary["uncertain"] + counts_summary["not_evaluated"]
    if unresolved_count > 0:
        details = []
        if counts_summary["uncertain"] > 0:
            details.append(f"{counts_summary['uncertain']} uncertain check(s)")
        if counts_summary["not_evaluated"] > 0:
            details.append(f"{counts_summary['not_evaluated']} check(s) could not be evaluated")
        reason = f"Operator Review Required: {', '.join(details)}."
        return "REVIEW", reason, combined_issue_summary, counts_summary, findings

    # 3. All checks satisfied -> PASS
    reason = f"Assembly Passed: All {len(findings)} inspection criteria satisfied."
    return "PASS", reason, combined_issue_summary, counts_summary, findings
