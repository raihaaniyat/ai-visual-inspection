"""
OpenCV-based supporting visual verification and difference extraction.

Derived in part from Automated-Optical-Inspection (MIT License):
https://github.com/decodelabs-internship-f/Automated-Optical-Inspection

NOTE ON INTENDED USE:
This module compares the RECTIFIED WORKPIECE PLANES (excluding background objects,
tools, tables, or jars) inside the valid overlapping workpiece boundary.
It renders a continuous "Reference difference map" with an understandable legend.
Residual differences along slat edges and shadows reflect appearance/lighting variations,
NOT confirmed defects. Bounding boxes are not splattered across slats.
If overall comparison quality is poor (excessive disparity), it marks the comparison unreliable.
"""
from __future__ import annotations

import base64
import cv2
import numpy as np
from PIL import Image

from backend.config import settings
from backend.schemas import NormalizedBox


class DifferenceTuple(tuple):
    """
    Backward-compatible 5-tuple for difference results:
    (has_diff, count, overlay_b64, mask_b64, notes)
    with added .reliable and .notice attributes.
    """
    def __new__(cls, has_diff: bool, count: int, overlay_b64: str | None, mask_b64: str | None, notes: str, reliable: bool = True, notice: str = ""):
        obj = super().__new__(cls, (has_diff, count, overlay_b64, mask_b64, notes))
        obj.has_diff = has_diff
        obj.count = count
        obj.overlay_b64 = overlay_b64
        obj.mask_b64 = mask_b64
        obj.notes = notes
        obj.reliable = reliable
        obj.notice = notice
        return obj


def compute_rectified_visual_difference(
    rectified_reference: Image.Image,
    rectified_test: Image.Image,
    diff_threshold: int | None = None,
    min_area: int | None = None,
) -> DifferenceTuple:
    """
    Compares two rectified workpiece images inside the valid workpiece area.
    Applies modest illumination normalization without smoothing away small components.
    Renders a continuous difference-intensity overlay with an understandable legend.
    Residual slat edges and shadows are treated as alignment/appearance differences,
    not confirmed component defects.
    """
    threshold_val = diff_threshold or settings.opencv_diff_threshold
    min_area_val = min_area or settings.opencv_min_defect_area

    ref_bgr = cv2.cvtColor(np.array(rectified_reference.convert("RGB")), cv2.COLOR_RGB2BGR)
    test_bgr = cv2.cvtColor(np.array(rectified_test.convert("RGB")), cv2.COLOR_RGB2BGR)

    ref_h, ref_w = ref_bgr.shape[:2]
    if test_bgr.shape[:2] != (ref_h, ref_w):
        test_bgr = cv2.resize(test_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

    # 1. Valid overlapping workpiece boundary mask (exclude 4% perimeter border artifacts)
    m_y, m_x = max(1, int(0.04 * ref_h)), max(1, int(0.04 * ref_w))
    valid_mask = np.zeros((ref_h, ref_w), dtype=np.uint8)
    valid_mask[m_y:ref_h - m_y, m_x:ref_w - m_x] = 255

    # 2. Modest illumination normalization (in LAB space)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    test_lab = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB)
    l_ref, a_ref, b_ref = cv2.split(ref_lab)
    l_test, a_test, b_test = cv2.split(test_lab)

    # Compute mean and standard deviation strictly inside the valid workpiece area
    mask_pixels_ref = l_ref[valid_mask > 0]
    mask_pixels_test = l_test[valid_mask > 0]

    mu_ref, sig_ref = float(mask_pixels_ref.mean()), float(mask_pixels_ref.std())
    mu_test, sig_test = float(mask_pixels_test.mean()), float(mask_pixels_test.std())

    # Modest gain and bias to match global exposure without wiping local contrast
    gain = float(np.clip(sig_ref / (sig_test + 1e-5), 0.75, 1.35))
    l_test_norm = np.clip((l_test.astype(np.float32) - mu_test) * gain + mu_ref, 0, 255).astype(np.uint8)

    # 3. Compute continuous difference without smoothing away small components (3x3 kernel only)
    l_ref_subtle = cv2.GaussianBlur(l_ref, (3, 3), 0)
    l_test_subtle = cv2.GaussianBlur(l_test_norm, (3, 3), 0)

    diff = cv2.absdiff(l_ref_subtle, l_test_subtle)
    diff[valid_mask == 0] = 0

    mean_diff = float(diff[valid_mask > 0].mean())
    high_diff_ratio = float((diff[valid_mask > 0] > 45).mean())
    mean_lum_disparity = abs(mu_ref - mu_test)

    # 4. Check Comparison Quality
    # If mean residual diff is excessive, illumination gap is extreme, or over 38% of workpiece has severe difference, mark unreliable
    is_reliable = (mean_diff <= 42.0) and (high_diff_ratio <= 0.38) and (mean_lum_disparity <= 75.0)
    if not is_reliable:
        reliability_notice = "Reference comparison unreliable: Excessive illumination or viewpoint disparity."
    else:
        reliability_notice = "Reference difference map reflects residual alignment and appearance differences."

    # 5. Extract significant concentrated hotspots (e.g. missing screw holes),
    # ignoring elongated slat ridges (aspect ratio > 3.2 or < 0.31)
    _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
    thresh[valid_mask == 0] = 0
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compact_hotspots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_val:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / float(max(1, h))
            # Keep only reasonably compact spots; discard long continuous wood slat edge lines
            if 0.30 <= aspect <= 3.3 and (w < ref_w * 0.40 and h < ref_h * 0.40):
                compact_hotspots.append(cnt)

    # 6. Render Continuous Heatmap Overlay with understandable legend
    # Scale difference map to [0, 255] with contrast stretching
    diff_vis = np.clip(diff.astype(np.float32) * (255.0 / 110.0), 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_TURBO)

    # Alpha blending: 0 alpha for noise floor (diff < 14), smoothly increasing to 0.58
    alpha_mask = np.clip((diff.astype(np.float32) - 14.0) / 60.0, 0.0, 0.58)
    alpha_mask[valid_mask == 0] = 0.0
    alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    overlay = (test_bgr.astype(np.float32) * (1.0 - alpha_3d) + heatmap.astype(np.float32) * alpha_3d).astype(np.uint8)

    # Render valid workpiece border in subtle cyan dashed line or thin border
    cv2.rectangle(overlay, (m_x, m_y), (ref_w - m_x, ref_h - m_y), (255, 200, 0), 1)

    # Title Banner
    cv2.putText(overlay, "Reference difference map", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "Reference difference map", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # If comparison quality is poor, display clear notice banner
    if not is_reliable:
        cv2.rectangle(overlay, (12, 38), (360, 68), (20, 24, 33), -1)
        cv2.rectangle(overlay, (12, 38), (360, 68), (0, 140, 255), 1)
        cv2.putText(overlay, "Reference comparison unreliable", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 1, cv2.LINE_AA)

    # Understandable Legend Bar in bottom-right corner
    leg_x, leg_y, leg_w, leg_h = ref_w - 190, ref_h - 44, 175, 34
    cv2.rectangle(overlay, (leg_x - 6, leg_y - 6), (leg_x + leg_w + 6, leg_y + leg_h + 6), (15, 23, 42), -1)
    cv2.rectangle(overlay, (leg_x - 6, leg_y - 6), (leg_x + leg_w + 6, leg_y + leg_h + 6), (70, 80, 100), 1)

    # Color gradient
    grad = np.linspace(0, 255, leg_w, dtype=np.uint8).reshape(1, leg_w)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    overlay[leg_y + 12:leg_y + 20, leg_x:leg_x + leg_w] = cv2.resize(grad_color, (leg_w, 8))

    cv2.putText(overlay, "Residual Difference Intensity", (leg_x, leg_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(overlay, "0 (Low)", (leg_x, leg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(overlay, "Max (High)", (leg_x + leg_w - 52, leg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    _, mask_buf = cv2.imencode(".png", thresh)
    mask_b64 = base64.b64encode(mask_buf.tobytes()).decode("utf-8")

    _, overlay_buf = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(overlay_buf.tobytes()).decode("utf-8")

    has_diff = len(compact_hotspots) > 0
    notes = f"Reference difference map computed on rectified workpiece plane. Residual difference mean: {mean_diff:.1f}px ({len(compact_hotspots)} candidate difference region(s))."

    return DifferenceTuple(
        has_diff=has_diff,
        count=len(compact_hotspots),
        overlay_b64=overlay_b64,
        mask_b64=mask_b64,
        notes=notes,
        reliable=is_reliable,
        notice=reliability_notice,
    )


def extract_difference_regions(
    rectified_reference: Image.Image,
    rectified_test: Image.Image,
    diff_threshold: int | None = None,
    min_area: int | None = None,
    max_hotspots: int = 6,
    max_regions: int | None = None,
) -> list[NormalizedBox]:
    """
    Extracts normalized bounding boxes of compact candidate difference hotspots
    (e.g. missing screw holes, unpopulated component pads) on the rectified workpiece plane.
    Elongated continuous slat edges and shadows are filtered out.
    """
    limit = max_regions or max_hotspots
    threshold_val = diff_threshold or settings.opencv_diff_threshold
    min_area_val = min_area or settings.opencv_min_defect_area

    ref_bgr = cv2.cvtColor(np.array(rectified_reference.convert("RGB")), cv2.COLOR_RGB2BGR)
    test_bgr = cv2.cvtColor(np.array(rectified_test.convert("RGB")), cv2.COLOR_RGB2BGR)

    ref_h, ref_w = ref_bgr.shape[:2]
    if test_bgr.shape[:2] != (ref_h, ref_w):
        test_bgr = cv2.resize(test_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

    m_y, m_x = max(1, int(0.04 * ref_h)), max(1, int(0.04 * ref_w))
    valid_mask = np.zeros((ref_h, ref_w), dtype=np.uint8)
    valid_mask[m_y:ref_h - m_y, m_x:ref_w - m_x] = 255

    # Illumination normalization
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    test_lab = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB)
    l_ref = cv2.split(ref_lab)[0]
    l_test = cv2.split(test_lab)[0]

    mu_ref, sig_ref = float(l_ref[valid_mask > 0].mean()), float(l_ref[valid_mask > 0].std())
    mu_test, sig_test = float(l_test[valid_mask > 0].mean()), float(l_test[valid_mask > 0].std())
    gain = float(np.clip(sig_ref / (sig_test + 1e-5), 0.75, 1.35))
    l_test_norm = np.clip((l_test.astype(np.float32) - mu_test) * gain + mu_ref, 0, 255).astype(np.uint8)

    diff = cv2.absdiff(cv2.GaussianBlur(l_ref, (3, 3), 0), cv2.GaussianBlur(l_test_norm, (3, 3), 0))
    diff[valid_mask == 0] = 0

    _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
    thresh[valid_mask == 0] = 0
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    boxes: list[NormalizedBox] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_val:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / float(max(1, h))
            # Discard elongated slat contours
            if 0.30 <= aspect <= 3.3 and w < ref_w * 0.40 and h < ref_h * 0.40:
                pad_x = int(w * 0.35)
                pad_y = int(h * 0.35)
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(ref_w, x + w + pad_x)
                y1 = min(ref_h, y + h + pad_y)
                boxes.append(
                    NormalizedBox(
                        ymin=round(y0 / ref_h, 3),
                        xmin=round(x0 / ref_w, 3),
                        ymax=round(y1 / ref_h, 3),
                        xmax=round(x1 / ref_w, 3),
                    )
                )
                if len(boxes) >= limit:
                    break
    return boxes
