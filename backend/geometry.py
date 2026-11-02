"""
Workpiece Geometry, Rectification, and Perspective Transformation.

Provides:
- 4-point perspective rectification of planar workpieces.
- Coordinate transformations between original camera coordinates and rectified workpiece coordinates.
- Mapping rectified bounding boxes to polygons on original inspection photographs.
- Fallback workpiece corner estimation using contour analysis.
"""
from __future__ import annotations

import base64
import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.schemas import (
    MappingDiagnosticPoint,
    MappingDiagnosticResponse,
    NormalizedBox,
    WorkpieceCorners,
)


RECTIFIED_WIDTH = 700
RECTIFIED_HEIGHT = 600


def ensure_correct_orientation(image: Image.Image) -> Image.Image:
    """Applies EXIF orientation so width/height and coordinates are accurate."""
    return ImageOps.exif_transpose(image).convert("RGB")


def corners_to_numpy(corners: WorkpieceCorners, width: int, height: int) -> np.ndarray:
    """Converts normalized corners [[x, y], ...] to absolute pixel numpy array."""
    pts = [
        [corners.top_left[0] * width, corners.top_left[1] * height],
        [corners.top_right[0] * width, corners.top_right[1] * height],
        [corners.bottom_right[0] * width, corners.bottom_right[1] * height],
        [corners.bottom_left[0] * width, corners.bottom_left[1] * height],
    ]
    return np.float32(pts)


def rectify_workpiece(
    image: Image.Image,
    corners: WorkpieceCorners | None = None,
    target_w: int = RECTIFIED_WIDTH,
    target_h: int = RECTIFIED_HEIGHT,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Rectifies a planar workpiece to standard board coordinates using a 4-point perspective transform.
    Returns:
        (rectified_image, M_forward, M_inverse)
    """
    img_corrected = ensure_correct_orientation(image)
    img_bgr = cv2.cvtColor(np.array(img_corrected), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if corners is None:
        # Default fallback: central 80% if corners unassigned
        corners = WorkpieceCorners(
            top_left=[0.10, 0.10],
            top_right=[0.90, 0.10],
            bottom_right=[0.90, 0.90],
            bottom_left=[0.10, 0.90],
        )

    src_pts = corners_to_numpy(corners, w, h)
    dst_pts = np.float32([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    warped_bgr = cv2.warpPerspective(img_bgr, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(warped_rgb), M, M_inv


def transform_box_to_original_polygon(
    box: NormalizedBox,
    M_inv: np.ndarray,
    rect_w: int = RECTIFIED_WIDTH,
    rect_h: int = RECTIFIED_HEIGHT,
    orig_w: int = 1,
    orig_h: int = 1,
) -> list[list[float]]:
    """
    Transforms a normalized bounding box on the rectified board into a 4-point polygon
    on the original photograph, normalized to [0.0, 1.0].
    """
    # 4 corners in rectified pixel coordinates: TL, TR, BR, BL
    x1, y1 = box.xmin * rect_w, box.ymin * rect_h
    x2, y2 = box.xmax * rect_w, box.ymax * rect_h

    rect_pts = np.float32([
        [[x1, y1]],
        [[x2, y1]],
        [[x2, y2]],
        [[x1, y2]],
    ])

    orig_pts = cv2.perspectiveTransform(rect_pts, M_inv)
    # Normalize back to 0.0..1.0
    poly = []
    for pt in orig_pts:
        px, py = pt[0]
        norm_x = max(0.0, min(1.0, float(px) / float(orig_w)))
        norm_y = max(0.0, min(1.0, float(py) / float(orig_h)))
        poly.append([norm_x, norm_y])

    return poly


def estimate_board_corners(image: Image.Image) -> WorkpieceCorners:
    """
    Heuristic workpiece corner detector for grooved wooden panels / boards.
    Finds the prominent quadrangular workpiece contour in the image.
    """
    img_corrected = ensure_correct_orientation(image)
    bgr = cv2.cvtColor(np.array(img_corrected), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Resize for faster edge processing
    scale = 600.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_poly = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (small.shape[0] * small.shape[1] * 0.15):  # at least 15% of image
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_poly = approx

    if best_poly is not None:
        # Scale back to full image and order corners
        pts = best_poly.reshape(4, 2) / scale
        ordered = _order_quad_points(pts)
        return WorkpieceCorners(
            top_left=[float(ordered[0][0]) / w, float(ordered[0][1]) / h],
            top_right=[float(ordered[1][0]) / w, float(ordered[1][1]) / h],
            bottom_right=[float(ordered[2][0]) / w, float(ordered[2][1]) / h],
            bottom_left=[float(ordered[3][0]) / w, float(ordered[3][1]) / h],
        )

    # Safe fallback if contour not distinct: center 80%
    return WorkpieceCorners(
        top_left=[0.10, 0.15],
        top_right=[0.90, 0.15],
        bottom_right=[0.90, 0.85],
        bottom_left=[0.10, 0.85],
    )


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL has smallest x+y
    rect[2] = pts[np.argmax(s)]  # BR has largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR has smallest y-x
    rect[3] = pts[np.argmax(diff)]  # BL has largest y-x
    return rect


# ==============================================================================
# AI-Independent Geometry Diagnostics & Multi-Scale Verification
# ==============================================================================

def transform_orig_point_to_rect(
    orig_norm_pt: list[float],
    M: np.ndarray,
    orig_w: int,
    orig_h: int,
    rect_w: int = RECTIFIED_WIDTH,
    rect_h: int = RECTIFIED_HEIGHT,
) -> list[float]:
    """Transforms a normalized point [norm_x, norm_y] in original photo to normalized [norm_x, norm_y] in rectified space."""
    px = orig_norm_pt[0] * orig_w
    py = orig_norm_pt[1] * orig_h
    pts = np.float32([[[px, py]]])
    warped = cv2.perspectiveTransform(pts, M)
    rx, ry = warped[0][0]
    return [
        max(0.0, min(1.0, float(rx) / float(rect_w))),
        max(0.0, min(1.0, float(ry) / float(rect_h))),
    ]


def transform_rect_point_to_orig(
    rect_norm_pt: list[float],
    M_inv: np.ndarray,
    rect_w: int = RECTIFIED_WIDTH,
    rect_h: int = RECTIFIED_HEIGHT,
    orig_w: int = 1,
    orig_h: int = 1,
) -> list[float]:
    """Transforms a normalized point [norm_x, norm_y] in rectified workpiece to normalized [norm_x, norm_y] in original photo."""
    rx = rect_norm_pt[0] * rect_w
    ry = rect_norm_pt[1] * rect_h
    pts = np.float32([[[rx, ry]]])
    orig = cv2.perspectiveTransform(pts, M_inv)
    px, py = orig[0][0]
    return [
        max(0.0, min(1.0, float(px) / float(orig_w))),
        max(0.0, min(1.0, float(py) / float(orig_h))),
    ]


def compute_mapping_diagnostics(
    image: Image.Image,
    corners: WorkpieceCorners | None = None,
    target_w: int = RECTIFIED_WIDTH,
    target_h: int = RECTIFIED_HEIGHT,
    display_dimensions: list[tuple[int, int]] | None = None,
) -> MappingDiagnosticResponse:
    """
    Evaluates geometric perspective mapping accuracy without AI:
    - 4 workpiece corners round-trip (TL, TR, BR, BL).
    - Center and edge point mappings.
    - Scale invariance tests across different photograph dimensions and display resolutions.
    - Measures Euclidean pixel error and normalized distance.
    - Produces a side-by-side diagnostic visualization overlay.
    """
    img_corrected = ensure_correct_orientation(image)
    orig_w, orig_h = img_corrected.size

    if corners is None:
        corners = estimate_board_corners(img_corrected)

    # 1. Perspective Transform
    rect_img, M, M_inv = rectify_workpiece(img_corrected, corners, target_w, target_h)

    # 2. Define known test points in both coordinate spaces
    test_points_def: list[tuple[str, str, list[float]]] = [
        ("Top-Left Corner", "orig", [corners.top_left[0], corners.top_left[1]]),
        ("Top-Right Corner", "orig", [corners.top_right[0], corners.top_right[1]]),
        ("Bottom-Right Corner", "orig", [corners.bottom_right[0], corners.bottom_right[1]]),
        ("Bottom-Left Corner", "orig", [corners.bottom_left[0], corners.bottom_left[1]]),
        ("Workpiece Centre", "rect", [0.50, 0.50]),
        ("Top Edge Midpoint", "rect", [0.50, 0.05]),
        ("Left Edge Midpoint", "rect", [0.05, 0.50]),
        ("Right Edge Midpoint", "rect", [0.95, 0.50]),
        ("Bottom Edge Midpoint", "rect", [0.50, 0.95]),
        ("Upper-Left Quadrant", "rect", [0.25, 0.25]),
        ("Upper-Right Quadrant", "rect", [0.75, 0.25]),
        ("Lower-Left Quadrant", "rect", [0.25, 0.75]),
        ("Lower-Right Quadrant", "rect", [0.75, 0.75]),
    ]

    diagnostic_points: list[MappingDiagnosticPoint] = []
    pixel_errors: list[float] = []
    norm_errors: list[float] = []

    for name, source_space, coords in test_points_def:
        if source_space == "orig":
            orig_p = coords
            rect_p = transform_orig_point_to_rect(orig_p, M, orig_w, orig_h, target_w, target_h)
            roundtrip_p = transform_rect_point_to_orig(rect_p, M_inv, target_w, target_h, orig_w, orig_h)
        else:
            rect_p = coords
            orig_p = transform_rect_point_to_orig(rect_p, M_inv, target_w, target_h, orig_w, orig_h)
            # Roundtrip: orig -> rect -> orig
            roundtrip_rect = transform_orig_point_to_rect(orig_p, M, orig_w, orig_h, target_w, target_h)
            roundtrip_p = transform_rect_point_to_orig(roundtrip_rect, M_inv, target_w, target_h, orig_w, orig_h)

        # Measure errors in camera pixel space
        px_diff = (orig_p[0] - roundtrip_p[0]) * orig_w
        py_diff = (orig_p[1] - roundtrip_p[1]) * orig_h
        px_err = float(np.hypot(px_diff, py_diff))

        norm_err = float(np.hypot(orig_p[0] - roundtrip_p[0], orig_p[1] - roundtrip_p[1]))

        pixel_errors.append(px_err)
        norm_errors.append(norm_err)

        diagnostic_points.append(
            MappingDiagnosticPoint(
                name=name,
                orig_point=[round(orig_p[0], 4), round(orig_p[1], 4)],
                rect_point=[round(rect_p[0], 4), round(rect_p[1], 4)],
                roundtrip_point=[round(roundtrip_p[0], 4), round(roundtrip_p[1], 4)],
                pixel_error=round(px_err, 4),
                norm_error=round(norm_err, 6),
            )
        )

    # 3. Multi-resolution scale invariance test
    resolutions = display_dimensions or [(orig_w, orig_h), (800, 600), (400, 300)]
    for rw, rh in resolutions:
        # Scale test to ensure normalization behaves identically at different frontend sizes
        test_scaled_corners = corners_to_numpy(corners, rw, rh)
        dst_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
        M_scale = cv2.getPerspectiveTransform(test_scaled_corners, dst_pts)
        M_scale_inv = cv2.getPerspectiveTransform(dst_pts, test_scaled_corners)
        # Verify center point roundtrip
        cp = np.float32([[[target_w * 0.5, target_h * 0.5]]])
        scaled_orig = cv2.perspectiveTransform(cp, M_scale_inv)
        scaled_rt = cv2.perspectiveTransform(scaled_orig, M_scale)
        rt_err = float(np.hypot(cp[0][0][0] - scaled_rt[0][0][0], cp[0][0][1] - scaled_rt[0][0][1]))
        pixel_errors.append(rt_err)

    max_px_err = float(np.max(pixel_errors))
    mean_px_err = float(np.mean(pixel_errors))
    max_norm_err = float(np.max(norm_errors))

    # 4. Generate side-by-side diagnostic visualization
    vis_b64 = _render_diagnostic_visualization(
        img_corrected=img_corrected,
        rect_img=rect_img,
        corners=corners,
        points=diagnostic_points,
        mean_px_err=mean_px_err,
        max_px_err=max_px_err,
    )

    board_notice = (
        "A board-plane homography aligns the flat board surface; protruding 3D screw heads "
        "under oblique camera angles exhibit perspective parallax. Preserve an inspection margin "
        "and do not assume pixel-perfect 3D correspondence."
    )

    return MappingDiagnosticResponse(
        status="ok",
        image_dimensions=[orig_w, orig_h],
        rectified_dimensions=[target_w, target_h],
        num_points_tested=len(diagnostic_points),
        max_pixel_error=round(max_px_err, 4),
        mean_pixel_error=round(mean_px_err, 4),
        max_norm_error=round(max_norm_err, 6),
        board_plane_notice=board_notice,
        points=diagnostic_points,
        diagnostic_visualization_png_base64=vis_b64,
    )


def _render_diagnostic_visualization(
    img_corrected: Image.Image,
    rect_img: Image.Image,
    corners: WorkpieceCorners,
    points: list[MappingDiagnosticPoint],
    mean_px_err: float,
    max_px_err: float,
) -> str:
    """Renders a side-by-side visual image showing original vs rectified point mappings and measured errors."""
    orig_bgr = cv2.cvtColor(np.array(img_corrected), cv2.COLOR_RGB2BGR)
    rect_bgr = cv2.cvtColor(np.array(rect_img), cv2.COLOR_RGB2BGR)

    orig_h, orig_w = orig_bgr.shape[:2]
    rect_h, rect_w = rect_bgr.shape[:2]

    # Standardize height for side-by-side display
    target_side_h = 550
    scale_orig = target_side_h / orig_h
    scale_rect = target_side_h / rect_h

    side_orig = cv2.resize(orig_bgr, (int(orig_w * scale_orig), target_side_h))
    side_rect = cv2.resize(rect_bgr, (int(rect_w * scale_rect), target_side_h))

    so_h, so_w = side_orig.shape[:2]
    sr_h, sr_w = side_rect.shape[:2]

    # Draw quadrilateral on original
    c_pts = np.array([
        [int(corners.top_left[0] * so_w), int(corners.top_left[1] * so_h)],
        [int(corners.top_right[0] * so_w), int(corners.top_right[1] * so_h)],
        [int(corners.bottom_right[0] * so_w), int(corners.bottom_right[1] * so_h)],
        [int(corners.bottom_left[0] * so_w), int(corners.bottom_left[1] * so_h)],
    ], dtype=np.int32)
    cv2.polylines(side_orig, [c_pts], isClosed=True, color=(255, 200, 0), thickness=2)

    # Draw test points
    palette = [
        (0, 255, 128), (0, 200, 255), (255, 100, 100), (255, 200, 0),
        (255, 0, 255), (0, 255, 255), (100, 150, 255), (150, 255, 100),
        (255, 150, 50), (180, 100, 255), (50, 200, 200), (200, 200, 200), (120, 255, 200),
    ]

    for idx, p in enumerate(points):
        col = palette[idx % len(palette)]
        # Left side (orig)
        ox, oy = int(p.orig_point[0] * so_w), int(p.orig_point[1] * so_h)
        cv2.circle(side_orig, (ox, oy), 6, col, -1)
        cv2.circle(side_orig, (ox, oy), 7, (0, 0, 0), 1)
        cv2.putText(side_orig, str(idx + 1), (ox + 8, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Right side (rect)
        rx, ry = int(p.rect_point[0] * sr_w), int(p.rect_point[1] * sr_h)
        cv2.circle(side_rect, (rx, ry), 6, col, -1)
        cv2.circle(side_rect, (rx, ry), 7, (0, 0, 0), 1)
        cv2.putText(side_rect, str(idx + 1), (rx + 8, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Add header bars to both views
    cv2.putText(side_orig, "ORIGINAL CAMERA PHOTOGRAPH", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(side_orig, "ORIGINAL CAMERA PHOTOGRAPH", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(side_rect, "RECTIFIED WORKPIECE PLANE", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(side_rect, "RECTIFIED WORKPIECE PLANE", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine side by side with divider
    divider = np.full((target_side_h, 4, 3), (80, 80, 80), dtype=np.uint8)
    combined = np.hstack([side_orig, divider, side_rect])

    # Footer Banner with measured errors and parallax disclaimer
    footer_h = 70
    footer = np.zeros((footer_h, combined.shape[1], 3), dtype=np.uint8)
    footer[:] = (20, 24, 33)

    err_text = f"Geometric Mapping Diagnostic (AI-Independent): {len(points)} Points Tested | Max Error: {max_px_err:.4f}px | Mean Error: {mean_px_err:.4f}px"
    cv2.putText(footer, err_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (46, 204, 113) if max_px_err < 1.0 else (0, 200, 255), 1, cv2.LINE_AA)

    disclaimer = "Notice: Board-plane homography aligns planar board surface; protruding 3D components under angle exhibit perspective parallax."
    cv2.putText(footer, disclaimer, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    final_img = np.vstack([combined, footer])

    _, buf = cv2.imencode(".png", final_img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")
