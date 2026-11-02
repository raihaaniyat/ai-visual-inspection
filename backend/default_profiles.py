"""
Default Inspection Profiles and Synthetic Demonstration Generator.

IMPORTANT:
- Synthetic illustrations are strictly isolated to Demo Mode.
- Normal inspection requires operator-uploaded or real physical reference images.
- The physical 6-screw experiment arrangement:
  upper centre; upper left and right; lower left and right; lower centre.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from backend.schemas import CheckItem, ExpectedRegion, NormalizedBox, WorkpieceCorners
from backend.storage import (
    create_or_save_profile,
    get_profile,
    list_profiles,
    save_image_file,
)
from backend.config import settings


# ==============================================================================
# Physical 6-Screw Experiment Configuration (Actual physical arrangement)
# ==============================================================================

def get_physical_6_screw_checklist() -> list[CheckItem]:
    """
    Acceptance criteria for the 6-screw physical board experiment:
    upper centre; upper left and right; lower left and right; lower centre.
    Notice: No unrequested checks such as hidden thread damage!
    """
    return [
        CheckItem(
            check_id="chk_count",
            check_type="count",
            target_count=6,
            description="Total 6 installed screws present on the workpiece",
            expected_requirement="The workpiece must contain exactly 6 installed screws. Exclude any loose screws outside the workpiece.",
            required=True,
        ),
        CheckItem(
            check_id="chk_upper_centre",
            check_type="presence",
            region_id="upper-centre",
            description="Upper-centre screw present and seated",
            expected_requirement="One metallic screw must be seated in the upper-centre hole.",
            required=True,
        ),
        CheckItem(
            check_id="chk_upper_left",
            check_type="presence",
            region_id="upper-left",
            description="Upper-left screw present and seated",
            expected_requirement="One metallic screw must be seated in the upper-left hole.",
            required=True,
        ),
        CheckItem(
            check_id="chk_upper_right",
            check_type="presence",
            region_id="upper-right",
            description="Upper-right screw present and seated",
            expected_requirement="One metallic screw must be seated in the upper-right hole.",
            required=True,
        ),
        CheckItem(
            check_id="chk_lower_left",
            check_type="presence",
            region_id="lower-left",
            description="Lower-left screw present and seated",
            expected_requirement="One metallic screw must be seated in the lower-left hole.",
            required=True,
        ),
        CheckItem(
            check_id="chk_lower_right",
            check_type="presence",
            region_id="lower-right",
            description="Lower-right screw present and seated",
            expected_requirement="One metallic screw must be seated in the lower-right hole.",
            required=True,
        ),
        CheckItem(
            check_id="chk_lower_centre",
            check_type="presence",
            region_id="lower-centre",
            description="Lower-centre screw present and seated",
            expected_requirement="One metallic screw must be seated in the lower-centre hole.",
            required=True,
        ),
    ]


def get_physical_6_screw_regions() -> list[ExpectedRegion]:
    """
    Normalized bounding boxes for the 6 screw positions in rectified board coordinates
    (normalized 0.0 to 1.0 within the rectified planar board).
    Arrangement:
    - Upper centre: ~[0.15, 0.38, 0.32, 0.58]
    - Upper left:   ~[0.26, 0.15, 0.44, 0.35]
    - Upper right:  ~[0.26, 0.62, 0.44, 0.82]
    - Lower left:   ~[0.54, 0.14, 0.72, 0.34]
    - Lower right:  ~[0.53, 0.63, 0.71, 0.83]
    - Lower centre: ~[0.67, 0.38, 0.85, 0.58]
    """
    return [
        ExpectedRegion(
            region_id="upper-centre",
            label="Upper-Centre Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.14, xmin=0.37, ymax=0.32, xmax=0.59),
        ),
        ExpectedRegion(
            region_id="upper-left",
            label="Upper-Left Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.25, xmin=0.14, ymax=0.43, xmax=0.36),
        ),
        ExpectedRegion(
            region_id="upper-right",
            label="Upper-Right Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.25, xmin=0.62, ymax=0.43, xmax=0.84),
        ),
        ExpectedRegion(
            region_id="lower-left",
            label="Lower-Left Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.53, xmin=0.13, ymax=0.71, xmax=0.35),
        ),
        ExpectedRegion(
            region_id="lower-right",
            label="Lower-Right Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.52, xmin=0.62, ymax=0.70, xmax=0.84),
        ),
        ExpectedRegion(
            region_id="lower-centre",
            label="Lower-Centre Screw",
            component_type="screw",
            box=NormalizedBox(ymin=0.66, xmin=0.37, ymax=0.84, xmax=0.59),
        ),
    ]


def seed_physical_experiment_profile() -> str:
    """
    Creates or updates the physical 6-screw experiment profile using the supplied photograph.
    Excludes background tabletop and loose screws.
    """
    profile_id = "prof_physical_6screw"
    existing = get_profile(profile_id)
    if existing and existing.reference_image_id:
        return profile_id

    # Check if physical reference photo exists
    ref_photo_path = Path("data/physical_samples/90645232-ae17-491d-a5f4-b9408128cead.jpg")
    ref_img = None
    ref_corners = None
    if ref_photo_path.exists():
        ref_img = Image.open(ref_photo_path).convert("RGB")
        # Accurate 4 board corners for the physical reference photograph:
        # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        ref_corners = WorkpieceCorners(
            top_left=[0.24, 0.17],
            top_right=[0.74, 0.16],
            bottom_right=[0.75, 0.81],
            bottom_left=[0.21, 0.81],
        )

    prof = create_or_save_profile(
        name="Physical 6-Screw Board",
        description="Physical plywood board assembly with 6 screws: upper-centre, upper-left/right, lower-left/right, lower-centre.",
        reference_image=ref_img,
        reference_corners=ref_corners,
        checklist=get_physical_6_screw_checklist(),
        regions=get_physical_6_screw_regions(),
        profile_id=profile_id,
    )
    return prof.profile_id


def seed_general_inspection_profile() -> str:
    """
    Creates a universal profile for inspecting any arbitrary assembly or image.
    Automatically detects what components exist and identifies what is missing or present.
    """
    profile_id = "prof_general_inspection"
    existing = get_profile(profile_id)
    if existing and existing.reference_image_id:
        return profile_id

    ref_photo_path = Path("data/physical_samples/90645232-ae17-491d-a5f4-b9408128cead.jpg")
    ref_img = Image.open(ref_photo_path).convert("RGB") if ref_photo_path.exists() else None
    ref_corners = WorkpieceCorners(
        top_left=[0.24, 0.17],
        top_right=[0.74, 0.16],
        bottom_right=[0.75, 0.81],
        bottom_left=[0.21, 0.81],
    ) if ref_img else None

    general_checklist = [
        CheckItem(
            check_id="chk_assembly_completeness",
            check_type="presence",
            description="All required components from approved reference present",
            expected_requirement="Verify that every component, fastener, and part visible in the approved reference is present in its required position on the workpiece.",
            required=True,
        ),
        CheckItem(
            check_id="chk_no_missing_parts",
            check_type="presence",
            description="No missing components or vacant mounting locations",
            expected_requirement="No empty holes, vacant mounting locations, unpopulated slots, or missing parts.",
            required=True,
        ),
    ]

    prof = create_or_save_profile(
        name="General Visual Inspection (Any Assembly / Image)",
        description="Universal inspection for any manufactured product, electronics, mechanical parts, or fasteners. Automatically detects components and reports what is missing and what is present.",
        reference_image=ref_img,
        reference_corners=ref_corners,
        checklist=general_checklist,
        regions=[],  # No hardcoded regions: dynamically discovers all components!
        profile_id=profile_id,
    )
    return prof.profile_id


# ==============================================================================
# Demo Mode Synthetic Generator (Isolated strictly to Demo Mode)
# ==============================================================================

def create_synthetic_demo_board_image(
    screws_present: list[str] | None = None,
    misplaced_screws: dict[str, tuple[float, float]] | None = None,
    blur_region: str | None = None,
    width: int = 800,
    height: int = 600,
) -> Image.Image:
    """
    Generates a synthetic wood-board illustration strictly for Demo Mode.
    Must never be silently used for real user profiles.
    """
    # 1. Plywood wood-grain background
    img = np.full((height, width, 3), (160, 195, 222), dtype=np.uint8)
    np.random.seed(42)
    grain = np.random.normal(0, 12, (height, width, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    for y in range(0, height, 4):
        alpha = np.random.uniform(0.92, 1.05)
        img[y:y+2, :] = np.clip(img[y:y+2, :].astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    cv2.rectangle(img, (40, 40), (width - 40, height - 40), (120, 150, 180), 3)

    # 6 positions in actual physical layout (upper-centre, upper-left/right, lower-left/right, lower-centre)
    demo_positions = {
        "upper-centre": (0.24, 0.50),
        "upper-left": (0.34, 0.25),
        "upper-right": (0.34, 0.75),
        "lower-left": (0.64, 0.25),
        "lower-right": (0.64, 0.75),
        "lower-centre": (0.76, 0.50),
    }

    if screws_present is None:
        screws_present = list(demo_positions.keys())

    for pos_id, (norm_y, norm_x) in demo_positions.items():
        cx = int(norm_x * width)
        cy = int(norm_y * height)

        if misplaced_screws and pos_id in misplaced_screws:
            shift_y, shift_x = misplaced_screws[pos_id]
            cx_screw = int(shift_x * width)
            cy_screw = int(shift_y * height)
            cv2.circle(img, (cx, cy), 22, (90, 110, 130), -1)
            cv2.circle(img, (cx, cy), 18, (45, 55, 65), -1)
            _draw_screw(img, cx_screw, cy_screw)
        elif pos_id in screws_present:
            _draw_screw(img, cx, cy)
        else:
            cv2.circle(img, (cx, cy), 22, (90, 110, 130), -1)
            cv2.circle(img, (cx, cy), 18, (45, 55, 65), -1)

    if blur_region and blur_region in demo_positions:
        norm_y, norm_x = demo_positions[blur_region]
        cx = int(norm_x * width)
        cy = int(norm_y * height)
        y1, y2 = max(0, cy - 65), min(height, cy + 65)
        x1, x2 = max(0, cx - 65), min(width, cx + 65)
        blurred_roi = cv2.GaussianBlur(img[y1:y2, x1:x2], (35, 35), 20)
        img[y1:y2, x1:x2] = blurred_roi

    # Watermark as DEMO
    cv2.putText(img, "SYNTHETIC DEMO SAMPLE", (50, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _draw_screw(img: np.ndarray, cx: int, cy: int) -> None:
    cv2.circle(img, (cx, cy), 24, (100, 120, 140), 2)
    cv2.circle(img, (cx, cy), 22, (190, 195, 200), -1)
    cv2.circle(img, (cx, cy), 19, (220, 225, 230), -1)
    cv2.circle(img, (cx, cy), 15, (180, 185, 190), -1)
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), (60, 60, 60), 3)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), (60, 60, 60), 3)

