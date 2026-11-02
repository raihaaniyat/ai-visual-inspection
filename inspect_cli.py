"""
Command-Line Interface (CLI) for VisionInspect AI.

Run assembly inspections directly from the terminal or batch scripts
without requiring any web server or Streamlit frontend.

Usage:
    # 1. Run inspection on an image file:
    python inspect_cli.py inspect --image path/to/board.png

    # 2. Run inspection on built-in synthetic test samples:
    python inspect_cli.py inspect --sample pass
    python inspect_cli.py inspect --sample missing
    python inspect_cli.py inspect --sample misplaced
    python inspect_cli.py inspect --sample obscured

    # 3. Save the annotated image to a file:
    python inspect_cli.py inspect --sample missing --output-annotated result.png

    # 4. List all saved inspection profiles:
    python inspect_cli.py list-profiles

    # 5. View inspection history & accuracy benchmarks:
    python inspect_cli.py history
    python inspect_cli.py accuracy
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import settings
from backend.default_profiles import (
    create_synthetic_demo_board_image,
    seed_physical_experiment_profile,
)
from backend.gemini_inspector import run_workpiece_inspection
from backend.schemas import WorkpieceCorners
from backend.storage import (
    get_accuracy_report,
    get_profile,
    init_db,
    list_inspections,
    list_profiles,
    load_image_file,
    save_inspection,
    update_ground_truth,
)


def cmd_inspect(args: argparse.Namespace) -> None:
    init_db()
    seed_physical_experiment_profile()

    profile_id = args.profile or "prof_physical_6screw"
    profile = get_profile(profile_id)
    if not profile:
        print(f"[ERROR] Profile '{profile_id}' not found. Available profiles:")
        for p in list_profiles():
            print(f"  - {p.profile_id} ({p.name})")
        sys.exit(1)

    if not profile.reference_image_id:
        print(f"[ERROR] Add an approved reference first for profile '{profile_id}'.")
        sys.exit(1)

    ref_img = load_image_file(profile.reference_image_id)
    if not ref_img:
        print(f"[ERROR] Reference image for profile '{profile_id}' missing from storage.")
        sys.exit(1)

    # Determine input image
    test_img = None
    sample_name = None

    if args.sample:
        sample_name = args.sample.lower()
        if sample_name == "pass":
            test_img = create_synthetic_demo_board_image()
        elif sample_name == "missing":
            screws = ["upper-centre", "upper-left", "upper-right", "lower-left", "lower-right"]
            test_img = create_synthetic_demo_board_image(screws_present=screws)
        elif sample_name == "misplaced":
            misplaced = {"lower-centre": (0.50, 0.50)}
            test_img = create_synthetic_demo_board_image(misplaced_screws=misplaced)
        elif sample_name in ("obscured", "review"):
            test_img = create_synthetic_demo_board_image(blur_region="upper-left")
        else:
            print(f"[ERROR] Unknown sample '{args.sample}'. Options: pass, missing, misplaced, obscured")
            sys.exit(1)
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image file '{args.image}' not found.")
            sys.exit(1)
        test_img = Image.open(img_path).convert("RGB")
    else:
        print("[ERROR] You must specify either --image <path> or --sample <pass|missing|misplaced|obscured>.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" VisionInspect AI — Inspection Execution")
    print(f"=======================================================")
    print(f" Profile:          {profile.name} (v{profile.version})")
    print(f" Acceptance Items: {len(profile.checklist)} checklist items")
    print(f" Defined Regions:  {len(profile.regions)} component positions")
    print(f" Engine:           Inspection Engine")
    if sample_name:
        print(f" Source:           Synthetic demo sample [{sample_name}]")
    else:
        print(f" Source:           File: {args.image}")
    print(f"-------------------------------------------------------\n")

    start = time.perf_counter()
    result = run_workpiece_inspection(
        reference_image=ref_img,
        test_image=test_img,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        checklist=profile.checklist,
        regions=profile.regions,
        reference_corners=profile.reference_corners,
    )
    total_time = (time.perf_counter() - start) * 1000.0

    # Save to SQLite database
    saved_res = save_inspection(
        profile_id=result.profile_id,
        profile_version=result.profile_version,
        verdict=result.verdict,
        verdict_reason=result.verdict_reason,
        findings=result.findings,
        counts_summary=result.counts_summary,
        engine=result.engine,
        model_identifier=result.model_identifier,
        inference_latency_ms=result.inference_latency_ms,
        total_latency_ms=total_time,
        inspection_issue_summary=result.inspection_issue_summary,
        reference_identity_ok=result.reference_identity_ok,
        image_quality_ok=result.image_quality_ok,
        geometric_alignment_ok=result.geometric_alignment_ok,
        rectified_ref_png_base64=result.rectified_ref_png_base64,
        rectified_test_png_base64=result.rectified_test_png_base64,
        annotated_image_png_base64=result.annotated_image_png_base64,
        opencv_diff_png_base64=result.opencv_diff_png_base64,
        ground_truth_verdict=args.ground_truth,
        ground_truth_notes=f"CLI inspection ({sample_name or args.image})",
    )

    # Save output annotated image if requested
    if args.output_annotated and result.annotated_image_png_base64:
        out_path = Path(args.output_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_bytes = base64.b64decode(result.annotated_image_png_base64)
        out_path.write_bytes(img_bytes)
        print(f"[SAVED] Annotated inspection image saved to: {out_path.resolve()}")

    if args.output_diff and result.opencv_diff_png_base64:
        out_diff = Path(args.output_diff)
        out_diff.parent.mkdir(parents=True, exist_ok=True)
        diff_bytes = base64.b64decode(result.opencv_diff_png_base64)
        out_diff.write_bytes(diff_bytes)
        print(f"[SAVED] OpenCV difference map saved to: {out_diff.resolve()}")

    if args.json:
        dump = saved_res.model_dump()
        dump["annotated_image_png_base64"] = None
        dump["opencv_diff_png_base64"] = None
        print(json.dumps(dump, indent=2))
        return

    # Print Clean Console Results
    verdict = result.verdict
    badge = f"*** {verdict} ***"
    if verdict == "PASS":
        badge = f"\033[92m[ PASS ]\033[0m" if sys.stdout.isatty() else "[ PASS ]"
    elif verdict == "REJECT":
        badge = f"\033[91m[ REJECT ]\033[0m" if sys.stdout.isatty() else "[ REJECT ]"
    else:
        badge = f"\033[93m[ REVIEW ]\033[0m" if sys.stdout.isatty() else "[ REVIEW ]"

    print(f"VERDICT: {badge}")
    print(f"REASON:  {result.verdict_reason}\n")

    print(f"COUNTS:")
    counts = result.counts_summary
    print(f"  Total Checks:   {counts.get('total_checks', 0)}")
    print(f"  Satisfied:      {counts.get('satisfied', 0)}")
    print(f"  Failed:         {counts.get('failed', 0)}")
    print(f"  Uncertain:      {counts.get('uncertain', 0)}")
    print(f"  Not Evaluated:  {counts.get('not_evaluated', 0)}\n")

    if result.inspection_issue_summary:
        print(f"INSPECTION ISSUE: {result.inspection_issue_summary}\n")

    print(f"FINDINGS BY CHECK:")
    print(f"  {'CHECK ID':<20} | {'LABEL':<22} | {'STATUS':<14} | EVIDENCE")
    print(f"  {'-'*20}-+-{'-'*22}-+-{'-'*14}-+-{'-'*35}")
    for f in result.findings:
        st_label = f.status.upper()
        print(f"  {f.check_id:<20} | {f.label[:22]:<22} | {st_label:<14} | {f.evidence}")

    print(f"\nLATENCY: Inference={result.inference_latency_ms:.1f}ms | Total={total_time:.1f}ms")
    print(f"SAVED INSPECTION ID: {saved_res.inspection_id}\n")


def cmd_list_profiles(_: argparse.Namespace) -> None:
    init_db()
    seed_physical_experiment_profile()
    profiles = list_profiles()

    print(f"\nSAVED INSPECTION PROFILES ({len(profiles)}):")
    print(f"  {'PROFILE ID':<24} | {'NAME':<30} | {'REGIONS':<8} | CREATED")
    print(f"  {'-'*24}-+-{'-'*30}-+-{'-'*8}-+-{'-'*20}")
    for p in profiles:
        print(f"  {p.profile_id:<24} | {p.name:<30} | {len(p.regions):<8} | {p.created_at[:19]}")
    print("")


def cmd_history(args: argparse.Namespace) -> None:
    init_db()
    inspections = list_inspections(limit=args.limit)

    print(f"\nRECENT INSPECTION HISTORY ({len(inspections)}):")
    print(f"  {'INSPECTION ID':<20} | {'PROFILE':<25} | {'VERDICT':<8} | {'GROUND TRUTH':<12} | TIMESTAMP")
    print(f"  {'-'*20}-+-{'-'*25}-+-{'-'*8}-+-{'-'*12}-+-{'-'*20}")
    for i in inspections:
        gt = i.ground_truth_verdict or "None"
        print(f"  {i.inspection_id:<20} | {i.profile_id[:25]:<25} | {i.verdict:<8} | {gt:<12} | {i.created_at[:19]}")
    print("")


def cmd_accuracy(_: argparse.Namespace) -> None:
    init_db()
    acc = get_accuracy_report()

    print(f"\nACCURACY BENCHMARK REPORT:")
    print(f"  Total Inspections:       {acc.total_inspections}")
    print(f"  Labelled Ground Truth:   {acc.ground_truth_count}")
    acc_pct = f"{acc.accuracy * 100:.1f}%" if acc.accuracy is not None else "N/A"
    print(f"  Verified Accuracy:       {acc_pct}")
    print(f"  Breakdown (P/R/Rev):     {acc.pass_count} PASS / {acc.reject_count} REJECT / {acc.review_count} REVIEW\n")

    if acc.ground_truth_count > 0:
        print("CONFUSION MATRIX (Ground Truth vs Model Output):")
        print("  Actual \\ Pred    PASS    REJECT  REVIEW")
        for actual in ("PASS", "REJECT", "REVIEW"):
            row = acc.confusion_matrix.get(actual, {})
            print(f"  {actual:<15} {row.get('PASS', 0):<7} {row.get('REJECT', 0):<7} {row.get('REVIEW', 0):<7}")
    print("")


def cmd_test_engine(_args: argparse.Namespace) -> None:
    from backend.app import check_engine_health
    print("\nVerifying Inspection Engine status...")
    res = check_engine_health()
    if res["success"]:
        print(f"[SUCCESS] {res['message']}")
        print(f"Roundtrip Latency: {res['latency_ms']}ms\n")
    else:
        print(f"[FAILED] {res['message']}\n")



def main():
    parser = argparse.ArgumentParser(
        description="VisionInspect AI — Assembly Quality Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: inspect
    p_inspect = subparsers.add_parser("inspect", help="Run visual inspection on an image or sample")
    p_inspect.add_argument("--image", "-i", type=str, help="Path to image file to inspect")
    p_inspect.add_argument("--sample", "-s", type=str, choices=["pass", "missing", "misplaced", "obscured"], help="Run against a synthetic test sample")
    p_inspect.add_argument("--profile", "-p", type=str, default="prof_physical_6screw", help="Profile ID (default: prof_physical_6screw)")
    p_inspect.add_argument("--output-annotated", "-o", type=str, help="Save annotated inspection PNG to this path")
    p_inspect.add_argument("--output-diff", type=str, help="Save OpenCV pixel difference overlay PNG to this path")
    p_inspect.add_argument("--ground-truth", "-g", choices=["PASS", "REJECT", "REVIEW"], help="Record operator expected verdict")
    p_inspect.add_argument("--no-opencv", action="store_true", help="Disable supporting OpenCV difference check")
    p_inspect.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_inspect.set_defaults(func=cmd_inspect)

    # Subcommand: list-profiles
    p_list = subparsers.add_parser("list-profiles", help="List all saved inspection profiles")
    p_list.set_defaults(func=cmd_list_profiles)

    # Subcommand: history
    p_hist = subparsers.add_parser("history", help="Show inspection history")
    p_hist.add_argument("--limit", "-n", type=int, default=20, help="Number of records to show")
    p_hist.set_defaults(func=cmd_history)

    # Subcommand: accuracy
    p_acc = subparsers.add_parser("accuracy", help="Show ground-truth accuracy report & confusion matrix")
    p_acc.set_defaults(func=cmd_accuracy)

    # Subcommand: test-engine (and alias test-key)
    p_test = subparsers.add_parser("test-engine", help="Verify whether the Inspection Engine is working properly")
    p_test.set_defaults(func=cmd_test_engine)

    p_test_legacy = subparsers.add_parser("test-key", help="Verify whether the Inspection Engine is working properly")
    p_test_legacy.set_defaults(func=cmd_test_engine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
