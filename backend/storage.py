"""
SQLite persistence and local file storage for AI-Visual-Inspector.

Stores:
- Inspection Profiles (metadata, reference images, checklists, expected normalized regions)
- Inspection Records (inspections, timestamps, model identifiers, structured findings, computed verdicts)
- Operator Ground Truth labels (for accuracy benchmarking, strictly isolated from inference inputs)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image, ImageOps

import httpx

from backend.config import settings

logger = logging.getLogger("backend.storage")
from backend.schemas import (
    CheckItem,
    ComponentFinding,
    ExpectedRegion,
    FailedCheckStat,
    GroundTruthMetrics,
    NormalizedBox,
    OperationalStatisticsResponse,
    RecentInspectionSummary,
    TrendPoint,
    WorkpieceCorners,
    ProfileResponse,
    AssemblyInspectionResponse,
    AccuracyReportResponse,
)


def _get_db_path() -> Path:
    return settings.storage_dir_path / "inspector.db"


def _get_images_dir() -> Path:
    d = settings.storage_dir_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_db() -> None:
    """Initialize database tables and run lightweight migrations if needed."""
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Profiles table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                version INTEGER DEFAULT 1,
                reference_image_id TEXT DEFAULT NULL,
                reference_corners_json TEXT DEFAULT NULL,
                checklist_json TEXT NOT NULL,
                regions_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Check if columns exist in profiles for backwards compatibility
        cursor.execute("PRAGMA table_info(profiles)")
        p_info = cursor.fetchall()
        for col in p_info:
            if col[1] == "reference_image_id" and col[3] == 1:
                cursor.execute("ALTER TABLE profiles RENAME TO _profiles_old")
                cursor.execute(
                    """
                    CREATE TABLE profiles (
                        profile_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        version INTEGER DEFAULT 1,
                        reference_image_id TEXT DEFAULT NULL,
                        reference_corners_json TEXT DEFAULT NULL,
                        checklist_json TEXT NOT NULL,
                        regions_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO profiles (profile_id, name, description, version, reference_image_id, reference_corners_json, checklist_json, regions_json, created_at, updated_at)
                    SELECT profile_id, name, description, COALESCE(version, 1), reference_image_id, reference_corners_json, checklist_json, regions_json, created_at, updated_at
                    FROM _profiles_old
                    """
                )
                cursor.execute("DROP TABLE _profiles_old")
                break

        cursor.execute("PRAGMA table_info(profiles)")
        cols = {row[1] for row in cursor.fetchall()}
        if "version" not in cols:
            cursor.execute("ALTER TABLE profiles ADD COLUMN version INTEGER DEFAULT 1")
        if "reference_corners_json" not in cols:
            cursor.execute("ALTER TABLE profiles ADD COLUMN reference_corners_json TEXT DEFAULT NULL")

        cursor.execute("PRAGMA table_info(inspections)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "profile_name" in existing_cols or "board_aligned" in existing_cols:
            cursor.execute("DROP TABLE IF EXISTS inspections")

        # Inspections table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                profile_version INTEGER DEFAULT 1,
                verdict TEXT NOT NULL,
                verdict_reason TEXT NOT NULL,
                inspection_issue_summary TEXT DEFAULT NULL,
                reference_identity_ok INTEGER DEFAULT 1,
                image_quality_ok INTEGER DEFAULT 1,
                geometric_alignment_ok INTEGER DEFAULT 1,
                counts_summary_json TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                rectified_ref_b64 TEXT DEFAULT NULL,
                rectified_test_b64 TEXT DEFAULT NULL,
                annotated_b64 TEXT DEFAULT NULL,
                opencv_diff_b64 TEXT DEFAULT NULL,
                engine TEXT NOT NULL,
                model_identifier TEXT NOT NULL,
                inference_latency_ms REAL NOT NULL,
                total_latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL,
                ground_truth_verdict TEXT DEFAULT NULL,
                ground_truth_notes TEXT DEFAULT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
            )
            """
        )
        cursor.execute("PRAGMA table_info(inspections)")
        icols = {row[1] for row in cursor.fetchall()}
        if "profile_version" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN profile_version INTEGER DEFAULT 1")

        # Sanitize legacy provider naming in inspection records
        cursor.execute("UPDATE inspections SET engine = 'Inspection Engine' WHERE engine LIKE '%GenAI%' OR engine LIKE '%Google%'")
        cursor.execute("UPDATE inspections SET model_identifier = 'Inspection Engine' WHERE model_identifier LIKE '%gemini%'")
        if "inspection_issue_summary" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN inspection_issue_summary TEXT DEFAULT NULL")
        if "reference_identity_ok" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN reference_identity_ok INTEGER DEFAULT 1")
        if "image_quality_ok" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN image_quality_ok INTEGER DEFAULT 1")
        if "geometric_alignment_ok" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN geometric_alignment_ok INTEGER DEFAULT 1")
        if "counts_summary_json" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN counts_summary_json TEXT DEFAULT '{}'")
        if "rectified_ref_b64" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN rectified_ref_b64 TEXT DEFAULT NULL")
        if "rectified_test_b64" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN rectified_test_b64 TEXT DEFAULT NULL")
        if "annotated_b64" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN annotated_b64 TEXT DEFAULT NULL")
        if "opencv_diff_b64" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN opencv_diff_b64 TEXT DEFAULT NULL")
        if "inspection_state" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN inspection_state TEXT DEFAULT 'COMPLETED'")
        if "failure_category" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN failure_category TEXT DEFAULT NULL")
        if "request_id" not in icols:
            cursor.execute("ALTER TABLE inspections ADD COLUMN request_id TEXT DEFAULT NULL")

        # Durable stored images table (preserves reference & test images across serverless container restarts)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stored_images (
                image_id TEXT PRIMARY KEY,
                image_bytes BLOB NOT NULL,
                format TEXT DEFAULT 'PNG',
                content_hash TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute("PRAGMA table_info(stored_images)")
        sicols = {row[1] for row in cursor.fetchall()}
        if "content_hash" not in sicols:
            cursor.execute("ALTER TABLE stored_images ADD COLUMN content_hash TEXT DEFAULT NULL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stored_images_content_hash ON stored_images(content_hash)")

        conn.commit()


def _supabase_headers() -> dict[str, str]:
    return {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
    }


def _is_supabase_configured() -> bool:
    return bool(settings.supabase_url and settings.supabase_key)


def _cache_image_bytes(file_id: str, img_bytes: bytes) -> None:
    """Best-effort disk cache (handles read-only / serverless filesystems gracefully)."""
    try:
        img_dir = _get_images_dir()
        img_dir.mkdir(parents=True, exist_ok=True)
        filepath = img_dir / file_id
        if not filepath.exists():
            with open(filepath, "wb") as f:
                f.write(img_bytes)
    except Exception:
        pass


def save_image_file(image: Image.Image, prefix: str = "img") -> str:
    """
    Saves PIL Image durably to database (BLOB) and caches locally on disk.
    
    DEDUPLICATION:
    Computes canonical SHA-256 hash of image bytes. If an image with the exact same
    content has already been stored in Supabase or local SQLite, the existing image ID
    is reused and no duplicate row or binary BLOB is created, preventing Supabase database bloat.
    """
    # 1. Convert to canonical PNG bytes and compute SHA-256 hash
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    img_bytes = buf.getvalue()
    content_hash = hashlib.sha256(img_bytes).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()
    deterministic_id = f"{prefix}_{content_hash[:20]}.png"

    # 2. Check local SQLite for existing identical image
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_id FROM stored_images WHERE content_hash = ? OR image_id = ? LIMIT 1",
            (content_hash, deterministic_id),
        )
        row = cursor.fetchone()
        if row and row[0]:
            existing_id = row[0]
            logger.info(
                "Image deduplicated in local database (hash=%s, id=%s). Reusing without duplicate storage.",
                content_hash[:12],
                existing_id,
            )
            _cache_image_bytes(existing_id, img_bytes)
            return existing_id

    # 3. Check Supabase for existing identical image
    if _is_supabase_configured():
        try:
            headers = _supabase_headers()
            # Check by deterministic image_id (primary key)
            r = httpx.get(
                f"{settings.supabase_url}/rest/v1/stored_images?image_id=eq.{deterministic_id}&select=image_id&limit=1",
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200 and r.json():
                existing_id = r.json()[0]["image_id"]
                logger.info(
                    "Image deduplicated in Supabase by ID (hash=%s, id=%s). Reusing without duplicate storage.",
                    content_hash[:12],
                    existing_id,
                )
                _cache_image_bytes(existing_id, img_bytes)
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR IGNORE INTO stored_images (image_id, image_bytes, format, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                        (existing_id, img_bytes, "PNG", content_hash, now_iso),
                    )
                    conn.commit()
                return existing_id

            # Also check by content_hash column if present
            r_hash = httpx.get(
                f"{settings.supabase_url}/rest/v1/stored_images?content_hash=eq.{content_hash}&select=image_id&limit=1",
                headers=headers,
                timeout=5,
            )
            if r_hash.status_code == 200 and r_hash.json():
                existing_id = r_hash.json()[0]["image_id"]
                logger.info(
                    "Image deduplicated in Supabase by content_hash (hash=%s, id=%s). Reusing without duplicate storage.",
                    content_hash[:12],
                    existing_id,
                )
                _cache_image_bytes(existing_id, img_bytes)
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR IGNORE INTO stored_images (image_id, image_bytes, format, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                        (existing_id, img_bytes, "PNG", content_hash, now_iso),
                    )
                    conn.commit()
                return existing_id
        except Exception as exc:
            logger.debug("Supabase deduplication lookup exception: %s", exc)

    # 4. If image is genuinely new, insert into local database
    file_id = deterministic_id
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO stored_images (image_id, image_bytes, format, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_id, img_bytes, "PNG", content_hash, now_iso),
        )
        conn.commit()

    # 5. Insert new image into Supabase (if configured)
    if _is_supabase_configured():
        try:
            headers = {
                **_supabase_headers(),
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates",
            }
            payload = {
                "image_id": file_id,
                "image_bytes": "\\x" + img_bytes.hex(),
                "format": "PNG",
                "content_hash": content_hash,
                "created_at": now_iso,
            }
            res = httpx.post(
                f"{settings.supabase_url}/rest/v1/stored_images",
                headers=headers,
                json=payload,
                timeout=10,
            )
            if res.status_code == 400 and "content_hash" in res.text:
                # Schema in Supabase might not have content_hash column yet, retry without it
                payload.pop("content_hash", None)
                res = httpx.post(
                    f"{settings.supabase_url}/rest/v1/stored_images",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
            if res.status_code in (200, 201):
                logger.info("Persisted new non-duplicate image to Supabase: %s", file_id)
        except Exception as exc:
            logger.warning("Could not persist new image to Supabase: %s", exc)

    # 6. Cache to disk
    _cache_image_bytes(file_id, img_bytes)

    return file_id


def load_image_file(file_id: str) -> Image.Image | None:
    """
    Loads PIL Image from local cache if present, falling back to durable database BLOB storage
    or Supabase storage. Corrects EXIF orientation.
    """
    # 1. Try local disk cache
    try:
        img_dir = _get_images_dir()
        filepath = img_dir / file_id
        if filepath.exists():
            raw = Image.open(filepath)
            return ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        pass

    # 2. Try local database storage
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_bytes FROM stored_images WHERE image_id = ?", (file_id,))
        row = cursor.fetchone()
        if row and row[0]:
            raw_bytes = row[0]
            raw = Image.open(io.BytesIO(raw_bytes))
            img = ImageOps.exif_transpose(raw).convert("RGB")
            _cache_image_bytes(file_id, raw_bytes)
            return img

    # 3. Fall back to Supabase durable storage
    if _is_supabase_configured():
        try:
            headers = _supabase_headers()
            r = httpx.get(
                f"{settings.supabase_url}/rest/v1/stored_images?image_id=eq.{file_id}&select=image_bytes&limit=1",
                headers=headers,
                timeout=8,
            )
            if r.status_code == 200 and r.json():
                raw_hex = r.json()[0].get("image_bytes", "")
                if isinstance(raw_hex, str):
                    if raw_hex.startswith("\\x"):
                        raw_bytes = bytes.fromhex(raw_hex[2:])
                    else:
                        raw_bytes = bytes.fromhex(raw_hex)
                else:
                    raw_bytes = bytes(raw_hex)
                raw = Image.open(io.BytesIO(raw_bytes))
                img = ImageOps.exif_transpose(raw).convert("RGB")
                # Cache to local SQLite & disk
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR IGNORE INTO stored_images (image_id, image_bytes, format, created_at) VALUES (?, ?, ?, ?)",
                        (file_id, raw_bytes, "PNG", datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                _cache_image_bytes(file_id, raw_bytes)
                return img
        except Exception as exc:
            logger.warning("Could not load image from Supabase: %s", exc)

    return None


# ==============================================================================
# Profile Operations
# ==============================================================================

def create_or_save_profile(
    name: str,
    description: str = "",
    reference_image: Image.Image | None = None,
    reference_corners: WorkpieceCorners | None = None,
    checklist: list[CheckItem] | None = None,
    regions: list[ExpectedRegion] | None = None,
    profile_id: str | None = None,
) -> ProfileResponse:
    """Creates or updates a profile."""
    init_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    pid = profile_id or f"prof_{uuid.uuid4().hex[:8]}"

    existing = get_profile(pid)
    version = (existing.version if existing else 1)
    ref_image_id = existing.reference_image_id if existing else None
    ref_corners_to_save = reference_corners or (existing.reference_corners if existing else None)

    if reference_image is not None:
        ref_image_id = save_image_file(reference_image, prefix="ref")

    checklist_items = checklist if checklist is not None else (existing.checklist if existing else [])
    region_items = regions if regions is not None else (existing.regions if existing else [])

    cl_json = json.dumps([item.model_dump() for item in checklist_items])
    reg_json = json.dumps([region.model_dump() for region in region_items])
    corners_json = json.dumps(ref_corners_to_save.model_dump()) if ref_corners_to_save else None

    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO profiles 
            (profile_id, name, description, version, reference_image_id, reference_corners_json, checklist_json, regions_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                name,
                description,
                version,
                ref_image_id,
                corners_json,
                cl_json,
                reg_json,
                existing.created_at if existing else now_iso,
                now_iso,
            ),
        )
        conn.commit()

    if _is_supabase_configured():
        try:
            headers = {
                **_supabase_headers(),
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            payload = {
                "profile_id": pid,
                "name": name,
                "description": description or "",
                "version": version,
                "reference_image_id": ref_image_id,
                "reference_corners_json": corners_json,
                "checklist_json": cl_json,
                "regions_json": reg_json,
                "created_at": existing.created_at if existing else now_iso,
                "updated_at": now_iso,
            }
            httpx.post(f"{settings.supabase_url}/rest/v1/profiles", headers=headers, json=payload, timeout=6)
        except Exception as exc:
            logger.debug("Supabase profile sync error: %s", exc)

    return ProfileResponse(
        profile_id=pid,
        name=name,
        description=description,
        version=version,
        reference_image_id=ref_image_id,
        reference_corners=ref_corners_to_save,
        checklist=checklist_items,
        regions=region_items,
        created_at=existing.created_at if existing else now_iso,
        updated_at=now_iso,
    )


def update_profile_reference(
    profile_id: str,
    reference_image: Image.Image,
    reference_corners: WorkpieceCorners | None = None,
) -> ProfileResponse:
    """
    Requirement 1: Replacing a reference must invalidate its old regions
    and any cached inspection results. Associate each reference, checklist
    and set of regions with a profile version.
    """
    init_db()
    existing = get_profile(profile_id)
    if not existing:
        raise ValueError(f"Profile {profile_id} does not exist")

    now_iso = datetime.now(timezone.utc).isoformat()
    new_version = existing.version + 1
    ref_image_id = save_image_file(reference_image, prefix="ref")

    # Invalidate old regions on reference replacement
    cleared_regions: list[ExpectedRegion] = []
    reg_json = json.dumps([])
    cl_json = json.dumps([item.model_dump() for item in existing.checklist])
    corners_json = json.dumps(reference_corners.model_dump()) if reference_corners else None

    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE profiles 
            SET reference_image_id = ?, reference_corners_json = ?, version = ?, regions_json = ?, updated_at = ?
            WHERE profile_id = ?
            """,
            (ref_image_id, corners_json, new_version, reg_json, now_iso, profile_id),
        )
        # Invalidate old cached inspections for this profile by deleting them or separating by version
        cursor.execute("DELETE FROM inspections WHERE profile_id = ?", (profile_id,))
        conn.commit()

    if _is_supabase_configured():
        try:
            headers = {
                **_supabase_headers(),
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            payload = {
                "profile_id": profile_id,
                "name": existing.name,
                "description": existing.description or "",
                "version": new_version,
                "reference_image_id": ref_image_id,
                "reference_corners_json": corners_json,
                "checklist_json": cl_json,
                "regions_json": reg_json,
                "created_at": existing.created_at,
                "updated_at": now_iso,
            }
            httpx.post(f"{settings.supabase_url}/rest/v1/profiles", headers=headers, json=payload, timeout=6)
        except Exception as exc:
            logger.debug("Supabase profile sync error: %s", exc)

    return ProfileResponse(
        profile_id=profile_id,
        name=existing.name,
        description=existing.description,
        version=new_version,
        reference_image_id=ref_image_id,
        reference_corners=reference_corners,
        checklist=existing.checklist,
        regions=cleared_regions,
        created_at=existing.created_at,
        updated_at=now_iso,
    )


def save_profile(
    name: str,
    description: str,
    reference_image: Image.Image | None,
    checklist: list[CheckItem],
    regions: list[ExpectedRegion],
    reference_corners: WorkpieceCorners | None = None,
    profile_id: str | None = None,
) -> ProfileResponse:
    """Backwards compatible alias for create_or_save_profile."""
    return create_or_save_profile(
        name=name,
        description=description,
        reference_image=reference_image,
        reference_corners=reference_corners,
        checklist=checklist,
        regions=regions,
        profile_id=profile_id,
    )


def get_profile(profile_id: str) -> ProfileResponse | None:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT profile_id, name, description, version, reference_image_id, reference_corners_json, checklist_json, regions_json, created_at, updated_at FROM profiles WHERE profile_id = ?",
            (profile_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        pid, name, desc, ver, ref_id, corners_j, cl_json, reg_json, created, updated = row
        corners = WorkpieceCorners(**json.loads(corners_j)) if corners_j else None
        return ProfileResponse(
            profile_id=pid,
            name=name,
            description=desc or "",
            version=ver or 1,
            reference_image_id=ref_id,
            reference_corners=corners,
            checklist=[CheckItem(**item) for item in json.loads(cl_json)],
            regions=[ExpectedRegion(**r) for r in json.loads(reg_json)],
            created_at=created,
            updated_at=updated,
        )


def list_profiles() -> list[ProfileResponse]:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT profile_id, name, description, version, reference_image_id, reference_corners_json, checklist_json, regions_json, created_at, updated_at FROM profiles ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()

    profiles = []
    for row in rows:
        pid, name, desc, ver, ref_id, corners_j, cl_json, reg_json, created, updated = row
        corners = WorkpieceCorners(**json.loads(corners_j)) if corners_j else None
        profiles.append(
            ProfileResponse(
                profile_id=pid,
                name=name,
                description=desc or "",
                version=ver or 1,
                reference_image_id=ref_id,
                reference_corners=corners,
                checklist=[CheckItem(**item) for item in json.loads(cl_json)],
                regions=[ExpectedRegion(**r) for r in json.loads(reg_json)],
                created_at=created,
                updated_at=updated,
            )
        )
    return profiles


def delete_profile(profile_id: str) -> bool:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
        cursor.execute("DELETE FROM inspections WHERE profile_id = ?", (profile_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

    if _is_supabase_configured():
        try:
            headers = _supabase_headers()
            httpx.delete(f"{settings.supabase_url}/rest/v1/profiles?profile_id=eq.{profile_id}", headers=headers, timeout=6)
        except Exception as exc:
            logger.debug("Supabase profile delete error: %s", exc)

    return deleted


# ==============================================================================
# Inspection Operations
# ==============================================================================

def save_inspection(
    profile_id: str,
    profile_version: int,
    verdict: str,
    verdict_reason: str,
    findings: list[ComponentFinding],
    counts_summary: dict[str, int],
    engine: str,
    model_identifier: str,
    inference_latency_ms: float,
    total_latency_ms: float,
    inspection_state: str = "COMPLETED",
    failure_category: str | None = None,
    request_id: str | None = None,
    inspection_issue_summary: str | None = None,
    reference_identity_ok: bool = True,
    image_quality_ok: bool = True,
    geometric_alignment_ok: bool = True,
    rectified_ref_png_base64: str | None = None,
    rectified_test_png_base64: str | None = None,
    annotated_image_png_base64: str | None = None,
    opencv_diff_png_base64: str | None = None,
    ground_truth_verdict: str | None = None,
    ground_truth_notes: str | None = None,
) -> AssemblyInspectionResponse:
    init_db()
    inspection_id = f"insp_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    counts_json = json.dumps(counts_summary)
    findings_json = json.dumps([f.model_dump() for f in findings])

    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO inspections (
                inspection_id, profile_id, profile_version,
                verdict, verdict_reason, inspection_issue_summary,
                reference_identity_ok, image_quality_ok, geometric_alignment_ok,
                counts_summary_json, findings_json,
                rectified_ref_b64, rectified_test_b64, annotated_b64, opencv_diff_b64,
                engine, model_identifier, inference_latency_ms,
                total_latency_ms, created_at, ground_truth_verdict, ground_truth_notes,
                inspection_state, failure_category, request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                profile_id,
                profile_version,
                verdict,
                verdict_reason,
                inspection_issue_summary,
                1 if reference_identity_ok else 0,
                1 if image_quality_ok else 0,
                1 if geometric_alignment_ok else 0,
                counts_json,
                findings_json,
                rectified_ref_png_base64,
                rectified_test_png_base64,
                annotated_image_png_base64,
                opencv_diff_png_base64,
                engine,
                model_identifier,
                inference_latency_ms,
                total_latency_ms,
                now_iso,
                ground_truth_verdict,
                ground_truth_notes,
                inspection_state,
                failure_category,
                request_id,
            ),
        )
        conn.commit()

    if _is_supabase_configured():
        try:
            headers = {
                **_supabase_headers(),
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            # Deduplicate image bloat: Reference workpiece image is already stored durably
            # on the profile. Omit redundant duplicate full-resolution base64 strings
            # to prevent Supabase database storage from getting choked out.
            insp_payload = {
                "inspection_id": inspection_id,
                "profile_id": profile_id,
                "profile_version": profile_version,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "inspection_issue_summary": inspection_issue_summary,
                "reference_identity_ok": 1 if reference_identity_ok else 0,
                "image_quality_ok": 1 if image_quality_ok else 0,
                "geometric_alignment_ok": 1 if geometric_alignment_ok else 0,
                "counts_summary_json": counts_json,
                "findings_json": findings_json,
                "rectified_ref_b64": None,
                "rectified_test_b64": None,
                "annotated_b64": None,
                "opencv_diff_b64": None,
                "engine": engine,
                "model_identifier": model_identifier,
                "inference_latency_ms": inference_latency_ms,
                "total_latency_ms": total_latency_ms,
                "created_at": now_iso,
                "ground_truth_verdict": ground_truth_verdict,
                "ground_truth_notes": ground_truth_notes,
                "inspection_state": inspection_state,
                "failure_category": failure_category,
                "request_id": request_id,
            }
            httpx.post(f"{settings.supabase_url}/rest/v1/inspections", headers=headers, json=insp_payload, timeout=6)
        except Exception as exc:
            logger.debug("Supabase inspection sync error: %s", exc)

    return AssemblyInspectionResponse(
        inspection_id=inspection_id,
        profile_id=profile_id,
        profile_version=profile_version,
        inspection_state=inspection_state,  # type: ignore
        failure_category=failure_category,  # type: ignore
        request_id=request_id,
        verdict=verdict,  # type: ignore
        verdict_reason=verdict_reason,
        inspection_issue_summary=inspection_issue_summary,
        reference_identity_ok=reference_identity_ok,
        image_quality_ok=image_quality_ok,
        geometric_alignment_ok=geometric_alignment_ok,
        findings=findings,
        counts_summary=counts_summary,
        rectified_ref_png_base64=rectified_ref_png_base64,
        rectified_test_png_base64=rectified_test_png_base64,
        annotated_image_png_base64=annotated_image_png_base64,
        opencv_diff_png_base64=opencv_diff_png_base64,
        engine=engine,
        model_identifier=model_identifier,
        inference_latency_ms=inference_latency_ms,
        total_latency_ms=total_latency_ms,
        created_at=now_iso,
        ground_truth_verdict=ground_truth_verdict,
        ground_truth_notes=ground_truth_notes,
    )


def list_inspections(profile_id: str | None = None, limit: int = 100) -> list[AssemblyInspectionResponse]:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query_base = """
            SELECT inspection_id, profile_id, profile_version, verdict, verdict_reason,
                   inspection_issue_summary, reference_identity_ok, image_quality_ok, geometric_alignment_ok,
                   counts_summary_json, findings_json, rectified_ref_b64, rectified_test_b64,
                   annotated_b64, opencv_diff_b64, engine, model_identifier,
                   inference_latency_ms, total_latency_ms, created_at,
                   ground_truth_verdict, ground_truth_notes,
                   COALESCE(inspection_state, 'COMPLETED'), failure_category, request_id
            FROM inspections
        """
        if profile_id:
            cursor.execute(f"{query_base} WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?", (profile_id, limit))
        else:
            cursor.execute(f"{query_base} ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()

    results = []
    for row in rows:
        (
            insp_id, pid, pver, verd, vreason,
            issue_sum, ref_id_ok, img_q_ok, geo_ok,
            counts_j, find_j, ref_b64, test_b64,
            ann_b64, diff_b64, eng, model_id,
            inf_lat, tot_lat, created,
            gt_verd, gt_notes,
            insp_state, fail_cat, req_id,
        ) = row
        findings = [ComponentFinding(**f) for f in json.loads(find_j)]
        counts = json.loads(counts_j)
        results.append(
            AssemblyInspectionResponse(
                inspection_id=insp_id,
                profile_id=pid,
                profile_version=pver or 1,
                inspection_state=insp_state or "COMPLETED",
                failure_category=fail_cat,
                request_id=req_id,
                verdict=verd,  # type: ignore
                verdict_reason=vreason,
                inspection_issue_summary=issue_sum,
                reference_identity_ok=bool(ref_id_ok),
                image_quality_ok=bool(img_q_ok),
                geometric_alignment_ok=bool(geo_ok),
                findings=findings,
                counts_summary=counts,
                rectified_ref_png_base64=ref_b64,
                rectified_test_png_base64=test_b64,
                annotated_image_png_base64=ann_b64,
                opencv_diff_png_base64=diff_b64,
                engine=eng,
                model_identifier=model_id,
                inference_latency_ms=inf_lat,
                total_latency_ms=tot_lat,
                created_at=created,
                ground_truth_verdict=gt_verd,
                ground_truth_notes=gt_notes,
            )
        )
    return results


def update_ground_truth(inspection_id: str, expected_verdict: str, notes: str = "") -> bool:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE inspections
            SET ground_truth_verdict = ?, ground_truth_notes = ?
            WHERE inspection_id = ?
            """,
            (expected_verdict, notes, inspection_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_accuracy_report() -> AccuracyReportResponse:
    """Calculates accuracy and confusion matrix only from inspections with recorded ground truth."""
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verdict, ground_truth_verdict FROM inspections"
        )
        rows = cursor.fetchall()

    total = len(rows)
    gt_count = 0
    pass_cnt = sum(1 for r in rows if r[0] == "PASS")
    rej_cnt = sum(1 for r in rows if r[0] == "REJECT")
    rev_cnt = sum(1 for r in rows if r[0] == "REVIEW")

    correct = 0
    confusion: dict[str, dict[str, int]] = {
        "PASS": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
        "REJECT": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
        "REVIEW": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
    }

    for verdict, gt in rows:
        if gt:
            gt_count += 1
            if gt in confusion and verdict in confusion[gt]:
                confusion[gt][verdict] += 1
            if verdict == gt:
                correct += 1

    accuracy = (correct / gt_count) if gt_count > 0 else None

    return AccuracyReportResponse(
        total_inspections=total,
        ground_truth_count=gt_count,
        pass_count=pass_cnt,
        reject_count=rej_cnt,
        review_count=rev_cnt,
        accuracy=round(accuracy, 4) if accuracy is not None else None,
        confusion_matrix=confusion,
    )


def get_operational_statistics(days: int | None = None) -> OperationalStatisticsResponse:
    """
    Computes production operational statistics from stored inspection records.
    Requirements:
    - Total attempts & completed inspections.
    - PASS, REJECT, and REVIEW counts and rates among completed inspections.
    - Processing error count (separated from product defects).
    - Average processing latencies.
    - Inspection volume and verdict trends grouped by date.
    - Most frequently failed checks with failure rates.
    - Recent inspection results.
    - Ground truth metrics computed ONLY on independently labelled records (accuracy, FAR, FRR, coverage).
    - Empty state when no data exists.
    """
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = """
            SELECT inspection_id, profile_id, profile_version, verdict,
                   COALESCE(inspection_state, 'COMPLETED'), failure_category,
                   findings_json, inference_latency_ms, total_latency_ms,
                   created_at, ground_truth_verdict
            FROM inspections
        """
        params = []
        if days is not None and days > 0:
            cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            query += " WHERE created_at >= ?"
            params.append(cutoff_iso)
        query += " ORDER BY created_at DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

    total_attempts = len(rows)

    if total_attempts == 0:
        return OperationalStatisticsResponse(
            selected_range_days=days,
            total_attempts=0,
            completed_inspections=0,
            processing_errors_count=0,
            pass_count=0,
            reject_count=0,
            review_count=0,
            pass_rate_percent=None,
            reject_rate_percent=None,
            review_rate_percent=None,
            avg_inference_latency_ms=None,
            avg_total_latency_ms=None,
            volume_trends=[],
            frequently_failed_checks=[],
            recent_inspections=[],
            ground_truth=GroundTruthMetrics(
                total_labelled=0,
                total_unlabelled=0,
                accuracy_percent=None,
                false_accept_count=0,
                false_accept_rate_percent=None,
                false_reject_count=0,
                false_reject_rate_percent=None,
                review_coverage_percent=None,
                error_coverage_percent=None,
                confusion_matrix={
                    "PASS": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
                    "REJECT": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
                    "REVIEW": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
                },
            ),
        )

    completed_rows = [r for r in rows if r[4] == "COMPLETED"]
    error_rows = [r for r in rows if r[4] == "ERROR"]

    completed_count = len(completed_rows)
    error_count = len(error_rows)

    pass_count = sum(1 for r in completed_rows if r[3] == "PASS")
    reject_count = sum(1 for r in completed_rows if r[3] == "REJECT")
    review_count = sum(1 for r in completed_rows if r[3] == "REVIEW")

    pass_rate = round(pass_count / completed_count * 100.0, 1) if completed_count > 0 else None
    reject_rate = round(reject_count / completed_count * 100.0, 1) if completed_count > 0 else None
    review_rate = round(review_count / completed_count * 100.0, 1) if completed_count > 0 else None

    avg_inf = round(sum(r[7] for r in completed_rows) / completed_count, 1) if completed_count > 0 else None
    avg_tot = round(sum(r[8] for r in completed_rows) / completed_count, 1) if completed_count > 0 else None

    # Volume & Verdict Trends by Date
    date_groups: dict[str, dict[str, int]] = {}
    for r in rows:
        d = r[9][:10] if r[9] else "Unknown"
        if d not in date_groups:
            date_groups[d] = {"total": 0, "completed": 0, "pass": 0, "reject": 0, "review": 0, "error": 0}
        date_groups[d]["total"] += 1
        if r[4] == "ERROR":
            date_groups[d]["error"] += 1
        else:
            date_groups[d]["completed"] += 1
            v = r[3]
            if v == "PASS":
                date_groups[d]["pass"] += 1
            elif v == "REJECT":
                date_groups[d]["reject"] += 1
            elif v == "REVIEW":
                date_groups[d]["review"] += 1

    volume_trends = [
        TrendPoint(
            date=d,
            total_attempts=counts["total"],
            completed=counts["completed"],
            pass_count=counts["pass"],
            reject_count=counts["reject"],
            review_count=counts["review"],
            error_count=counts["error"],
        )
        for d, counts in sorted(date_groups.items())
    ]

    # Frequently Failed Checks
    fail_counts: dict[str, dict[str, str | int]] = {}
    for r in completed_rows:
        findings_raw = r[6]
        if findings_raw:
            try:
                findings_list = json.loads(findings_raw)
                for f in findings_list:
                    if f.get("status") == "failed":
                        cid = f.get("check_id") or f.get("region_id") or "check"
                        desc = f.get("description") or cid
                        key = f"{cid}::{desc}"
                        if key not in fail_counts:
                            fail_counts[key] = {"check_id": cid, "description": desc, "count": 0}
                        fail_counts[key]["count"] += 1
            except Exception:
                pass

    frequently_failed = []
    for item in sorted(fail_counts.values(), key=lambda x: int(x["count"]), reverse=True)[:10]:
        fail_cnt = int(item["count"])
        frequently_failed.append(
            FailedCheckStat(
                check_id=str(item["check_id"]),
                description=str(item["description"]),
                failure_count=fail_cnt,
                failure_rate_percent=round(fail_cnt / completed_count * 100.0, 1) if completed_count > 0 else 0.0,
            )
        )

    # Recent inspections
    recent_inspections = [
        RecentInspectionSummary(
            inspection_id=r[0],
            profile_id=r[1],
            profile_version=r[2],
            created_at=r[9],
            verdict=r[3],
            inspection_state=r[4],
            total_latency_ms=round(r[8], 1),
            ground_truth_verdict=r[10],
        )
        for r in rows[:10]
    ]

    # Ground Truth Benchmarking (strictly on labelled records)
    labelled_rows = [r for r in rows if r[10]]
    total_labelled = len(labelled_rows)
    total_unlabelled = total_attempts - total_labelled

    if total_labelled > 0:
        correct = sum(1 for r in labelled_rows if r[3] == r[10])
        accuracy_percent = round(correct / total_labelled * 100.0, 1)

        gt_rejects = [r for r in labelled_rows if r[10] == "REJECT"]
        fa_count = sum(1 for r in gt_rejects if r[3] == "PASS")
        fa_rate = round(fa_count / len(gt_rejects) * 100.0, 1) if gt_rejects else None

        gt_passes = [r for r in labelled_rows if r[10] == "PASS"]
        fr_count = sum(1 for r in gt_passes if r[3] == "REJECT")
        fr_rate = round(fr_count / len(gt_passes) * 100.0, 1) if gt_passes else None

        rev_cov = round(sum(1 for r in labelled_rows if r[3] == "REVIEW") / total_labelled * 100.0, 1)
        err_cov = round(sum(1 for r in labelled_rows if r[4] == "ERROR") / total_labelled * 100.0, 1)

        confusion = {
            "PASS": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
            "REJECT": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
            "REVIEW": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
        }
        for r in labelled_rows:
            gt = r[10]
            verd = r[3]
            if gt in confusion and verd in confusion[gt]:
                confusion[gt][verd] += 1

        gt_metrics = GroundTruthMetrics(
            total_labelled=total_labelled,
            total_unlabelled=total_unlabelled,
            accuracy_percent=accuracy_percent,
            false_accept_count=fa_count,
            false_accept_rate_percent=fa_rate,
            false_reject_count=fr_count,
            false_reject_rate_percent=fr_rate,
            review_coverage_percent=rev_cov,
            error_coverage_percent=err_cov,
            confusion_matrix=confusion,
        )
    else:
        gt_metrics = GroundTruthMetrics(
            total_labelled=0,
            total_unlabelled=total_attempts,
            accuracy_percent=None,
            false_accept_count=0,
            false_accept_rate_percent=None,
            false_reject_count=0,
            false_reject_rate_percent=None,
            review_coverage_percent=None,
            error_coverage_percent=None,
            confusion_matrix={
                "PASS": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
                "REJECT": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
                "REVIEW": {"PASS": 0, "REJECT": 0, "REVIEW": 0},
            },
        )

    return OperationalStatisticsResponse(
        selected_range_days=days,
        total_attempts=total_attempts,
        completed_inspections=completed_count,
        processing_errors_count=error_count,
        pass_count=pass_count,
        reject_count=reject_count,
        review_count=review_count,
        pass_rate_percent=pass_rate,
        reject_rate_percent=reject_rate,
        review_rate_percent=review_rate,
        avg_inference_latency_ms=avg_inf,
        avg_total_latency_ms=avg_tot,
        volume_trends=volume_trends,
        frequently_failed_checks=frequently_failed,
        recent_inspections=recent_inspections,
        ground_truth=gt_metrics,
    )
