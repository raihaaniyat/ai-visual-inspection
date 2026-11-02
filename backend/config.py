"""
Central configuration for AI-Visual-Inspector.

Supports both:
1. Gemini Vision mode (cloud-based assembly verification using Google GenAI SDK).
2. Legacy local Anomaly Detection + Ollama VLM mode (optional).

All sensitive credentials and paths are read from environment variables (.env).
"""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vision API (Server-side vision inspection)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_key_fallback: str = Field(default="", alias="GEMINI_API_KEY_FALLBACK")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")

    # Durable Storage: PostgreSQL / Supabase
    database_url: str = Field(default="", alias="DATABASE_URL")
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")

    # Security & Access Controls for public deployment
    inspection_api_key: str = Field(default="", alias="INSPECTION_API_KEY")
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    max_upload_size_bytes: int = Field(default=4194304, alias="MAX_UPLOAD_SIZE_BYTES")  # 4.0 MB limit for Vercel

    # Assembly Inspection Storage & Persistence
    storage_dir: str = Field(default="./backend/data/assembly_storage", alias="INSPECTOR_STORAGE_DIR")
    opencv_diff_threshold: int = Field(default=40, alias="OPENCV_DIFF_THRESHOLD")
    opencv_min_defect_area: int = Field(default=150, alias="OPENCV_MIN_DEFECT_AREA")

    # Legacy Ollama / Local VLM (optional)
    ollama_host: str = Field(default="http://localhost:11434", alias="INSPECTOR_OLLAMA_HOST")
    vlm_model: str = "qwen2.5vl:7b"
    vlm_enabled: bool = False

    # Legacy Anomaly detection (optional)
    ad_backend: str = "patchcore"  # "patchcore" | "efficient_ad"
    ad_checkpoint_dir: str = "./backend/data/checkpoints"
    anomaly_threshold: float = 0.5

    # Dataset
    mvtec_root: str = "./backend/data/mvtec_ad"
    mvtec_categories: str = "bottle,cable,metal_nut,screw"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Optional email notification
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    notify_email: str = ""

    @property
    def categories(self) -> list[str]:
        return [c.strip() for c in self.mvtec_categories.split(",") if c.strip()]

    @property
    def checkpoint_dir_path(self) -> Path:
        return Path(self.ad_checkpoint_dir)

    @property
    def mvtec_root_path(self) -> Path:
        return Path(self.mvtec_root)

    @property
    def storage_dir_path(self) -> Path:
        # In serverless environments (Vercel, AWS Lambda), root filesystem is read-only except /tmp
        if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            p = Path("/tmp/assembly_storage")
        else:
            p = Path(self.storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
