"""Typed configuration loaded from environment variables / .env file.

Centralises every external knob the evaluation framework needs. Modules
should depend on :class:`Settings` instead of reading ``os.environ`` directly,
so that tests can construct an isolated config instance.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the evaluation framework.

    Values are read from environment variables, with a ``.env`` file at the
    repository root acting as a fallback. Field names match the variable
    names declared in ``.env.example`` (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MAS-Ops endpoint -------------------------------------------------------
    masops_host: str = Field(default="", description="Private VPC IP of EC2-mas-ops.")
    masops_port: int = Field(default=8000, description="Port of the MAS-Ops eval API.")
    masops_timeout: int = Field(default=1200, description="HTTP timeout in seconds.")
    masops_health_path: str = Field(default="/health", description="Health endpoint path.")

    # SWE-bench dataset / harness -------------------------------------------
    swebench_dataset: str = Field(
        default="princeton-nlp/SWE-bench_Verified",
        description="Hugging Face dataset identifier.",
    )
    swebench_harness_run_id_prefix: str = Field(
        default="eval",
        description="Prefix for the SWE-bench harness run_id.",
    )

    # LLM ---------------------------------------------------------------------
    anthropic_api_key: str = Field(default="", description="Anthropic API key for narratives.")

    # Local paths ------------------------------------------------------------
    results_dir: Path = Field(default=Path("./results"), description="Per-execution outputs.")
    consolidated_dir: Path = Field(
        default=Path("./consolidated"),
        description="Aggregated reports and charts.",
    )

    # Logging ----------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Python logging level name.")

    # --- Derived helpers ----------------------------------------------------

    def masops_url(self) -> str:
        """Return ``http://{host}:{port}`` with no trailing slash."""
        return f"http://{self.masops_host}:{self.masops_port}"

    def masops_eval_endpoint(self) -> str:
        """Return the full URL of the ``/eval/pr-review`` endpoint."""
        return f"{self.masops_url()}/eval/pr-review"

    def masops_health_endpoint(self) -> str:
        """Return the full URL of the health endpoint."""
        path = self.masops_health_path if self.masops_health_path.startswith("/") else (
            f"/{self.masops_health_path}"
        )
        return f"{self.masops_url()}{path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance (cached)."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging with a rich handler.

    Args:
        level: Optional log level override. Defaults to ``settings.log_level``.
    """
    from rich.logging import RichHandler

    effective_level = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=effective_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
