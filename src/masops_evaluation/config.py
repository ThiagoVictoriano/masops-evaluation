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

    # Langfuse (read-only telemetry source for end-to-end token accounting) ---
    langfuse_url: str = Field(
        default="",
        description=(
            "Base URL of the self-hosted Langfuse instance. Defaults to "
            "http://{masops_host}:3000 when empty (Langfuse runs alongside "
            "MAS-Ops on the same EC2)."
        ),
    )
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public key (Basic Auth username). Optional.",
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse secret key (Basic Auth password). Optional.",
    )
    langfuse_project_id: str = Field(
        default="masops-project",
        description="Langfuse project identifier; used for logging context.",
    )

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

    def effective_langfuse_url(self) -> str:
        """Return the Langfuse base URL, deriving from ``masops_host`` when unset.

        Langfuse self-hosted runs on the same EC2 as MAS-Ops on port 3000, so
        the default is computed from ``masops_host`` rather than duplicated
        in the env file.
        """
        if self.langfuse_url:
            return self.langfuse_url.rstrip("/")
        if self.masops_host:
            return f"http://{self.masops_host}:3000"
        return ""

    def has_langfuse_credentials(self) -> bool:
        """Return True when both Langfuse keys are configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


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
