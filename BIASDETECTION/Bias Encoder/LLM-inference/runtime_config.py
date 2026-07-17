#!/usr/bin/env python
"""Explicit, non-logging runtime configuration for retrieval operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class RuntimeConfigurationError(ValueError):
    """A required provider setting is absent or an env file is invalid."""


@dataclass(frozen=True)
class RuntimeReadiness:
    ready: bool
    required_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "required_keys": list(self.required_keys),
            "missing_keys": list(self.missing_keys),
        }


def load_explicit_env_file(path: Path | str | None) -> Path | None:
    """Load only an explicitly supplied dotenv file without overriding the process.

    Values are intentionally never returned or logged.  Importing this module has
    no configuration side effects.
    """

    if path is None:
        return None
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise RuntimeConfigurationError(f"Environment file does not exist: {env_path}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - exercised without optional deps
        raise RuntimeConfigurationError(
            "python-dotenv is required when --env-file is used; install requirements-qdrant.txt"
        ) from exc
    try:
        loaded = load_dotenv(dotenv_path=env_path, override=False, verbose=False)
    except Exception as exc:
        raise RuntimeConfigurationError("Could not parse the explicit environment file") from exc
    if not loaded and not _contains_assignments(env_path):
        raise RuntimeConfigurationError("The explicit environment file contains no assignments")
    return env_path


def provider_required_keys(mode: str, *, include_source_fetch: bool = True) -> tuple[str, ...]:
    normalized = str(mode or "").strip().casefold()
    keys: list[str] = []
    if normalized in {"web", "hybrid", "all"}:
        keys.append("BRAVE_SEARCH_API_KEY")
        if include_source_fetch:
            keys.append("CORPUS_FETCH_CONTACT")
    if normalized in {"corpus", "hybrid", "all"}:
        keys.extend(("QDRANT_URL", "QDRANT_API_KEY"))
    return tuple(dict.fromkeys(keys))


def check_runtime_readiness(
    mode: str = "all",
    *,
    environment: Mapping[str, str] | None = None,
    include_source_fetch: bool = True,
) -> RuntimeReadiness:
    env = environment if environment is not None else os.environ
    required = provider_required_keys(mode, include_source_fetch=include_source_fetch)
    missing = tuple(key for key in required if not str(env.get(key) or "").strip())
    return RuntimeReadiness(not missing, required, missing)


def require_runtime_readiness(
    mode: str = "all",
    *,
    environment: Mapping[str, str] | None = None,
    include_source_fetch: bool = True,
) -> None:
    readiness = check_runtime_readiness(
        mode,
        environment=environment,
        include_source_fetch=include_source_fetch,
    )
    if readiness.missing_keys:
        raise RuntimeConfigurationError(
            "Missing required environment variables: " + ", ".join(readiness.missing_keys)
        )


def redact_diagnostic_mapping(
    values: Mapping[str, object],
    *,
    secret_keys: Sequence[str] = (
        "authorization",
        "api_key",
        "key",
        "token",
        "x-subscription-token",
    ),
) -> dict[str, object]:
    """Return a shallow diagnostic copy with credential-shaped fields removed."""

    secrets = tuple(value.casefold() for value in secret_keys)
    return {
        str(key): ("[redacted]" if any(part in str(key).casefold() for part in secrets) else value)
        for key, value in values.items()
    }


def _contains_assignments(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RuntimeConfigurationError("Could not read the explicit environment file") from exc
    return any(
        "=" in line and line.strip() and not line.lstrip().startswith("#")
        for line in lines
    )

