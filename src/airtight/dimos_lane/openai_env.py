"""Load OPENAI_API_KEY from the HackMIT .env without logging the value."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ENV = Path("/home/ronil/HackMIT/.env")
_KEY_NAMES = ("OPENAI_API_KEY", "OPEN_AI_KEY")


def _load_key_from_file(path: Path) -> bool:
    if not path.is_file():
        return False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            name, value = line.split(":", 1)
        elif "=" in line:
            name, value = line.split("=", 1)
        else:
            continue
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export ") :].strip()
        if name in _KEY_NAMES:
            value = value.split(" #", 1)[0].strip().strip("'\"")
            if value:
                os.environ["OPENAI_API_KEY"] = value
                return True
    return False


def ensure_openai_key(*, env_path: Path | None = None, reload_file: bool = False) -> bool:
    """Set OPENAI_API_KEY in-process from env or a YAML/dotenv file.

    Returns True if a non-empty key is available afterwards. Never prints
    or writes the key.
    """
    if os.environ.get("OPENAI_API_KEY") and not reload_file:
        return True
    return _load_key_from_file(env_path or _DEFAULT_ENV)


def openai_auth_error(*, timeout: float = 20.0) -> str | None:
    """Return a short error token if the key is missing/rejected, else None.

    Does not print or return the key.
    """
    if not ensure_openai_key():
        return "missing"
    err = _probe_openai(timeout)
    if err == "401":
        os.environ.pop("OPENAI_API_KEY", None)
        if ensure_openai_key(reload_file=True):
            err = _probe_openai(timeout)
    return err


def _probe_openai(timeout: float) -> str | None:
    try:
        from openai import OpenAI

        OpenAI(timeout=timeout).models.retrieve("gpt-4o-mini")
    except Exception as exc:
        text = str(exc)
        if "401" in text or "Incorrect API key" in text:
            return "401"
        return type(exc).__name__
    return None
