"""Config loading: YAML desk config + .env secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "desk.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env")
    with open(path) as f:
        return yaml.safe_load(f)


def require_secret(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing secret {name}: fill it in {REPO_ROOT / '.env'}"
        )
    return val
