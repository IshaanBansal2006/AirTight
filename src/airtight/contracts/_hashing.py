from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


def content_hash(model: BaseModel) -> str:
    """Stable 12-hex digest of a model's canonical JSON, used to key caches and logs."""
    payload = model.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
