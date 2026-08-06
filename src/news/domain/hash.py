from __future__ import annotations

import hashlib
import re


def normalize_source_id(source_id: str) -> str:
    return re.sub(r"\s+", " ", source_id.strip().lower())


def calculate_raw_content_hash(source_id: str, raw_content: str) -> str:
    payload = f"{normalize_source_id(source_id)}\0{raw_content}".encode()
    return hashlib.sha256(payload).hexdigest()
