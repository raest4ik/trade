from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from src.evaluation.domain.enums import DatasetSplit


class FrozenTestGuardError(ValueError):
    pass


def validate_test_access(
    *,
    split: DatasetSplit,
    allow_frozen_test: bool,
    frozen_config_path: Path | None,
    expected_config: dict[str, object],
) -> dict[str, object] | None:
    if split != DatasetSplit.TEST:
        return None
    if not allow_frozen_test:
        raise FrozenTestGuardError("TEST requires --allow-frozen-test")
    if frozen_config_path is None:
        raise FrozenTestGuardError("TEST requires --frozen-config")
    if not frozen_config_path.is_file():
        raise FrozenTestGuardError("frozen config file does not exist")
    loaded = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise FrozenTestGuardError("frozen config must be a JSON object")
    config = cast("dict[str, object]", loaded)
    mismatches = [key for key, expected in expected_config.items() if config.get(key) != expected]
    if mismatches:
        raise FrozenTestGuardError("frozen config mismatch: " + ", ".join(sorted(mismatches)))
    return config
