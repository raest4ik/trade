from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

from src.ai_events.application.ports import AIModelCompletion
from src.ai_events.domain.schema import AIEventOutput


class JsonFileAIEventCache:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def get(self, key: str) -> AIModelCompletion | None:
        return await asyncio.to_thread(self._read, key)

    async def put(self, key: str, completion: AIModelCompletion) -> None:
        await asyncio.to_thread(self._write, key, completion)

    def _read(self, key: str) -> AIModelCompletion | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        return AIModelCompletion(
            output=AIEventOutput.model_validate(payload["output"]),
            response_id=str(payload["response_id"]),
            actual_model=str(payload["actual_model"]),
            latency_ms=int(str(payload["latency_ms"])),
            input_tokens=_optional_int(payload.get("input_tokens")),
            output_tokens=_optional_int(payload.get("output_tokens")),
            total_tokens=_optional_int(payload.get("total_tokens")),
        )

    def _write(self, key: str, completion: AIModelCompletion) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        payload = {
            "actual_model": completion.actual_model,
            "input_tokens": completion.input_tokens,
            "latency_ms": completion.latency_ms,
            "output": completion.output.model_dump(mode="json"),
            "output_tokens": completion.output_tokens,
            "response_id": completion.response_id,
            "total_tokens": completion.total_tokens,
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}.json"


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))
