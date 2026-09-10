from __future__ import annotations

import json
import time
from typing import Any, cast

import httpx
from pydantic import ValidationError

from src.ai_trading_agent_v1.domain import (
    AgentModelRequest,
    AgentModelResponse,
    StructuredAgentOutput,
    ToolCallRequest,
)
from src.production_readonly_adapters_v1.domain import AGENT_ADAPTER_ID

MAX_RESPONSE_BYTES = 1_000_000


class OllamaAgentError(RuntimeError):
    pass


class OllamaAgentTimeoutError(OllamaAgentError):
    pass


class OllamaAgentUnavailableError(OllamaAgentError):
    pass


class OllamaAgentInvalidResponseError(OllamaAgentError):
    pass


class OllamaAgentModel:
    adapter_id = AGENT_ADAPTER_ID

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        think: bool,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
        random_seed: int,
        context_length: int,
        http_client: httpx.Client | None = None,
    ) -> None:
        parsed = httpx.URL(base_url.rstrip("/"))
        allowed_hosts = {"localhost", "127.0.0.1", "host.docker.internal"}
        if parsed.scheme not in {"http", "https"} or parsed.host not in allowed_hosts:
            raise ValueError("Ollama base URL must use localhost")
        if not model.strip():
            raise ValueError("Ollama model must not be empty")
        self._base_url = str(parsed).rstrip("/")
        self._model = model
        self._think = think
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._max_output_tokens = max_output_tokens
        self._random_seed = random_seed
        self._context_length = context_length
        self._client = http_client
        self.model_id = f"ollama:{model}"
        self.last_metadata: dict[str, Any] = {}

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        payload = self._payload(request)
        started = time.perf_counter()
        response = self._post_with_retries(payload)
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise OllamaAgentInvalidResponseError("OLLAMA_RESPONSE_TOO_LARGE")
        try:
            body = cast("dict[str, Any]", response.json())
            message = cast("dict[str, Any]", body["message"])
            result = _agent_response(message)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise OllamaAgentInvalidResponseError("OLLAMA_INVALID_RESPONSE") from exc
        self.last_metadata = {
            "adapter_id": self.adapter_id,
            "model": str(body.get("model") or self._model),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "prompt_eval_count": _optional_int(body.get("prompt_eval_count")),
            "eval_count": _optional_int(body.get("eval_count")),
        }
        return result

    def smoke(self) -> dict[str, Any]:
        response = self._post_with_retries(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": '{"status":"READY"}'},
                ],
                "stream": False,
                "think": False,
                "format": {"type": "object"},
                "options": {"temperature": 0, "num_predict": 32, "seed": self._random_seed},
            }
        )
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise OllamaAgentInvalidResponseError("OLLAMA_RESPONSE_TOO_LARGE")
        try:
            body = cast("dict[str, Any]", response.json())
            content = cast("dict[str, Any]", body["message"])["content"]
            if not isinstance(content, str) or not isinstance(json.loads(content), dict):
                raise TypeError("smoke response is not a JSON object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaAgentInvalidResponseError("OLLAMA_INVALID_SMOKE_RESPONSE") from exc
        return {"status": "READY", "model_id": self.model_id, "adapter_id": self.adapter_id}

    def _payload(self, request: AgentModelRequest) -> dict[str, Any]:
        user_payload: dict[str, Any] = {
            "prompt_version": request.prompt_version,
            "allowed_universe": request.allowed_universe,
            "transcript": request.transcript,
            "safety_policy": request.safety_policy,
            "response_contract": {
                "tool_calls": [{"name": "read_only_tool", "arguments": {}}],
                "final_output_schema": StructuredAgentOutput.model_json_schema(),
            },
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
            ],
            "stream": False,
            "think": self._think,
            "format": StructuredAgentOutput.model_json_schema(),
            "options": {
                "temperature": 0,
                "seed": self._random_seed,
                "num_ctx": self._context_length,
                "num_predict": self._max_output_tokens,
            },
        }
        tools = _read_only_tools(request.safety_policy)
        if tools:
            payload["tools"] = tools
        return payload

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._post(payload)
            except httpx.TimeoutException as exc:
                if attempt >= self._max_retries:
                    raise OllamaAgentTimeoutError("AGENT_MODEL_TIMEOUT") from exc
                continue
            except httpx.RequestError as exc:
                if attempt >= self._max_retries:
                    raise OllamaAgentUnavailableError("AGENT_MODEL_UNAVAILABLE") from exc
                continue
            if response.status_code >= 500 and attempt < self._max_retries:
                continue
            if response.status_code >= 400:
                raise OllamaAgentUnavailableError(f"OLLAMA_HTTP_{response.status_code}")
            return response
        raise OllamaAgentUnavailableError("AGENT_MODEL_UNAVAILABLE")

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self._base_url}/api/chat"
        if self._client is not None:
            return self._client.post(url, json=payload, timeout=self._timeout)
        with httpx.Client(
            timeout=self._timeout, headers={"User-Agent": AGENT_ADAPTER_ID}
        ) as client:
            return client.post(url, json=payload)


def _agent_response(message: dict[str, Any]) -> AgentModelResponse:
    native_calls = message.get("tool_calls")
    if native_calls:
        calls: list[ToolCallRequest] = []
        if not isinstance(native_calls, list):
            raise TypeError("tool_calls must be a list")
        for item in cast("list[dict[str, Any]]", native_calls):
            function = cast("dict[str, Any]", item["function"])
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCallRequest(name=str(function["name"]), arguments=arguments))
        return AgentModelResponse(tool_calls=calls)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise TypeError("message.content must be a non-empty string")
    parsed = json.loads(content)
    if isinstance(parsed, dict) and ({"tool_calls", "final_output"} & parsed.keys()):
        return AgentModelResponse.model_validate(parsed)
    return AgentModelResponse(final_output=json.dumps(parsed, ensure_ascii=False, sort_keys=True))


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_only_tools(safety_policy: dict[str, Any]) -> list[dict[str, Any]]:
    registered = safety_policy.get("registered_tools")
    if not isinstance(registered, list):
        return []
    forbidden = {str(name) for name in safety_policy.get("forbidden_tools", [])}
    tools: list[dict[str, Any]] = []
    for item in cast("list[object]", registered):
        if not isinstance(item, dict):
            continue
        tool = cast("dict[str, Any]", item)
        name = tool.get("name")
        parameters = tool.get("arguments_schema")
        if not isinstance(name, str) or not isinstance(parameters, dict) or name in forbidden:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Read-only deterministic operation context.",
                    "parameters": parameters,
                },
            }
        )
    return tools
