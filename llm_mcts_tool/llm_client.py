from __future__ import annotations

import json
import os
import re
import shlex
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GLM_DEFAULT_MODEL = "glm-5-2-260617"
GLM_DEFAULT_API_KEY_ENV = "ARK_API_KEY"
GLM_DEFAULT_TIMEOUT = 1800
GLM_DEFAULT_MAX_TOKENS = 65536
GLM_DEFAULT_TEMPERATURE = 1.0


def is_glm_model(model: str | None) -> bool:
    name = str(model or "").strip().lower()
    return name.startswith("glm-")


GPT_LUNA_DEFAULT_MODEL = "gpt-5.6-luna"
GPT_LUNA_DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
GPT_LUNA_DEFAULT_BASE_URL = "http://63.141.241.66/v1"
GPT_LUNA_DEFAULT_TIMEOUT = 180
GPT_LUNA_DEFAULT_TEMPERATURE = 0.2
GPT_LUNA_API_URL_FILE = Path(__file__).resolve().parents[1] / "GPT-API" / "api-url.txt"
DASHSCOPE_COMPAT_URLS = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://llm-84jio1rtskombrcq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
}


def is_gpt_luna_model(model: str | None) -> bool:
    name = str(model or "").strip().lower()
    return name == GPT_LUNA_DEFAULT_MODEL or name.startswith("gpt-5.6")


class LLMClient:
    last_call: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        raise NotImplementedError


def load_env_file(path: Path | str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def load_gpt_api_url_file(path: Path | str | None = None) -> None:
    env_path = Path(path) if path is not None else GPT_LUNA_API_URL_FILE
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if not value:
            continue
        if key in {"api", "api_key", "openai_api_key"}:
            os.environ.setdefault("OPENAI_API_KEY", value)
        elif key in {"url", "base_url", "openai_base_url"}:
            os.environ.setdefault("OPENAI_BASE_URL", value.rstrip("/"))


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 120,
        max_retries: int = 0,
        retry_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        request_summary = {
            "url": f"{self.base_url}/chat/completions",
            "model": self.model,
            "messages": payload["messages"],
            "response_format": payload["response_format"],
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self.last_call = {
            "schema_name": schema_name,
            "request": request_summary,
            "attempts": [],
        }

        raw = ""
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                self.last_call["attempts"].append({"attempt": attempt + 1, "status": "success"})
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self.last_call["attempts"].append(
                    {"attempt": attempt + 1, "status": "http_error", "code": exc.code, "body": body}
                )
                raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
            except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
                self.last_call["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "status": "retryable_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt >= self.max_retries:
                    raise TimeoutError(
                        f"LLM API request timed out or failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(self.retry_backoff * float(attempt + 1))

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if self.last_call is not None:
            self.last_call["raw_response_text"] = raw
            self.last_call["raw_response_json"] = data
            self.last_call["message_content"] = content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON content for {schema_name}: {content[:500]}") from exc
        if self.last_call is not None:
            self.last_call["parsed_json"] = parsed
        return parsed


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                text = getattr(item, "text", None)
                parts.append(str(text) if text is not None else str(item))
        return "".join(parts)
    return str(content)


def _loads_json_object(content: str, schema_name: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM returned non-JSON content for {schema_name}: {text[:500]}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-object JSON for {schema_name}: {text[:500]}")
    return parsed


class GptLunaLLMClient(LLMClient):
    """OpenAI-compatible proxy client for gpt-5.6-luna. Does not change DashScope calls."""

    def __init__(
        self,
        *,
        model: str = GPT_LUNA_DEFAULT_MODEL,
        base_url: str = GPT_LUNA_DEFAULT_BASE_URL,
        api_key_env: str = GPT_LUNA_DEFAULT_API_KEY_ENV,
        timeout: int = GPT_LUNA_DEFAULT_TIMEOUT,
        max_retries: int = 0,
        retry_backoff: float = 2.0,
        temperature: float = GPT_LUNA_DEFAULT_TEMPERATURE,
        reasoning_effort: str | None = "low",
        urlopen: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = str(base_url).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.temperature = float(temperature)
        self.reasoning_effort = reasoning_effort
        self._urlopen = urlopen or urllib.request.urlopen

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        request_summary = {
            "provider": "gpt-luna",
            "url": f"{self.base_url}/chat/completions",
            "model": self.model,
            "messages": payload["messages"],
            "temperature": self.temperature,
            "reasoning_effort": payload.get("reasoning_effort"),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self.last_call = {
            "schema_name": schema_name,
            "request": request_summary,
            "attempts": [],
        }

        raw = ""
        data: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self._urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                self.last_call["attempts"].append({"attempt": attempt + 1, "status": "success"})
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self.last_call["attempts"].append(
                    {"attempt": attempt + 1, "status": "http_error", "code": exc.code, "body": body}
                )
                if (
                    "reasoning_effort" in payload
                    and exc.code in {400, 422}
                    and "reasoning" in body.lower()
                ):
                    payload.pop("reasoning_effort", None)
                    continue
                raise RuntimeError(f"GPT Luna HTTP {exc.code}: {body}") from exc
            except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
                self.last_call["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "status": "retryable_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt >= self.max_retries:
                    raise TimeoutError(
                        f"GPT Luna request timed out or failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(self.retry_backoff * float(attempt + 1))

        if data is None:
            raise RuntimeError("GPT Luna request returned no response")
        message = data["choices"][0]["message"]
        content = _message_text(message.get("content"))
        if not str(content or "").strip():
            content = _message_text(message.get("reasoning_content") or message.get("reasoning"))
        if self.last_call is not None:
            self.last_call["raw_response_text"] = raw
            self.last_call["raw_response_json"] = data
            self.last_call["message_content"] = content
        parsed = _loads_json_object(content, schema_name)
        if self.last_call is not None:
            self.last_call["parsed_json"] = parsed
        return parsed


class VolcengineArkLLMClient(LLMClient):
    """Volcengine Ark client for GLM 5.2. Used only when v2 `--provider glm`."""

    def __init__(
        self,
        *,
        model: str = GLM_DEFAULT_MODEL,
        api_key_env: str = GLM_DEFAULT_API_KEY_ENV,
        timeout: int = GLM_DEFAULT_TIMEOUT,
        max_retries: int = 0,
        retry_backoff: float = 2.0,
        thinking: bool = True,
        max_tokens: int = GLM_DEFAULT_MAX_TOKENS,
        temperature: float = GLM_DEFAULT_TEMPERATURE,
        ark_client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.thinking = bool(thinking)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._ark_client = ark_client

    def _ark(self) -> Any:
        if self._ark_client is not None:
            return self._ark_client
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise RuntimeError(
                "GLM 5.2 requires volcengine-python-sdk[ark]. "
                "Install with: pip install -U 'volcengine-python-sdk[ark]'"
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        self._ark_client = Ark(api_key=api_key, timeout=self.timeout)
        return self._ark_client

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request_summary = {
            "provider": "glm",
            "sdk": "volcenginesdkarkruntime",
            "model": self.model,
            "messages": payload["messages"],
            "thinking": payload["thinking"],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self.last_call = {
            "schema_name": schema_name,
            "request": request_summary,
            "attempts": [],
        }

        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._ark().chat.completions.create(**payload)
                self.last_call["attempts"].append({"attempt": attempt + 1, "status": "success"})
                break
            except (socket.timeout, TimeoutError) as exc:
                self.last_call["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "status": "retryable_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt >= self.max_retries:
                    raise TimeoutError(
                        f"GLM Ark request timed out after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(self.retry_backoff * float(attempt + 1))

        if response is None:
            raise RuntimeError("GLM Ark request returned no response")
        message = response.choices[0].message
        content = _message_text(getattr(message, "content", None))
        raw_json: Any
        if hasattr(response, "model_dump"):
            raw_json = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw_json = response.to_dict()
        else:
            raw_json = {"repr": repr(response)}
        if self.last_call is not None:
            self.last_call["raw_response_json"] = raw_json
            self.last_call["message_content"] = content
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                self.last_call["reasoning_content"] = reasoning
        parsed = _loads_json_object(content, schema_name)
        if self.last_call is not None:
            self.last_call["parsed_json"] = parsed
        return parsed
